"""
Modality analysis for combo1 best model (bbb_train.py architecture, 0.879 AUC).

1. Embedding representational quality:
   - UMAP of each modality's projected token (512-d) colored by BBB+/-
   - Silhouette score per modality

2. Modality gating weights across 10 seeds:
   - skip_gate  : sigmoid scalar (backbone vs pre-backbone)
   - cross_pool_gate : sigmoid scalar (static alpha-pool vs cross-attn pool)
   - softmax(alpha)  : static per-modality weights (4 values)
   - mean cross-attn weights : data-dependent, averaged over all samples

Run:
  python analyze_modality.py
"""

import os, sys, pickle
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, '/home/minji/bbb-combo1')

# ---------------------------------------------------------------------------
# Reproduce model architecture from bbb_train.py (iter55+74+90 lineage)
# ---------------------------------------------------------------------------

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len, attn_drop=0.1):
        super().__init__()
        self.norm_u    = nn.LayerNorm(d_ffn)
        self.norm_v    = nn.LayerNorm(d_ffn)
        self.q_proj    = nn.Linear(d_ffn, d_ffn)
        self.k_proj    = nn.Linear(d_ffn, d_ffn)
        self.v_proj    = nn.Linear(d_ffn, d_ffn)
        nn.init.eye_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        import math
        self.log_temp  = nn.Parameter(torch.tensor(-0.5 * math.log(d_ffn)))
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        u = self.norm_u(u)
        v = self.norm_v(v)
        Q = self.q_proj(v); K = self.k_proj(v); V = self.v_proj(v)
        scores = (Q @ K.transpose(-1, -2)) * self.log_temp.exp()
        seq_len = scores.size(-1)
        diag = torch.eye(seq_len, device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(diag.unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        return u * (attn @ V)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim()-1)).bernoulli_(keep).div_(keep)
        return x * mask


class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, drop_path=0.0):
        super().__init__()
        self.norm          = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu           = SpatialGatingUnit(d_ffn, seq_len)
        self.drop_path     = DropPath(drop_path)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        return self.drop_path(x) + residual


class gMLP(nn.Module):
    def __init__(self, d_model=512, d_ffn=1048, seq_len=4, num_layers=4):
        super().__init__()
        dp_rates = [0.12 * i / max(1, num_layers - 1) for i in range(num_layers)]
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len, drop_path=dp) for dp in dp_rates]
        )

    def forward(self, x):
        return self.model(x)


class MultiModalGMLPFromFlat(nn.Module):
    def __init__(self, mod_dims: OrderedDict, d_model=512, d_ffn=1048, depth=4,
                 dropout=0.2, use_gated_pool=True):
        super().__init__()
        self.mod_names      = list(mod_dims.keys())
        self.mod_dims_list  = [mod_dims[n] for n in self.mod_names]
        self.seq_len        = len(self.mod_names)
        self.use_gated_pool = use_gated_pool

        self.proj = nn.ModuleDict({
            name: nn.Linear(in_dim, d_model)
            for name, in_dim in zip(self.mod_names, self.mod_dims_list)
        })
        self.backbone = gMLP(d_model=d_model, d_ffn=d_ffn,
                             seq_len=self.seq_len, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        if use_gated_pool:
            self.alpha           = nn.Parameter(torch.zeros(self.seq_len))
            self.q_pool          = nn.Parameter(torch.zeros(d_model))
            self.cross_pool_gate = nn.Parameter(torch.zeros(1))
        self.head      = nn.Linear(d_model, 1)
        self.drop      = nn.Dropout(dropout)
        self.skip_gate = nn.Parameter(torch.zeros(1))
        self._d_model  = d_model

    def forward(self, x):
        chunks = torch.split(x, self.mod_dims_list, dim=1)
        tokens = [self.proj[name](chunk)
                  for name, chunk in zip(self.mod_names, chunks)]
        X0 = torch.stack(tokens, dim=1)
        X  = self.backbone(X0)
        gate = torch.sigmoid(self.skip_gate)
        X = (1.0 - gate) * X + gate * X0
        if self.use_gated_pool:
            w      = torch.softmax(self.alpha, dim=0)
            Xp_gated = (X * w.view(1, -1, 1)).sum(dim=1)
            scores   = (X @ self.q_pool) / (self._d_model ** 0.5)
            attn_w   = torch.softmax(scores, dim=1)
            Xp_ca    = (X * attn_w.unsqueeze(-1)).sum(dim=1)
            cgate    = torch.sigmoid(self.cross_pool_gate)
            Xp = (1.0 - cgate) * Xp_gated + cgate * Xp_ca
        else:
            Xp = X.mean(dim=1)
        Xp = self.drop(self.norm(Xp))
        return self.head(Xp).squeeze(-1)

    def get_projected_tokens(self, x):
        """Return pre-backbone projected tokens (B, seq_len, d_model) — no grad."""
        chunks = torch.split(x, self.mod_dims_list, dim=1)
        tokens = [self.proj[name](chunk)
                  for name, chunk in zip(self.mod_names, chunks)]
        return torch.stack(tokens, dim=1)  # (B, seq_len, d_model)

    def get_pool_weights(self, x):
        """Return (alpha_w, cross_attn_w, skip_g, cross_g) for a batch."""
        chunks = torch.split(x, self.mod_dims_list, dim=1)
        tokens = [self.proj[name](chunk)
                  for name, chunk in zip(self.mod_names, chunks)]
        X0 = torch.stack(tokens, dim=1)
        X  = self.backbone(X0)
        gate  = torch.sigmoid(self.skip_gate)
        X = (1.0 - gate) * X + gate * X0

        alpha_w = torch.softmax(self.alpha, dim=0).detach().cpu()   # (seq_len,)
        scores  = (X @ self.q_pool) / (self._d_model ** 0.5)
        cross_w = torch.softmax(scores, dim=1).detach().cpu()       # (B, seq_len)
        skip_g  = torch.sigmoid(self.skip_gate).item()
        cross_g = torch.sigmoid(self.cross_pool_gate).item()
        return alpha_w.numpy(), cross_w.numpy(), skip_g, cross_g


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARTIFACT_DIR = '/home/minji/bbb-combo1/bbb_artifacts/best'
SEEDS        = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
MOD_DIMS     = OrderedDict([('maccs', 166), ('avalon', 512), ('rdkit', 217), ('mole', 768)])
MOD_NAMES    = list(MOD_DIMS.keys())
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_PATH   = '/home/minji/BBB/holdout_splits/internal_curated_holdout_10pct_seed42/internal_curated_label_remaining.csv'

# ---------------------------------------------------------------------------
# Load data from dataset cache
# ---------------------------------------------------------------------------

def load_data():
    cache_path = '/home/minji/bbb-combo1/dataset_cache/internal.pt'
    data = torch.load(cache_path, weights_only=False)
    X = data.features   # (N, total_dim)
    y = data.labels     # (N,)
    return X, y

# ---------------------------------------------------------------------------
# Load one seed's model
# ---------------------------------------------------------------------------

def load_model(seed, mod_dims=MOD_DIMS):
    seed_dir = os.path.join(ARTIFACT_DIR, f'seed_{seed}')
    cfg_path  = os.path.join(seed_dir, 'config.json')
    import json
    with open(cfg_path) as f:
        cfg = json.load(f)
    bc = cfg.get('base_config', cfg.get('hparams', {}))

    model = MultiModalGMLPFromFlat(
        mod_dims   = mod_dims,
        d_model    = bc.get('d_model', 512),
        d_ffn      = bc.get('d_ffn', 1048),
        depth      = bc.get('depth', 4),
        dropout    = bc.get('dropout', 0.2),
        use_gated_pool = True,
    )
    state = torch.load(os.path.join(seed_dir, 'model.pth'), map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    model.eval()

    # Apply rdkit scaler
    scaler_path = os.path.join(seed_dir, 'rdkit_scaler.pkl')
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    return model, scaler

def apply_scaler(X, scaler):
    if scaler is None:
        return X
    # rdkit slice: maccs(166) + avalon(512) = 678, then rdkit(217)
    rd_start, rd_end = 166 + 512, 166 + 512 + 217
    X = X.clone()
    X[:, rd_start:rd_end] = torch.tensor(
        scaler.transform(X[:, rd_start:rd_end].numpy()), dtype=torch.float32
    )
    return X

# ---------------------------------------------------------------------------
# Analysis 1: Embedding representational quality (UMAP)
# ---------------------------------------------------------------------------

def analyze_embeddings(X, y):
    print('\n[1/2] Embedding representational quality (UMAP)...')
    try:
        import umap
    except ImportError:
        print('  umap-learn not installed. Skipping UMAP. Install with: pip install umap-learn')
        return
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler as SKScaler

    # Use seed_42 model for projections
    model, scaler = load_model(42)
    Xs = apply_scaler(X, scaler)

    with torch.no_grad():
        tokens = model.get_projected_tokens(Xs.to(device)).cpu().numpy()  # (N, 4, 512)

    labels = y.numpy().astype(int)
    label_names = {0: 'BBB-', 1: 'BBB+'}
    colors = {0: '#E74C3C', 1: '#2ECC71'}

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Modality Projected Token Space (UMAP, 512-d → 2-d)\ncolored by BBB label',
                 fontsize=13, fontweight='bold')

    sil_scores = {}
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)

    for i, (name, ax) in enumerate(zip(MOD_NAMES, axes)):
        emb = tokens[:, i, :]  # (N, 512)
        emb_scaled = SKScaler().fit_transform(emb)
        umap_2d = reducer.fit_transform(emb_scaled)

        for lbl in [0, 1]:
            mask = labels == lbl
            ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                       c=colors[lbl], label=label_names[lbl],
                       alpha=0.4, s=8, rasterized=True)

        sil = silhouette_score(umap_2d, labels)
        sil_scores[name] = sil
        ax.set_title(f'{name}\nSilhouette={sil:.3f}', fontsize=11)
        ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
        ax.legend(markerscale=3, fontsize=9)
        print(f'  {name}: silhouette={sil:.4f}')

    plt.tight_layout()
    out = '/home/minji/bbb-combo1/modality_umap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')
    return sil_scores


# ---------------------------------------------------------------------------
# Analysis 2: Modality gating weights across 10 seeds
# ---------------------------------------------------------------------------

def analyze_gating(X, y):
    print('\n[2/2] Modality gating weights (10 seeds)...')

    all_alpha   = []   # (10, 4) — static softmax(alpha) weights
    all_cross   = []   # (10, 4) — mean cross-attn weights per modality
    all_skip    = []   # (10,)   — skip_gate scalar
    all_cgate   = []   # (10,)   — cross_pool_gate scalar

    for seed in SEEDS:
        model, scaler = load_model(seed)
        Xs = apply_scaler(X, scaler).to(device)
        model = model.to(device)
        with torch.no_grad():
            alpha_w, cross_w, skip_g, cross_g = model.get_pool_weights(Xs)
        all_alpha.append(alpha_w)
        all_cross.append(cross_w.mean(axis=0))  # average over samples
        all_skip.append(skip_g)
        all_cgate.append(cross_g)
        print(f'  seed={seed:>4d}  skip_gate={skip_g:.3f}  cross_gate={cross_g:.3f}  '
              f'alpha={alpha_w.round(3)}', flush=True)

    all_alpha = np.array(all_alpha)   # (10, 4)
    all_cross = np.array(all_cross)   # (10, 4)
    all_skip  = np.array(all_skip)    # (10,)
    all_cgate = np.array(all_cgate)   # (10,)

    # Combined effective weight per modality
    # effective_w = (1-cross_gate) * alpha_w + cross_gate * cross_w
    eff = (1 - all_cgate[:, None]) * all_alpha + all_cgate[:, None] * all_cross  # (10, 4)

    print('\n  --- Gating Summary (mean ± std across 10 seeds) ---')
    print(f'  skip_gate  (backbone mix)  : {all_skip.mean():.4f} ± {all_skip.std():.4f}')
    print(f'  cross_pool_gate (CA ratio) : {all_cgate.mean():.4f} ± {all_cgate.std():.4f}')
    print()
    print(f'  {"Modality":<10} {"alpha_w":>10} {"cross_attn_w":>14} {"effective_w":>13}')
    print('  ' + '-'*52)
    for j, name in enumerate(MOD_NAMES):
        a = f'{all_alpha[:,j].mean():.4f}±{all_alpha[:,j].std():.4f}'
        c = f'{all_cross[:,j].mean():.4f}±{all_cross[:,j].std():.4f}'
        e = f'{eff[:,j].mean():.4f}±{eff[:,j].std():.4f}'
        print(f'  {name:<10} {a:>10}   {c:>14}   {e:>13}')

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Modality Gating Weights — combo1 best (10 seeds)', fontsize=13, fontweight='bold')
    bar_colors = ['#3498DB', '#E67E22', '#2ECC71', '#9B59B6']

    def bar_with_err(ax, vals, title, ylabel):
        mu  = vals.mean(axis=0)
        std = vals.std(axis=0)
        bars = ax.bar(MOD_NAMES, mu, yerr=std, color=bar_colors,
                      capsize=6, alpha=0.85, edgecolor='black', linewidth=0.7)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, min(1.0, mu.max() * 1.5 + std.max()))
        for bar, m, s in zip(bars, mu, std):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.005,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    bar_with_err(axes[0], all_alpha, 'Static Alpha Weights\nsoftmax(α)', 'weight')
    bar_with_err(axes[1], all_cross, 'Cross-Attn Weights\nmean over samples', 'weight')
    bar_with_err(axes[2], eff,       'Effective Weights\n(1-cgate)·α + cgate·CA', 'weight')

    # Scalar gates as text
    fig.text(0.5, -0.02,
             f'skip_gate: {all_skip.mean():.3f}±{all_skip.std():.3f}  |  '
             f'cross_pool_gate: {all_cgate.mean():.3f}±{all_cgate.std():.3f}',
             ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    out = '/home/minji/bbb-combo1/modality_gating.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved: {out}')

    return {
        'alpha': all_alpha, 'cross': all_cross, 'effective': eff,
        'skip_gate': all_skip, 'cross_pool_gate': all_cgate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Loading dataset cache...')
    X, y = load_data()
    print(f'  X={X.shape}, y={y.shape}, BBB+={int(y.sum())}, BBB-={int((y==0).sum())}')

    sil = analyze_embeddings(X, y)
    gating = analyze_gating(X, y)

    print('\nDone.')

if __name__ == '__main__':
    main()
