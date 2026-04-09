"""
BBB gMLP model — architecture optimization.
This is the file the autoresearch agent modifies.

The agent may freely change:
- Model architecture: SpatialGatingUnit, gMLPBlock, gMLP, MultiModalGMLPFromFlat
- Training details inside train_model() (optimizer type, scheduler, etc.)

The agent must NOT change:
- bbb_prepare.py (data loading, splitting, evaluation)
- Any hyperparameter value in BASE_CONFIG
- The output format (the --- block at the end of __main__)

Usage:
    conda run -n rapids-25.02 python bbb_train.py > bbb_run_combo<N>.log 2>&1
"""

import os
import sys
import json
import pickle
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.optim as optim
from bbb_prepare import *  # fixed utilities, constants, device
from copy import deepcopy

# Artifact output directory (current run — agent copies to best/ on keep)
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bbb_artifacts', 'current')

# Dataset cache directory — computed once, reloaded on every subsequent run
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_cache')

# Time budget per seed training loop (seconds).
# Acts as a safeguard against architectures that are too heavy for this dataset.
# Normal training completes in ~20-40s/seed; 300s allows ~10x headroom.
TIME_BUDGET = 300

# ---------------------------------------------------------------------------
# Fixed hyperparameters — DO NOT MODIFY
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    'd_model':      512,
    'd_ffn':        1048,
    'depth':        4,
    'dropout':      0.2,
    'lr':           1e-4,
    'weight_decay': 1e-5,
    'num_epochs':   NUM_EPOCHS,   # 50, from bbb_prepare
    'patience':     PATIENCE,     # 10, from bbb_prepare
    'batch_size':   BATCH_SIZE,   # 128, from bbb_prepare
}

SEEDS       = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
SPLIT_MODE  = 'scaffold'   # fixed — random_scaffold not used for architecture search

# External remaining dataset paths (fixed — do not modify)
_EXT_DIR = '/home/minji/BBB/holdout_splits/external_cls_only_holdout_10pct_seed42'
EXT_LABEL_PATH  = f'{_EXT_DIR}/external_cls_only_label_remaining.csv'
EXT_EMBED_PATHS = {
    'scage1': f'{_EXT_DIR}/external_cls_only_scage1_remaining.csv',
    'scage2': f'{_EXT_DIR}/external_cls_only_scage2_remaining.csv',
    'mole':   f'{_EXT_DIR}/external_cls_only_mole_remaining.csv',
}

# Merged holdout dataset paths (fixed — do not modify)
_HOLDOUT_DIR        = '/home/minji/BBB/holdout_splits/merged_holdout_10pct_seed42'
HOLDOUT_LABEL_PATH  = f'{_HOLDOUT_DIR}/label_holdout.csv'
HOLDOUT_EMBED_PATHS = {
    'scage1': f'{_HOLDOUT_DIR}/scage1_holdout.csv',
    'scage2': f'{_HOLDOUT_DIR}/scage2_holdout.csv',
    'mole':   f'{_HOLDOUT_DIR}/mole_holdout.csv',
}

# ---------------------------------------------------------------------------
# Model Architecture  (agent modifies this section)
# ---------------------------------------------------------------------------

class SpatialGatingUnit(nn.Module):
    """Attention-based SGU with pre-norm on both u and v + learned temperature + attn dropout.

    Builds on ae66a27 (0.8563): pre-norm u+v + stoch_depth 0.05 + learned_temp + pos_weight.
    Adds attention dropout (p=0.1) to the attention weights for additional regularization.
    With seq_len=4, dropout on a 4x4 attention matrix adds noise to cross-modal mixing.
    iter57: mask diagonal in attention (no self-attention; force cross-modal only).
    """
    def __init__(self, d_ffn, seq_len, attn_drop=0.1):
        super().__init__()
        self.norm_v    = nn.LayerNorm(d_ffn)   # normalize v (gate computation)
        self.norm_u    = nn.LayerNorm(d_ffn)   # normalize u (input signal)
        self.d_ffn     = d_ffn
        # Learnable log-temperature: init so scale ≈ 1/sqrt(d_ffn) at start
        import math
        self.log_temp  = nn.Parameter(torch.tensor(-0.5 * math.log(d_ffn)))
        self.attn_drop = nn.Dropout(attn_drop)
        # Self-attention Q, K, V projections for the v gate
        self.q_proj    = nn.Linear(d_ffn, d_ffn)
        self.k_proj    = nn.Linear(d_ffn, d_ffn)
        self.v_proj    = nn.Linear(d_ffn, d_ffn)
        # Init near-identity for stable start
        nn.init.eye_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)        # (B, seq_len, d_ffn) each
        u = self.norm_u(u)               # normalize u before gating
        v = self.norm_v(v)               # normalize v before attention
        # Self-attention gating with learned temperature + attention dropout
        Q = self.q_proj(v)               # (B, seq_len, d_ffn)
        K = self.k_proj(v)               # (B, seq_len, d_ffn)
        V = self.v_proj(v)               # (B, seq_len, d_ffn)
        scale = self.log_temp.exp()      # learned scalar temperature
        scores = Q @ K.transpose(-1, -2) * scale   # (B, seq_len, seq_len)
        # Mask diagonal to force cross-modal attention only (no self-attention)
        seq_len = scores.size(-1)
        diag_mask = torch.eye(seq_len, device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))
        attn  = torch.softmax(scores, dim=-1)       # (B, seq_len, seq_len)
        attn  = self.attn_drop(attn)     # dropout on attention weights
        v_out = attn @ V                 # (B, seq_len, d_ffn)
        return u * v_out


class gMLPBlock(nn.Module):
    """gMLP block with optional stochastic depth regularization.

    Stochastic depth: during training, the block output is dropped with
    probability `drop_prob` (residual path always preserved). At inference,
    the full block runs. Linear schedule: earlier layers have lower drop prob.
    """
    def __init__(self, d_model, d_ffn, seq_len, drop_prob=0.0):
        super().__init__()
        self.norm          = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu           = SpatialGatingUnit(d_ffn, seq_len)
        self.drop_prob     = drop_prob

    def forward(self, x):
        residual = x
        if self.training and self.drop_prob > 0.0:
            # Bernoulli drop: entire block dropped for some samples in batch
            keep_prob = 1.0 - self.drop_prob
            shape     = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1)
            mask      = torch.rand(shape, device=x.device) < keep_prob
            # Scale surviving blocks to keep expectation correct
            if not mask.any():
                return residual
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        if self.training and self.drop_prob > 0.0:
            x = x * mask / keep_prob
        return x + residual


class gMLP(nn.Module):
    def __init__(self, d_model=512, d_ffn=1048, seq_len=4, num_layers=4,
                 stochastic_depth_rate=0.1):
        super().__init__()
        # Linear schedule: block 0 gets 0, block (num_layers-1) gets max rate
        drop_probs = [stochastic_depth_rate * i / max(num_layers - 1, 1)
                      for i in range(num_layers)]
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len, drop_prob=dp)
              for dp in drop_probs]
        )

    def forward(self, x):
        return self.model(x)


class MultiModalGMLPFromFlat(nn.Module):
    def __init__(self, mod_dims: OrderedDict,
                 d_model=512, d_ffn=1048, depth=4,
                 dropout=0.2, use_gated_pool=True,
                 stochastic_depth_rate=0.05):
        super().__init__()
        self.mod_names      = list(mod_dims.keys())
        self.mod_dims       = [mod_dims[n] for n in self.mod_names]
        self.seq_len        = len(self.mod_names)
        self.use_gated_pool = use_gated_pool

        self.proj = nn.ModuleDict({
            name: nn.Linear(in_dim, d_model)
            for name, in_dim in zip(self.mod_names, self.mod_dims)
        })
        self.backbone = gMLP(
            d_model=d_model, d_ffn=d_ffn,
            seq_len=self.seq_len, num_layers=depth,
            stochastic_depth_rate=stochastic_depth_rate,
        )
        self.norm = nn.LayerNorm(d_model)
        if use_gated_pool:
            self.alpha = nn.Parameter(torch.zeros(self.seq_len))
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        chunks = torch.split(x, self.mod_dims, dim=1)
        tokens = [self.proj[name](chunk)
                  for name, chunk in zip(self.mod_names, chunks)]
        X = torch.stack(tokens, dim=1)          # (B, seq_len, d_model)
        X = self.backbone(X)
        if self.use_gated_pool:
            w  = torch.softmax(self.alpha, dim=0)
            Xp = (X * w.view(1, -1, 1)).sum(dim=1)
        else:
            Xp = X.mean(dim=1)
        Xp = self.drop(self.norm(Xp))
        return self.head(Xp).squeeze(-1)


# ---------------------------------------------------------------------------
# Training loop  (agent may modify optimizer, scheduler, etc.)
# ---------------------------------------------------------------------------

def train_model(model, optimizer, train_loader, val_loader, loss_fn,
                num_epochs=BASE_CONFIG['num_epochs'],
                patience=BASE_CONFIG['patience']):
    best_val   = float('inf')
    best_state = None
    bad        = 0
    t_start    = time.time()

    for epoch in range(num_epochs):
        elapsed = time.time() - t_start
        if elapsed > TIME_BUDGET:
            print(f'    [timeout] TIME_BUDGET={TIME_BUDGET}s exceeded at epoch {epoch} ({elapsed:.1f}s elapsed)')
            break

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += loss_fn(model(x), y).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = deepcopy(model.state_dict())
            bad        = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Dataset caching — computed once per worktree, reloaded on every run
# ---------------------------------------------------------------------------

def load_cached_dataset(cache_name, smiles_path, embed_paths, fp_types, expected_dims=None):
    """Load dataset from cache if available, otherwise compute and cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f'{cache_name}.pt')
    if os.path.exists(cache_path):
        print(f'  [cache] Loading {cache_name} from {cache_path}')
        return torch.load(cache_path, weights_only=False)
    print(f'  [cache] Computing {cache_name} (will be cached for future runs)...')
    dataset = ScageConcatDataset(smiles_path, embed_paths, fp_types, expected_dims)
    torch.save(dataset, cache_path)
    print(f'  [cache] Saved to {cache_path}')
    return dataset


# ---------------------------------------------------------------------------
# Helper: apply scaler to rdkit slice of a raw feature tensor
# ---------------------------------------------------------------------------

def apply_scaler(X: torch.Tensor, scaler, rd_start, rd_end) -> torch.Tensor:
    """Return a scaled copy of X. Safe to call with scaler=None."""
    if scaler is None or rd_start is None:
        return X
    X = X.clone()
    X[:, rd_start:rd_end] = torch.tensor(
        scaler.transform(X[:, rd_start:rd_end].numpy()),
        dtype=torch.float32,
    )
    return X


# ---------------------------------------------------------------------------
# Evaluation: scaffold split, 10 seeds
# ---------------------------------------------------------------------------

def run_evaluation(dataset, ext_dataset, holdout_dataset, mod_dims):
    """
    Train with scaffold split across all SEEDS.
    Returns:
        mean internal test ROC-AUC  (float)
        mean external ROC-AUC       (float)
        mean holdout ROC-AUC        (float)
    """
    int_aucs, ext_aucs, holdout_aucs = [], [], []

    for seed in SEEDS:
        set_seed(seed)
        train_ds, val_ds, test_ds, scaler, (rd_start, rd_end), _ = split_then_normalize(
            dataset, split_mode=SPLIT_MODE,
            train_ratio=0.8, val_ratio=0.1, seed=seed,
        )

        # Apply same scaler (fitted on this seed's train set) to external and holdout
        ext_X     = apply_scaler(ext_dataset.features,     scaler, rd_start, rd_end)
        holdout_X = apply_scaler(holdout_dataset.features, scaler, rd_start, rd_end)

        bs = BASE_CONFIG['batch_size']
        train_loader   = data.DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=4)
        val_loader     = data.DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=4)
        test_loader    = data.DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=4)
        ext_loader     = data.DataLoader(
            data.TensorDataset(ext_X,     ext_dataset.labels),
            batch_size=bs, shuffle=False, num_workers=4,
        )
        holdout_loader = data.DataLoader(
            data.TensorDataset(holdout_X, holdout_dataset.labels),
            batch_size=bs, shuffle=False, num_workers=4,
        )

        set_seed(seed)
        model = MultiModalGMLPFromFlat(
            mod_dims=mod_dims,
            d_model=BASE_CONFIG['d_model'],
            d_ffn=BASE_CONFIG['d_ffn'],
            depth=BASE_CONFIG['depth'],
            dropout=BASE_CONFIG['dropout'],
            use_gated_pool=True,
            stochastic_depth_rate=0.05,
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=BASE_CONFIG['lr'],
            weight_decay=BASE_CONFIG['weight_decay'],
        )
        # pos_weight: fixed at 0.12 (continuing lower from 0.16-keep)
        pos_weight = torch.tensor([0.12]).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model = train_model(model, optimizer, train_loader, val_loader, loss_fn)

        int_auc     = eval_model(model, test_loader)['roc_auc']
        ext_auc     = eval_model(model, ext_loader)['roc_auc']
        holdout_auc = eval_model(model, holdout_loader)['roc_auc']

        int_aucs.append(int_auc)
        ext_aucs.append(ext_auc)
        holdout_aucs.append(holdout_auc)
        print(f"  seed={seed:>4d}  int={int_auc:.4f}  ext={ext_auc:.4f}  holdout={holdout_auc:.4f}")

        # ── Save artifacts for this seed ──────────────────────────────────
        seed_dir = os.path.join(ARTIFACT_DIR, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(seed_dir, 'model.pth'))

        if scaler is not None:
            with open(os.path.join(seed_dir, 'rdkit_scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)

        config = {
            'seed':        seed,
            'split_mode':  SPLIT_MODE,
            'fp_types':    FP_TYPES,
            'mod_dims':    {k: int(v) for k, v in mod_dims.items()},
            'base_config': {k: (v if not hasattr(v, '__name__') else str(v))
                            for k, v in BASE_CONFIG.items()},
            'metrics': {
                'roc_auc_scaffold': int_auc,
                'roc_auc_external': ext_auc,
                'roc_auc_holdout':  holdout_auc,
            },
        }
        with open(os.path.join(seed_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        # ─────────────────────────────────────────────────────────────────

        del model, optimizer
        del train_loader, val_loader, test_loader, ext_loader, holdout_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean = lambda lst: float(sum(lst) / len(lst))
    return mean(int_aucs), mean(ext_aucs), mean(holdout_aucs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t_start = time.time()

    print('Loading internal dataset...')
    dataset       = load_cached_dataset('internal', LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    expected_dims = dataset.expected_dims
    mod_dims      = OrderedDict((t, expected_dims[t]) for t in FP_TYPES)
    print(f'Internal : {len(dataset)} samples, feature_dim={dataset.features.shape[1]}')

    print('Loading external dataset...')
    ext_dataset = load_cached_dataset(
        'external', EXT_LABEL_PATH, EXT_EMBED_PATHS,
        fp_types=FP_TYPES, expected_dims=expected_dims,
    )
    print(f'External : {len(ext_dataset)} samples')

    print('Loading holdout dataset...')
    holdout_dataset = load_cached_dataset(
        'holdout', HOLDOUT_LABEL_PATH, HOLDOUT_EMBED_PATHS,
        fp_types=FP_TYPES, expected_dims=expected_dims,
    )
    print(f'Holdout  : {len(holdout_dataset)} samples')
    print(f'Mod dims : {dict(mod_dims)}')
    print(f'Split    : {SPLIT_MODE}, Seeds: {SEEDS}')
    print(f'BASE_CONFIG: {BASE_CONFIG}')

    print(f"\n{'='*60}")
    print(f"Evaluating | split_mode={SPLIT_MODE} | n_seeds={len(SEEDS)}")
    print('='*60)
    int_auc, ext_auc, holdout_auc = run_evaluation(
        dataset, ext_dataset, holdout_dataset, mod_dims,
    )
    print(f"\nMean  int={int_auc:.4f}  ext={ext_auc:.4f}  holdout={holdout_auc:.4f}")

    # -----------------------------------------------------------------------
    # Output block — DO NOT CHANGE FORMAT (parsed by autoresearch loop)
    # -----------------------------------------------------------------------
    peak_vram_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024
                    if torch.cuda.is_available() else 0.0)
    print('\n---')
    print(f'roc_auc_scaffold:    {int_auc:.6f}')
    print(f'roc_auc_external:    {ext_auc:.6f}')
    print(f'roc_auc_holdout:     {holdout_auc:.6f}')
    print(f'total_seconds:       {time.time() - t_start:.1f}')
    print(f'peak_vram_mb:        {peak_vram_mb:.1f}')
    print(f'n_seeds:             {len(SEEDS)}')
