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
import shutil
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.optim as optim
from bbb_prepare import *  # fixed utilities, constants, device
from copy import deepcopy
from sklearn.metrics import roc_auc_score

# Artifact directories — keep a consistent best/current/archive layout.
ARTIFACT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bbb_artifacts')
BEST_ARTIFACT_DIR = os.path.join(ARTIFACT_ROOT, 'best')
CURRENT_ARTIFACT_DIR = os.path.join(ARTIFACT_ROOT, 'current')
ARCHIVE_ARTIFACT_DIR = os.path.join(ARTIFACT_ROOT, 'archive')
ARTIFACT_DIR = CURRENT_ARTIFACT_DIR

# Dataset cache directory — computed once, reloaded on every subsequent run
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_cache')

# Time budget per seed training loop (seconds).
# Acts as a safeguard against architectures that are too heavy for this dataset.
# Normal training completes in ~20-40s/seed; 300s allows ~10x headroom.
TIME_BUDGET = 300


def _unique_archive_path(name: str) -> str:
    base = os.path.join(ARCHIVE_ARTIFACT_DIR, name)
    if not os.path.exists(base):
        return base
    idx = 1
    while True:
        candidate = f'{base}_{idx}'
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def ensure_artifact_layout():
    expected_seed_dirs = {f'seed_{seed}' for seed in SEEDS}
    os.makedirs(BEST_ARTIFACT_DIR, exist_ok=True)
    os.makedirs(CURRENT_ARTIFACT_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_ARTIFACT_DIR, exist_ok=True)

    readme_path = os.path.join(ARTIFACT_ROOT, 'README.txt')
    readme = (
        'Layout:\n'
        '  best/    active kept best seed artifacts\n'
        '  current/ latest run artifacts before keep decision\n'
        '  archive/ older snapshots or non-standard directories moved aside\n'
    )
    with open(readme_path, 'w') as f:
        f.write(readme)

    for name in os.listdir(ARTIFACT_ROOT):
        if name in {'best', 'current', 'archive', 'README.txt'}:
            continue
        src = os.path.join(ARTIFACT_ROOT, name)
        shutil.move(src, _unique_archive_path(name))

    for parent in [BEST_ARTIFACT_DIR, CURRENT_ARTIFACT_DIR]:
        for name in os.listdir(parent):
            if name in expected_seed_dirs:
                continue
            src = os.path.join(parent, name)
            shutil.move(src, _unique_archive_path(f'{os.path.basename(parent)}_{name}'))

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
    """2-head attention-based SGU with learnable diagonal bias mask.
    Diagonal bias encourages each modality token to focus on nearby tokens,
    acting as a soft locality prior while still allowing global attention.
    Attention dropout p=0.1 per head.
    """
    def __init__(self, d_ffn, seq_len, n_heads=2):
        super().__init__()
        assert d_ffn % n_heads == 0
        self.norm      = nn.LayerNorm(d_ffn)
        self.d_ffn     = d_ffn
        self.n_heads   = n_heads
        self.head_dim  = d_ffn // n_heads
        self.seq_len   = seq_len
        # Self-attention Q, K, V projections for the v gate
        self.q_proj    = nn.Linear(d_ffn, d_ffn)
        self.k_proj    = nn.Linear(d_ffn, d_ffn)
        self.v_proj    = nn.Linear(d_ffn, d_ffn)
        self.attn_drop = nn.Dropout(p=0.1)
        # Learnable diagonal bias: shape (seq_len, seq_len), init to 0
        # At softmax, this adds a learned locality prior to attention logits
        self.diag_bias = nn.Parameter(torch.zeros(seq_len, seq_len))
        # Init near-identity for stable start
        nn.init.eye_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)        # (B, seq_len, d_ffn) each
        B, S, D = v.shape
        v = self.norm(v)
        # Multi-head self-attention gating
        Q = self.q_proj(v).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, hd)
        K = self.k_proj(v).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** 0.5
        logits = Q @ K.transpose(-1, -2) / scale  # (B, H, S, S)
        # Add learnable diagonal bias (broadcast over batch and heads)
        logits = logits + self.diag_bias.unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(logits, dim=-1)  # (B, H, S, S)
        attn = self.attn_drop(attn)
        v_out = (attn @ V).transpose(1, 2).contiguous().view(B, S, D)  # (B, S, d_ffn)
        return u * v_out


class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, drop_path_prob=0.0):
        super().__init__()
        self.norm          = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu           = SpatialGatingUnit(d_ffn, seq_len)
        self.drop_path_prob = drop_path_prob

    def forward(self, x):
        residual = x
        out = self.norm(x)
        out = F.gelu(self.channel_proj1(out))
        out = self.sgu(out)
        out = self.channel_proj2(out)
        # Stochastic depth: randomly skip block during training
        if self.training and self.drop_path_prob > 0.0:
            # Sample a single Bernoulli per batch item
            keep = torch.bernoulli(
                torch.full((x.size(0), 1, 1), 1.0 - self.drop_path_prob, device=x.device)
            )
            out = out * keep / (1.0 - self.drop_path_prob)
        return out + residual


class gMLP(nn.Module):
    def __init__(self, d_model=512, d_ffn=1048, seq_len=4, num_layers=4,
                 drop_path_prob=0.1):
        super().__init__()
        # Linearly increase drop_path_prob from 0 to drop_path_prob across layers
        dpr = [drop_path_prob * i / max(num_layers - 1, 1)
               for i in range(num_layers)]
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len, drop_path_prob=dpr[i])
              for i in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)


class MultiModalGMLPFromFlat(nn.Module):
    def __init__(self, mod_dims: OrderedDict,
                 d_model=512, d_ffn=1048, depth=4,
                 dropout=0.2, use_gated_pool=True):
        super().__init__()
        self.mod_names      = list(mod_dims.keys())
        self.mod_dims       = [mod_dims[n] for n in self.mod_names]
        self.seq_len        = len(self.mod_names)
        self.use_gated_pool = use_gated_pool

        self.proj = nn.ModuleDict({
            name: nn.Linear(in_dim, d_model)
            for name, in_dim in zip(self.mod_names, self.mod_dims)
        })
        # Pre-projection LN only for continuous embed modalities (not binary fps)
        _embed_mods_set = {'scage1', 'scage2', 'mole'}
        self.embed_prenorm = nn.ModuleDict({
            name: nn.LayerNorm(in_dim)
            for name, in_dim in zip(self.mod_names, self.mod_dims)
            if name in _embed_mods_set
        })
        self.backbone = gMLP(
            d_model=d_model, d_ffn=d_ffn,
            seq_len=self.seq_len, num_layers=depth,
        )
        self.norm = nn.LayerNorm(d_model)
        # Attention pooling: input-dependent query replaces static gated pool
        self.pool_query = nn.Parameter(torch.zeros(d_model))
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)
        self.skip_gate = nn.Parameter(torch.zeros(1))  # gate_init=0; learned convex mix of backbone + pre-backbone
        # Dual-residual: embed and fp token means added to pooled output (init 0.1 each)
        _embed_mods = {'scage1', 'scage2', 'mole'}
        self._n_fp = sum(1 for n in self.mod_names if n not in _embed_mods)
        self.em_gate = nn.Parameter(torch.tensor([0.1]))
        self.fp_gate = nn.Parameter(torch.tensor([0.1]))
        self.em_max_gate = nn.Parameter(torch.tensor([0.1]))
        self.std_gate = nn.Parameter(torch.tensor([0.1]))
        self.token_scale = nn.Parameter(torch.ones(self.seq_len))
        # Parameter-free cross-attention: fp tokens attend to embed tokens (init=0 gate)
        self.cross_gate = nn.Parameter(torch.zeros(1))
        # Hadamard product residual: fp_mean * em_mean (multiplicative cross-modal interaction)
        self.cross_prod_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        chunks = torch.split(x, self.mod_dims, dim=1)
        tokens = []
        for name, chunk in zip(self.mod_names, chunks):
            if name in self.embed_prenorm:
                chunk = self.embed_prenorm[name](chunk)
            tokens.append(self.proj[name](chunk))
        X0 = torch.stack(tokens, dim=1)         # (B, seq_len, d_model) — pre-backbone tokens
        # fp tokens attend to embed tokens via raw dot-product (no projection weights)
        fp_t = X0[:, :self._n_fp, :]
        em_t = X0[:, self._n_fp:, :]
        scale_ca = X0.shape[-1] ** 0.5
        ca_attn = torch.softmax(fp_t @ em_t.transpose(-1, -2) / scale_ca, dim=-1)  # (B, n_fp, n_em)
        fp_cross = ca_attn @ em_t                                                    # (B, n_fp, d)
        fp_t = fp_t + self.cross_gate * fp_cross
        X0 = torch.cat([fp_t, em_t], dim=1)
        X  = self.backbone(X0)
        gate = torch.sigmoid(self.skip_gate)
        X = (1.0 - gate) * X + gate * X0        # learned convex combination
        X = X * self.token_scale.unsqueeze(0).unsqueeze(-1)  # per-position scale
        # Attention pooling: scores = softmax(X @ q / sqrt(d))
        scale = X.shape[-1] ** 0.5
        scores = torch.softmax(X @ self.pool_query / scale, dim=1)  # (B, seq_len)
        Xp = (scores.unsqueeze(-1) * X).sum(dim=1)                  # (B, d_model)
        Xp_em = X[:, self._n_fp:, :].mean(dim=1)           # embed mean
        Xp_em_max = X[:, self._n_fp:, :].max(dim=1).values  # embed max
        Xp_fp = X[:, :self._n_fp, :].mean(dim=1)           # fp mean
        Xp_std = X.std(dim=1)                               # token std (diversity signal)
        Xp_cross_prod = Xp_fp * Xp_em                       # multiplicative fp x embed interaction
        Xp = Xp + self.em_gate * Xp_em + self.em_max_gate * Xp_em_max + self.fp_gate * Xp_fp + self.std_gate * Xp_std + self.cross_prod_gate * Xp_cross_prod
        Xp = self.drop(self.norm(Xp))
        return self.head(Xp).squeeze(-1)


# ---------------------------------------------------------------------------
# Training loop  (agent may modify optimizer, scheduler, etc.)
# ---------------------------------------------------------------------------

def train_model(model, optimizer, train_loader, val_loader, loss_fn,
                num_epochs=BASE_CONFIG['num_epochs'],
                patience=BASE_CONFIG['patience']):
    best_val   = -1.0   # best val_roc_auc (higher = better)
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
            optimizer.step()

        # Early stopping on val_roc_auc (more robust to scaffold distribution shift than val_loss)
        model.eval()
        y_true_val, y_prob_val = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                prob = torch.sigmoid(model(x)).cpu().numpy()
                y_prob_val.extend(prob.tolist())
                y_true_val.extend(y.numpy().tolist())
        val_auc = roc_auc_score(y_true_val, y_prob_val) if len(set(y_true_val)) > 1 else 0.0

        if val_auc > best_val:
            best_val   = val_auc
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


def predict_probs(model, loader) -> np.ndarray:
    """Return sigmoid probabilities for the full loader in dataset order."""
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            probs.extend(torch.sigmoid(model(x)).cpu().numpy())
    return np.asarray(probs, dtype=np.float32)


def roc_auc_from_probs(y_true: torch.Tensor, probs: np.ndarray) -> float:
    y_true_np = y_true.cpu().numpy()
    if len(set(y_true_np.tolist())) <= 1:
        return 0.0
    return round(float(roc_auc_score(y_true_np, probs)), 4)


# ---------------------------------------------------------------------------
# Evaluation: scaffold split, 10 seeds
# ---------------------------------------------------------------------------

def run_evaluation(dataset, ext_dataset, holdout_dataset, mod_dims):
    """
    Train with scaffold split across all SEEDS.
    Returns:
        mean internal test ROC-AUC           (float)
        soft-voting ensemble external ROC-AUC (float)
        soft-voting ensemble holdout ROC-AUC  (float)
    """
    int_aucs = []
    ext_seed_probs, holdout_seed_probs = [], []

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
        ).to(device)

        lr = BASE_CONFIG['lr']
        wd = BASE_CONFIG['weight_decay']
        fast_names = {'em_gate', 'fp_gate', 'em_max_gate', 'std_gate',
                      'token_scale', 'skip_gate', 'pool_query', 'cross_gate', 'cross_prod_gate'}
        fast_params, base_params = [], []
        for name, p in model.named_parameters():
            if any(fn in name for fn in fast_names):
                fast_params.append(p)
            else:
                base_params.append(p)
        optimizer = optim.Adam(
            [{'params': base_params, 'lr': lr},
             {'params': fast_params, 'lr': lr * 1.2}],
            weight_decay=wd,
            amsgrad=True,
        )
        # pos_weight: iter71: try 0.20 (tuning below 0.22)
        pos_weight = torch.tensor([0.20]).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model = train_model(model, optimizer, train_loader, val_loader, loss_fn)

        int_auc = eval_model(model, test_loader)['roc_auc']
        ext_probs = predict_probs(model, ext_loader)
        holdout_probs = predict_probs(model, holdout_loader)
        ext_auc = roc_auc_from_probs(ext_dataset.labels, ext_probs)
        holdout_auc = roc_auc_from_probs(holdout_dataset.labels, holdout_probs)

        int_aucs.append(int_auc)
        ext_seed_probs.append(ext_probs)
        holdout_seed_probs.append(holdout_probs)
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
    ext_ensemble_probs = np.mean(np.stack(ext_seed_probs, axis=0), axis=0)
    holdout_ensemble_probs = np.mean(np.stack(holdout_seed_probs, axis=0), axis=0)
    ext_auc = roc_auc_from_probs(ext_dataset.labels, ext_ensemble_probs)
    holdout_auc = roc_auc_from_probs(holdout_dataset.labels, holdout_ensemble_probs)
    print(f"  [ensemble] ext={ext_auc:.4f}  holdout={holdout_auc:.4f}")
    return mean(int_aucs), ext_auc, holdout_auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t_start = time.time()
    ensure_artifact_layout()

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
