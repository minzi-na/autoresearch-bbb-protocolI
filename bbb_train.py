"""
BBB gMLP model — architecture optimization (Phase 1 v2 starting point).

Architecture: original gMLP (a489d67 — Conv1d SGU, gated_pool, no SiLU/RMSNorm).
Training dynamics: original (Adam, val_loss ES, no scheduler, no smoothing, no grad clip).
Loss: BCEWithLogitsLoss with pos_weight = n_neg/n_pos (auto-computed per seed).
Eval: internal scaffold test 10-seed mean; ext/holdout 10-seed soft voting ensemble.

This is the file the autoresearch agent modifies.

The agent may freely change:
- Model architecture: SpatialGatingUnit, gMLPBlock, gMLP, MultiModalGMLPFromFlat

The agent must NOT change:
- bbb_prepare.py (data loading, splitting, evaluation)
- Any hyperparameter value in BASE_CONFIG
- pos_weight (auto-computed; do not hardcode or tune)
- Training dynamics (Adam optimizer, val_loss ES, etc. — locked at original gMLP baseline)
- The output format (the --- block at the end of __main__, 9 metrics)

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
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'bbb_artifacts', 'current')

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
    """iter52: iter90-style enhanced attention SGU.
    Pre-norm u+v, Q/K/V proj with V identity init, learnable log-temperature,
    diagonal-mask (force cross-modal), attention dropout.
    """
    def __init__(self, d_ffn, seq_len, attn_drop=0.1):
        super().__init__()
        import math
        self.norm_u    = nn.LayerNorm(d_ffn)
        self.norm_v    = nn.LayerNorm(d_ffn)
        self.q_proj    = nn.Linear(d_ffn, d_ffn)
        self.k_proj    = nn.Linear(d_ffn, d_ffn)
        self.v_proj    = nn.Linear(d_ffn, d_ffn)
        nn.init.eye_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        self.log_temp  = nn.Parameter(torch.tensor(-0.5 * math.log(d_ffn)))
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        u = self.norm_u(u)
        v = self.norm_v(v)
        Q = self.q_proj(v)
        K = self.k_proj(v)
        V = self.v_proj(v)
        scores = (Q @ K.transpose(-1, -2)) * self.log_temp.exp()
        # diagonal mask: force cross-modal attention only
        seq_len = scores.size(-1)
        diag = torch.eye(seq_len, device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(diag.unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        v_out = attn @ V
        return u * v_out


class DropPath(nn.Module):
    """iter10: per-sample stochastic depth on residual branch."""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)
        return x * mask


class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, drop_path=0.1):
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
        # iter21: linear-decay drop_path 0 -> 0.12
        dp_rates = [0.12 * i / max(1, num_layers - 1) for i in range(num_layers)]
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len, drop_path=dp) for dp in dp_rates]
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
        self.backbone = gMLP(
            d_model=d_model, d_ffn=d_ffn,
            seq_len=self.seq_len, num_layers=depth,
        )
        self.norm = nn.LayerNorm(d_model)
        if use_gated_pool:
            self.alpha = nn.Parameter(torch.zeros(self.seq_len))
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)
        # iter55: skip connection sigmoid convex combine (init=0 → pure backbone)
        self.skip_gate = nn.Parameter(torch.zeros(1))
        # iter73: cross-modality self-context FiLM (between proj and backbone)
        # Output: gamma+beta per modality token. zero-init → identity (gamma=1, beta=0)
        self.film_mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2 * self.seq_len * d_model),
        )
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

    def forward(self, x):
        chunks = torch.split(x, self.mod_dims, dim=1)
        tokens = [self.proj[name](chunk)
                  for name, chunk in zip(self.mod_names, chunks)]
        X0 = torch.stack(tokens, dim=1)         # (B, seq_len, d_model) - pre-backbone
        # iter73: self-context FiLM modulation
        ctx = X0.mean(dim=1)                                # (B, d_model)
        film = self.film_mlp(ctx)                           # (B, 2*seq_len*d_model)
        film = film.view(film.size(0), 2, self.seq_len, -1) # (B, 2, seq_len, d_model)
        gamma = film[:, 0]                                  # (B, seq_len, d_model)
        beta  = film[:, 1]
        X0 = (1.0 + gamma) * X0 + beta                      # identity at init
        X = self.backbone(X0)
        # iter55: sigmoid convex combine — init=0 → pure backbone, learns to mix in X0
        gate = torch.sigmoid(self.skip_gate)
        X = (1.0 - gate) * X + gate * X0
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
# Helpers for ensemble evaluation
# ---------------------------------------------------------------------------

def predict_probs(model, loader) -> np.ndarray:
    """Return sigmoid probabilities for the full loader in dataset order."""
    model.eval()
    probs = []
    with torch.no_grad():
        for x, *_ in loader:
            probs.extend(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.asarray(probs, dtype=np.float32)


def metrics_from_probs(y_true: torch.Tensor, probs: np.ndarray, threshold: float = 0.5):
    """Return ROC-AUC + threshold-based MCC/F1/ACC from probs."""
    y_true_np = y_true.cpu().numpy().ravel()
    y_pred = (probs > threshold).astype(int)
    return {
        'roc_auc':  float(roc_auc_score(y_true_np, probs)),
        'mcc':      float(matthews_corrcoef(y_true_np, y_pred)),
        'f1':       float(f1_score(y_true_np, y_pred, zero_division=0)),
        'accuracy': float(accuracy_score(y_true_np, y_pred)),
    }


# ---------------------------------------------------------------------------
# Evaluation: scaffold split, 10 seeds
# ---------------------------------------------------------------------------

def run_evaluation(dataset, ext_dataset, holdout_dataset, mod_dims):
    """
    Train with scaffold split across all SEEDS.
    Returns 9 metrics:
      internal scaffold test (10-seed mean):     roc_auc, mcc, f1, accuracy
      external soft voting ensemble:             roc_auc
      holdout soft voting ensemble:              roc_auc, mcc, f1, accuracy
    """
    int_aucs, int_mccs, int_f1s, int_accs = [], [], [], []
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

        optimizer = optim.Adam(
            model.parameters(),
            lr=BASE_CONFIG['lr'],
            weight_decay=BASE_CONFIG['weight_decay'],
        )
        # pos_weight = n_neg/n_pos (auto-computed; the only patch vs a489d67)
        y_train = train_ds.tensors[1] if hasattr(train_ds, 'tensors') else train_ds.labels
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pw = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

        model = train_model(model, optimizer, train_loader, val_loader, loss_fn)

        # Internal scaffold test: full metrics per seed
        int_metrics = eval_model(model, test_loader)
        int_aucs.append(int_metrics['roc_auc'])
        int_mccs.append(int_metrics['mcc'])
        int_f1s.append(int_metrics['f1'])
        int_accs.append(int_metrics['accuracy'])

        # Ext / holdout: store probabilities for ensemble
        ext_probs     = predict_probs(model, ext_loader)
        holdout_probs = predict_probs(model, holdout_loader)
        ext_seed_probs.append(ext_probs)
        holdout_seed_probs.append(holdout_probs)

        # Per-seed sanity print (per-seed ext/holdout AUC, not used for final metrics)
        ext_auc_seed     = float(roc_auc_score(ext_dataset.labels.cpu().numpy().ravel(),     ext_probs))
        holdout_auc_seed = float(roc_auc_score(holdout_dataset.labels.cpu().numpy().ravel(), holdout_probs))
        print(f"  seed={seed:>4d}  int={int_metrics['roc_auc']:.4f}  "
              f"ext={ext_auc_seed:.4f}  holdout={holdout_auc_seed:.4f}")

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
                'roc_auc_scaffold': int_metrics['roc_auc'],
                'mcc_scaffold':     int_metrics['mcc'],
                'f1_scaffold':      int_metrics['f1'],
                'acc_scaffold':     int_metrics['accuracy'],
            },
        }
        with open(os.path.join(seed_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        # ─────────────────────────────────────────────────────────────────

        del model, optimizer
        del train_loader, val_loader, test_loader, ext_loader, holdout_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Soft voting ensemble for ext/holdout
    ext_ensemble     = np.mean(np.stack(ext_seed_probs,     axis=0), axis=0)
    holdout_ensemble = np.mean(np.stack(holdout_seed_probs, axis=0), axis=0)

    ext_metrics     = metrics_from_probs(ext_dataset.labels,     ext_ensemble)
    holdout_metrics = metrics_from_probs(holdout_dataset.labels, holdout_ensemble)

    mean = lambda lst: float(sum(lst) / len(lst))
    return {
        'roc_s':       mean(int_aucs),
        'mcc_s':       mean(int_mccs),
        'f1_s':        mean(int_f1s),
        'acc_s':       mean(int_accs),
        'roc_ext':     ext_metrics['roc_auc'],
        'roc_holdout': holdout_metrics['roc_auc'],
        'mcc_holdout': holdout_metrics['mcc'],
        'f1_holdout':  holdout_metrics['f1'],
        'acc_holdout': holdout_metrics['accuracy'],
    }


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
    m = run_evaluation(dataset, ext_dataset, holdout_dataset, mod_dims)
    print(f"\nMean  int={m['roc_s']:.4f}  ext={m['roc_ext']:.4f}  holdout={m['roc_holdout']:.4f}")
    print(f"[internal] mcc={m['mcc_s']:.4f}  f1={m['f1_s']:.4f}  acc={m['acc_s']:.4f}")
    print(f"[holdout]  mcc={m['mcc_holdout']:.4f}  f1={m['f1_holdout']:.4f}  acc={m['acc_holdout']:.4f}")

    # -----------------------------------------------------------------------
    # Output block — DO NOT CHANGE FORMAT (parsed by autoresearch loop)
    # -----------------------------------------------------------------------
    peak_vram_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024
                    if torch.cuda.is_available() else 0.0)
    print('\n---')
    print(f'roc_auc_scaffold:    {m["roc_s"]:.6f}')
    print(f'roc_auc_external:    {m["roc_ext"]:.6f}')
    print(f'roc_auc_holdout:     {m["roc_holdout"]:.6f}')
    print(f'mcc_scaffold:        {m["mcc_s"]:.6f}')
    print(f'f1_scaffold:         {m["f1_s"]:.6f}')
    print(f'acc_scaffold:        {m["acc_s"]:.6f}')
    print(f'mcc_holdout:         {m["mcc_holdout"]:.6f}')
    print(f'f1_holdout:          {m["f1_holdout"]:.6f}')
    print(f'acc_holdout:         {m["acc_holdout"]:.6f}')
    print(f'total_seconds:       {time.time() - t_start:.1f}')
    print(f'peak_vram_mb:        {peak_vram_mb:.1f}')
    print(f'n_seeds:             {len(SEEDS)}')
