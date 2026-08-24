"""
Full metric evaluation for all saved combo1 models.

Models:
  phase2_best      : iter90 arch + P2_BEST hparams + pw=0.08
  phase2_best_auto : iter90 arch + P2_BEST hparams + pw=auto
  iter90_base_auto : iter90 arch + BASE_CONFIG     + pw=auto  (0.8611 reference)
  best             : iter55 arch + BASE_CONFIG     + pw=auto

Metrics per dataset:
  internal test set : 10-seed MEAN  (roc_auc, mcc, f1, accuracy)
  external          : soft voting ensemble
  holdout           : soft voting ensemble

Run:
  python eval_all_models.py
"""

import os, sys, json, pickle, math
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    accuracy_score, confusion_matrix,
)
from bbb_prepare import (
    set_seed, split_then_normalize, eval_model,
    LABEL_PATH, EMBED_PATHS, FP_TYPES, device,
)
from bbb_iter90 import (
    load_cached_dataset, apply_scaler, predict_probs,
    SEEDS, SPLIT_MODE,
    EXT_LABEL_PATH, EXT_EMBED_PATHS,
    HOLDOUT_LABEL_PATH, HOLDOUT_EMBED_PATHS,
)

ARTIFACT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bbb_artifacts')

MODELS = OrderedDict([
    ('iter90_pw008',       os.path.join(ARTIFACT_BASE, 'phase2_best')),
    ('iter90_p2_auto',     os.path.join(ARTIFACT_BASE, 'phase2_best_auto')),
    ('iter90_base_auto',   os.path.join(ARTIFACT_BASE, 'iter90_base_auto')),
    ('iter55_base_auto',   os.path.join(ARTIFACT_BASE, 'best')),
])

# ---------------------------------------------------------------------------
# Architecture definitions
# ---------------------------------------------------------------------------

class SpatialGatingUnit_iter90(nn.Module):
    """iter90: pre-norm u+v, Q/K/V attention, diagonal mask, learned temperature."""
    def __init__(self, d_ffn, seq_len, attn_drop=0.1):
        super().__init__()
        self.norm_u    = nn.LayerNorm(d_ffn)
        self.norm_v    = nn.LayerNorm(d_ffn)
        self.q_proj    = nn.Linear(d_ffn, d_ffn)
        self.k_proj    = nn.Linear(d_ffn, d_ffn)
        self.v_proj    = nn.Linear(d_ffn, d_ffn)
        nn.init.eye_(self.v_proj.weight); nn.init.zeros_(self.v_proj.bias)
        self.log_temp  = nn.Parameter(torch.tensor(-0.5 * math.log(d_ffn)))
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        u = self.norm_u(u); v = self.norm_v(v)
        Q = self.q_proj(v); K = self.k_proj(v); V = self.v_proj(v)
        scores = (Q @ K.transpose(-1, -2)) * self.log_temp.exp()
        seq_len = scores.size(-1)
        diag = torch.eye(seq_len, device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(diag.unsqueeze(0), float('-inf'))
        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        return u * (attn @ V)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean centering). Used in iter90."""
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps   = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


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


class gMLPBlock_iter90(nn.Module):
    """iter90 gMLP block: RMSNorm + SiLU activation."""
    def __init__(self, d_model, d_ffn, seq_len, drop_prob=0.0):
        super().__init__()
        self.norm          = RMSNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu           = SpatialGatingUnit_iter90(d_ffn, seq_len)
        self.drop_prob     = drop_prob

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.silu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        return x + residual


class gMLPBlock_iter55(nn.Module):
    """iter55 gMLP block: LayerNorm + DropPath."""
    def __init__(self, d_model, d_ffn, seq_len, drop_path=0.0):
        super().__init__()
        self.norm          = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu           = SpatialGatingUnit_iter90(d_ffn, seq_len)
        self.drop_path     = DropPath(drop_path)

    def forward(self, x):
        r = x
        x = F.gelu(self.channel_proj1(self.norm(x)))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        return self.drop_path(x) + r


class gMLP_iter90(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, num_layers=4, stochastic_depth_rate=0.05):
        super().__init__()
        dp = [stochastic_depth_rate * i / max(num_layers-1, 1) for i in range(num_layers)]
        self.model = nn.Sequential(
            *[gMLPBlock_iter90(d_model, d_ffn, seq_len, drop_prob=p) for p in dp]
        )

    def forward(self, x): return self.model(x)


class gMLP_iter55(nn.Module):
    """iter55: linear-decay drop_path 0->0.12."""
    def __init__(self, d_model, d_ffn, seq_len, num_layers=4):
        super().__init__()
        dp = [0.12 * i / max(num_layers-1, 1) for i in range(num_layers)]
        self.model = nn.Sequential(
            *[gMLPBlock_iter55(d_model, d_ffn, seq_len, drop_path=p) for p in dp]
        )

    def forward(self, x): return self.model(x)


class MultiModal_iter90(nn.Module):
    """iter90: pool_query (input-dependent) + skip_gate."""
    def __init__(self, mod_dims, d_model=512, d_ffn=1048, depth=4,
                 dropout=0.2, stochastic_depth_rate=0.05):
        super().__init__()
        self.mod_names     = list(mod_dims.keys())
        self.mod_dims_list = [mod_dims[n] for n in self.mod_names]
        self.seq_len       = len(self.mod_names)
        self.proj = nn.ModuleDict({
            n: nn.Linear(d, d_model)
            for n, d in zip(self.mod_names, self.mod_dims_list)
        })
        self.backbone  = gMLP_iter90(d_model, d_ffn, self.seq_len, depth, stochastic_depth_rate)
        self.norm      = nn.LayerNorm(d_model)
        self.pool_query = nn.Parameter(torch.zeros(d_model))
        self.skip_gate  = nn.Parameter(torch.zeros(1))
        self.head       = nn.Linear(d_model, 1)
        self.drop       = nn.Dropout(dropout)
        self._d         = d_model

    def forward(self, x):
        chunks = torch.split(x, self.mod_dims_list, dim=1)
        tokens = [self.proj[n](c) for n, c in zip(self.mod_names, chunks)]
        X0 = torch.stack(tokens, dim=1)
        X  = self.backbone(X0)
        g  = torch.sigmoid(self.skip_gate)
        X  = (1-g)*X + g*X0
        scores = (X @ self.pool_query) / (self._d ** 0.5)
        w  = torch.softmax(scores, dim=-1)
        Xp = (X * w.unsqueeze(-1)).sum(dim=1)
        return self.head(self.drop(self.norm(Xp))).squeeze(-1)


class MultiModal_iter55(nn.Module):
    """iter55: alpha (static softmax weights) + skip_gate."""
    def __init__(self, mod_dims, d_model=512, d_ffn=1048, depth=4, dropout=0.2):
        super().__init__()
        self.mod_names     = list(mod_dims.keys())
        self.mod_dims_list = [mod_dims[n] for n in self.mod_names]
        self.seq_len       = len(self.mod_names)
        self.proj = nn.ModuleDict({
            n: nn.Linear(d, d_model)
            for n, d in zip(self.mod_names, self.mod_dims_list)
        })
        self.backbone  = gMLP_iter55(d_model, d_ffn, self.seq_len, depth)
        self.norm      = nn.LayerNorm(d_model)
        self.alpha     = nn.Parameter(torch.zeros(self.seq_len))
        self.skip_gate = nn.Parameter(torch.zeros(1))
        self.head      = nn.Linear(d_model, 1)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x):
        chunks = torch.split(x, self.mod_dims_list, dim=1)
        tokens = [self.proj[n](c) for n, c in zip(self.mod_names, chunks)]
        X0 = torch.stack(tokens, dim=1)
        X  = self.backbone(X0)
        g  = torch.sigmoid(self.skip_gate)
        X  = (1-g)*X + g*X0
        w  = torch.softmax(self.alpha, dim=0)
        Xp = (X * w.view(1,-1,1)).sum(dim=1)
        return self.head(self.drop(self.norm(Xp))).squeeze(-1)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(artifact_dir, seed, mod_dims):
    seed_dir  = os.path.join(artifact_dir, f'seed_{seed}')
    state     = torch.load(os.path.join(seed_dir, 'model.pth'),
                           map_location='cpu', weights_only=False)
    with open(os.path.join(seed_dir, 'config.json')) as f:
        cfg = json.load(f)
    hp = cfg.get('hparams', cfg.get('base_config', {}))

    # Detect architecture from state dict keys
    if 'pool_query' in state:
        model = MultiModal_iter90(
            mod_dims,
            d_model  = hp.get('d_model', 512),
            d_ffn    = hp.get('d_ffn', 1048),
            depth    = hp.get('depth', 4),
            dropout  = hp.get('dropout', 0.1),
            stochastic_depth_rate = hp.get('stochastic_depth_rate', 0.05),
        )
    else:
        model = MultiModal_iter55(
            mod_dims,
            d_model  = hp.get('d_model', 512),
            d_ffn    = hp.get('d_ffn', 1048),
            depth    = hp.get('depth', 4),
            dropout  = hp.get('dropout', 0.2),
        )

    model.load_state_dict(state)
    model.eval()

    scaler = None
    scaler_path = os.path.join(seed_dir, 'rdkit_scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    return model, scaler, cfg


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true)
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = specificity = 0.0

    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    return {
        'roc_auc':  round(float(auc), 4),
        'mcc':      round(float(matthews_corrcoef(y_true, y_pred)), 4),
        'f1':       round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
    }


def fmt(m):
    return (f"roc_auc={m['roc_auc']:.4f}  mcc={m['mcc']:.4f}  "
            f"f1={m['f1']:.4f}  acc={m['accuracy']:.4f}")


# ---------------------------------------------------------------------------
# Evaluation per model
# ---------------------------------------------------------------------------

def evaluate_model(model_name, artifact_dir, dataset, ext_dataset, holdout_dataset, mod_dims):
    print(f'\n{"="*60}')
    print(f'  {model_name}')
    print(f'{"="*60}')

    int_seed_metrics = []
    ext_seed_probs   = []
    hold_seed_probs  = []
    y_ext   = ext_dataset.labels.numpy()
    y_hold  = holdout_dataset.labels.numpy()

    for seed in SEEDS:
        set_seed(seed)
        _, _, test_ds, scaler, (rd_start, rd_end), _ = split_then_normalize(
            dataset, split_mode=SPLIT_MODE,
            train_ratio=0.8, val_ratio=0.1, seed=seed,
        )

        ext_X  = apply_scaler(ext_dataset.features,     scaler, rd_start, rd_end)
        hold_X = apply_scaler(holdout_dataset.features, scaler, rd_start, rd_end)

        model, _, _ = load_model(artifact_dir, seed, mod_dims)
        model = model.to(device)

        bs = 256
        test_loader = data.DataLoader(test_ds,  batch_size=bs, shuffle=False)
        ext_loader  = data.DataLoader(
            data.TensorDataset(ext_X,  ext_dataset.labels),  batch_size=bs, shuffle=False)
        hold_loader = data.DataLoader(
            data.TensorDataset(hold_X, holdout_dataset.labels), batch_size=bs, shuffle=False)

        y_te   = test_ds.tensors[1].numpy()
        prob_te = predict_probs(model, test_loader)
        m_int   = compute_metrics(y_te, prob_te)
        int_seed_metrics.append(m_int)

        ext_seed_probs.append(predict_probs(model, ext_loader))
        hold_seed_probs.append(predict_probs(model, hold_loader))

        print(f"  seed={seed:>4d}  int_auc={m_int['roc_auc']:.4f}  "
              f"mcc={m_int['mcc']:.4f}  f1={m_int['f1']:.4f}  acc={m_int['accuracy']:.4f}",
              flush=True)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Internal: 10-seed mean
    int_mean = {k: round(float(np.mean([m[k] for m in int_seed_metrics])), 4)
                for k in int_seed_metrics[0]}
    int_std  = {k: round(float(np.std( [m[k] for m in int_seed_metrics])), 4)
                for k in int_seed_metrics[0]}

    # External / holdout: soft voting ensemble
    ext_ens  = np.mean(np.stack(ext_seed_probs,  axis=0), axis=0)
    hold_ens = np.mean(np.stack(hold_seed_probs, axis=0), axis=0)
    ext_m    = compute_metrics(y_ext,  ext_ens)
    hold_m   = compute_metrics(y_hold, hold_ens)

    print(f'\n  [Internal 10-seed mean]  {fmt(int_mean)}')
    print(f'  [Internal 10-seed std ]  '
          f"roc_auc={int_std['roc_auc']:.4f}  mcc={int_std['mcc']:.4f}  "
          f"f1={int_std['f1']:.4f}  acc={int_std['accuracy']:.4f}")
    print(f'  [External soft-voting]   {fmt(ext_m)}')
    print(f'  [Holdout  soft-voting]   {fmt(hold_m)}')

    return {
        'model':    model_name,
        'int_mean': int_mean,
        'int_std':  int_std,
        'ext':      ext_m,
        'holdout':  hold_m,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results):
    print(f'\n{"="*100}')
    print('  SUMMARY')
    print(f'{"="*100}')
    hdr = (f'{"Model":<22} | {"Int AUC":>8} {"Int MCC":>8} {"Int F1":>7} {"Int ACC":>8} | '
           f'{"Ext AUC":>8} {"Ext MCC":>8} {"Ext F1":>7} {"Ext ACC":>8} | '
           f'{"Hold AUC":>9} {"Hold MCC":>9} {"Hold F1":>8} {"Hold ACC":>9}')
    print(hdr)
    print('-' * len(hdr))
    for r in results:
        mi, me, mh = r['int_mean'], r['ext'], r['holdout']
        print(
            f'{r["model"]:<22} | '
            f'{mi["roc_auc"]:>8.4f} {mi["mcc"]:>8.4f} {mi["f1"]:>7.4f} {mi["accuracy"]:>8.4f} | '
            f'{me["roc_auc"]:>8.4f} {me["mcc"]:>8.4f} {me["f1"]:>7.4f} {me["accuracy"]:>8.4f} | '
            f'{mh["roc_auc"]:>9.4f} {mh["mcc"]:>9.4f} {mh["f1"]:>8.4f} {mh["accuracy"]:>9.4f}'
        )

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_all_models_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Check which model directories exist
    available = {k: v for k, v in MODELS.items()
                 if os.path.exists(os.path.join(v, 'seed_42', 'model.pth'))}
    missing   = [k for k in MODELS if k not in available]
    if missing:
        print(f'[WARN] Not yet trained, skipping: {missing}')
    print(f'Evaluating: {list(available.keys())}')

    print('\nLoading datasets...')
    dataset       = load_cached_dataset('internal', LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    expected_dims = dataset.expected_dims
    mod_dims      = OrderedDict((t, expected_dims[t]) for t in FP_TYPES)
    ext_dataset   = load_cached_dataset(
        'external', EXT_LABEL_PATH, EXT_EMBED_PATHS,
        fp_types=FP_TYPES, expected_dims=expected_dims)
    holdout_dataset = load_cached_dataset(
        'holdout', HOLDOUT_LABEL_PATH, HOLDOUT_EMBED_PATHS,
        fp_types=FP_TYPES, expected_dims=expected_dims)
    print(f'  internal={len(dataset)}  external={len(ext_dataset)}  holdout={len(holdout_dataset)}')

    results = []
    for name, art_dir in available.items():
        r = evaluate_model(name, art_dir, dataset, ext_dataset, holdout_dataset, mod_dims)
        results.append(r)

    print_summary(results)


if __name__ == '__main__':
    main()
