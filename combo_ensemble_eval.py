"""
Soft-voting ensemble evaluation for combo best models.

For each combo:
  - Loads 10 saved models from bbb_artifacts/best/seed_*/
  - ext / holdout: each seed's scaler → inference → average probs → single ROC-AUC
  - internal:      per-seed test split (scaffold) → per-seed ROC-AUC → mean
                   (test sets differ per seed so ensemble doesn't apply cleanly)

Usage:
    conda run -n rapids-25.02 python combo_ensemble_eval.py --combo combo1
    conda run -n rapids-25.02 python combo_ensemble_eval.py --combo combo2
    conda run -n rapids-25.02 python combo_ensemble_eval.py --combo combo3
    conda run -n rapids-25.02 python combo_ensemble_eval.py  # all combos
"""

import os
import sys
import json
import pickle
import argparse
import importlib
import importlib.util
import inspect
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Combo definitions
# ---------------------------------------------------------------------------

COMBO_DIRS = {
    'combo1': '/home/minji/bbb-combo1',
    'combo2': '/home/minji/bbb-combo2',
    'combo3': '/home/minji/bbb-combo3',
}

SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
BATCH_SIZE = 128
ROOT = '/home/minji/autoresearch'


# ---------------------------------------------------------------------------
# Dynamic import helper
# ---------------------------------------------------------------------------

def import_from_dir(module_name, directory):
    """Import a module from a specific directory (not on sys.path by default)."""
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(directory, f'{module_name}.py'),
    )
    mod = importlib.util.module_from_spec(spec)
    # Insert directory so relative imports (e.g. from bbb_prepare import *) work
    sys.path.insert(0, directory)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Inference helper: collect raw probabilities
# ---------------------------------------------------------------------------

def get_probs(model, X: torch.Tensor, y: torch.Tensor, device, batch_size=BATCH_SIZE):
    """Return numpy array of sigmoid probabilities for the given dataset."""
    model.eval()
    loader = data.DataLoader(
        data.TensorDataset(X, y),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    all_probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs)


# ---------------------------------------------------------------------------
# Per-combo evaluation
# ---------------------------------------------------------------------------

def evaluate_combo(combo_name: str, combo_dir: str, best_subdir: str, output_dir: str):
    print(f"\n{'='*65}")
    print(f"  Combo: {combo_name}  ({combo_dir})")
    print(f"{'='*65}")

    # -- import combo-specific modules --------------------------------------
    bbb_prepare = import_from_dir('bbb_prepare', combo_dir)
    bbb_train   = import_from_dir('bbb_train',   combo_dir)

    device = bbb_prepare.device
    FP_TYPES = bbb_prepare.FP_TYPES

    # -- load dataset caches (already built by autoresearch loop) -----------
    cache_dir = os.path.join(combo_dir, 'dataset_cache')

    def load_cache(name):
        path = os.path.join(cache_dir, f'{name}.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Cache not found: {path}')
        return torch.load(path, weights_only=False)

    print('Loading cached datasets...')
    internal_ds = load_cache('internal')
    ext_ds      = load_cache('external')
    holdout_ds  = load_cache('holdout')
    print(f'  internal={len(internal_ds)}  ext={len(ext_ds)}  holdout={len(holdout_ds)}')

    mod_dims = OrderedDict(
        (t, internal_ds.expected_dims[t]) for t in FP_TYPES
    )

    # Find rdkit slice for scaler application
    rd_start, rd_end, offset = None, None, 0
    for t in FP_TYPES:
        dim = internal_ds.expected_dims[t]
        if t == 'rdkit':
            rd_start, rd_end = offset, offset + dim
            break
        offset += dim

    # -- load artifacts -----------------------------------------------------
    best_dir = os.path.join(combo_dir, 'bbb_artifacts', best_subdir)

    def load_seed(seed):
        seed_dir = os.path.join(best_dir, f'seed_{seed}')
        model_path  = os.path.join(seed_dir, 'model.pth')
        scaler_path = os.path.join(seed_dir, 'rdkit_scaler.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model not found: {model_path}')

        # Load scaler (may not exist if no rdkit in combo)
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

        # Rebuild model with same architecture as bbb_train.py current best
        with open(os.path.join(seed_dir, 'config.json')) as f:
            cfg = json.load(f)
        base = cfg['base_config']

        # Load state dict first to detect architecture from checkpoint
        state = torch.load(model_path, map_location=device, weights_only=True)
        use_gated_pool = 'pool_queries' in state  # attention pooling iff checkpoint has pool_queries

        # Build kwargs based on what this combo's class actually accepts
        cls_params = inspect.signature(bbb_train.MultiModalGMLPFromFlat.__init__).parameters
        kwargs = dict(
            mod_dims=mod_dims,
            d_model=base['d_model'],
            d_ffn=base['d_ffn'],
            depth=base['depth'],
            dropout=base['dropout'],
            use_gated_pool=use_gated_pool,
        )
        if 'stochastic_depth_rate' in cls_params:
            kwargs['stochastic_depth_rate'] = 0.05

        model = bbb_train.MultiModalGMLPFromFlat(**kwargs).to(device)
        incompatible = model.load_state_dict(state, strict=False)
        allowed_missing = {'aux_scale', 'aux_norm.weight', 'aux_norm.bias', 'aux_head.weight', 'aux_head.bias'}
        unexpected = set(incompatible.unexpected_keys)
        missing = set(incompatible.missing_keys)
        if unexpected:
            print(f'  [WARN] Ignoring unexpected checkpoint keys for seed {seed}: {sorted(unexpected)}')
        if missing - allowed_missing:
            raise RuntimeError(f'Unsupported missing keys in checkpoint for seed {seed}: {sorted(missing)}')
        model.eval()

        return model, scaler, cfg

    def apply_scaler(X: torch.Tensor, scaler):
        if scaler is None or rd_start is None:
            return X
        X = X.clone()
        X[:, rd_start:rd_end] = torch.tensor(
            scaler.transform(X[:, rd_start:rd_end].numpy()),
            dtype=torch.float32,
        )
        return X

    # -- collect predictions -----------------------------------------------
    ext_probs_list     = []
    holdout_probs_list = []
    int_aucs           = []

    for seed in SEEDS:
        print(f'  Loading seed={seed}...', end=' ', flush=True)
        model, scaler, cfg = load_seed(seed)

        # ext
        X_ext = apply_scaler(ext_ds.features, scaler)
        ext_probs_list.append(get_probs(model, X_ext, ext_ds.labels, device))

        # holdout
        X_holdout = apply_scaler(holdout_ds.features, scaler)
        holdout_probs_list.append(get_probs(model, X_holdout, holdout_ds.labels, device))

        # Use saved internal scaffold ROC-AUC from the artifact config.
        # Reconstructing the test split later is not guaranteed to match the
        # exact evaluation environment used when the checkpoint was kept.
        int_auc = float(cfg['metrics']['roc_auc_scaffold'])
        int_aucs.append(int_auc)

        # per-seed individual ext/holdout AUC for comparison
        ext_auc_seed     = roc_auc_score(ext_ds.labels.numpy(),     ext_probs_list[-1])
        holdout_auc_seed = roc_auc_score(holdout_ds.labels.numpy(), holdout_probs_list[-1])
        print(f'int={int_auc:.4f}  ext={ext_auc_seed:.4f}  holdout={holdout_auc_seed:.4f}')

        del model

    # -- ensemble ROC-AUC --------------------------------------------------
    ext_probs_mean     = np.mean(ext_probs_list,     axis=0)
    holdout_probs_mean = np.mean(holdout_probs_list, axis=0)

    roc_ext_ensemble     = roc_auc_score(ext_ds.labels.numpy(),     ext_probs_mean)
    roc_holdout_ensemble = roc_auc_score(holdout_ds.labels.numpy(), holdout_probs_mean)
    roc_int_mean         = float(np.mean(int_aucs))
    roc_ext_mean         = float(np.mean([roc_auc_score(ext_ds.labels.numpy(), p)
                                          for p in ext_probs_list]))
    roc_holdout_mean     = float(np.mean([roc_auc_score(holdout_ds.labels.numpy(), p)
                                          for p in holdout_probs_list]))

    print(f'\n  {"─"*55}')
    print(f'  {"metric":<22}  {"10-seed mean":>12}  {"ensemble":>12}  {"gain":>8}')
    print(f'  {"─"*55}')
    print(f'  {"roc_s (scaffold)":<22}  {roc_int_mean:>12.6f}  {"(n/a)":>12}  {"":>8}')
    print(f'  {"roc_ext":<22}  {roc_ext_mean:>12.6f}  {roc_ext_ensemble:>12.6f}  {roc_ext_ensemble - roc_ext_mean:>+8.6f}')
    print(f'  {"roc_holdout":<22}  {roc_holdout_mean:>12.6f}  {roc_holdout_ensemble:>12.6f}  {roc_holdout_ensemble - roc_holdout_mean:>+8.6f}')
    print(f'  {"─"*55}')

    summary = {
        'combo': combo_name,
        'best_subdir': best_subdir,
        'roc_s': round(roc_int_mean, 4),
        'roc_ext': round(roc_ext_ensemble, 4),
        'roc_holdout': round(roc_holdout_ensemble, 4),
        'internal_test': {
            'roc_auc_mean': round(roc_int_mean, 4),
        },
        'external_ensemble': {
            'roc_auc': round(roc_ext_ensemble, 4),
            'roc_auc_seed_mean': round(roc_ext_mean, 4),
            'roc_auc_ensemble_gain': round(roc_ext_ensemble - roc_ext_mean, 4),
        },
        'holdout_ensemble': {
            'roc_auc': round(roc_holdout_ensemble, 4),
            'roc_auc_seed_mean': round(roc_holdout_mean, 4),
            'roc_auc_ensemble_gain': round(roc_holdout_ensemble - roc_holdout_mean, 4),
        },
        'roc_int_mean': roc_int_mean,
        'roc_ext_mean': roc_ext_mean,
        'roc_ext_ensemble': roc_ext_ensemble,
        'roc_holdout_mean': roc_holdout_mean,
        'roc_holdout_ensemble': roc_holdout_ensemble,
    }

    combo_out_dir = os.path.join(output_dir, combo_name)
    os.makedirs(combo_out_dir, exist_ok=True)

    with open(os.path.join(combo_out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([summary]).to_csv(os.path.join(combo_out_dir, 'best_summary.csv'), index=False)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--combo', choices=['combo1', 'combo2', 'combo3'],
                        help='Which combo to evaluate (default: all)')
    parser.add_argument('--best-subdir', default='best',
                        help='Best model subdir inside bbb_artifacts. Default: best')
    parser.add_argument('--output-dir', default=os.path.join(ROOT, 'combo_ensemble_eval_outputs'),
                        help='Directory to write evaluation outputs.')
    args = parser.parse_args()

    targets = ([args.combo] if args.combo else ['combo1', 'combo2', 'combo3'])

    results = []
    for combo_name in targets:
        combo_dir = COMBO_DIRS[combo_name]
        result = evaluate_combo(combo_name, combo_dir, args.best_subdir, args.output_dir)
        results.append(result)

    print(f"\n{'='*65}")
    print('SUMMARY')
    print(f"{'='*65}")
    print(f"  {'combo':<8}  {'roc_s':>8}  {'roc_ext':>10}  {'roc_holdout':>12}")
    print(f"  {'─'*80}")
    for r in results:
        print(f"  {r['combo']:<8}  "
              f"{r['roc_s']:>8.4f}  "
              f"{r['roc_ext']:>10.4f}  "
              f"{r['roc_holdout']:>12.4f}")

    pd.DataFrame(
        [
            {
                'combo': r['combo'],
                'best_subdir': r['best_subdir'],
                'roc_s': r['roc_s'],
                'roc_ext': r['roc_ext'],
                'roc_holdout': r['roc_holdout'],
            }
            for r in results
        ]
    ).to_csv(os.path.join(args.output_dir, 'all_combo_summary.csv'), index=False)
