"""
Per-seed (mean +/- std) evaluation of exp-1 best models on:
  - MoleculeNet internal test set (per-seed scaffold test split)
  - GM-BBB external set (8109, EXT_LABEL_PATH = external_cls_only_remaining)

Fills the per-seed gap left by eval_all_models.py (which stored internal as
10-seed mean of roc/mcc/f1/acc only, and external as soft-voting ensemble only).

Metric convention matches eval_holdout_subset_888_329.py (threshold y_prob > 0.5),
incl. auprc, so numbers are directly comparable to the 888/nn05 per-seed table.

Run:
  /home/minji/anaconda3/bin/python eval_perseed_int_ext.py
"""

import os, sys, json
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data as data
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bbb_prepare import set_seed, split_then_normalize, LABEL_PATH, EMBED_PATHS, FP_TYPES, device
from bbb_iter90 import (
    load_cached_dataset, apply_scaler, predict_probs, SEEDS, SPLIT_MODE,
    EXT_LABEL_PATH, EXT_EMBED_PATHS,
)
from eval_all_models import load_model

ARTIFACT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bbb_artifacts')

EXP1_MODELS = OrderedDict([
    ('exp1_phase2_best_auto',  os.path.join(ARTIFACT_BASE, 'phase2_best_auto')),
    ('exp1_phase2_best_pw008', os.path.join(ARTIFACT_BASE, 'phase2_best')),
])

AGG_KEYS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc', 'auprc', 'specificity']


def metrics_from_probs(y_true, y_prob):
    """Match eval_holdout_subset_888_329.metrics_from_probs (threshold > 0.5)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, _, _ = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
    has_both = len(set(y_true.tolist())) > 1
    return {
        'n':           int(len(y_true)),
        'pos':         int(y_true.sum()),
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'precision':   float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':      float(recall_score(y_true, y_pred, zero_division=0)),
        'f1':          float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc':     float(roc_auc_score(y_true, y_prob) if has_both else 0.0),
        'mcc':         float(matthews_corrcoef(y_true, y_pred)),
        'auprc':       float(average_precision_score(y_true, y_prob) if has_both else 0.0),
        'specificity': float(specificity),
    }


def agg(per_seed_list):
    mean = {k: round(float(np.mean([m[k] for m in per_seed_list])), 6) for k in AGG_KEYS}
    std  = {k: round(float(np.std ([m[k] for m in per_seed_list], ddof=0)), 6) for k in AGG_KEYS}
    return mean, std


def main():
    print('Loading internal + external(8109) ...')
    internal = load_cached_dataset('internal', LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    expected_dims = internal.expected_dims
    mod_dims = OrderedDict((t, expected_dims[t]) for t in FP_TYPES)
    ext = load_cached_dataset('external', EXT_LABEL_PATH, EXT_EMBED_PATHS,
                              fp_types=FP_TYPES, expected_dims=expected_dims)
    y_ext = ext.labels.numpy().astype(int)
    print(f'  internal={len(internal)}  external={len(ext)}  (ext pos={int(y_ext.sum())})')
    print(f'  EXT_LABEL_PATH = {EXT_LABEL_PATH}')

    results = {}
    for model_name, art_dir in EXP1_MODELS.items():
        if not os.path.exists(os.path.join(art_dir, 'seed_42', 'model.pth')):
            print(f'[skip] {model_name}: no saved seeds')
            continue
        print(f'\n=== {model_name} ===')
        int_ps, ext_ps = [], []
        for seed in SEEDS:
            set_seed(seed)
            _, _, test_ds, scaler, (rd_start, rd_end), _ = split_then_normalize(
                internal, split_mode=SPLIT_MODE, train_ratio=0.8, val_ratio=0.1, seed=seed)
            ext_X = apply_scaler(ext.features, scaler, rd_start, rd_end)
            model, _, _ = load_model(art_dir, seed, mod_dims)
            model = model.to(device)

            test_loader = data.DataLoader(test_ds, batch_size=256, shuffle=False)
            ext_loader  = data.DataLoader(
                data.TensorDataset(ext_X, ext.labels), batch_size=256, shuffle=False)

            y_te = test_ds.tensors[1].numpy().astype(int)
            m_int = metrics_from_probs(y_te, predict_probs(model, test_loader))
            m_ext = metrics_from_probs(y_ext, predict_probs(model, ext_loader))
            int_ps.append(m_int); ext_ps.append(m_ext)
            print(f'  seed={seed:>4d}  int roc={m_int["roc_auc"]:.4f} auprc={m_int["auprc"]:.4f}  '
                  f'ext roc={m_ext["roc_auc"]:.4f} auprc={m_ext["auprc"]:.4f}', flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        int_mean, int_std = agg(int_ps)
        ext_mean, ext_std = agg(ext_ps)
        results[model_name] = {
            'internal_test': {'n_seeds': len(SEEDS), 'per_seed_mean': int_mean, 'per_seed_std': int_std},
            'external_8109': {'n': int(len(y_ext)), 'pos': int(y_ext.sum()),
                              'n_seeds': len(SEEDS), 'per_seed_mean': ext_mean, 'per_seed_std': ext_std},
        }
        print(f'  [internal test] roc={int_mean["roc_auc"]:.4f}±{int_std["roc_auc"]:.4f}  '
              f'auprc={int_mean["auprc"]:.4f}±{int_std["auprc"]:.4f}  '
              f'mcc={int_mean["mcc"]:.4f}±{int_std["mcc"]:.4f}  '
              f'f1={int_mean["f1"]:.4f}±{int_std["f1"]:.4f}  acc={int_mean["accuracy"]:.4f}±{int_std["accuracy"]:.4f}')
        print(f'  [external 8109] roc={ext_mean["roc_auc"]:.4f}±{ext_std["roc_auc"]:.4f}  '
              f'auprc={ext_mean["auprc"]:.4f}±{ext_std["auprc"]:.4f}  '
              f'mcc={ext_mean["mcc"]:.4f}±{ext_std["mcc"]:.4f}  '
              f'f1={ext_mean["f1"]:.4f}±{ext_std["f1"]:.4f}  acc={ext_mean["accuracy"]:.4f}±{ext_std["accuracy"]:.4f}')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_perseed_int_ext_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved -> {out}')


if __name__ == '__main__':
    main()
