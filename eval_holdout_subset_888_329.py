"""
Evaluate experiment-1 best models (phase2_best, phase2_best_auto) on the
COMMON holdout subsets used by experiment-2 (combos_v2):

  - total : /home/minji/holdout_subset/merged_holdout_10pct_seed42_simfilter09_total  (888)
  - nn05  : /home/minji/holdout_subset/merged_holdout_10pct_seed42_simfilter09_nn05   (329)

Method (inference only, no retraining):
  1. Load exp-1's cached holdout dataset (1089 mols, training-consistent features).
  2. Subset rows by canonical SMILES membership in the 888 / 329 label files.
  3. Per seed: refit the rdkit scaler on the internal scaffold-train split
     (identical to eval_all_models.py), apply to the holdout features,
     run the saved seed model -> sigmoid probs over the full 1089 holdout.
  4. Soft-voting ensemble = mean of per-seed probs (row-wise), then slice to
     each subset and compute metrics. (Row-wise mean => slicing the ensemble
     equals ensembling the slice; identical to combos_v2 final_holdout_eval.)
  5. Sanity: full-1089 metrics must reproduce eval_all_models_results.json.

Metrics match combos_v2 metrics_from_probs (threshold y_prob > 0.5):
  accuracy, precision, recall, f1, roc_auc, mcc, auprc, specificity.

Run:
  /home/minji/anaconda3/bin/python eval_holdout_subset_888_329.py
"""

import os, sys, json
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data as data
from rdkit import Chem
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bbb_prepare import set_seed, split_then_normalize, LABEL_PATH, EMBED_PATHS, FP_TYPES, device
from bbb_iter90 import load_cached_dataset, apply_scaler, predict_probs, SEEDS, SPLIT_MODE
from eval_all_models import load_model

ARTIFACT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bbb_artifacts')

# Experiment-1 best-model candidates (10 saved seeds each)
EXP1_MODELS = OrderedDict([
    ('exp1_phase2_best_pw008', os.path.join(ARTIFACT_BASE, 'phase2_best')),
    ('exp1_phase2_best_auto',  os.path.join(ARTIFACT_BASE, 'phase2_best_auto')),
])

SUBSET_ROOT = '/home/minji/holdout_subset'
SUBSET_DIRS = {
    'total': f'{SUBSET_ROOT}/merged_holdout_10pct_seed42_simfilter09_total/label_holdout.csv',
    'nn05':  f'{SUBSET_ROOT}/merged_holdout_10pct_seed42_simfilter09_nn05/label_holdout.csv',
}

# exp-1 cached holdout = BBB/holdout_splits/merged_holdout_10pct_seed42 (1089)
HOLDOUT_LABEL_PATH  = '/home/minji/BBB/holdout_splits/merged_holdout_10pct_seed42/label_holdout.csv'
HOLDOUT_EMBED_PATHS = {
    'scage1': '/home/minji/BBB/holdout_splits/merged_holdout_10pct_seed42/scage1_holdout.csv',
    'scage2': '/home/minji/BBB/holdout_splits/merged_holdout_10pct_seed42/scage2_holdout.csv',
    'mole':   '/home/minji/BBB/holdout_splits/merged_holdout_10pct_seed42/mole_holdout.csv',
}


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def load_subset_smiles(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    cs = set()
    for s in df['smiles']:
        c = canon(s)
        if c is not None:
            cs.add(c)
    return cs


def metrics_from_probs(y_true, y_prob):
    """Match combos_v2 final_holdout_eval.metrics_from_probs (threshold > 0.5)."""
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
        'accuracy':    round(float(accuracy_score(y_true, y_pred)), 6),
        'precision':   round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        'recall':      round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        'f1':          round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        'roc_auc':     round(float(roc_auc_score(y_true, y_prob) if has_both else 0.0), 6),
        'mcc':         round(float(matthews_corrcoef(y_true, y_pred)), 6),
        'auprc':       round(float(average_precision_score(y_true, y_prob) if has_both else 0.0), 6),
        'specificity': round(float(specificity), 6),
    }


def main():
    print('Loading internal (for per-seed scaler) + cached holdout (1089) ...')
    internal = load_cached_dataset('internal', LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    expected_dims = internal.expected_dims
    mod_dims = OrderedDict((t, expected_dims[t]) for t in FP_TYPES)
    holdout = load_cached_dataset(
        'holdout', HOLDOUT_LABEL_PATH, HOLDOUT_EMBED_PATHS,
        fp_types=FP_TYPES, expected_dims=expected_dims)
    print(f'  internal={len(internal)}  holdout={len(holdout)}')

    # Canonical smiles aligned with holdout.features rows
    hold_smiles = list(holdout.df['smiles'])
    assert len(hold_smiles) == len(holdout), 'df/feature length mismatch'
    y_hold = holdout.labels.numpy().astype(int)

    # Build subset row-index masks
    masks = {'full': np.ones(len(holdout), dtype=bool)}
    for name, path in SUBSET_DIRS.items():
        want = load_subset_smiles(path)
        mask = np.array([s in want for s in hold_smiles], dtype=bool)
        n_matched = int(mask.sum())
        print(f'  subset {name}: target={len(want)}  matched_in_cache={n_matched}')
        masks[name] = mask

    results = {}
    for model_name, art_dir in EXP1_MODELS.items():
        if not os.path.exists(os.path.join(art_dir, 'seed_42', 'model.pth')):
            print(f'[skip] {model_name}: no saved seeds')
            continue
        print(f'\n=== {model_name} ===')
        seed_probs = []
        for seed in SEEDS:
            set_seed(seed)
            # Refit scaler on this seed's internal scaffold-train split
            _, _, _, scaler, (rd_start, rd_end), _ = split_then_normalize(
                internal, split_mode=SPLIT_MODE, train_ratio=0.8, val_ratio=0.1, seed=seed)
            hold_X = apply_scaler(holdout.features, scaler, rd_start, rd_end)
            model, _, _ = load_model(art_dir, seed, mod_dims)
            model = model.to(device)
            loader = data.DataLoader(
                data.TensorDataset(hold_X, holdout.labels), batch_size=256, shuffle=False)
            seed_probs.append(predict_probs(model, loader))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        probs_stack = np.stack(seed_probs, axis=0)           # (n_seeds, 1089)
        ens = probs_stack.mean(axis=0)                        # (1089,)

        results[model_name] = {}
        agg_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc', 'auprc', 'specificity']
        for name, mask in masks.items():
            # (a) soft-voting ensemble: metrics on the row-wise mean prob
            ens_m = metrics_from_probs(y_hold[mask], ens[mask])
            # (b) per-seed: metrics per seed, then mean +/- std across seeds
            per_seed_m = [metrics_from_probs(y_hold[mask], probs_stack[i][mask])
                          for i in range(probs_stack.shape[0])]
            per_seed_mean = {k: round(float(np.mean([m[k] for m in per_seed_m])), 6) for k in agg_keys}
            per_seed_std  = {k: round(float(np.std ([m[k] for m in per_seed_m], ddof=0)), 6) for k in agg_keys}
            results[model_name][name] = {
                'n': ens_m['n'], 'pos': ens_m['pos'],
                'ensemble':      ens_m,
                'per_seed_mean': per_seed_mean,
                'per_seed_std':  per_seed_std,
                'n_seeds':       int(probs_stack.shape[0]),
            }
            print(f'  [{name:>5}] n={ens_m["n"]:>4} pos={ens_m["pos"]:>4}  '
                  f'ENS roc_auc={ens_m["roc_auc"]:.4f} mcc={ens_m["mcc"]:.4f} '
                  f'f1={ens_m["f1"]:.4f} acc={ens_m["accuracy"]:.4f} auprc={ens_m["auprc"]:.4f}')
            print(f'        {"":>9} per-seed roc_auc={per_seed_mean["roc_auc"]:.4f}'
                  f'±{per_seed_std["roc_auc"]:.4f}  mcc={per_seed_mean["mcc"]:.4f}±{per_seed_std["mcc"]:.4f}  '
                  f'f1={per_seed_mean["f1"]:.4f}±{per_seed_std["f1"]:.4f}  '
                  f'acc={per_seed_mean["accuracy"]:.4f}±{per_seed_std["accuracy"]:.4f}')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'eval_holdout_subset_888_329_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved -> {out}')


if __name__ == '__main__':
    main()
