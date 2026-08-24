"""
Standard 4-eval-set per-seed evaluation for a combo's BEST AutoResearch model.

Produces the SAME structure / metric convention as combo1's
eval_perseed_int_ext.py + eval_holdout_subset_888_329.py so combo1/2/3 are
directly comparable:

  - internal_test  : MoleculeNet per-seed scaffold test split
  - external_8109  : GM-BBB external set (EXT_LABEL_PATH, 8109)
  - holdout full   : 1089 merged holdout
  - holdout total  : 888  (simfilter09 total subset)
  - holdout nn05   : 329  (simfilter09 nn05 subset)

Inference only (loads saved per-seed models). Uses this combo's own
bbb_train_best architecture + saved rdkit scaler, matching how the models were
trained/validated. Metric threshold y_prob > 0.5 (matches combo1).

Run (from this combo dir):
  conda run -n rapids-25.02 python eval_standard_4set.py \
      --model_dir bbb_artifacts/best --output eval_standard_4set_results.json
"""

import os, sys, json, pickle, argparse, inspect
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

from bbb_prepare import (
    set_seed, split_then_normalize, LABEL_PATH, EMBED_PATHS, FP_TYPES,
)
# Best-commit architecture (matches saved weights) lives in bbb_train_best.py
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    'bbb_train_best',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bbb_train_best.py'))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MultiModalGMLPFromFlat = _mod.MultiModalGMLPFromFlat
predict_probs       = _mod.predict_probs
load_cached_dataset = _mod.load_cached_dataset
apply_scaler        = _mod.apply_scaler
SEEDS               = _mod.SEEDS
SPLIT_MODE          = _mod.SPLIT_MODE
BASE_CONFIG         = _mod.BASE_CONFIG
EXT_LABEL_PATH      = _mod.EXT_LABEL_PATH
EXT_EMBED_PATHS     = _mod.EXT_EMBED_PATHS
HOLDOUT_LABEL_PATH  = _mod.HOLDOUT_LABEL_PATH
HOLDOUT_EMBED_PATHS = _mod.HOLDOUT_EMBED_PATHS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SUBSET_ROOT = '/home/minji/holdout_subset'
SUBSET_DIRS = {
    'total': f'{SUBSET_ROOT}/merged_holdout_10pct_seed42_simfilter09_total/label_holdout.csv',
    'nn05':  f'{SUBSET_ROOT}/merged_holdout_10pct_seed42_simfilter09_nn05/label_holdout.csv',
}

AGG_KEYS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc', 'auprc', 'specificity']


def load_model(seed_dir, mod_dims):
    with open(os.path.join(seed_dir, 'config.json')) as f:
        cfg = json.load(f)
    hparams = cfg.get('hparams', cfg.get('base_config', {}))
    init_params = inspect.signature(MultiModalGMLPFromFlat.__init__).parameters
    kwargs = dict(
        mod_dims=mod_dims,
        d_model=hparams.get('d_model', BASE_CONFIG['d_model']),
        d_ffn=hparams.get('d_ffn', BASE_CONFIG['d_ffn']),
        depth=hparams.get('depth', BASE_CONFIG['depth']),
        dropout=hparams.get('dropout', BASE_CONFIG['dropout']),
        use_gated_pool=True,
    )
    if 'stochastic_depth_rate' in init_params:
        kwargs['stochastic_depth_rate'] = hparams.get('stochastic_depth_rate', 0.05)
    model = MultiModalGMLPFromFlat(**kwargs).to(DEVICE)
    state = torch.load(os.path.join(seed_dir, 'model.pth'),
                       map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


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
    """Match combo1 eval_*.metrics_from_probs (threshold > 0.5)."""
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


def main(model_dir, output_path):
    print('Loading internal + external(8109) + holdout(1089) ...')
    internal = load_cached_dataset('internal', LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    expected_dims = internal.expected_dims
    mod_dims = OrderedDict((t, expected_dims[t]) for t in FP_TYPES)
    ext = load_cached_dataset('external', EXT_LABEL_PATH, EXT_EMBED_PATHS,
                              fp_types=FP_TYPES, expected_dims=expected_dims)
    holdout = load_cached_dataset('holdout', HOLDOUT_LABEL_PATH, HOLDOUT_EMBED_PATHS,
                                  fp_types=FP_TYPES, expected_dims=expected_dims)
    y_ext = ext.labels.numpy().astype(int)
    y_hold = holdout.labels.numpy().astype(int)
    print(f'  internal={len(internal)}  external={len(ext)} (pos={int(y_ext.sum())})  holdout={len(holdout)}')

    # holdout subset masks (canonical-SMILES membership)
    hold_smiles = list(holdout.df['smiles'])
    assert len(hold_smiles) == len(holdout), 'df/feature length mismatch'
    masks = {'full': np.ones(len(holdout), dtype=bool)}
    for name, path in SUBSET_DIRS.items():
        want = load_subset_smiles(path)
        mask = np.array([s in want for s in hold_smiles], dtype=bool)
        print(f'  subset {name}: target={len(want)}  matched={int(mask.sum())}')
        masks[name] = mask

    int_ps, ext_ps = [], []
    ext_seed_probs, hold_seed_probs = [], []
    for seed in SEEDS:
        seed_dir = os.path.join(model_dir, f'seed_{seed}')
        if not os.path.exists(os.path.join(seed_dir, 'model.pth')):
            print(f'  [skip] seed={seed}: no model')
            continue
        set_seed(seed)
        _, _, test_ds, scaler_fit, (rd_start, rd_end), _ = split_then_normalize(
            internal, split_mode=SPLIT_MODE, train_ratio=0.8, val_ratio=0.1, seed=seed)
        # Prefer the saved scaler (fitted during training) for ext/holdout
        scaler_path = os.path.join(seed_dir, 'rdkit_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler_fit = pickle.load(f)
        ext_X  = apply_scaler(ext.features,     scaler_fit, rd_start, rd_end)
        hold_X = apply_scaler(holdout.features, scaler_fit, rd_start, rd_end)

        model = load_model(seed_dir, mod_dims)
        test_loader = data.DataLoader(test_ds, batch_size=256, shuffle=False)
        ext_loader  = data.DataLoader(data.TensorDataset(ext_X, ext.labels),
                                      batch_size=256, shuffle=False)
        hold_loader = data.DataLoader(data.TensorDataset(hold_X, holdout.labels),
                                      batch_size=256, shuffle=False)

        y_te = test_ds.tensors[1].numpy().astype(int)
        ext_p = predict_probs(model, ext_loader)
        m_int = metrics_from_probs(y_te, predict_probs(model, test_loader))
        m_ext = metrics_from_probs(y_ext, ext_p)
        int_ps.append(m_int); ext_ps.append(m_ext)
        ext_seed_probs.append(ext_p)
        hold_seed_probs.append(predict_probs(model, hold_loader))
        print(f'  seed={seed:>4d}  int={m_int["roc_auc"]:.4f}  ext={m_ext["roc_auc"]:.4f}', flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    int_mean, int_std = agg(int_ps)
    ext_mean, ext_std = agg(ext_ps)
    ext_ens = metrics_from_probs(y_ext, np.stack(ext_seed_probs, axis=0).mean(axis=0))
    probs_stack = np.stack(hold_seed_probs, axis=0)   # (n_seeds, 1089)
    ens = probs_stack.mean(axis=0)

    results = {
        'internal_test': {'n_seeds': len(int_ps), 'per_seed_mean': int_mean, 'per_seed_std': int_std},
        'external_8109': {'n': int(len(y_ext)), 'pos': int(y_ext.sum()),
                          'n_seeds': len(ext_ps), 'ensemble': ext_ens,
                          'per_seed_mean': ext_mean, 'per_seed_std': ext_std},
        'holdout': {},
    }
    for name, mask in masks.items():
        ens_m = metrics_from_probs(y_hold[mask], ens[mask])
        per_seed_m = [metrics_from_probs(y_hold[mask], probs_stack[i][mask])
                      for i in range(probs_stack.shape[0])]
        ps_mean, ps_std = agg(per_seed_m)
        results['holdout'][name] = {
            'n': ens_m['n'], 'pos': ens_m['pos'], 'n_seeds': int(probs_stack.shape[0]),
            'ensemble': ens_m, 'per_seed_mean': ps_mean, 'per_seed_std': ps_std,
        }

    print(f'\n  internal roc={int_mean["roc_auc"]:.4f}±{int_std["roc_auc"]:.4f}')
    print(f'  ext-8109 roc={ext_mean["roc_auc"]:.4f}±{ext_std["roc_auc"]:.4f}  (ens {ext_ens["roc_auc"]:.4f})')
    for name in ['full', 'total', 'nn05']:
        h = results['holdout'][name]
        print(f'  holdout {name:>5} (n={h["n"]}) roc={h["per_seed_mean"]["roc_auc"]:.4f}'
              f'±{h["per_seed_std"]["roc_auc"]:.4f}  (ens {h["ensemble"]["roc_auc"]:.4f})')

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved -> {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--output', default='eval_standard_4set_results.json')
    args = parser.parse_args()
    main(args.model_dir, args.output)
