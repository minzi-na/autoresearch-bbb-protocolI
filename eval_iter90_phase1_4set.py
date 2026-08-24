"""Per-seed 4-set ROC-AUC for candidate Protocol I COMBO1 PHASE-1 (iter90) models.

Goal: identify which saved iter90 checkpoint reproduces the decision-time
phase-1 internal ROC (~0.876) and report its untouched-set values
(GM-BBB 8109, holdout 888, holdout nn05 329) per-seed, so the #1.3 transfer
table can use the true phase-1 (iter90) row instead of a phase-2 variant.

Reuses the exact pipeline of eval_holdout_subset_888_329.py + eval_perseed_int_ext.py.
Inference only. Metric threshold prob > 0.5; ROC is threshold-free.
"""
import os, sys, json
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data as data
from rdkit import Chem
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bbb_prepare import set_seed, split_then_normalize, LABEL_PATH, EMBED_PATHS, FP_TYPES, device
from bbb_iter90 import (load_cached_dataset, apply_scaler, predict_probs, SEEDS, SPLIT_MODE,
                        EXT_LABEL_PATH, EXT_EMBED_PATHS, MultiModalGMLPFromFlat)


import torch.nn as nn


class Iter90AlphaModel(MultiModalGMLPFromFlat):
    """Matches the archived best_116735a checkpoint, which uses the fixed-softmax
    `alpha` pooling (the state immediately before commit 116735a swapped alpha ->
    pool_query; the two score 0.87567 vs 0.87573, i.e. identical within noise)."""
    def __init__(self, mod_dims, **kw):
        super().__init__(mod_dims, **kw)
        if self.use_gated_pool:
            del self.pool_query
            self.alpha = nn.Parameter(torch.zeros(self.seq_len))

    def forward(self, x):
        chunks = torch.split(x, self.mod_dims, dim=1)
        tokens = [self.proj[name](chunk) for name, chunk in zip(self.mod_names, chunks)]
        X0 = torch.stack(tokens, dim=1)
        X = self.backbone(X0)
        gate = torch.sigmoid(self.skip_gate)
        X = (1.0 - gate) * X + gate * X0
        if self.use_gated_pool:
            w = torch.softmax(self.alpha, dim=0)
            Xp = (X * w.view(1, -1, 1)).sum(dim=1)
        else:
            Xp = X.mean(dim=1)
        Xp = self.drop(self.norm(Xp))
        return self.head(Xp).squeeze(-1)


def load_model(art, seed, mod_dims):
    """Load the archived iter90 (fixed-softmax-alpha) checkpoint."""
    seed_dir = os.path.join(art, f'seed_{seed}')
    state = torch.load(os.path.join(seed_dir, 'model.pth'), map_location='cpu', weights_only=False)
    cfg = json.load(open(os.path.join(seed_dir, 'config.json')))
    hp = cfg.get('base_config', cfg.get('hparams', {}))
    model = Iter90AlphaModel(
        mod_dims, d_model=hp.get('d_model', 512), d_ffn=hp.get('d_ffn', 1048),
        depth=hp.get('depth', 4), dropout=hp.get('dropout', 0.1),
        use_gated_pool=True, stochastic_depth_rate=0.05)
    model.load_state_dict(state)
    model.eval()
    return model, None, cfg

ARTIFACT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bbb_artifacts')
CANDIDATES = OrderedDict([
    ('iter90_archive_116735a', os.path.join(ARTIFACT_BASE, 'archive', 'best_116735a_iter90_phase1')),
])

HOLDOUT_LABEL_PATH = '/home/minji/BBB/holdout_splits/merged_holdout_10pct_seed42/label_holdout.csv'
HOLDOUT_EMBED_PATHS = {
    'scage1': '/home/minji/BBB/holdout_splits/merged_holdout_10pct_seed42/scage1_holdout.csv',
    'scage2': '/home/minji/BBB/holdout_splits/merged_holdout_10pct_seed42/scage2_holdout.csv',
    'mole':   '/home/minji/BBB/holdout_splits/merged_holdout_10pct_seed42/mole_holdout.csv',
}
SUBSET_ROOT = '/home/minji/holdout_subset'
SUBSET_DIRS = {
    'total': f'{SUBSET_ROOT}/merged_holdout_10pct_seed42_simfilter09_total/label_holdout.csv',
    'nn05':  f'{SUBSET_ROOT}/merged_holdout_10pct_seed42_simfilter09_nn05/label_holdout.csv',
}


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def load_subset_smiles(path):
    import pandas as pd
    return {c for c in (canon(s) for s in pd.read_csv(path)['smiles']) if c}


def msd(vals):
    a = np.array(vals)
    return float(a.mean()), float(a.std(ddof=1))


def main():
    internal = load_cached_dataset('internal', LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    expected = internal.expected_dims
    mod_dims = OrderedDict((t, expected[t]) for t in FP_TYPES)
    ext = load_cached_dataset('external', EXT_LABEL_PATH, EXT_EMBED_PATHS,
                              fp_types=FP_TYPES, expected_dims=expected)
    holdout = load_cached_dataset('holdout', HOLDOUT_LABEL_PATH, HOLDOUT_EMBED_PATHS,
                                  fp_types=FP_TYPES, expected_dims=expected)
    y_ext = ext.labels.numpy().astype(int)
    y_hold = holdout.labels.numpy().astype(int)
    hold_smiles = list(holdout.df['smiles'])
    masks = {}
    for name, path in SUBSET_DIRS.items():
        want = load_subset_smiles(path)
        masks[name] = np.array([s in want for s in hold_smiles], dtype=bool)
        print(f'  subset {name}: matched {int(masks[name].sum())}')

    results = {}
    for mname, art in CANDIDATES.items():
        if not os.path.exists(os.path.join(art, 'seed_42', 'model.pth')):
            print(f'[skip] {mname}'); continue
        print(f'\n=== {mname} ({art}) ===')
        per = {k: [] for k in ['internal', 'ext', 'total', 'nn05']}
        for seed in SEEDS:
            set_seed(seed)
            _, _, test_ds, scaler, (rd_s, rd_e), _ = split_then_normalize(
                internal, split_mode=SPLIT_MODE, train_ratio=0.8, val_ratio=0.1, seed=seed)
            model, _, _ = load_model(art, seed, mod_dims)
            model = model.to(device)
            # internal test
            y_te = test_ds.tensors[1].numpy().astype(int)
            p_te = predict_probs(model, data.DataLoader(test_ds, batch_size=256))
            per['internal'].append(roc_auc_score(y_te, p_te))
            # external
            eX = apply_scaler(ext.features, scaler, rd_s, rd_e)
            p_ext = predict_probs(model, data.DataLoader(data.TensorDataset(eX, ext.labels), batch_size=256))
            per['ext'].append(roc_auc_score(y_ext, p_ext))
            # holdout subsets
            hX = apply_scaler(holdout.features, scaler, rd_s, rd_e)
            p_h = predict_probs(model, data.DataLoader(data.TensorDataset(hX, holdout.labels), batch_size=256))
            per['total'].append(roc_auc_score(y_hold[masks['total']], p_h[masks['total']]))
            per['nn05'].append(roc_auc_score(y_hold[masks['nn05']], p_h[masks['nn05']]))
            print(f'  seed={seed:>4d}  int={per["internal"][-1]:.4f}  ext={per["ext"][-1]:.4f}  '
                  f'total={per["total"][-1]:.4f}  nn05={per["nn05"][-1]:.4f}', flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results[mname] = {k: {'mean': msd(v)[0], 'std_ddof1': msd(v)[1]} for k, v in per.items()}
        print(f'  -> internal {msd(per["internal"])[0]:.4f}±{msd(per["internal"])[1]:.4f}  '
              f'GM-BBB {msd(per["ext"])[0]:.4f}±{msd(per["ext"])[1]:.4f}  '
              f'total {msd(per["total"])[0]:.4f}±{msd(per["total"])[1]:.4f}  '
              f'nn05 {msd(per["nn05"])[0]:.4f}±{msd(per["nn05"])[1]:.4f}')

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'eval_iter90_phase1_4set_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nsaved eval_iter90_phase1_4set_results.json')


if __name__ == '__main__':
    main()
