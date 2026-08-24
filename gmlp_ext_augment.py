"""
GateMol-BBB combo3 external augmentation: train on internal_train + external, evaluate roc_s & roc_holdout.

- Scaffold split fixed per seed (same as baseline)
- External data appended to train set only
- Scaler fitted on combined train (internal_train + external)
- Evaluates: roc_s (internal scaffold test, 10-seed mean), roc_holdout (soft voting ensemble)
- Optimizer: AdamW + CosineAnnealingLR (T_max=50, eta_min=1e-6)
- pos_weight: auto-computed from combined train set (n_neg/n_pos, clamped [0.1, 10.0])
- Architecture: 8593ba3 (bbb_train_best.py), RMSNorm + SiLU + Conv1d SGU

Usage:
    conda run -n rapids-25.02 python gmlp_ext_augment.py --output gmlp_ext_metrics.json
"""

import os, sys, json, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from collections import OrderedDict

from bbb_prepare import (
    split_then_normalize, set_seed,
    LABEL_PATH, EMBED_PATHS, FP_TYPES,
    NUM_EPOCHS, PATIENCE, BATCH_SIZE,
)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('bbb_train_best',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bbb_train_best.py'))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MultiModalGMLPFromFlat = _mod.MultiModalGMLPFromFlat
train_model            = _mod.train_model
load_cached_dataset    = _mod.load_cached_dataset
SEEDS                  = _mod.SEEDS
SPLIT_MODE             = _mod.SPLIT_MODE
BASE_CONFIG            = _mod.BASE_CONFIG
EXT_LABEL_PATH         = _mod.EXT_LABEL_PATH
EXT_EMBED_PATHS        = _mod.EXT_EMBED_PATHS
HOLDOUT_LABEL_PATH     = _mod.HOLDOUT_LABEL_PATH
HOLDOUT_EMBED_PATHS    = _mod.HOLDOUT_EMBED_PATHS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'bbb_artifacts', 'ext_augmented')


def get_rd_slice(fp_types, expected_dims):
    offset = 0
    for t in fp_types:
        dim = expected_dims[t]
        if t == 'rdkit':
            return offset, offset + dim
        offset += dim
    return None, None


def get_probs(model, X: torch.Tensor) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = X[i:i+256].to(DEVICE)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs)


def main(output_path):
    print('Loading datasets...')
    int_ds  = load_cached_dataset('internal', LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    expected_dims = int_ds.expected_dims
    mod_dims = OrderedDict((t, expected_dims[t]) for t in FP_TYPES)

    ext_ds  = load_cached_dataset('external', EXT_LABEL_PATH, EXT_EMBED_PATHS,
                                   fp_types=FP_TYPES, expected_dims=expected_dims)
    hold_ds = load_cached_dataset('holdout', HOLDOUT_LABEL_PATH, HOLDOUT_EMBED_PATHS,
                                   fp_types=FP_TYPES, expected_dims=expected_dims)

    rd_start, rd_end = get_rd_slice(FP_TYPES, expected_dims)

    X_int  = int_ds.features.numpy()
    y_int  = int_ds.labels.numpy()
    X_ext  = ext_ds.features.numpy()
    y_ext  = ext_ds.labels.numpy()
    X_hold = hold_ds.features.numpy()
    y_hold = hold_ds.labels.numpy()

    print(f'Internal: {len(X_int)}  External: {len(X_ext)}  Holdout: {len(X_hold)}')

    int_aucs = []
    hold_probs_list = []

    bs = BASE_CONFIG['batch_size']

    for seed in SEEDS:
        set_seed(seed)
        _, _, _, _, (_, _), split_info = split_then_normalize(
            int_ds, split_mode=SPLIT_MODE,
            train_ratio=0.8, val_ratio=0.1, seed=seed,
        )
        train_idx, val_idx, test_idx = split_info

        X_train = np.concatenate([X_int[train_idx].copy(), X_ext.copy()], axis=0)
        y_train = np.concatenate([y_int[train_idx].copy(), y_ext.copy()], axis=0)
        X_val   = X_int[val_idx].copy()
        y_val   = y_int[val_idx].copy()
        X_test  = X_int[test_idx].copy()
        y_test  = y_int[test_idx].copy()
        X_ho    = X_hold.copy()

        if rd_start is not None:
            scaler = StandardScaler()
            X_train[:, rd_start:rd_end] = scaler.fit_transform(X_train[:, rd_start:rd_end])
            X_val  [:, rd_start:rd_end] = scaler.transform(X_val[:, rd_start:rd_end])
            X_test [:, rd_start:rd_end] = scaler.transform(X_test[:, rd_start:rd_end])
            X_ho   [:, rd_start:rd_end] = scaler.transform(X_ho[:, rd_start:rd_end])

        to_tensor = lambda x, y: data.TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        train_loader = data.DataLoader(to_tensor(X_train, y_train), batch_size=bs, shuffle=True,  num_workers=4)
        val_loader   = data.DataLoader(to_tensor(X_val,   y_val),   batch_size=bs, shuffle=False, num_workers=4)
        test_loader  = data.DataLoader(to_tensor(X_test,  y_test),  batch_size=bs, shuffle=False, num_workers=4)

        set_seed(seed)
        model = MultiModalGMLPFromFlat(
            mod_dims=mod_dims,
            d_model=BASE_CONFIG['d_model'],
            d_ffn=BASE_CONFIG['d_ffn'],
            depth=BASE_CONFIG['depth'],
            dropout=BASE_CONFIG['dropout'],
            use_gated_pool=True,
        ).to(DEVICE)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=BASE_CONFIG['lr'],
            weight_decay=BASE_CONFIG['weight_decay'],
        )

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pw = float(np.clip(n_neg / n_pos, 0.1, 10.0))
        pos_weight = torch.tensor([pw]).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model = train_model(model, optimizer, train_loader, val_loader, loss_fn,
                            num_epochs=BASE_CONFIG['num_epochs'],
                            patience=BASE_CONFIG['patience'])

        model.eval()
        test_probs_list, test_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                p = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
                test_probs_list.append(p)
                test_labels.append(yb.numpy())
        test_probs  = np.concatenate(test_probs_list)
        test_labels = np.concatenate(test_labels).astype(int)
        int_auc = round(float(roc_auc_score(test_labels, test_probs)), 4)

        ho_probs = get_probs(model, torch.tensor(X_ho, dtype=torch.float32))
        ho_auc   = round(float(roc_auc_score(y_hold.astype(int), ho_probs)), 4)

        int_aucs.append(int_auc)
        hold_probs_list.append(ho_probs)
        print(f'  seed={seed:4d}  roc_s={int_auc:.4f}  roc_holdout={ho_auc:.4f}'
              f'  n_train={len(X_train)}  pos_weight={pw:.3f}')

        seed_dir = os.path.join(SAVE_DIR, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(seed_dir, 'model.pth'))
        if rd_start is not None:
            with open(os.path.join(seed_dir, 'rdkit_scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
        with open(os.path.join(seed_dir, 'config.json'), 'w') as f:
            json.dump({'seed': seed, 'fp_types': FP_TYPES,
                       'mod_dims': {k: int(v) for k, v in mod_dims.items()},
                       'base_config': BASE_CONFIG,
                       'pos_weight': pw,
                       'metrics': {'roc_s': int_auc, 'roc_holdout_seed': ho_auc}}, f, indent=2)

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    roc_s_mean  = round(float(np.mean(int_aucs)), 4)
    roc_holdout = round(float(roc_auc_score(y_hold.astype(int),
                                             np.mean(hold_probs_list, axis=0))), 4)

    result = {
        'fp_types':    FP_TYPES,
        'n_seeds':     len(int_aucs),
        'roc_s':       roc_s_mean,
        'roc_holdout': roc_holdout,
        'seed_roc_s':  int_aucs,
        'base_config': BASE_CONFIG,
    }
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n  [result] roc_s={roc_s_mean:.4f}  roc_holdout={roc_holdout:.4f}')
    print(f'  Saved -> {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='gmlp_ext_metrics.json')
    args = parser.parse_args()
    main(args.output)
