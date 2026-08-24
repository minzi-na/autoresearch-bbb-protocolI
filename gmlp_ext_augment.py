"""
GateMol-BBB external augmentation: train on internal_train + external, evaluate roc_s & roc_holdout.

- Scaffold split fixed per seed (same as baseline)
- External data appended to train set only
- Scaler fitted on combined train (internal_train + external)
- Evaluates: roc_s (internal scaffold test, 10-seed mean), roc_holdout (soft voting ensemble)

Usage:
    conda run -n rapids-25.02 python gmlp_ext_augment.py --output gmlp_ext_metrics.json
"""

import os, sys, json, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, accuracy_score
from collections import OrderedDict
from copy import deepcopy

from bbb_prepare import (
    split_then_normalize, set_seed,
    LABEL_PATH, EMBED_PATHS, FP_TYPES,
    NUM_EPOCHS, PATIENCE, BATCH_SIZE,
)
from bbb_train import (
    MultiModalGMLPFromFlat, train_model, predict_probs, roc_auc_from_probs,
    load_cached_dataset, apply_scaler,
    SEEDS, SPLIT_MODE, BASE_CONFIG,
    EXT_LABEL_PATH, EXT_EMBED_PATHS,
    HOLDOUT_LABEL_PATH, HOLDOUT_EMBED_PATHS,
    TIME_BUDGET,
)
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Phase 2 best hyperparameters (combo1)
HPARAMS = {
    'dropout':               0.046019393915327604,
    'lr':                    0.00011022334100107642,
    'weight_decay':          3.88007962016146e-06,
    'd_model':               512,
    'd_ffn':                 1048,
    'depth':                 4,
    'stochastic_depth_rate': 0.05,
    'batch_size':            128,
    'num_epochs':            NUM_EPOCHS,
    'patience':              PATIENCE,
}

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
    int_mccs, int_f1s, int_accs = [], [], []
    hold_probs_list = []

    for seed in SEEDS:
        set_seed(seed)
        _, _, _, _, (_, _), split_info = split_then_normalize(
            int_ds, split_mode=SPLIT_MODE,
            train_ratio=0.8, val_ratio=0.1, seed=seed,
        )
        train_idx, val_idx, test_idx = split_info

        # Concatenate external to train
        X_train = np.concatenate([X_int[train_idx].copy(), X_ext.copy()], axis=0)
        y_train = np.concatenate([y_int[train_idx].copy(), y_ext.copy()], axis=0)
        X_val   = X_int[val_idx].copy()
        y_val   = y_int[val_idx].copy()
        X_test  = X_int[test_idx].copy()
        y_test  = y_int[test_idx].copy()
        X_ho    = X_hold.copy()

        # Fit scaler on combined train
        if rd_start is not None:
            scaler = StandardScaler()
            X_train[:, rd_start:rd_end] = scaler.fit_transform(X_train[:, rd_start:rd_end])
            X_val  [:, rd_start:rd_end] = scaler.transform(X_val[:, rd_start:rd_end])
            X_test [:, rd_start:rd_end] = scaler.transform(X_test[:, rd_start:rd_end])
            X_ho   [:, rd_start:rd_end] = scaler.transform(X_ho[:, rd_start:rd_end])

        bs = HPARAMS['batch_size']
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
            d_model=HPARAMS['d_model'],
            d_ffn=HPARAMS['d_ffn'],
            depth=HPARAMS['depth'],
            dropout=HPARAMS['dropout'],
            use_gated_pool=True,
            stochastic_depth_rate=HPARAMS['stochastic_depth_rate'],
        ).to(DEVICE)

        optimizer = optim.Adam(model.parameters(),
                               lr=HPARAMS['lr'], weight_decay=HPARAMS['weight_decay'])
        n_pos = float(y_train.sum())
        n_neg = float(len(y_train) - n_pos)
        pw = torch.tensor([min(max(n_neg / n_pos, 0.1), 10.0)]).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

        model = train_model(model, optimizer, train_loader, val_loader, loss_fn,
                            num_epochs=HPARAMS['num_epochs'], patience=HPARAMS['patience'])

        # Evaluate
        model.eval()
        test_probs_list, test_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                p = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
                test_probs_list.append(p)
                test_labels.append(yb.numpy())
        test_probs  = np.concatenate(test_probs_list)
        test_labels = np.concatenate(test_labels).astype(int)
        int_auc  = round(float(roc_auc_score(test_labels, test_probs)), 4)
        test_pred = (test_probs > 0.5).astype(int)
        int_mcc  = round(float(matthews_corrcoef(test_labels, test_pred)), 4)
        int_f1   = round(float(f1_score(test_labels, test_pred, zero_division=0)), 4)
        int_acc  = round(float(accuracy_score(test_labels, test_pred)), 4)

        ho_probs = get_probs(model, torch.tensor(X_ho, dtype=torch.float32))
        ho_auc   = round(float(roc_auc_score(y_hold.astype(int), ho_probs)), 4)

        int_aucs.append(int_auc)
        int_mccs.append(int_mcc)
        int_f1s.append(int_f1)
        int_accs.append(int_acc)
        hold_probs_list.append(ho_probs)
        print(f'  seed={seed:4d}  roc_s={int_auc:.4f}  mcc={int_mcc:.4f}  f1={int_f1:.4f}'
              f'  acc={int_acc:.4f}  roc_holdout={ho_auc:.4f}  n_train={len(X_train)}')

        # Save artifact
        seed_dir = os.path.join(SAVE_DIR, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(seed_dir, 'model.pth'))
        if rd_start is not None:
            with open(os.path.join(seed_dir, 'rdkit_scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
        with open(os.path.join(seed_dir, 'config.json'), 'w') as f:
            json.dump({'seed': seed, 'fp_types': FP_TYPES,
                       'mod_dims': {k: int(v) for k, v in mod_dims.items()},
                       'hparams': HPARAMS,
                       'metrics': {'roc_s': int_auc, 'roc_holdout_seed': ho_auc}}, f, indent=2)

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Ensemble holdout
    roc_s_mean   = round(float(np.mean(int_aucs)), 4)
    mcc_s_mean   = round(float(np.mean(int_mccs)), 4)
    f1_s_mean    = round(float(np.mean(int_f1s)), 4)
    acc_s_mean   = round(float(np.mean(int_accs)), 4)

    hold_ensemble_probs = np.mean(hold_probs_list, axis=0)
    hold_true = y_hold.astype(int).ravel()
    hold_pred = (hold_ensemble_probs > 0.5).astype(int)
    roc_holdout     = round(float(roc_auc_score(hold_true, hold_ensemble_probs)), 4)
    holdout_mcc     = round(float(matthews_corrcoef(hold_true, hold_pred)), 4)
    holdout_f1      = round(float(f1_score(hold_true, hold_pred, zero_division=0)), 4)
    holdout_acc     = round(float(accuracy_score(hold_true, hold_pred)), 4)

    result = {
        'fp_types':    FP_TYPES,
        'n_seeds':     len(int_aucs),
        'internal': {
            'roc_auc': roc_s_mean,
            'mcc':     mcc_s_mean,
            'f1':      f1_s_mean,
            'accuracy': acc_s_mean,
        },
        'holdout': {
            'roc_auc': roc_holdout,
            'mcc':     holdout_mcc,
            'f1':      holdout_f1,
            'accuracy': holdout_acc,
        },
        'seed_roc_s':  int_aucs,
        'hparams':     HPARAMS,
    }
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n  [internal] roc_s={roc_s_mean:.4f}  mcc={mcc_s_mean:.4f}  f1={f1_s_mean:.4f}  acc={acc_s_mean:.4f}')
    print(f'  [holdout]  roc={roc_holdout:.4f}   mcc={holdout_mcc:.4f}  f1={holdout_f1:.4f}  acc={holdout_acc:.4f}')
    print(f'  Saved -> {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='gmlp_ext_metrics.json')
    args = parser.parse_args()
    main(args.output)
