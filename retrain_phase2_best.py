"""
Retrain combo1 with Phase 2 best hyperparameters and save models.

Phase 2 best (roc_s=0.87848):
  dropout=0.046019, lr=1.1022e-4, wd=3.88e-6, warmup_epochs=0

Architecture: iter90 (same as bbb_train.py)
Output: bbb_artifacts/phase2_best/seed_*/
"""

import os
import sys
import json
import pickle
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.optim as optim
from bbb_prepare import *
from bbb_train import (
    SpatialGatingUnit, gMLPBlock, gMLP, MultiModalGMLPFromFlat,
    train_model, predict_probs, roc_auc_from_probs,
    load_cached_dataset, apply_scaler,
    SEEDS, SPLIT_MODE,
    LABEL_PATH, EMBED_PATHS, FP_TYPES,
    EXT_LABEL_PATH, EXT_EMBED_PATHS,
    HOLDOUT_LABEL_PATH, HOLDOUT_EMBED_PATHS,
)

# Phase 2 best hyperparameters
P2_BEST = {
    'dropout':       0.046019393915327604,
    'lr':            0.00011022334100107642,
    'weight_decay':  3.88007962016146e-06,
    'd_model':       512,
    'd_ffn':         1048,
    'depth':         4,
    'batch_size':    128,
    'num_epochs':    50,
    'patience':      10,
    'pos_weight':    0.08,
    'stochastic_depth_rate': 0.05,
}

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'bbb_artifacts', 'phase2_best')


def run_retrain(dataset, ext_dataset, holdout_dataset, mod_dims):
    int_aucs = []
    ext_seed_probs, holdout_seed_probs = [], []

    for seed in SEEDS:
        set_seed(seed)
        train_ds, val_ds, test_ds, scaler, (rd_start, rd_end), _ = split_then_normalize(
            dataset, split_mode=SPLIT_MODE,
            train_ratio=0.8, val_ratio=0.1, seed=seed,
        )

        ext_X     = apply_scaler(ext_dataset.features,     scaler, rd_start, rd_end)
        holdout_X = apply_scaler(holdout_dataset.features, scaler, rd_start, rd_end)

        bs = P2_BEST['batch_size']
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
            d_model=P2_BEST['d_model'],
            d_ffn=P2_BEST['d_ffn'],
            depth=P2_BEST['depth'],
            dropout=P2_BEST['dropout'],
            use_gated_pool=True,
            stochastic_depth_rate=P2_BEST['stochastic_depth_rate'],
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=P2_BEST['lr'],
            weight_decay=P2_BEST['weight_decay'],
        )
        pos_weight = torch.tensor([P2_BEST['pos_weight']]).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model = train_model(model, optimizer, train_loader, val_loader, loss_fn,
                            num_epochs=P2_BEST['num_epochs'],
                            patience=P2_BEST['patience'])

        int_auc       = eval_model(model, test_loader)['roc_auc']
        ext_probs     = predict_probs(model, ext_loader)
        holdout_probs = predict_probs(model, holdout_loader)
        ext_auc     = roc_auc_from_probs(ext_dataset.labels,     ext_probs)
        holdout_auc = roc_auc_from_probs(holdout_dataset.labels, holdout_probs)

        int_aucs.append(int_auc)
        ext_seed_probs.append(ext_probs)
        holdout_seed_probs.append(holdout_probs)
        print(f"  seed={seed:>4d}  int={int_auc:.4f}  ext={ext_auc:.4f}  holdout={holdout_auc:.4f}")

        seed_dir = os.path.join(SAVE_DIR, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(seed_dir, 'model.pth'))
        if scaler is not None:
            with open(os.path.join(seed_dir, 'rdkit_scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
        config = {
            'seed':       seed,
            'split_mode': SPLIT_MODE,
            'fp_types':   FP_TYPES,
            'mod_dims':   {k: int(v) for k, v in mod_dims.items()},
            'hparams':    P2_BEST,
            'metrics': {
                'roc_auc_scaffold': int_auc,
                'roc_auc_external': ext_auc,
                'roc_auc_holdout':  holdout_auc,
            },
        }
        with open(os.path.join(seed_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        del model, optimizer
        del train_loader, val_loader, test_loader, ext_loader, holdout_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean = lambda lst: float(sum(lst) / len(lst))
    ext_ensemble_probs     = np.mean(np.stack(ext_seed_probs,     axis=0), axis=0)
    holdout_ensemble_probs = np.mean(np.stack(holdout_seed_probs, axis=0), axis=0)
    ext_ens_auc     = roc_auc_from_probs(ext_dataset.labels,     ext_ensemble_probs)
    holdout_ens_auc = roc_auc_from_probs(holdout_dataset.labels, holdout_ensemble_probs)

    print(f"\n  [ensemble] ext={ext_ens_auc:.4f}  holdout={holdout_ens_auc:.4f}")
    print(f"  [mean internal] roc_s={mean(int_aucs):.5f}")

    summary = {
        'roc_auc_scaffold_mean': mean(int_aucs),
        'roc_auc_external_ensemble': ext_ens_auc,
        'roc_auc_holdout_ensemble':  holdout_ens_auc,
        'hparams': P2_BEST,
    }
    with open(os.path.join(SAVE_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nModels saved to: {SAVE_DIR}")


if __name__ == '__main__':
    print('Loading internal dataset...')
    dataset       = load_cached_dataset('internal', LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    expected_dims = dataset.expected_dims
    mod_dims      = OrderedDict((t, expected_dims[t]) for t in FP_TYPES)
    print(f'Internal : {len(dataset)} samples')

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

    print(f'\nRetraining with Phase 2 best hparams -> {SAVE_DIR}')
    run_retrain(dataset, ext_dataset, holdout_dataset, mod_dims)
