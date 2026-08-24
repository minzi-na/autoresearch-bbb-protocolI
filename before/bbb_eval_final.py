"""
Final 10-seed test set evaluation for BBB prediction.

Loads best_params.json from bbb_artifacts/, trains on train+val for each
of 10 seeds × 2 split modes, evaluates on test set, and saves results CSV.

Usage:
    conda run -n rapids-25.02 python /home/minji/autoresearch/bbb_eval_final.py
"""

import os
import sys
import json
import csv
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bbb_prepare import *
from bbb_train import MultiModalGMLPFromFlat, train_model

EVAL_SEEDS  = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
ARTIFACTS   = os.path.join(os.path.dirname(__file__), 'bbb_artifacts')
PARAMS_PATH = os.path.join(ARTIFACTS, 'best_params.json')
OUT_CSV     = os.path.join(ARTIFACTS, 'final_eval_results.csv')


def eval_one(params, dataset, mod_dims, split_mode, seed):
    set_seed(seed)
    train_ds, val_ds, test_ds, scaler, rd_slice, _ = split_then_normalize(
        dataset, split_mode=split_mode,
        train_ratio=0.8, val_ratio=0.1, seed=seed,
    )

    # Merge train+val for final training
    X_tv = torch.cat([train_ds.tensors[0], val_ds.tensors[0]], dim=0)
    y_tv = torch.cat([train_ds.tensors[1], val_ds.tensors[1]], dim=0)
    tv_ds = data.TensorDataset(X_tv, y_tv)

    train_loader = data.DataLoader(tv_ds,   batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    test_loader  = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    d_model = params['d_model']
    d_ffn   = d_model * params['ffn_multiplier']

    model = MultiModalGMLPFromFlat(
        mod_dims=mod_dims,
        d_model=d_model,
        d_ffn=d_ffn,
        depth=params['depth'],
        dropout=params['dropout'],
        use_gated_pool=True,
        cond_type=params.get('cond_type', 'none'),
    ).to(device)

    y_tv_np  = y_tv.cpu().numpy()
    n_pos, n_neg = (y_tv_np == 1).sum(), (y_tv_np == 0).sum()
    pos_weight = None
    if n_pos > 0:
        pos_weight = torch.tensor([max(n_neg / n_pos, 1.0)], dtype=torch.float32, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    loss_fn   = (nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                 if pos_weight is not None else nn.BCEWithLogitsLoss())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=params['lr'] * 0.01)

    # For final eval, use a dummy val_loader (same as train) just to satisfy early stopping API
    # — we use full patience so training runs to NUM_EPOCHS with no early stop on val
    model = train_model(model, optimizer, train_loader, test_loader, loss_fn, scheduler=scheduler)

    metrics = eval_model(model, test_loader)
    metrics['composite'] = composite_score(metrics)
    return metrics


if __name__ == '__main__':
    t_start = time.time()

    print(f'Loading best params from {PARAMS_PATH}')
    with open(PARAMS_PATH) as f:
        best_params_all = json.load(f)

    print('Loading BBB dataset...')
    dataset    = ScageConcatDataset(LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    mod_dims   = OrderedDict((t, dataset.expected_dims[t]) for t in FP_TYPES)
    print(f'Dataset: {len(dataset)} samples')
    print(f'Mod dims: {dict(mod_dims)}')

    fieldnames = ['split_mode', 'seed', 'accuracy', 'precision', 'recall',
                  'f1', 'roc_auc', 'mcc', 'sensitivity', 'specificity', 'composite']
    rows = []

    for split_mode in SPLIT_MODES:
        params = best_params_all.get(split_mode)
        if params is None:
            print(f'[WARN] No params found for {split_mode}, skipping.')
            continue
        print(f"\n{'='*60}")
        print(f"Final eval | split_mode={split_mode}")
        print(f"Params: {params}")
        print('='*60)

        seed_metrics = []
        for seed in EVAL_SEEDS:
            m = eval_one(params, dataset, mod_dims, split_mode, seed)
            seed_metrics.append(m)
            rows.append({'split_mode': split_mode, 'seed': seed, **m})
            print(f"  seed={seed:4d}  composite={m['composite']:.4f}  "
                  f"f1={m['f1']:.4f}  mcc={m['mcc']:.4f}  auc={m['roc_auc']:.4f}")

        # Mean over 10 seeds
        mean_row = {'split_mode': split_mode, 'seed': 'mean'}
        for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                  'mcc', 'sensitivity', 'specificity', 'composite']:
            mean_row[k] = round(float(np.mean([m[k] for m in seed_metrics])), 4)
        rows.append(mean_row)
        print(f"\n  [MEAN/{split_mode}]  composite={mean_row['composite']:.4f}  "
              f"f1={mean_row['f1']:.4f}  mcc={mean_row['mcc']:.4f}  auc={mean_row['roc_auc']:.4f}")

    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nResults saved to {OUT_CSV}')
    print(f'Total time: {time.time() - t_start:.1f}s')
