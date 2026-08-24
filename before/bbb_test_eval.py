"""
Final test-set evaluation with best params from Exp31 autoresearch.

Protocol:
  - 80/10/10 train/val/test split (scaffold-group-aware)
  - Train on 80%, early-stop on val 10% (composite score), eval on test 10%
  - 10 seeds → report mean ± std
  - Optimizer / loss / EMA identical to bbb_train.py (Adam, label smoothing, EMA)

Usage:
    conda run -n rapids-25.02 python /home/minji/autoresearch/bbb_test_eval.py
"""

import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bbb_prepare import *
from bbb_train import MultiModalGMLPFromFlat, train_model

EVAL_SEEDS  = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
ARTIFACTS   = os.path.join(os.path.dirname(__file__), 'bbb_artifacts')
PARAMS_PATH = os.path.join(ARTIFACTS, 'best_params.json')
OUT_CSV     = os.path.join(ARTIFACTS, 'test_eval_results.csv')


def eval_one(params, dataset, mod_dims, split_mode, seed):
    """Train with best params on train split, early-stop on val, evaluate on test."""
    set_seed(seed)
    train_ds, val_ds, test_ds, _, _, _ = split_then_normalize(
        dataset, split_mode=split_mode,
        train_ratio=0.8, val_ratio=0.1, seed=seed,
    )

    train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = data.DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    d_model      = params['d_model']
    d_ffn        = d_model * params['ffn_multiplier']
    dropout      = params['dropout']
    lr           = params['lr']
    weight_decay = params['weight_decay']
    cond_type    = params.get('cond_type', 'none')
    modal_drop_p = params.get('modal_drop_p', 0.0)
    ls_eps       = params.get('ls_eps', 0.0)

    model = MultiModalGMLPFromFlat(
        mod_dims=mod_dims,
        d_model=d_model,
        d_ffn=d_ffn,
        depth=params['depth'],
        dropout=dropout,
        use_gated_pool=True,
        cond_type=cond_type,
        modal_drop_p=modal_drop_p,
    ).to(device)

    # Pos weight from train labels (same as bbb_train.py objective)
    y_train  = train_ds.tensors[1].cpu().numpy()
    n_pos, n_neg = (y_train == 1).sum(), (y_train == 0).sum()
    pos_weight = None
    if n_pos > 0:
        pos_weight = torch.tensor([max(n_neg / n_pos, 1.0)], dtype=torch.float32, device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    base_loss = (nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
                 if pos_weight is not None else nn.BCEWithLogitsLoss(reduction='none'))

    def loss_fn(logits, targets):
        smooth = targets * (1.0 - ls_eps) + 0.5 * ls_eps
        return base_loss(logits, smooth).mean()

    # train_model: EMA + composite-score early stopping on val
    trained_model = train_model(model, optimizer, train_loader, val_loader, loss_fn)

    metrics = eval_model(trained_model, test_loader)
    metrics['composite'] = composite_score(metrics)
    metrics['n_train']   = len(train_ds)
    metrics['n_val']     = len(val_ds)
    metrics['n_test']    = len(test_ds)
    return metrics


if __name__ == '__main__':
    t_start = time.time()

    print(f'Loading best params from {PARAMS_PATH}')
    with open(PARAMS_PATH) as f:
        best_params_all = json.load(f)

    print('Loading BBB dataset...')
    dataset  = ScageConcatDataset(LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    mod_dims = OrderedDict((t, dataset.expected_dims[t]) for t in FP_TYPES)
    print(f'Dataset: {len(dataset)} samples')
    print(f'Mod dims: {dict(mod_dims)}')

    import csv
    fieldnames = ['split_mode', 'seed', 'n_train', 'n_val', 'n_test',
                  'accuracy', 'precision', 'recall', 'f1',
                  'roc_auc', 'mcc', 'sensitivity', 'specificity', 'composite']
    all_rows = []

    for split_mode in ['scaffold']:  # random_scaffold skipped per user request
        params = best_params_all.get(split_mode)
        if params is None:
            print(f'[WARN] No params for {split_mode}, skipping.')
            continue

        print(f"\n{'='*60}")
        print(f"Test eval | split_mode={split_mode} | 10 seeds")
        print(f"Params: {params}")
        print('='*60)

        seed_metrics = []
        for seed in EVAL_SEEDS:
            m = eval_one(params, dataset, mod_dims, split_mode, seed)
            seed_metrics.append(m)
            all_rows.append({'split_mode': split_mode, 'seed': seed, **m})
            print(f"  seed={seed:4d}  composite={m['composite']:.4f}  "
                  f"f1={m['f1']:.4f}  mcc={m['mcc']:.4f}  auc={m['roc_auc']:.4f}  "
                  f"acc={m['accuracy']:.4f}  n_test={m['n_test']}")

        # Summary
        print(f"\n  {'─'*50}")
        for k in ['composite', 'f1', 'mcc', 'roc_auc', 'accuracy', 'precision', 'recall', 'specificity']:
            vals = [m[k] for m in seed_metrics]
            print(f"  [{split_mode}] {k:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        mean_row = {'split_mode': split_mode, 'seed': 'MEAN'}
        for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                  'mcc', 'sensitivity', 'specificity', 'composite']:
            mean_row[k] = round(float(np.mean([m[k] for m in seed_metrics])), 4)
        mean_row['n_train'] = seed_metrics[0]['n_train']
        mean_row['n_val']   = seed_metrics[0]['n_val']
        mean_row['n_test']  = seed_metrics[0]['n_test']
        all_rows.append(mean_row)

    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'\nResults saved to {OUT_CSV}')
    print(f'Total time: {time.time() - t_start:.1f}s')
