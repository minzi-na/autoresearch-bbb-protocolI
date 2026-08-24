"""
External validation on external_dataset_filtered_v2.csv (9272 samples).

Protocol (same training setup as bbb_test_eval.py):
  - Scaffold split bench_label.csv → 80% train / 10% val / 10% (discarded)
  - Train on 80%, early-stop on val 10% (composite score, EMA)
  - Evaluate trained model on external_dataset_filtered_v2.csv
  - RDKit scaler fitted on train split, applied to external set
  - 10 seeds → mean ± std

Usage:
    conda run -n rapids-25.02 python /home/minji/autoresearch/bbb_external_eval.py
"""

import os, sys, json, time, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bbb_prepare import *
from bbb_train import MultiModalGMLPFromFlat, train_model

EVAL_SEEDS  = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
ARTIFACTS   = os.path.join(os.path.dirname(__file__), 'bbb_artifacts')
PARAMS_PATH = os.path.join(ARTIFACTS, 'best_params.json')
OUT_CSV     = os.path.join(ARTIFACTS, 'external_eval_results.csv')

EXT_LABEL_PATH  = '/home/minji/BBB/data_process/external_dataset_filtered_v2.csv'
EXT_EMBED_PATHS = {
    'scage1': '/home/minji/BBB/data_process/embeddings_v2/external_dataset_filtered_v2_scage1.csv',
    'scage2': '/home/minji/BBB/data_process/embeddings_v2/external_dataset_filtered_v2_scage2.csv',
    'mole':   '/home/minji/BBB/data_process/embeddings_v2/external_dataset_filtered_v2_mole.csv',
}


def build_external_dataset(ext_label_path, ext_embed_paths, fp_types, expected_dims):
    """Load external dataset using the same feature schema as bench_label.csv."""
    df = pd.read_csv(ext_label_path)
    if 'p_np' in df.columns:
        df = df[['smiles', 'p_np']].rename(columns={'p_np': 'label'})
        df['label'] = df['label'].replace({'BBB-': 0, 'BBB+': 1})
    elif 'BBclass' in df.columns:
        df = df[['smiles', 'BBclass']].rename(columns={'BBclass': 'label'})
    else:
        raise ValueError("Label column not found")

    df = df.drop_duplicates(subset='smiles').reset_index(drop=True)

    def canon(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, canonical=True) if m else None
    df['smiles'] = df['smiles'].apply(canon)
    df = df.dropna(subset=['smiles']).reset_index(drop=True)

    embed_dicts, _ = load_molecular_embeddings(ext_embed_paths)

    features, labels, failed = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Building external features'):
        smi = row['smiles']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed.append(smi); continue
        feat = make_feature_vector(mol, smi, fp_types, expected_dims, embed_dicts)
        if feat is None or feat.ndim != 1:
            failed.append(smi); continue
        features.append(feat)
        labels.append(int(row['label']))

    print(f'  External: {len(features)} valid / {len(failed)} failed')
    X = torch.tensor(np.stack(features), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    return X, y  # raw tensors (scaler applied per-seed below)


def eval_one(params, train_dataset, X_ext_raw, y_ext, mod_dims, seed):
    set_seed(seed)

    # Scaffold split of bench_label.csv — same as bbb_test_eval.py
    train_ds, val_ds, _, scaler, rd_slice, _ = split_then_normalize(
        train_dataset, split_mode='scaffold',
        train_ratio=0.8, val_ratio=0.1, seed=seed,
    )

    train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Apply scaler (fitted on train split) to external dataset
    X_ext = X_ext_raw.clone()
    if scaler is not None:
        rd_start, rd_end = rd_slice
        X_ext[:, rd_start:rd_end] = torch.tensor(
            scaler.transform(X_ext[:, rd_start:rd_end]), dtype=torch.float32
        )
    ext_ds     = data.TensorDataset(X_ext, y_ext)
    ext_loader = data.DataLoader(ext_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

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
        d_model=d_model, d_ffn=d_ffn, depth=params['depth'],
        dropout=dropout, use_gated_pool=True,
        cond_type=cond_type, modal_drop_p=modal_drop_p,
    ).to(device)

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

    trained_model = train_model(model, optimizer, train_loader, val_loader, loss_fn)

    metrics = eval_model(trained_model, ext_loader)
    metrics['composite'] = composite_score(metrics)
    metrics['n_train']   = len(train_ds)
    return metrics


if __name__ == '__main__':
    t_start = time.time()

    print(f'Loading best params from {PARAMS_PATH}')
    with open(PARAMS_PATH) as f:
        best_params_all = json.load(f)

    print('Loading training dataset (bench_label.csv)...')
    train_dataset = ScageConcatDataset(LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    mod_dims      = OrderedDict((t, train_dataset.expected_dims[t]) for t in FP_TYPES)
    print(f'Training dataset: {len(train_dataset)} samples')

    print('Building external dataset...')
    X_ext_raw, y_ext = build_external_dataset(
        EXT_LABEL_PATH, EXT_EMBED_PATHS, FP_TYPES, train_dataset.expected_dims
    )
    print(f'External dataset: {len(y_ext)} samples  '
          f'(BBB+: {int(y_ext.sum())}  BBB-: {int((y_ext==0).sum())})')

    fieldnames = ['split_mode', 'seed', 'n_train', 'accuracy', 'precision', 'recall',
                  'f1', 'roc_auc', 'mcc', 'sensitivity', 'specificity', 'composite']
    all_rows = []

    # Use scaffold split params (as requested)
    for split_mode in ['scaffold']:
        params = best_params_all.get(split_mode)
        if params is None:
            print(f'[WARN] No params for {split_mode}, skipping.')
            continue

        print(f"\n{'='*60}")
        print(f"External eval | scaffold split params | 10 seeds")
        print(f"Params: {params}")
        print('='*60)

        seed_metrics = []
        for seed in EVAL_SEEDS:
            m = eval_one(params, train_dataset, X_ext_raw, y_ext, mod_dims, seed)
            seed_metrics.append(m)
            all_rows.append({'split_mode': split_mode, 'seed': seed, **m})
            print(f"  seed={seed:4d}  composite={m['composite']:.4f}  "
                  f"f1={m['f1']:.4f}  mcc={m['mcc']:.4f}  auc={m['roc_auc']:.4f}  "
                  f"acc={m['accuracy']:.4f}  n_train={m['n_train']}")

        print(f"\n  {'─'*50}")
        for k in ['composite', 'f1', 'mcc', 'roc_auc', 'accuracy', 'precision', 'recall', 'specificity']:
            vals = [m[k] for m in seed_metrics]
            print(f"  [external/{split_mode}] {k:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        mean_row = {'split_mode': split_mode, 'seed': 'MEAN', 'n_train': seed_metrics[0]['n_train']}
        for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                  'mcc', 'sensitivity', 'specificity', 'composite']:
            mean_row[k] = round(float(np.mean([m[k] for m in seed_metrics])), 4)
        all_rows.append(mean_row)

    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'\nResults saved to {OUT_CSV}')
    print(f'Total time: {time.time() - t_start:.1f}s')
