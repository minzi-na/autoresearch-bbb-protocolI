"""
TabPFN v2 baseline for BBB prediction.

Feature configs (all within TabPFN's 500-d limit):
  - maccs      : 166-d
  - rdkit      : 217-d
  - maccs+rdkit: 383-d

Evaluation mirrors bbb_train.py:
  - Internal scaffold test: 10-seed mean
  - External (soft voting ensemble across 10 seeds)
  - Holdout  (soft voting ensemble across 10 seeds)

Run with:
  conda run -n tabpfn python tabpfn_baseline.py
"""

import os, random, warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths (mirrors bbb_train.py)
# ---------------------------------------------------------------------------
_INT_DIR      = '/home/minji/BBB/holdout_splits/internal_curated_holdout_10pct_seed42'
INT_LABEL     = f'{_INT_DIR}/internal_curated_label_remaining.csv'

_EXT_DIR      = '/home/minji/BBB/holdout_splits/external_cls_only_holdout_10pct_seed42'
EXT_LABEL     = f'{_EXT_DIR}/external_cls_only_label_remaining.csv'

_HOLDOUT_DIR  = '/home/minji/BBB/holdout_splits/merged_holdout_10pct_seed42'
HOLDOUT_LABEL = f'{_HOLDOUT_DIR}/label_holdout.csv'

SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]

FEATURE_CONFIGS = OrderedDict([
    ('maccs',       ['maccs']),
    ('rdkit',       ['rdkit']),
    ('maccs+rdkit', ['maccs', 'rdkit']),
])

# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None

def get_maccs(mol):
    bv = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(bv.GetNumBits(), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr[1:]  # drop bit 0 (always 0), gives 166-d

def get_rdkit_desc(mol):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(
        [d[0] for d in Descriptors._descList]
    )
    try:
        descs = np.array(calc.CalcDescriptors(mol), dtype=np.float32)
        return np.nan_to_num(descs, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return np.zeros(len(Descriptors._descList), dtype=np.float32)

RDKIT_DIM = len(Descriptors._descList)

def make_feature(mol, fp_types):
    chunks = []
    for t in fp_types:
        if t == 'maccs':
            chunks.append(get_maccs(mol))
        elif t == 'rdkit':
            chunks.append(get_rdkit_desc(mol))
    return np.concatenate(chunks, axis=0).astype(np.float32)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(label_path, fp_types):
    df = pd.read_csv(label_path)
    df = df[['smiles', 'p_np']].rename(columns={'p_np': 'label'}).copy()
    if df['label'].dtype == object:
        df['label'] = df['label'].map({'BBB+': 1, 'BBB-': 0})
    df['label'] = df['label'].astype(int)
    df['smiles'] = df['smiles'].apply(canon)
    df = df.dropna(subset=['smiles']).drop_duplicates(subset='smiles').reset_index(drop=True)

    X, y, smiles_list = [], [], []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None:
            continue
        feat = make_feature(mol, fp_types)
        X.append(feat)
        y.append(row['label'])
        smiles_list.append(row['smiles'])

    return np.stack(X), np.array(y, dtype=int), smiles_list

# ---------------------------------------------------------------------------
# Scaffold split (mirrors bbb_prepare.py split_then_normalize)
# ---------------------------------------------------------------------------

def scaffold_split(smiles_list, y, seed, train_ratio=0.8, val_ratio=0.1):
    df = pd.DataFrame({'smiles': smiles_list, 'label': y})

    def get_scaffold(smi):
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) if m else None

    df['scaffold'] = df['smiles'].apply(get_scaffold)
    groups = sorted(df.groupby('scaffold').groups.values(), key=lambda g: len(g), reverse=True)

    n = len(df)
    train_cap = int(round(train_ratio * n))
    val_cap   = int(round(val_ratio * n))

    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        g = list(g)
        if   len(train_idx) + len(g) <= train_cap: train_idx += g
        elif len(val_idx)   + len(g) <= val_cap:   val_idx   += g
        else:                                        test_idx  += g

    return (np.array(train_idx), np.array(val_idx), np.array(test_idx))

# ---------------------------------------------------------------------------
# Metrics (mirrors bbb_prepare.py eval_model)
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        specificity = sensitivity = 0.0

    return {
        'accuracy':    round(float(accuracy_score(y_true, y_pred)), 4),
        'precision':   round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall':      round(float(sensitivity), 4),
        'f1':          round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'roc_auc':     round(float(roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0), 4),
        'mcc':         round(float(matthews_corrcoef(y_true, y_pred)), 4),
        'sensitivity': round(float(sensitivity), 4),
        'specificity': round(float(specificity), 4),
    }

def composite_score(m):
    return m['f1'] + m['mcc'] + m['roc_auc']

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_config(config_name, fp_types):
    print(f'\n{"="*60}', flush=True)
    print(f'  Feature config: {config_name}  ({"+".join(fp_types)})', flush=True)
    print(f'{"="*60}', flush=True)

    from tabpfn import TabPFNClassifier

    print('Loading internal dataset...', flush=True)
    X_int, y_int, smi_int = load_dataset(INT_LABEL, fp_types)
    print(f'  internal: {X_int.shape[0]} samples, {X_int.shape[1]} features', flush=True)

    print('Loading external dataset...', flush=True)
    X_ext, y_ext, _ = load_dataset(EXT_LABEL, fp_types)
    print(f'  external: {X_ext.shape[0]} samples', flush=True)

    print('Loading holdout dataset...', flush=True)
    X_hold, y_hold, _ = load_dataset(HOLDOUT_LABEL, fp_types)
    print(f'  holdout:  {X_hold.shape[0]} samples', flush=True)

    int_seed_metrics = []
    ext_seed_probs   = []
    hold_seed_probs  = []

    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed)

        train_idx, val_idx, test_idx = scaffold_split(smi_int, y_int, seed=seed)

        # Always copy so in-place normalization doesn't bleed across seeds
        X_tr, y_tr = X_int[train_idx].copy(), y_int[train_idx]
        X_te, y_te = X_int[test_idx].copy(),  y_int[test_idx]
        X_ext_s  = X_ext.copy()
        X_hold_s = X_hold.copy()

        # Normalize rdkit continuous features (fit on train only)
        if 'rdkit' in fp_types:
            rdkit_start = 166 if 'maccs' in fp_types else 0
            rdkit_end   = rdkit_start + RDKIT_DIM
            scaler = StandardScaler().fit(X_tr[:, rdkit_start:rdkit_end])
            for arr in [X_tr, X_te, X_ext_s, X_hold_s]:
                arr[:, rdkit_start:rdkit_end] = scaler.transform(arr[:, rdkit_start:rdkit_end]).astype(np.float32)

        clf = TabPFNClassifier(n_estimators=8, random_state=seed)
        clf.fit(X_tr, y_tr)

        # Internal scaffold test metrics
        if len(test_idx) > 0:
            prob_te = clf.predict_proba(X_te)[:, 1]
            int_seed_metrics.append(compute_metrics(y_te, prob_te))

        # Soft-voting probs for external and holdout
        ext_seed_probs.append(clf.predict_proba(X_ext_s)[:, 1])
        hold_seed_probs.append(clf.predict_proba(X_hold_s)[:, 1])

        auc_int = int_seed_metrics[-1]['roc_auc'] if len(test_idx) > 0 else float('nan')
        auc_ext = float(roc_auc_score(y_ext, ext_seed_probs[-1]))
        print(f'  seed={seed:>4d}  int={auc_int:.4f}  ext={auc_ext:.4f}', flush=True)

    # Internal: 10-seed mean
    int_mean = {k: round(float(np.mean([m[k] for m in int_seed_metrics])), 4)
                for k in int_seed_metrics[0]}

    # External / holdout: soft voting
    ext_prob_ensemble  = np.mean(ext_seed_probs,  axis=0)
    hold_prob_ensemble = np.mean(hold_seed_probs, axis=0)
    ext_metrics  = compute_metrics(y_ext,  ext_prob_ensemble)
    hold_metrics = compute_metrics(y_hold, hold_prob_ensemble)

    print(f'\n  --- {config_name} Results ---', flush=True)
    print(f'  Internal (10-seed mean):', flush=True)
    print(f'    roc_auc={int_mean["roc_auc"]:.4f}  f1={int_mean["f1"]:.4f}  '
          f'mcc={int_mean["mcc"]:.4f}  acc={int_mean["accuracy"]:.4f}')
    print(f'  External (soft voting):', flush=True)
    print(f'    roc_auc={ext_metrics["roc_auc"]:.4f}  f1={ext_metrics["f1"]:.4f}  '
          f'mcc={ext_metrics["mcc"]:.4f}  acc={ext_metrics["accuracy"]:.4f}')
    print(f'  Holdout (soft voting):', flush=True)
    print(f'    roc_auc={hold_metrics["roc_auc"]:.4f}  f1={hold_metrics["f1"]:.4f}  '
          f'mcc={hold_metrics["mcc"]:.4f}  acc={hold_metrics["accuracy"]:.4f}')

    return {
        'config': config_name,
        'int_mean': int_mean,
        'ext': ext_metrics,
        'holdout': hold_metrics,
    }


def main():
    results = []
    for config_name, fp_types in FEATURE_CONFIGS.items():
        r = run_config(config_name, fp_types)
        results.append(r)

    # Summary table
    print(f'\n{"="*80}', flush=True)
    print('  SUMMARY', flush=True)
    print(f'{"="*80}', flush=True)
    header = (f'{"Config":<15} {"Int_AUC":>8} {"Int_F1":>7} {"Int_MCC":>8} {"Int_ACC":>8} | '
              f'{"Ext_AUC":>8} {"Ext_F1":>7} {"Ext_MCC":>8} {"Ext_ACC":>8} | '
              f'{"Hold_AUC":>9} {"Hold_F1":>8} {"Hold_MCC":>9} {"Hold_ACC":>9}')
    print(header, flush=True)
    print('-' * len(header), flush=True)
    for r in results:
        m_int  = r['int_mean']
        m_ext  = r['ext']
        m_hold = r['holdout']
        print(
            f'{r["config"]:<15} '
            f'{m_int["roc_auc"]:>8.4f} {m_int["f1"]:>7.4f} {m_int["mcc"]:>8.4f} {m_int["accuracy"]:>8.4f} | '
            f'{m_ext["roc_auc"]:>8.4f} {m_ext["f1"]:>7.4f} {m_ext["mcc"]:>8.4f} {m_ext["accuracy"]:>8.4f} | '
            f'{m_hold["roc_auc"]:>9.4f} {m_hold["f1"]:>8.4f} {m_hold["mcc"]:>9.4f} {m_hold["accuracy"]:>9.4f}'
        )

    # Save TSV
    rows = []
    for r in results:
        for split_key, metrics in [('int', r['int_mean']), ('ext', r['ext']), ('holdout', r['holdout'])]:
            rows.append({
                'config': r['config'],
                'split': split_key,
                'roc_auc':  metrics['roc_auc'],
                'f1':       metrics['f1'],
                'mcc':      metrics['mcc'],
                'accuracy': metrics['accuracy'],
            })
    out_path = os.path.join(os.path.dirname(__file__), 'tabpfn_baseline_results.tsv')
    pd.DataFrame(rows).to_csv(out_path, sep='\t', index=False)
    print(f'\nResults saved to {out_path}', flush=True)


if __name__ == '__main__':
    main()
