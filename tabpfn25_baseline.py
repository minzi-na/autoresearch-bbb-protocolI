"""
TabPFN-2.5 baseline for BBB prediction (combo1 features).

Feature config: maccs + avalon + rdkit + mole  (166 + 512 + 217 + 768 = 1663-d)
  - Within TabPFN-2.5's 2000-feature limit.

Evaluation mirrors bbb_train.py / bbb_prepare.py:
  - Scaffold split (sort groups by size desc; train 0.8 / val 0.1 / test 0.1)
  - Internal scaffold test: 10-seed mean
  - External : 10-seed soft-voting ensemble
  - Holdout  : 10-seed soft-voting ensemble
  - RDKit descriptors normalized with StandardScaler fit on train only.

Run with:
  conda run -n tabpfn python tabpfn25_baseline.py
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
from rdkit.Avalon import pyAvalonTools

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths (mirror bbb_train.py / bbb_prepare.py)
# ---------------------------------------------------------------------------
_INT_DIR  = '/home/minji/BBB/holdout_splits/internal_curated_holdout_10pct_seed42'
_EXT_DIR  = '/home/minji/BBB/holdout_splits/external_cls_only_holdout_10pct_seed42'
_HOLD_DIR = '/home/minji/BBB/holdout_splits/merged_holdout_10pct_seed42'

INT_LABEL  = f'{_INT_DIR}/internal_curated_label_remaining.csv'
INT_MOLE   = f'{_INT_DIR}/internal_curated_mole_remaining.csv'

EXT_LABEL  = f'{_EXT_DIR}/external_cls_only_label_remaining.csv'
EXT_MOLE   = f'{_EXT_DIR}/external_cls_only_mole_remaining.csv'

HOLD_LABEL = f'{_HOLD_DIR}/label_holdout.csv'
HOLD_MOLE  = f'{_HOLD_DIR}/mole_holdout.csv'

SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
FP_TYPES = ['maccs', 'avalon', 'rdkit', 'mole']
CONFIG_NAME = '+'.join(FP_TYPES)

AVALON_BITS = 512
MACCS_DIM = 166
RDKIT_DIM = len(Descriptors._descList)
MOLE_DIM = 768  # default if file missing/empty

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
    return arr[1:]  # drop bit 0

def get_avalon(mol, nbits=AVALON_BITS):
    bv = pyAvalonTools.GetAvalonFP(mol, nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

_RDKIT_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(
    [d[0] for d in Descriptors._descList]
)

def get_rdkit_desc(mol):
    try:
        descs = np.array(_RDKIT_CALC.CalcDescriptors(mol), dtype=np.float32)
        return np.nan_to_num(descs, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return np.zeros(RDKIT_DIM, dtype=np.float32)

# ---------------------------------------------------------------------------
# mole embedding loading (smiles -> 768-d vector)
# ---------------------------------------------------------------------------

def load_mole_embeddings(path):
    df = pd.read_csv(path)
    df['smiles'] = df['smiles'].apply(canon)
    df = df.dropna(subset=['smiles']).reset_index(drop=True)
    cols = [c for c in df.columns if c != 'smiles']
    table = {}
    for _, row in df.iterrows():
        v = row[cols].to_numpy(dtype=np.float32, copy=False)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        table[row['smiles']] = v
    dim = len(cols) if cols else MOLE_DIM
    return table, dim

def safe_mole(table, dim, smi):
    v = table.get(smi)
    if v is None:
        return np.zeros(dim, dtype=np.float32)
    if v.shape[0] == dim:
        return v.astype(np.float32, copy=False)
    if v.shape[0] < dim:
        return np.concatenate([v, np.zeros(dim - v.shape[0], dtype=np.float32)])
    return v[:dim].astype(np.float32, copy=False)

# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def load_dataset(label_path, mole_path):
    df = pd.read_csv(label_path)
    df = df[['smiles', 'p_np']].rename(columns={'p_np': 'label'}).copy()
    if df['label'].dtype == object:
        df['label'] = df['label'].map({'BBB+': 1, 'BBB-': 0})
    df['label'] = df['label'].astype(int)
    df['smiles'] = df['smiles'].apply(canon)
    df = df.dropna(subset=['smiles']).drop_duplicates(subset='smiles').reset_index(drop=True)

    mole_table, mole_dim = load_mole_embeddings(mole_path)

    X, y, smis = [], [], []
    for _, row in df.iterrows():
        smi = row['smiles']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        chunks = [
            get_maccs(mol),
            get_avalon(mol),
            get_rdkit_desc(mol),
            safe_mole(mole_table, mole_dim, smi),
        ]
        feat = np.concatenate(chunks, axis=0).astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        X.append(feat)
        y.append(row['label'])
        smis.append(smi)
    return np.stack(X), np.array(y, dtype=int), smis, mole_dim

# ---------------------------------------------------------------------------
# Scaffold split (mirrors bbb_prepare.split_then_normalize, mode='scaffold')
# ---------------------------------------------------------------------------

def scaffold_split(smiles_list, y, seed, train_ratio=0.8, val_ratio=0.1):
    random.seed(seed); np.random.seed(seed)
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
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)

# ---------------------------------------------------------------------------
# Metrics
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f'\n{"="*70}', flush=True)
    print(f'  TabPFN-2.5  Feature config: {CONFIG_NAME}', flush=True)
    print(f'{"="*70}', flush=True)

    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    print('Loading internal dataset...', flush=True)
    X_int, y_int, smi_int, mole_dim = load_dataset(INT_LABEL, INT_MOLE)
    print(f'  internal: {X_int.shape[0]} samples, {X_int.shape[1]} features (mole_dim={mole_dim})', flush=True)

    print('Loading external dataset...', flush=True)
    X_ext, y_ext, _, _ = load_dataset(EXT_LABEL, EXT_MOLE)
    print(f'  external: {X_ext.shape[0]} samples', flush=True)

    print('Loading holdout dataset...', flush=True)
    X_hold, y_hold, _, _ = load_dataset(HOLD_LABEL, HOLD_MOLE)
    print(f'  holdout:  {X_hold.shape[0]} samples', flush=True)

    # rdkit slice = after maccs(166) + avalon(512), length RDKIT_DIM
    rd_start = MACCS_DIM + AVALON_BITS
    rd_end   = rd_start + RDKIT_DIM

    int_seed_metrics = []
    ext_seed_probs = []
    hold_seed_probs = []

    for seed in SEEDS:
        train_idx, val_idx, test_idx = scaffold_split(smi_int, y_int, seed=seed)

        X_tr, y_tr = X_int[train_idx].copy(), y_int[train_idx]
        X_te, y_te = X_int[test_idx].copy(),  y_int[test_idx]
        X_ext_s  = X_ext.copy()
        X_hold_s = X_hold.copy()

        # Normalize rdkit slice (fit on train only)
        scaler = StandardScaler().fit(X_tr[:, rd_start:rd_end])
        for arr in [X_tr, X_te, X_ext_s, X_hold_s]:
            arr[:, rd_start:rd_end] = scaler.transform(
                arr[:, rd_start:rd_end]
            ).astype(np.float32)

        clf = TabPFNClassifier(
            n_estimators=8,
            random_state=seed,
            ignore_pretraining_limits=False,  # within v2.5 limits
        )
        clf.fit(X_tr, y_tr)

        if len(test_idx) > 0:
            prob_te = clf.predict_proba(X_te)[:, 1]
            int_seed_metrics.append(compute_metrics(y_te, prob_te))

        ext_seed_probs.append(clf.predict_proba(X_ext_s)[:, 1])
        hold_seed_probs.append(clf.predict_proba(X_hold_s)[:, 1])

        auc_int = int_seed_metrics[-1]['roc_auc'] if len(test_idx) > 0 else float('nan')
        auc_ext = float(roc_auc_score(y_ext, ext_seed_probs[-1]))
        auc_hold = float(roc_auc_score(y_hold, hold_seed_probs[-1]))
        print(f'  seed={seed:>4d}  int={auc_int:.4f}  ext={auc_ext:.4f}  hold={auc_hold:.4f}', flush=True)

    int_mean = {k: round(float(np.mean([m[k] for m in int_seed_metrics])), 4)
                for k in int_seed_metrics[0]}
    ext_metrics  = compute_metrics(y_ext,  np.mean(ext_seed_probs,  axis=0))
    hold_metrics = compute_metrics(y_hold, np.mean(hold_seed_probs, axis=0))

    # --- Summary table (matches user-requested format) ---
    print(f'\n{"="*120}', flush=True)
    print('  SUMMARY  (TabPFN-2.5, combo1 features: maccs+avalon+rdkit+mole)', flush=True)
    print(f'{"="*120}', flush=True)
    header = (
        f'{"Config":<25} '
        f'{"Int_AUC":>8} {"Int_F1":>8} {"Int_MCC":>8} | '
        f'{"Ext_AUC":>8} {"Ext_F1":>8} {"Ext_MCC":>8} | '
        f'{"Hold_AUC":>9} {"Hold_F1":>9} {"Hold_MCC":>9}'
    )
    print(header, flush=True)
    print('-' * len(header), flush=True)
    print(
        f'{CONFIG_NAME:<25} '
        f'{int_mean["roc_auc"]:>8.4f} {int_mean["f1"]:>8.4f} {int_mean["mcc"]:>8.4f} | '
        f'{ext_metrics["roc_auc"]:>8.4f} {ext_metrics["f1"]:>8.4f} {ext_metrics["mcc"]:>8.4f} | '
        f'{hold_metrics["roc_auc"]:>9.4f} {hold_metrics["f1"]:>9.4f} {hold_metrics["mcc"]:>9.4f}',
        flush=True,
    )

    # Save TSV
    rows = [
        {'config': CONFIG_NAME, 'split': 'int',     **int_mean},
        {'config': CONFIG_NAME, 'split': 'ext',     **ext_metrics},
        {'config': CONFIG_NAME, 'split': 'holdout', **hold_metrics},
    ]
    out_path = os.path.join(os.path.dirname(__file__), 'tabpfn25_baseline_results.tsv')
    pd.DataFrame(rows).to_csv(out_path, sep='\t', index=False)
    print(f'\nResults saved to {out_path}', flush=True)


if __name__ == '__main__':
    main()
