"""
BBB prediction - fixed constants, data utilities, and evaluation.
DO NOT MODIFY THIS FILE.

Provides:
- Data loading and multi-modal feature engineering
- ScageConcatDataset: fingerprint + learned embedding dataset
- split_then_normalize: scaffold / random_scaffold splitting with RDKit normalization
- eval_model: binary classification metrics (returns dict)
- composite_score: the fixed autoresearch metric (f1 + mcc + roc_auc, higher is better)
"""

import os, json, pickle, random
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score,
)

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# ---------------------------------------------------------------------------
# Fixed constants (do not modify during autoresearch loop)
# Each worktree sets LABEL_PATH, EMBED_PATHS, FP_TYPES once for its combo,
# then leaves this file untouched for the duration of the experiment.
#
# combo1: maccs+avalon+rdkit+mole
# combo2: ecfp+maccs+avalon+tt+rdkit+scage1+mole
# combo3: ecfp+maccs+avalon+tt+rdkit+mole
# ---------------------------------------------------------------------------

_HOLDOUT_DIR = '/home/minji/BBB/holdout_splits/internal_curated_holdout_10pct_seed42'

LABEL_PATH  = f'{_HOLDOUT_DIR}/internal_curated_label_remaining.csv'
EMBED_PATHS = {
    'scage1': f'{_HOLDOUT_DIR}/internal_curated_scage1_remaining.csv',
    'mole':   f'{_HOLDOUT_DIR}/internal_curated_mole_remaining.csv',
}

# combo2: ecfp+maccs+avalon+tt+rdkit+scage1+mole
FP_TYPES    = ['ecfp', 'maccs', 'avalon', 'tt', 'rdkit', 'scage1', 'mole']

SPLIT_MODES = ['scaffold']

NUM_EPOCHS  = 50
PATIENCE    = 10
BATCH_SIZE  = 128
BASE_SEED   = 600

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed=700):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        torch.use_deterministic_algorithms(True, warn_only=True)

set_seed(BASE_SEED)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def to_numpy_bitvect(bitvect, n_bits=None, drop_first=False):
    if n_bits is None:
        n_bits = bitvect.GetNumBits()
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    if drop_first:
        arr = arr[1:]
    return arr.astype(np.float32)

def get_ecfp(mol, radius=2, nbits=1024):
    return to_numpy_bitvect(
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits), n_bits=nbits
    )

def get_maccs(mol):
    bv = MACCSkeys.GenMACCSKeys(mol)
    return to_numpy_bitvect(bv, n_bits=bv.GetNumBits(), drop_first=True)

def get_avalon(mol, nbits=512):
    from rdkit.Avalon import pyAvalonTools
    return to_numpy_bitvect(pyAvalonTools.GetAvalonFP(mol, nbits), n_bits=nbits)

def get_topological_torsion(mol, nbits=1024):
    bv = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nbits)
    return to_numpy_bitvect(bv, n_bits=nbits)

def get_rdkit_desc(mol):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(
        [d[0] for d in Descriptors._descList]
    )
    try:
        descs = np.array(calc.CalcDescriptors(mol), dtype=np.float32)
        descs = np.nan_to_num(descs, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        descs = np.zeros(len(Descriptors._descList), dtype=np.float32)
    return descs

def get_rdkit_descriptor_length():
    return len(Descriptors._descList)

# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_molecular_embeddings(embed_paths: dict):
    embed_data, embed_dims = {}, {}
    for name, path in embed_paths.items():
        try:
            df = pd.read_csv(path)
        except Exception:
            embed_data[name] = {}
            embed_dims[name] = 0
            continue

        def canon(s):
            m = Chem.MolFromSmiles(s)
            return Chem.MolToSmiles(m, canonical=True) if m else None

        df['smiles'] = df['smiles'].apply(canon)
        df = df.dropna(subset=['smiles']).reset_index(drop=True)
        embed_cols = [c for c in df.columns if c != 'smiles']
        embed_dims[name] = len(embed_cols)
        embed_data[name] = {
            row['smiles']: row[embed_cols].to_numpy(dtype=np.float32, copy=False)
            for _, row in df.iterrows()
        }
    return embed_data, embed_dims

# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------

def compute_expected_dims(fp_types, embed_dims: dict):
    expected = OrderedDict()
    for t in fp_types:
        if   t == 'ecfp':   expected[t] = 1024
        elif t == 'avalon':  expected[t] = 512
        elif t == 'maccs':   expected[t] = 166
        elif t == 'tt':      expected[t] = 1024
        elif t == 'rdkit':   expected[t] = get_rdkit_descriptor_length()
        elif 'mole' in t:
            d = embed_dims.get(t, 0)
            expected[t] = d if d > 0 else 768
        elif 'scage' in t:
            d = embed_dims.get(t, 0)
            expected[t] = d if d > 0 else 512
        else:
            raise ValueError(f'Unknown fp_type: {t}')
    return expected

def safe_fit_to_dim(vec, target_dim: int):
    if vec is None:
        return np.zeros(target_dim, dtype=np.float32)
    vec = vec.astype(np.float32, copy=False)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    cur = vec.shape[0]
    if cur == target_dim:  return vec
    if cur < target_dim:   return np.concatenate([vec, np.zeros(target_dim - cur, dtype=np.float32)])
    return vec[:target_dim]

def make_feature_vector(mol, smiles, fp_types, expected_dims, embed_dicts):
    chunks = []
    for t in fp_types:
        dim = expected_dims[t]
        try:
            if   t == 'ecfp':   vec = get_ecfp(mol, radius=2, nbits=dim)
            elif t == 'avalon':  vec = get_avalon(mol, nbits=dim)
            elif t == 'maccs':   vec = get_maccs(mol)
            elif t == 'tt':      vec = get_topological_torsion(mol, nbits=dim)
            elif t == 'rdkit':   vec = get_rdkit_desc(mol)
            elif 'scage' in t or 'mole' in t:
                vec = embed_dicts.get(t, {}).get(smiles, None)
            else:
                vec = None
        except Exception:
            vec = None
        chunks.append(safe_fit_to_dim(vec, dim))
    feat = np.concatenate(chunks, axis=0)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ScageConcatDataset(data.Dataset):
    def __init__(self, smiles_path, embed_paths: dict, fp_types, expected_dims=None):
        df = pd.read_csv(smiles_path)
        if 'p_np' not in df.columns or 'smiles' not in df.columns:
            raise ValueError("'smiles'와 'p_np' 컬럼이 모두 필요합니다.")

        df = (df[['smiles', 'p_np']]
              .rename(columns={'p_np': 'label'})
              .assign(label=lambda d: d['label'].replace({'BBB-': 0, 'BBB+': 1}))
              .drop_duplicates(subset='smiles')
              .reset_index(drop=True))

        self.embed_dicts, embed_dims = load_molecular_embeddings(embed_paths)
        if expected_dims is None:
            expected_dims = compute_expected_dims(fp_types, embed_dims)
        self.expected_dims = expected_dims
        self.fp_types = list(fp_types)

        def canon(s):
            m = Chem.MolFromSmiles(s)
            return Chem.MolToSmiles(m, canonical=True) if m else None

        df['smiles'] = df['smiles'].apply(canon)
        df = df.dropna(subset=['smiles']).reset_index(drop=True)

        features, labels, failed = [], [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc='Generating features'):
            smi = row['smiles']
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                failed.append(smi); continue
            feat = make_feature_vector(mol, smi, self.fp_types, self.expected_dims, self.embed_dicts)
            if feat is None or feat.ndim != 1:
                failed.append(smi); continue
            features.append(feat)
            labels.append(row['label'])

        self.features = torch.tensor(np.stack(features), dtype=torch.float32)
        self.labels   = torch.tensor(labels, dtype=torch.float32)
        self.df       = df[~df['smiles'].isin(failed)].reset_index(drop=True)

    def __len__(self):          return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

# ---------------------------------------------------------------------------
# Split + RDKit normalization
# ---------------------------------------------------------------------------

def split_then_normalize(dataset, split_mode='scaffold',
                         train_ratio=0.8, val_ratio=0.1, seed=700):
    set_seed(seed)
    df = dataset.df.copy()

    def get_scaffold(smi):
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) if m else None

    df['scaffold'] = df['smiles'].apply(get_scaffold)
    groups = list(df.groupby('scaffold').groups.values())

    if split_mode == 'scaffold':
        groups = sorted(groups, key=lambda g: len(g), reverse=True)
    elif split_mode == 'random_scaffold':
        random.Random(seed).shuffle(groups)
    else:
        raise ValueError("split_mode must be 'scaffold' or 'random_scaffold'")

    n = len(df)
    train_cap = int(round(train_ratio * n))
    val_cap   = int(round(val_ratio   * n))

    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        g = list(g)
        if   len(train_idx) + len(g) <= train_cap: train_idx += g
        elif len(val_idx)   + len(g) <= val_cap:   val_idx   += g
        else:                                        test_idx  += g

    def pick(idxs):
        return dataset.features[idxs], dataset.labels[idxs]

    X_train, y_train = pick(train_idx)
    X_val,   y_val   = pick(val_idx)
    X_test,  y_test  = pick(test_idx)

    # Find RDKit descriptor slice for normalization
    rd_start, rd_end, offset = None, None, 0
    for t in dataset.fp_types:
        dim = dataset.expected_dims[t]
        if t == 'rdkit':
            rd_start, rd_end = offset, offset + dim
            break
        offset += dim

    scaler = None
    if rd_start is not None:
        scaler = StandardScaler().fit(X_train[:, rd_start:rd_end])
        for X in [X_train, X_val]:
            X[:, rd_start:rd_end] = torch.tensor(
                scaler.transform(X[:, rd_start:rd_end]), dtype=torch.float32
            )
        if X_test.shape[0] > 0:
            X_test[:, rd_start:rd_end] = torch.tensor(
                scaler.transform(X_test[:, rd_start:rd_end]), dtype=torch.float32
            )

    return (
        data.TensorDataset(X_train, y_train),
        data.TensorDataset(X_val,   y_val),
        data.TensorDataset(X_test,  y_test),
        scaler, (rd_start, rd_end),
        (train_idx, val_idx, test_idx),
    )

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def eval_model(model, loader):
    model.eval()
    y_true, y_prob, y_pred = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            y_prob.extend(probs)
            y_pred.extend((probs > 0.5).astype(int))
            y_true.extend(y.numpy())

    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
    else:
        specificity = sensitivity = 0.0

    return {
        'accuracy':    round(accuracy_score(y_true, y_pred), 4),
        'precision':   round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall':      round(sensitivity, 4),
        'f1':          round(f1_score(y_true, y_pred, zero_division=0), 4),
        'roc_auc':     round(roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0, 4),
        'mcc':         round(matthews_corrcoef(y_true, y_pred), 4),
        'sensitivity': sensitivity,
        'specificity': specificity,
    }

def composite_score(metrics: dict) -> float:
    """Fixed autoresearch metric: f1 + mcc + roc_auc (higher is better, max ≈ 3.0)."""
    return metrics['f1'] + metrics['mcc'] + metrics['roc_auc']
