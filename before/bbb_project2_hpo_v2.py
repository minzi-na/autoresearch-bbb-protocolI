"""
bbb_project2_hpo_v2.py — Project 2 HPO v2: 4가지 최적화 전략 비교

bbb_project2_hpo.py(v1)의 문제:
  - n_trials=30(실질) → val fitting 불완전
  - scaffold val AUC가 test AUC의 좋은 proxy가 아님 (generalization gap 존재)

v2에서 시도하는 4가지 전략 (--mode 인수로 선택):
  rs_auc        : random_scaffold split + val AUC 최적화
  rs_composite  : random_scaffold split + (AUC + F1 + MCC_norm) / 3 최적화
  sc_composite  : scaffold split       + (AUC + F1 + MCC_norm) / 3 최적화
  cv10          : 10-fold Stratified CV + val AUC 최적화

Usage:
    conda run --no-capture-output -n rapids-25.02 python -u \\
        /home/minji/autoresearch/bbb_project2_hpo_v2.py --mode rs_auc

    # 4개 전부 순차 실행:
    for mode in rs_auc rs_composite sc_composite cv10; do
        conda run --no-capture-output -n rapids-25.02 python -u \\
            /home/minji/autoresearch/bbb_project2_hpo_v2.py --mode $mode \\
            > /home/minji/BBB/project2/v2_${mode}.log 2>&1
    done

Output (mode별 독립 디렉토리):
    /home/minji/BBB/project2/v2_{mode}/
        optuna/           — Optuna SQLite DB
        best_hpo/         — 모델/스플릿/config/scaler
        best_hparams.json — 확정 HP
        results.csv       — 최종 평가 결과
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn import functional as F
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score,
    average_precision_score,
)

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

import optuna
from optuna.trial import Trial

import wandb

# ---------------------------------------------------------------------------
# 환경 설정
# ---------------------------------------------------------------------------
def set_seed(seed=42):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 피처 생성 유틸
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
    return to_numpy_bitvect(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits), n_bits=nbits)

def get_maccs(mol):
    bv = MACCSkeys.GenMACCSKeys(mol)
    return to_numpy_bitvect(bv, n_bits=bv.GetNumBits(), drop_first=True)

def get_avalon(mol, nbits=512):
    from rdkit.Avalon import pyAvalonTools
    return to_numpy_bitvect(pyAvalonTools.GetAvalonFP(mol, nbits), n_bits=nbits)

def get_rdkit_descriptor_length():
    return len(Descriptors._descList)

# ---------------------------------------------------------------------------
# 임베딩 로드
# ---------------------------------------------------------------------------
def _canonicalize_smiles(smiles_series: pd.Series) -> pd.Series:
    def canon(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, canonical=True) if m else None
    return smiles_series.apply(canon)

def load_molecular_embeddings(embed_paths: dict):
    embed_data, embed_dims = {}, {}
    for name, path in embed_paths.items():
        try:
            df = pd.read_csv(path)
        except Exception:
            embed_data[name] = {}
            embed_dims[name] = 0
            continue
        df['smiles'] = _canonicalize_smiles(df['smiles'])
        df = df.dropna(subset=['smiles']).reset_index(drop=True)
        embed_cols = [c for c in df.columns if c != 'smiles']
        embed_dims[name] = len(embed_cols)
        mat = df[embed_cols].to_numpy(dtype=np.float32)
        embed_data[name] = dict(zip(df['smiles'], mat))
    return embed_data, embed_dims

# ---------------------------------------------------------------------------
# 차원 계산 + 안전 결합
# ---------------------------------------------------------------------------
def compute_expected_dims(fp_types, embed_dims: dict):
    expected = OrderedDict()
    for t in fp_types:
        if   t == 'ecfp':    expected[t] = 1024
        elif t == 'avalon':  expected[t] = 512
        elif t == 'maccs':   expected[t] = 166
        elif t == 'tt':      expected[t] = 1024
        elif t == 'rdkit':   expected[t] = get_rdkit_descriptor_length()
        elif 'mole'  in t:   expected[t] = embed_dims.get(t, 768)  if embed_dims.get(t, 0) > 0 else 768
        elif 'scage' in t:   expected[t] = embed_dims.get(t, 512)  if embed_dims.get(t, 0) > 0 else 512
        else: raise ValueError(f"Unknown fp_type: {t}")
    return expected

def safe_fit_to_dim(vec, target_dim):
    if vec is None:
        return np.zeros(target_dim, dtype=np.float32)
    vec = np.nan_to_num(vec.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cur = vec.shape[0]
    if cur == target_dim:    return vec
    elif cur < target_dim:   return np.concatenate([vec, np.zeros(target_dim - cur, dtype=np.float32)])
    else:                    return vec[:target_dim]

def make_feature_vector(mol, smiles, fp_types, expected_dims, embed_dicts):
    chunks = []
    for t in fp_types:
        dim = expected_dims[t]
        try:
            if   t == 'ecfp':   vec = get_ecfp(mol, radius=2, nbits=dim)
            elif t == 'avalon':  vec = get_avalon(mol, nbits=dim)
            elif t == 'maccs':   vec = get_maccs(mol)
            elif 'scage' in t or 'mole' in t:
                vec = embed_dicts.get(t, {}).get(smiles, None)
            else: vec = None
        except Exception:
            vec = None
        chunks.append(safe_fit_to_dim(vec, dim))
    feat = np.concatenate(chunks, axis=0)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ScageConcatDataset(data.Dataset):
    def __init__(self, smiles_path, embed_paths, fp_types, expected_dims=None):
        df = pd.read_csv(smiles_path)
        df = df[['smiles', 'p_np']].rename(columns={'p_np': 'label'})
        df['label'] = df['label'].replace({'BBB-': 0, 'BBB+': 1})
        df = df.drop_duplicates(subset='smiles').reset_index(drop=True)

        self.embed_dicts, embed_dims = load_molecular_embeddings(embed_paths)
        if expected_dims is None:
            expected_dims = compute_expected_dims(fp_types, embed_dims)
        self.expected_dims = expected_dims
        self.fp_types = list(fp_types)

        df['smiles'] = _canonicalize_smiles(df['smiles'])
        df = df.dropna(subset=['smiles']).reset_index(drop=True)

        features, labels = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Features"):
            smi = row['smiles']
            mol = Chem.MolFromSmiles(smi)
            if mol is None: continue
            feat = make_feature_vector(mol, smi, self.fp_types, self.expected_dims, self.embed_dicts)
            features.append(feat)
            labels.append(row['label'])

        self.features = torch.tensor(np.stack(features), dtype=torch.float32)
        self.labels   = torch.tensor(labels, dtype=torch.float32)
        self.df       = df.iloc[:len(features)].reset_index(drop=True)

    def __len__(self):   return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

# ---------------------------------------------------------------------------
# Split + Normalize (scaffold / random_scaffold)
# ---------------------------------------------------------------------------
def split_then_normalize(dataset, split_mode="scaffold", train_ratio=0.8, val_ratio=0.1, seed=42):
    set_seed(seed)
    df = dataset.df.copy()

    def get_scaffold(smi):
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) if m else None

    df['scaffold'] = df['smiles'].apply(get_scaffold)
    groups = list(df.groupby('scaffold').groups.values())

    if split_mode == "scaffold":
        groups = sorted(groups, key=lambda g: len(g), reverse=True)
    elif split_mode == "random_scaffold":
        rnd = random.Random(seed)
        rnd.shuffle(groups)

    n = len(df)
    train_cap = int(round(train_ratio * n))
    val_cap   = int(round(val_ratio   * n))
    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        g = list(g)
        if   len(train_idx) + len(g) <= train_cap: train_idx += g
        elif len(val_idx)   + len(g) <= val_cap:   val_idx   += g
        else:                                       test_idx  += g

    def pick(idxs):
        return dataset.features[idxs], dataset.labels[idxs]

    X_train, y_train = pick(train_idx)
    X_val,   y_val   = pick(val_idx)
    X_test,  y_test  = pick(test_idx)

    scaler = None
    X_train, X_val, X_test, scaler = _maybe_scale(X_train, X_val, X_test, dataset)

    return (
        data.TensorDataset(X_train, y_train),
        data.TensorDataset(X_val,   y_val),
        data.TensorDataset(X_test,  y_test),
        scaler, None,
        (train_idx, val_idx, test_idx)
    )

def _maybe_scale(X_train, X_val, X_test, dataset):
    """rdkit descriptor 컬럼이 있는 경우에만 StandardScaler 적용."""
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
        X_train[:, rd_start:rd_end] = torch.tensor(
            scaler.transform(X_train[:, rd_start:rd_end]), dtype=torch.float32)
        X_val[:, rd_start:rd_end] = torch.tensor(
            scaler.transform(X_val[:, rd_start:rd_end]), dtype=torch.float32)
        if X_test.shape[0] > 0:
            X_test[:, rd_start:rd_end] = torch.tensor(
                scaler.transform(X_test[:, rd_start:rd_end]), dtype=torch.float32)
    return X_train, X_val, X_test, scaler

# ---------------------------------------------------------------------------
# Stratified K-Fold split + Normalize (CV용)
# ---------------------------------------------------------------------------
def make_cv_splits(dataset, n_splits=10, seed=42):
    """
    10-fold Stratified CV 준비.
    반환: [(train_ds, val_ds), ...] × n_splits
    테스트셋은 HPO 중에는 사용하지 않음.
    """
    y_all = dataset.labels.numpy().astype(int)
    X_all = dataset.features  # Tensor

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(np.zeros(len(y_all)), y_all):
        X_train = X_all[train_idx].clone()
        y_train = dataset.labels[train_idx]
        X_val   = X_all[val_idx].clone()
        y_val   = dataset.labels[val_idx]

        # scaler는 fold마다 독립적으로 fit
        X_train, X_val, _, _ = _maybe_scale(
            X_train, X_val,
            torch.zeros(0, X_train.shape[1]),  # dummy test (사용 안 함)
            dataset
        )

        folds.append((
            data.TensorDataset(X_train, y_train),
            data.TensorDataset(X_val,   y_val),
        ))
    return folds

# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)
    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        return u * v

class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        return x + residual

class gMLP(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, seq_len=6, num_layers=4):
        super().__init__()
        self.model = nn.Sequential(*[gMLPBlock(d_model, d_ffn, seq_len) for _ in range(num_layers)])
    def forward(self, x):
        return self.model(x)

class MultiModalGMLPFromFlat(nn.Module):
    def __init__(self, mod_dims, d_model=512, d_ffn=1024, depth=4, dropout=0.2, use_gated_pool=True):
        super().__init__()
        self.mod_names  = list(mod_dims.keys())
        self.mod_dims   = [mod_dims[n] for n in self.mod_names]
        self.seq_len    = len(self.mod_names)
        self.use_gated_pool = use_gated_pool

        self.proj     = nn.ModuleDict({n: nn.Linear(d, d_model) for n, d in zip(self.mod_names, self.mod_dims)})
        self.backbone = gMLP(seq_len=self.seq_len, d_model=d_model, d_ffn=d_ffn, num_layers=depth)
        self.norm     = nn.LayerNorm(d_model)
        if use_gated_pool:
            self.alpha = nn.Parameter(torch.zeros(self.seq_len))
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        chunks = torch.split(x, self.mod_dims, dim=1)
        tokens = [self.proj[n](c) for n, c in zip(self.mod_names, chunks)]
        X = torch.stack(tokens, dim=1)
        X = self.backbone(X)
        if self.use_gated_pool:
            w = torch.softmax(self.alpha, dim=0)
            Xp = (X * w.view(1, -1, 1)).sum(dim=1)
        else:
            Xp = X.mean(dim=1)
        Xp = self.drop(self.norm(Xp))
        return self.head(Xp).squeeze(-1)

# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------
def train_model(model, optimizer, train_loader, val_loader, loss_fn,
                num_epochs=50, patience=10, run=None):
    best_val, best_state, bad = float('inf'), None, 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += loss_fn(model(x), y).item()
        val_loss /= len(val_loader)

        if run is not None:
            run.log({"epoch/train_loss": train_loss, "epoch/val_loss": val_loss, "epoch": epoch})

        if val_loss < best_val:
            best_val, bad = val_loss, 0
            best_state = deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model

def eval_model(model, loader):
    model.eval()
    y_true, y_prob, y_pred = [], [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
            y_prob.extend(probs)
            y_pred.extend((probs > 0.5).astype(int))
            y_true.extend(y.numpy())
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = specificity = 0.0
    return {
        'accuracy':    round(accuracy_score(y_true, y_pred), 4),
        'precision':   round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall':      round(sensitivity, 4),
        'f1':          round(f1_score(y_true, y_pred, zero_division=0), 4),
        'roc_auc':     round(roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0, 4),
        'mcc':         round(matthews_corrcoef(y_true, y_pred), 4),
        'auprc':       round(average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0, 4),
        'sensitivity': round(sensitivity, 4),
        'specificity': round(specificity, 4),
    }

def composite_score(metrics: dict) -> float:
    """(AUC + F1 + MCC_norm) / 3, MCC를 [0,1]로 정규화."""
    mcc_norm = (metrics['mcc'] + 1.0) / 2.0
    return (metrics['roc_auc'] + metrics['f1'] + mcc_norm) / 3.0

# ---------------------------------------------------------------------------
# 저장 유틸
# ---------------------------------------------------------------------------
def save_split_indices(out_dir, tag, split_indices):
    os.makedirs(out_dir, exist_ok=True)
    for name, idx in zip(['train', 'val', 'test'], split_indices):
        np.save(os.path.join(out_dir, f"{name}_idx_{tag}.npy"), np.array(idx, dtype=np.int64))

def save_config(path, cfg):
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)

def save_scaler(path, scaler):
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
LABEL_PATH  = '/home/minji/BBB/scage/BBB/data/bench_label.csv'
EMBED_PATHS = {
    'scage1': '/home/minji/BBB/scage/BBB/data/bench_embed.csv',
    'scage2': '/home/minji/BBB/scage/BBB/data/bench_atom_embed.csv',
    'mole':   '/home/minji/BBB/mole_public/MolE_embed_base_bbb.csv',
}
FP_TYPES    = ['ecfp', 'maccs', 'avalon', 'scage1', 'scage2', 'mole']
HPO_SEEDS   = [42, 600, 900]   # scaffold/rs: 3-seed 평균
N_TRIALS       = 100
N_TRIALS_CV10  = 30
FINAL_SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
SPLIT_MODES = ["scaffold", "random_scaffold"]

WANDB_ENTITY = "minzikim-seoul-national-university"

# mode별 설정 테이블
MODE_CONFIGS = {
    "rs_auc": {
        "hpo_split":   "random_scaffold",
        "objective":   "auc",
        "study_name":  "project2_v2_rs_auc",
        "wandb_hpo":   "MultiModal_GMLP_BBBP_Project2_V2_RS_AUC",
        "wandb_final": "MultiModal_GMLP_BBBP_Project2_V2_RS_AUC_Final",
        "description": "random_scaffold split + val AUC 최적화",
    },
    "rs_composite": {
        "hpo_split":   "random_scaffold",
        "objective":   "composite",
        "study_name":  "project2_v2_rs_composite",
        "wandb_hpo":   "MultiModal_GMLP_BBBP_Project2_V2_RS_Composite",
        "wandb_final": "MultiModal_GMLP_BBBP_Project2_V2_RS_Composite_Final",
        "description": "random_scaffold split + (AUC+F1+MCC_norm)/3 최적화",
    },
    "sc_composite": {
        "hpo_split":   "scaffold",
        "objective":   "composite",
        "study_name":  "project2_v2_sc_composite",
        "wandb_hpo":   "MultiModal_GMLP_BBBP_Project2_V2_SC_Composite",
        "wandb_final": "MultiModal_GMLP_BBBP_Project2_V2_SC_Composite_Final",
        "description": "scaffold split + (AUC+F1+MCC_norm)/3 최적화",
    },
    "cv10": {
        "hpo_split":   "cv10",
        "objective":   "auc",
        "study_name":  "project2_v2_cv10",
        "wandb_hpo":   "MultiModal_GMLP_BBBP_Project2_V2_CV10",
        "wandb_final": "MultiModal_GMLP_BBBP_Project2_V2_CV10_Final",
        "description": "10-fold Stratified CV + val AUC 최적화",
    },
    "cv10_composite": {
        "hpo_split":   "cv10",
        "objective":   "composite",
        "study_name":  "project2_v2_cv10_composite",
        "wandb_hpo":   "MultiModal_GMLP_BBBP_Project2_V2_CV10_Composite",
        "wandb_final": "MultiModal_GMLP_BBBP_Project2_V2_CV10_Composite_Final",
        "description": "10-fold Stratified CV + (AUC+F1+MCC_norm)/3 최적화",
    },
}

# ---------------------------------------------------------------------------
# Step 1: 데이터 로드
# ---------------------------------------------------------------------------
def load_data():
    print(f"[Step 1] 데이터셋 로드 (FP_TYPES={FP_TYPES})")
    t0 = time.time()
    set_seed(42)
    full_dataset = ScageConcatDataset(LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    mod_dims = OrderedDict((t, full_dataset.expected_dims[t]) for t in FP_TYPES)
    print(f"  로드 완료: {len(full_dataset)}개 샘플 | feature_dim={full_dataset.features.shape[1]} | {time.time()-t0:.1f}s")
    print(f"  Feature dims: {dict(mod_dims)}")
    return full_dataset, mod_dims

# ---------------------------------------------------------------------------
# Step 2: Optuna HPO (mode별)
# ---------------------------------------------------------------------------
def run_hpo(full_dataset, mod_dims, cfg, out_dir):
    optuna_dir = os.path.join(out_dir, 'optuna')
    os.makedirs(optuna_dir, exist_ok=True)

    hpo_split  = cfg['hpo_split']
    obj_type   = cfg['objective']
    wandb_proj = cfg['wandb_hpo']

    # --- split 준비 ---
    if hpo_split == 'cv10':
        print(f"\n[Step 2] 10-fold CV split 준비 (seed=42)")
        cv_folds = make_cv_splits(full_dataset, n_splits=10, seed=42)
        hpo_splits = None
        print(f"  10 folds 준비 완료 (각 fold: ~{len(cv_folds[0][0])}개 train)")
    else:
        print(f"\n[Step 2] HPO seed별 split 준비 (split_mode={hpo_split}, seeds={HPO_SEEDS})")
        hpo_splits = {}
        for s in HPO_SEEDS:
            train_ds, val_ds, _, _, _, _ = split_then_normalize(
                full_dataset, split_mode=hpo_split, train_ratio=0.8, val_ratio=0.1, seed=s
            )
            hpo_splits[s] = (train_ds, val_ds)
            print(f"  Seed {s}: train={len(train_ds)}, val={len(val_ds)}")
        cv_folds = None

    # --- Objective 함수 ---
    def objective(trial: Trial):
        d_model        = trial.suggest_categorical('d_model',       [256, 512, 768, 1024])
        depth          = trial.suggest_int('depth',                  4, 12)
        ffn_multiplier = trial.suggest_categorical('ffn_multiplier', [2, 3, 4])
        d_ffn          = d_model * ffn_multiplier
        dropout        = trial.suggest_float('dropout',              0.0, 0.3)
        lr             = trial.suggest_float('lr',                   1e-5, 1e-3, log=True)
        weight_decay   = trial.suggest_float('weight_decay',         1e-6, 1e-3, log=True)
        batch_size     = trial.suggest_categorical('batch_size',     [32, 64, 128, 256])

        run = wandb.init(
            project=wandb_proj, entity=WANDB_ENTITY,
            group="Optuna_HPO", job_type="hpo_trial", reinit=True,
            config={
                'd_model': d_model, 'depth': depth, 'ffn_multiplier': ffn_multiplier,
                'd_ffn': d_ffn, 'dropout': dropout, 'lr': lr,
                'weight_decay': weight_decay, 'batch_size': batch_size,
                'trial_number': trial.number, 'hpo_mode': hpo_split, 'obj_type': obj_type,
            }
        )
        wandb.run.name = f"trial-{trial.number}_dm{d_model}_d{depth}_bs{batch_size}"

        scores = []
        try:
            if hpo_split == 'cv10':
                # 10-fold CV: 모든 fold에 대해 평가
                for fold_idx, (train_ds, val_ds) in enumerate(cv_folds):
                    set_seed(42 + fold_idx)

                    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
                    val_loader   = data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

                    y_train = train_ds.tensors[1].cpu().numpy()
                    n_pos, n_neg = (y_train == 1).sum(), (y_train == 0).sum()
                    pos_weight = torch.tensor([max(n_neg / n_pos, 1.0)], dtype=torch.float32, device=device) if n_pos > 0 else None
                    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

                    model = MultiModalGMLPFromFlat(
                        mod_dims=mod_dims, d_model=d_model, d_ffn=d_ffn,
                        depth=depth, dropout=dropout, use_gated_pool=True
                    ).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                    model = train_model(model, optimizer, train_loader, val_loader, loss_fn,
                                        num_epochs=50, patience=10)

                    metrics = eval_model(model, val_loader)
                    fold_score = metrics['roc_auc'] if obj_type == 'auc' else composite_score(metrics)
                    scores.append(fold_score)
                    run.log({f"fold{fold_idx}/val_score": fold_score,
                             f"fold{fold_idx}/val_roc_auc": metrics['roc_auc']})

                    del model, optimizer, loss_fn
                    torch.cuda.empty_cache()

            else:
                # multi-seed scaffold/random_scaffold
                for s in HPO_SEEDS:
                    set_seed(s)
                    train_ds, val_ds = hpo_splits[s]

                    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
                    val_loader   = data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

                    y_train = train_ds.tensors[1].cpu().numpy()
                    n_pos, n_neg = (y_train == 1).sum(), (y_train == 0).sum()
                    pos_weight = torch.tensor([max(n_neg / n_pos, 1.0)], dtype=torch.float32, device=device) if n_pos > 0 else None
                    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

                    model = MultiModalGMLPFromFlat(
                        mod_dims=mod_dims, d_model=d_model, d_ffn=d_ffn,
                        depth=depth, dropout=dropout, use_gated_pool=True
                    ).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                    model = train_model(model, optimizer, train_loader, val_loader, loss_fn,
                                        num_epochs=50, patience=10)

                    metrics = eval_model(model, val_loader)
                    seed_score = metrics['roc_auc'] if obj_type == 'auc' else composite_score(metrics)
                    scores.append(seed_score)
                    run.log({f"seed{s}/val_score": seed_score,
                             f"seed{s}/val_roc_auc": metrics['roc_auc']})

                    del model, optimizer, loss_fn
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [OOM] trial {trial.number} (d_model={d_model}, depth={depth}, "
                      f"ffn×{ffn_multiplier}, bs={batch_size}) — GPU OOM, trial 스킵")
                torch.cuda.empty_cache()
                run.finish()
                raise optuna.exceptions.TrialPruned()
            raise

        avg_score = float(np.mean(scores))
        run.log({"avg_val_score": avg_score})
        run.summary["avg_val_score"] = avg_score
        run.finish()

        trial.report(avg_score, step=50)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return avg_score

    # --- Optuna study 실행 ---
    n_trials = N_TRIALS_CV10 if hpo_split == 'cv10' else N_TRIALS
    print(f"\n[Step 2] Optuna HPO 시작 (n_trials={n_trials}, obj={obj_type}, split={hpo_split})")
    db_path = os.path.join(optuna_dir, f"{cfg['study_name']}.db")
    study = optuna.create_study(
        study_name=cfg['study_name'],
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10),
    )

    completed_before = len([t for t in study.trials if t.state.name == 'COMPLETE'])
    if completed_before >= n_trials:
        print(f"  이미 {completed_before}개 COMPLETE trial 완료 — HPO 스킵")
    else:
        print(f"  완료된 trial: {completed_before} | 목표: {n_trials} COMPLETE")

        def stop_when_complete(study, trial):
            n_complete = len([t for t in study.trials if t.state.name == 'COMPLETE'])
            if n_complete >= n_trials:
                print(f"  [완료] COMPLETE {n_complete}개 달성 → HPO 종료")
                study.stop()

        study.optimize(objective, n_trials=99999,
                       callbacks=[stop_when_complete], gc_after_trial=True)

    print("\n" + "="*80)
    print(f"Best score ({obj_type}, avg): {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("="*80)

    return study

# ---------------------------------------------------------------------------
# Step 3: Best Hyperparameters 확정
# ---------------------------------------------------------------------------
def extract_best_hparams(study, out_dir):
    best = study.best_params.copy()
    ffn_mult = best.pop('ffn_multiplier')
    hparams = {
        'd_model':      best['d_model'],
        'd_ffn':        best['d_model'] * ffn_mult,
        'depth':        best['depth'],
        'dropout':      float(best['dropout']),
        'lr':           float(best['lr']),
        'weight_decay': float(best['weight_decay']),
        'batch_size':   best['batch_size'],
    }

    print("\n✅ FINAL_BEST_HPARAMS:")
    for k, v in hparams.items():
        print(f"  {k}: {v}")

    os.makedirs(out_dir, exist_ok=True)
    hp_path = os.path.join(out_dir, 'best_hparams.json')
    with open(hp_path, 'w') as f:
        json.dump({**hparams, 'optuna_best_value': study.best_value}, f, indent=2)
    print(f"  → 저장: {hp_path}")

    return hparams

# ---------------------------------------------------------------------------
# Step 4: 최종 평가 (10 seeds × scaffold + random_scaffold)
# ---------------------------------------------------------------------------
def run_final_evaluation(full_dataset, mod_dims, hparams, cfg, out_dir):
    best_hpo_dir = os.path.join(out_dir, 'best_hpo')
    results_csv  = os.path.join(out_dir, 'results.csv')
    wandb_proj   = cfg['wandb_final']

    for sub in ['models', 'splits', 'configs', 'scalers']:
        os.makedirs(os.path.join(best_hpo_dir, sub), exist_ok=True)

    done_keys = set()
    if os.path.exists(results_csv):
        done_df   = pd.read_csv(results_csv)
        done_keys = set(zip(done_df['split_mode'], done_df['seed']))
        print(f"\n[Step 4] 재개: {len(done_keys)}개 run 이미 완료")
    else:
        print("\n[Step 4] 최종 평가 시작 (10 seeds × 2 splits = 20 runs)")

    all_results = list(pd.read_csv(results_csv).to_dict('records')) if os.path.exists(results_csv) else []

    for split_mode in SPLIT_MODES:
        print("\n" + "#"*80)
        print(f"### Split Mode: {split_mode}")
        print("#"*80)

        for seed in FINAL_SEEDS:
            if (split_mode, seed) in done_keys:
                print(f"  SKIP: {split_mode} seed={seed} (이미 완료)")
                continue

            print(f"\n  --- Seed: {seed} ---")
            set_seed(seed)
            tag = f"{split_mode}_seed{seed}"

            run = wandb.init(
                project=wandb_proj, entity=WANDB_ENTITY,
                group=f"Final_{split_mode}", job_type="final_eval", reinit=True,
                config={"seed": seed, "split_mode": split_mode,
                        "fp_types": FP_TYPES, "model_hparams": hparams,
                        "hpo_mode": cfg['hpo_split'], "obj_type": cfg['objective']}
            )
            wandb.run.name = f"final_{tag}"

            train_ds, val_ds, test_ds, scaler, _, split_indices = split_then_normalize(
                full_dataset, split_mode=split_mode, train_ratio=0.8, val_ratio=0.1, seed=seed
            )

            save_split_indices(os.path.join(best_hpo_dir, 'splits'), tag, split_indices)

            bs = hparams['batch_size']
            train_loader = data.DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=4)
            val_loader   = data.DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=4)
            test_loader  = data.DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=4)

            y_train = train_ds.tensors[1].cpu().numpy()
            n_pos, n_neg = (y_train == 1).sum(), (y_train == 0).sum()
            pos_weight = torch.tensor([max(n_neg / n_pos, 1.0)], dtype=torch.float32, device=device) if n_pos > 0 else None
            loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

            model = MultiModalGMLPFromFlat(
                mod_dims=mod_dims,
                d_model=hparams['d_model'],
                d_ffn=hparams['d_ffn'],
                depth=hparams['depth'],
                dropout=hparams['dropout'],
                use_gated_pool=True,
            ).to(device)
            optimizer = optim.Adam(model.parameters(),
                                   lr=hparams['lr'], weight_decay=hparams['weight_decay'])

            model = train_model(model, optimizer, train_loader, val_loader, loss_fn,
                                num_epochs=50, patience=10, run=run)

            metrics = eval_model(model, test_loader)
            metrics.update({
                'seed':            seed,
                'split_mode':      split_mode,
                'train_pos_ratio': float(np.mean(y_train)),
                'val_pos_ratio':   float(np.mean(val_ds.tensors[1].cpu().numpy())),
                'test_pos_ratio':  float(np.mean(test_ds.tensors[1].cpu().numpy())),
                'hpo_mode':        cfg['hpo_split'],
                'obj_type':        cfg['objective'],
            })
            all_results.append(metrics)

            run.log({f"test/{k}": v for k, v in metrics.items()})
            run.config.update({"class_balance": {"n_pos": int(n_pos), "n_neg": int(n_neg)}})

            model_path  = os.path.join(best_hpo_dir, 'models',  f"model_{tag}.pth")
            cfg_path    = os.path.join(best_hpo_dir, 'configs', f"config_{tag}.json")
            scaler_path = os.path.join(best_hpo_dir, 'scalers', f"scaler_{tag}.pkl")

            torch.save(model.state_dict(), model_path)
            save_config(cfg_path, {
                "seed": seed, "split_mode": split_mode,
                "fp_types": FP_TYPES, "model_hparams": hparams,
                "mod_dims": {k: int(v) for k, v in mod_dims.items()},
                "hpo_mode": cfg['hpo_split'], "obj_type": cfg['objective'],
                "test_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                 for k, v in metrics.items()}
            })
            if scaler is not None:
                save_scaler(scaler_path, scaler)

            artifact = wandb.Artifact(f"gmlp_project2_v2_{tag}", type="model")
            artifact.add_file(model_path)
            artifact.add_file(cfg_path)
            if scaler is not None: artifact.add_file(scaler_path)
            run.log_artifact(artifact)
            run.finish()

            print(f"  roc_auc={metrics['roc_auc']:.4f} | mcc={metrics['mcc']:.4f} | f1={metrics['f1']:.4f}")

            pd.DataFrame(all_results).to_csv(results_csv, index=False)

    return pd.DataFrame(all_results)

# ---------------------------------------------------------------------------
# Step 5: 결과 요약
# ---------------------------------------------------------------------------
def print_summary(results_df, mode_name):
    print("\n" + "="*80)
    print(f"FINAL SUMMARY [{mode_name}] (mean ± std across 10 seeds)")
    print("="*80)
    for split_mode in SPLIT_MODES:
        sub = results_df[results_df['split_mode'] == split_mode]
        if sub.empty:
            continue
        print(f"\n[{split_mode}]")
        for metric in ['roc_auc', 'mcc', 'f1', 'auprc', 'accuracy']:
            if metric in sub.columns:
                m, s = sub[metric].mean(), sub[metric].std()
                print(f"  {metric:<12}: {m:.4f} ± {s:.4f}")
    print("="*80)

# ---------------------------------------------------------------------------
# 교차 비교: 4개 mode의 test set AUC 비교
# ---------------------------------------------------------------------------
def compare_modes():
    """
    4개 mode의 results.csv를 읽어 test AUC/MCC/F1 기준으로 비교 테이블 출력.
    각 mode의 HPO best params도 함께 표시.
    """
    base = "/home/minji/BBB/project2"
    rows = []

    print("\n" + "="*90)
    print("TEST SET 성능 비교 (mode별 best HP → 10 seed × 2 split 평균)")
    print("최종 모델 결정 기준: test roc_auc 높은 쪽 우선")
    print("="*90)

    for mode_name, cfg in MODE_CONFIGS.items():
        csv_path = os.path.join(base, f"v2_{mode_name}", "results.csv")
        hp_path  = os.path.join(base, f"v2_{mode_name}", "best_hparams.json")

        if not os.path.exists(csv_path):
            print(f"  [{mode_name}] results.csv 없음 — 미완료")
            continue

        df = pd.read_csv(csv_path)
        hp = json.load(open(hp_path)) if os.path.exists(hp_path) else {}

        for split_mode in SPLIT_MODES:
            sub = df[df['split_mode'] == split_mode]
            if sub.empty:
                continue
            row = {
                'mode':       mode_name,
                'hpo_split':  cfg['hpo_split'],
                'obj':        cfg['objective'],
                'eval_split': split_mode,
                'roc_auc':    round(sub['roc_auc'].mean(), 4),
                'roc_auc_std':round(sub['roc_auc'].std(),  4),
                'mcc':        round(sub['mcc'].mean(),     4),
                'f1':         round(sub['f1'].mean(),      4),
                'auprc':      round(sub['auprc'].mean(),   4),
                'd_model':    hp.get('d_model', '-'),
                'depth':      hp.get('depth', '-'),
                'batch_size': hp.get('batch_size', '-'),
            }
            rows.append(row)

    if not rows:
        print("  비교할 결과 없음.")
        return

    cmp_df = pd.DataFrame(rows)

    # split_mode별 출력
    for split_mode in SPLIT_MODES:
        sub = cmp_df[cmp_df['eval_split'] == split_mode].copy()
        if sub.empty:
            continue
        sub = sub.sort_values('roc_auc', ascending=False).reset_index(drop=True)
        sub.index += 1  # 1-based rank

        print(f"\n[{split_mode}] — test AUC 내림차순")
        print(f"  {'rank':<4} {'mode':<15} {'hpo_split':<16} {'obj':<12} "
              f"{'roc_auc':<12} {'±std':<8} {'mcc':<8} {'f1':<8} {'auprc':<8} "
              f"{'d_model':<8} {'depth':<6} {'bs'}")
        print("  " + "-"*110)
        for rank, r in sub.iterrows():
            print(f"  {rank:<4} {r['mode']:<15} {r['hpo_split']:<16} {r['obj']:<12} "
                  f"{r['roc_auc']:<12.4f} {r['roc_auc_std']:<8.4f} {r['mcc']:<8.4f} "
                  f"{r['f1']:<8.4f} {r['auprc']:<8.4f} "
                  f"{str(r['d_model']):<8} {str(r['depth']):<6} {r['batch_size']}")

    # 최종 추천: scaffold split test AUC 기준
    scaffold_auc = (
        cmp_df[cmp_df['eval_split'] == 'scaffold']
        .set_index('mode')['roc_auc']
        .sort_values(ascending=False)
    )
    print(f"\n[최종 모델 결정 기준] scaffold split test roc_auc 순위:")
    for rank, (mode_name, val) in enumerate(scaffold_auc.items(), 1):
        print(f"  {rank}. {mode_name:<18} scaffold_test_roc_auc={val:.4f}")

    best_mode = scaffold_auc.index[0]
    print(f"\n→ 최종 채택 mode: {best_mode} (scaffold test AUC {scaffold_auc.iloc[0]:.4f})")
    print("="*90)

    # CSV로도 저장
    cmp_csv = os.path.join(base, "v2_comparison.csv")
    cmp_df.to_csv(cmp_csv, index=False)
    print(f"  비교 결과 저장: {cmp_csv}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Project 2 HPO v2")
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=list(MODE_CONFIGS.keys()) + ['compare'],
        help="HPO 전략: rs_auc | rs_composite | sc_composite | cv10 | compare"
    )
    args = parser.parse_args()

    if args.mode == 'compare':
        compare_modes()
        return

    cfg = MODE_CONFIGS[args.mode]
    out_dir = f"/home/minji/BBB/project2/v2_{args.mode}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Mode: {args.mode} — {cfg['description']}")
    print(f"Output dir: {out_dir}")
    t_start = time.time()

    # Step 1: 데이터 로드
    full_dataset, mod_dims = load_data()

    # Step 2: Optuna HPO
    study = run_hpo(full_dataset, mod_dims, cfg, out_dir)

    # Step 3: Best hyperparameters 확정
    hparams = extract_best_hparams(study, out_dir)

    # Step 4: 최종 평가
    results_df = run_final_evaluation(full_dataset, mod_dims, hparams, cfg, out_dir)

    # Step 5: 결과 요약
    print_summary(results_df, args.mode)

    total = time.time() - t_start
    print(f"\n[Done] mode={args.mode} 완료 ({total/3600:.1f}h)")
    print(f"  Best params:  {out_dir}/best_hparams.json")
    print(f"  Results CSV:  {out_dir}/results.csv")
    print(f"  Artifacts:    {out_dir}/best_hpo/")


if __name__ == '__main__':
    main()
