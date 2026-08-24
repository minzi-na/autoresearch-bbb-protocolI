"""
bbb_project2_hpo.py — Project 2: Hyperparameter Optimization (HPO)

project2_hpo.md 기준:
- 피쳐 조합: ecfp+maccs+avalon+scage1+scage2+mole (고정)
- Optuna (TPE, n_trials=100) × 3 seed 평균 val AUC-ROC
- Best params 확정 후 10 seed × scaffold + random_scaffold 최종 평가

Usage:
    conda run -n rapids-25.02 python /home/minji/autoresearch/bbb_project2_hpo.py

Output (auto-created):
    /home/minji/BBB/project2/optuna/project2_hpo.db   — Optuna SQLite DB
    /home/minji/BBB/project2/best_hpo/                — 모델/스플릿/config/scaler
    /home/minji/BBB/project2/results_hpo.csv           — 최종 평가 결과
"""

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
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score,
    average_precision_score,
)

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
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
    """SMILES 시리즈를 벡터화 방식으로 canonical 변환."""
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
        # iterrows() 대신 numpy 행렬로 한번에 추출 후 dict 구성
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
            elif t == 'tt':
                bv = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=dim)
                vec = to_numpy_bitvect(bv, n_bits=dim)
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

        # canonical 변환 1회 (공유 함수 사용)
        df['smiles'] = _canonicalize_smiles(df['smiles'])
        df = df.dropna(subset=['smiles']).reset_index(drop=True)

        # iterrows() 대신 zip으로 순회 (Series 생성 오버헤드 제거)
        features, labels, valid_idx = [], [], []
        smiles_list = df['smiles'].tolist()
        labels_list = df['label'].tolist()
        for i, (smi, lbl) in enumerate(tqdm(zip(smiles_list, labels_list),
                                             total=len(df), desc="Generating Features")):
            mol = Chem.MolFromSmiles(smi)
            if mol is None: continue
            feat = make_feature_vector(mol, smi, self.fp_types, self.expected_dims, self.embed_dicts)
            features.append(feat)
            labels.append(lbl)
            valid_idx.append(i)

        self.features = torch.tensor(np.stack(features), dtype=torch.float32)
        self.labels   = torch.tensor(labels, dtype=torch.float32)
        self.df       = df.iloc[valid_idx].reset_index(drop=True)

    def __len__(self):   return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

# ---------------------------------------------------------------------------
# Split + Normalize
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
        X_train[:, rd_start:rd_end] = torch.tensor(scaler.transform(X_train[:, rd_start:rd_end]), dtype=torch.float32)
        X_val[:, rd_start:rd_end]   = torch.tensor(scaler.transform(X_val[:, rd_start:rd_end]),   dtype=torch.float32)
        if X_test.shape[0] > 0:
            X_test[:, rd_start:rd_end] = torch.tensor(scaler.transform(X_test[:, rd_start:rd_end]), dtype=torch.float32)

    return (
        data.TensorDataset(X_train, y_train),
        data.TensorDataset(X_val,   y_val),
        data.TensorDataset(X_test,  y_test),
        scaler, (rd_start, rd_end),
        (train_idx, val_idx, test_idx)
    )

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
# 상수 (project2_hpo.md 기준)
# ---------------------------------------------------------------------------
LABEL_PATH  = '/home/minji/BBB/scage/BBB/data/bench_label.csv'
EMBED_PATHS = {
    'scage1': '/home/minji/BBB/scage/BBB/data/bench_embed.csv',
    'scage2': '/home/minji/BBB/scage/BBB/data/bench_atom_embed.csv',
    'mole':   '/home/minji/BBB/mole_public/MolE_embed_base_bbb.csv',
}
FP_TYPES      = ['ecfp', 'maccs', 'avalon', 'scage1', 'scage2', 'mole']
HPO_SEEDS     = [42, 600, 900]
N_TRIALS      = 150
HPO_SPLIT     = "scaffold"
FINAL_SEEDS   = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
SPLIT_MODES   = ["scaffold", "random_scaffold"]

PROJECT2_DIR  = '/home/minji/BBB/project2'
OPTUNA_DIR    = os.path.join(PROJECT2_DIR, 'optuna')
BEST_HPO_DIR  = os.path.join(PROJECT2_DIR, 'best_hpo')
RESULTS_CSV   = os.path.join(PROJECT2_DIR, 'results_hpo.csv')
BEST_HP_JSON  = os.path.join(PROJECT2_DIR, 'best_hparams.json')

WANDB_ENTITY        = "minzikim-seoul-national-university"
WANDB_HPO_PROJECT   = "MultiModal_GMLP_BBBP_Project2_HPO"
WANDB_FINAL_PROJECT = "MultiModal_GMLP_BBBP_Project2_Final"

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
# Step 2: Optuna HPO
# ---------------------------------------------------------------------------
def run_hpo(full_dataset, mod_dims):
    os.makedirs(OPTUNA_DIR, exist_ok=True)

    print(f"\n[Step 2] HPO seed별 split 준비 (seeds={HPO_SEEDS})")
    hpo_splits = {}
    for s in HPO_SEEDS:
        train_ds, val_ds, _, _, _, _ = split_then_normalize(
            full_dataset, split_mode=HPO_SPLIT, train_ratio=0.8, val_ratio=0.1, seed=s
        )
        hpo_splits[s] = (train_ds, val_ds)
        print(f"  Seed {s}: train={len(train_ds)}, val={len(val_ds)}")

    def objective(trial: Trial):
        d_model        = trial.suggest_categorical('d_model',       [128, 256, 512, 768])
        depth          = trial.suggest_int('depth',                  4, 10)
        ffn_multiplier = trial.suggest_categorical('ffn_multiplier', [2, 3, 4])
        d_ffn          = d_model * ffn_multiplier
        dropout        = trial.suggest_float('dropout',              0.0, 0.3)
        lr             = trial.suggest_float('lr',                   1e-5, 1e-3, log=True)
        weight_decay   = trial.suggest_float('weight_decay',         1e-6, 1e-3, log=True)
        batch_size     = trial.suggest_categorical('batch_size',     [32, 64, 128, 256])

        run = wandb.init(
            project=WANDB_HPO_PROJECT, entity=WANDB_ENTITY,
            group="Optuna_HPO", job_type="hpo_trial", reinit=True,
            config={
                'd_model': d_model, 'depth': depth, 'ffn_multiplier': ffn_multiplier,
                'd_ffn': d_ffn, 'dropout': dropout, 'lr': lr,
                'weight_decay': weight_decay, 'batch_size': batch_size,
                'trial_number': trial.number, 'hpo_seeds': HPO_SEEDS,
            }
        )
        wandb.run.name = f"trial-{trial.number}_dm{d_model}_d{depth}_bs{batch_size}"

        seed_aucs = []
        try:
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
                                    num_epochs=50, patience=10, run=None)

                metrics = eval_model(model, val_loader)
                seed_aucs.append(metrics['roc_auc'])
                run.log({f"seed{s}/val_roc_auc": metrics['roc_auc']})

                # seed 완료 후 GPU 메모리 즉시 해제
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

        avg_auc = float(np.mean(seed_aucs))
        run.log({"avg_val_roc_auc": avg_auc})
        run.summary["avg_val_roc_auc"] = avg_auc
        run.finish()

        trial.report(avg_auc, step=50)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return avg_auc

    print(f"\n[Step 2] Optuna HPO 시작 (n_trials={N_TRIALS}, seeds={HPO_SEEDS})")
    study = optuna.create_study(
        study_name="project2_hpo_multiseed",
        storage=f"sqlite:///{OPTUNA_DIR}/project2_hpo.db",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10),
    )

    completed_before = len([t for t in study.trials if t.state.name == 'COMPLETE'])
    if completed_before >= N_TRIALS:
        print(f"  이미 {completed_before}개 COMPLETE trial 완료 — HPO 스킵")
    else:
        print(f"  완료된 trial: {completed_before} | 목표: {N_TRIALS} COMPLETE")

        def stop_when_complete(study, trial):
            n_complete = len([t for t in study.trials if t.state.name == 'COMPLETE'])
            if n_complete >= N_TRIALS:
                print(f"  [완료] COMPLETE {n_complete}개 달성 → HPO 종료")
                study.stop()

        study.optimize(objective, n_trials=99999,
                       callbacks=[stop_when_complete], gc_after_trial=True)

    print("\n" + "="*80)
    print(f"Best val AUC-ROC (avg 3 seeds): {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("="*80)

    return study

# ---------------------------------------------------------------------------
# Step 3: Best Hyperparameters 확정
# ---------------------------------------------------------------------------
def extract_best_hparams(study):
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

    os.makedirs(PROJECT2_DIR, exist_ok=True)
    with open(BEST_HP_JSON, 'w') as f:
        json.dump({**hparams, 'optuna_best_value': study.best_value}, f, indent=2)
    print(f"  → 저장: {BEST_HP_JSON}")

    return hparams

# ---------------------------------------------------------------------------
# Step 4: 최종 평가 (10 seeds × 2 split modes)
# ---------------------------------------------------------------------------
def run_final_evaluation(full_dataset, mod_dims, hparams):
    for sub in ['models', 'splits', 'configs', 'scalers']:
        os.makedirs(os.path.join(BEST_HPO_DIR, sub), exist_ok=True)

    # 이미 완료된 (split_mode, seed) 확인
    done_keys = set()
    if os.path.exists(RESULTS_CSV):
        done_df = pd.read_csv(RESULTS_CSV)
        done_keys = set(zip(done_df['split_mode'], done_df['seed']))
        print(f"\n[Step 4] 재개: {len(done_keys)}개 run 이미 완료")
    else:
        print("\n[Step 4] 최종 평가 시작 (10 seeds × 2 splits = 20 runs)")

    all_results = list(pd.read_csv(RESULTS_CSV).to_dict('records')) if os.path.exists(RESULTS_CSV) else []

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
                project=WANDB_FINAL_PROJECT, entity=WANDB_ENTITY,
                group=f"Final_{split_mode}", job_type="final_eval", reinit=True,
                config={"seed": seed, "split_mode": split_mode,
                        "fp_types": FP_TYPES, "model_hparams": hparams}
            )
            wandb.run.name = f"final_{tag}"

            train_ds, val_ds, test_ds, scaler, _, split_indices = split_then_normalize(
                full_dataset, split_mode=split_mode, train_ratio=0.8, val_ratio=0.1, seed=seed
            )

            save_split_indices(os.path.join(BEST_HPO_DIR, 'splits'), tag, split_indices)

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
            })
            all_results.append(metrics)

            run.log({f"test/{k}": v for k, v in metrics.items()})
            run.config.update({"class_balance": {"n_pos": int(n_pos), "n_neg": int(n_neg)}})

            # 저장
            model_path  = os.path.join(BEST_HPO_DIR, 'models',  f"model_{tag}.pth")
            cfg_path    = os.path.join(BEST_HPO_DIR, 'configs', f"config_{tag}.json")
            scaler_path = os.path.join(BEST_HPO_DIR, 'scalers', f"scaler_{tag}.pkl")

            torch.save(model.state_dict(), model_path)
            save_config(cfg_path, {
                "seed": seed, "split_mode": split_mode,
                "fp_types": FP_TYPES, "model_hparams": hparams,
                "mod_dims": {k: int(v) for k, v in mod_dims.items()},
                "test_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                 for k, v in metrics.items()}
            })
            if scaler is not None:
                save_scaler(scaler_path, scaler)

            artifact = wandb.Artifact(f"gmlp_project2_{tag}", type="model")
            artifact.add_file(model_path)
            artifact.add_file(cfg_path)
            if scaler is not None: artifact.add_file(scaler_path)
            run.log_artifact(artifact)
            run.finish()

            print(f"  roc_auc={metrics['roc_auc']:.4f} | mcc={metrics['mcc']:.4f} | f1={metrics['f1']:.4f}")

            # 즉시 저장 (중단 후 재개 가능)
            pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)

    return pd.DataFrame(all_results)

# ---------------------------------------------------------------------------
# Step 5: 결과 요약
# ---------------------------------------------------------------------------
def print_summary(results_df):
    print("\n" + "="*80)
    print("FINAL SUMMARY (mean ± std across 10 seeds)")
    print("="*80)
    for split_mode in SPLIT_MODES:
        sub = results_df[results_df['split_mode'] == split_mode]
        print(f"\n[{split_mode}]")
        for metric in ['roc_auc', 'mcc', 'f1', 'auprc', 'accuracy']:
            if metric in sub.columns:
                m, s = sub[metric].mean(), sub[metric].std()
                print(f"  {metric:<12}: {m:.4f} ± {s:.4f}")
    print("="*80)

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {device}")
    os.makedirs(PROJECT2_DIR, exist_ok=True)

    # Step 1: 데이터 로드
    full_dataset, mod_dims = load_data()

    # Step 2: Optuna HPO
    study = run_hpo(full_dataset, mod_dims)

    # Step 3: Best hyperparameters 확정
    hparams = extract_best_hparams(study)

    # Step 4: 최종 평가
    results_df = run_final_evaluation(full_dataset, mod_dims, hparams)

    # Step 5: 결과 요약
    print_summary(results_df)

    print(f"\n[Done] Project 2 HPO 완료.")
    print(f"  Optuna DB:    {OPTUNA_DIR}/project2_hpo.db")
    print(f"  Best params:  {BEST_HP_JSON}")
    print(f"  Results CSV:  {RESULTS_CSV}")
    print(f"  Artifacts:    {BEST_HPO_DIR}")


if __name__ == '__main__':
    main()
