"""
bbb_project2_8hpo_eval.py

8피쳐 HPO에서 찾은 최적 하이퍼파라미터를 6피쳐 조합에 그대로 적용하여
10 seeds × scaffold + random_scaffold 최종 평가 수행.

HPO params (8피쳐 실험에서 도출):
  d_model=768, d_ffn=2304, depth=8, dropout=0.1005,
  lr=2.948e-5, weight_decay=1.571e-6, batch_size=128

Usage:
    conda run --no-capture-output -n rapids-25.02 python -u /home/minji/autoresearch/bbb_project2_8hpo_eval.py
"""

import json, os, pickle, random, sys, time
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
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
)
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

import wandb

# ---------------------------------------------------------------------------
# 하이퍼파라미터 (8피쳐 HPO 결과)
# ---------------------------------------------------------------------------
FIXED_HPARAMS = {
    'd_model':      768,
    'd_ffn':        2304,
    'depth':        8,
    'dropout':      0.10053908929502889,
    'lr':           2.948201095082773e-05,
    'weight_decay': 1.5709031332962757e-06,
    'batch_size':   128,
}

# ---------------------------------------------------------------------------
# 데이터 경로 / 설정
# ---------------------------------------------------------------------------
LABEL_PATH  = '/home/minji/BBB/scage/BBB/data/bench_label.csv'
EMBED_PATHS = {
    'scage1': '/home/minji/BBB/scage/BBB/data/bench_embed.csv',
    'scage2': '/home/minji/BBB/scage/BBB/data/bench_atom_embed.csv',
    'mole':   '/home/minji/BBB/mole_public/MolE_embed_base_bbb.csv',
}
FP_TYPES    = ['ecfp', 'maccs', 'avalon', 'scage1', 'scage2', 'mole']
FINAL_SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
SPLIT_MODES = ['scaffold', 'random_scaffold']

OUT_DIR     = '/home/minji/BBB/project2/eval_8hpo'
RESULTS_CSV = os.path.join(OUT_DIR, 'results_8hpo_on_6feat.csv')

WANDB_ENTITY  = "minzikim-seoul-national-university"
WANDB_PROJECT = "MultiModal_GMLP_BBBP_Project2_8HPO_Eval"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 유틸 (bbb_project2_hpo.py와 동일)
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

def _canonicalize_smiles(smiles_series):
    def canon(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, canonical=True) if m else None
    return smiles_series.apply(canon)

def load_molecular_embeddings(embed_paths):
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

def compute_expected_dims(fp_types, embed_dims):
    expected = OrderedDict()
    for t in fp_types:
        if   t == 'ecfp':    expected[t] = 1024
        elif t == 'avalon':  expected[t] = 512
        elif t == 'maccs':   expected[t] = 166
        elif t == 'rdkit':   expected[t] = get_rdkit_descriptor_length()
        elif 'mole'  in t:   expected[t] = embed_dims.get(t, 768) if embed_dims.get(t, 0) > 0 else 768
        elif 'scage' in t:   expected[t] = embed_dims.get(t, 512) if embed_dims.get(t, 0) > 0 else 512
        else: raise ValueError(f"Unknown fp_type: {t}")
    return expected

def safe_fit_to_dim(vec, target_dim):
    if vec is None:
        return np.zeros(target_dim, dtype=np.float32)
    vec = np.nan_to_num(vec.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cur = vec.shape[0]
    if cur == target_dim:  return vec
    elif cur < target_dim: return np.concatenate([vec, np.zeros(target_dim - cur, dtype=np.float32)])
    else:                  return vec[:target_dim]

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
    return np.nan_to_num(np.concatenate(chunks), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

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

        features, labels, valid_idx = [], [], []
        for i, (smi, lbl) in enumerate(tqdm(zip(df['smiles'].tolist(), df['label'].tolist()),
                                             total=len(df), desc="Generating Features")):
            mol = Chem.MolFromSmiles(smi)
            if mol is None: continue
            features.append(make_feature_vector(mol, smi, self.fp_types, self.expected_dims, self.embed_dicts))
            labels.append(lbl)
            valid_idx.append(i)

        self.features = torch.tensor(np.stack(features), dtype=torch.float32)
        self.labels   = torch.tensor(labels, dtype=torch.float32)
        self.df       = df.iloc[valid_idx].reset_index(drop=True)

    def __len__(self):   return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

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
    rd_start, rd_end, offset = None, None, 0
    for t in dataset.fp_types:
        dim = dataset.expected_dims[t]
        if t == 'rdkit':
            rd_start, rd_end = offset, offset + dim
            break
        offset += dim

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
# main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {device}")
    print(f"FP_TYPES: {FP_TYPES}")
    print(f"FIXED_HPARAMS: {FIXED_HPARAMS}")

    for sub in ['models', 'splits', 'configs', 'scalers']:
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

    # 데이터 로드
    print("\n[Step 1] 데이터셋 로드...")
    set_seed(42)
    full_dataset = ScageConcatDataset(LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    mod_dims = OrderedDict((t, full_dataset.expected_dims[t]) for t in FP_TYPES)
    print(f"  {len(full_dataset)}개 샘플 | feature_dim={full_dataset.features.shape[1]}")
    print(f"  Feature dims: {dict(mod_dims)}")

    # 이미 완료된 run 확인
    done_keys = set()
    if os.path.exists(RESULTS_CSV):
        done_df = pd.read_csv(RESULTS_CSV)
        done_keys = set(zip(done_df['split_mode'], done_df['seed']))
        print(f"\n[재개] {len(done_keys)}개 run 이미 완료")

    all_results = list(pd.read_csv(RESULTS_CSV).to_dict('records')) if os.path.exists(RESULTS_CSV) else []

    # 최종 평가
    print("\n[Step 2] 최종 평가 (10 seeds × scaffold + random_scaffold)")
    for split_mode in SPLIT_MODES:
        print("\n" + "#"*80)
        print(f"### Split Mode: {split_mode}")
        print("#"*80)

        for seed in FINAL_SEEDS:
            if (split_mode, seed) in done_keys:
                print(f"  SKIP: {split_mode} seed={seed}")
                continue

            print(f"\n  --- Seed: {seed} ---")
            set_seed(seed)
            tag = f"{split_mode}_seed{seed}"

            run = wandb.init(
                project=WANDB_PROJECT, entity=WANDB_ENTITY,
                group=f"8HPO_{split_mode}", job_type="eval", reinit=True,
                config={"seed": seed, "split_mode": split_mode,
                        "fp_types": FP_TYPES, "model_hparams": FIXED_HPARAMS}
            )
            wandb.run.name = f"8hpo_{tag}"

            train_ds, val_ds, test_ds, scaler, _, split_indices = split_then_normalize(
                full_dataset, split_mode=split_mode, train_ratio=0.8, val_ratio=0.1, seed=seed
            )

            # split indices 저장
            splits_dir = os.path.join(OUT_DIR, 'splits')
            for name, idx in zip(['train', 'val', 'test'], split_indices):
                np.save(os.path.join(splits_dir, f"{name}_idx_{tag}.npy"), np.array(idx, dtype=np.int64))

            bs = FIXED_HPARAMS['batch_size']
            train_loader = data.DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=4)
            val_loader   = data.DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=4)
            test_loader  = data.DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=4)

            y_train = train_ds.tensors[1].cpu().numpy()
            n_pos, n_neg = (y_train == 1).sum(), (y_train == 0).sum()
            pos_weight = torch.tensor([max(n_neg / n_pos, 1.0)], dtype=torch.float32, device=device) if n_pos > 0 else None
            loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

            model = MultiModalGMLPFromFlat(
                mod_dims=mod_dims,
                d_model=FIXED_HPARAMS['d_model'],
                d_ffn=FIXED_HPARAMS['d_ffn'],
                depth=FIXED_HPARAMS['depth'],
                dropout=FIXED_HPARAMS['dropout'],
                use_gated_pool=True,
            ).to(device)
            optimizer = optim.Adam(model.parameters(),
                                   lr=FIXED_HPARAMS['lr'],
                                   weight_decay=FIXED_HPARAMS['weight_decay'])

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
            run.finish()

            # 모델/config 저장
            torch.save(model.state_dict(), os.path.join(OUT_DIR, 'models', f"model_{tag}.pth"))
            with open(os.path.join(OUT_DIR, 'configs', f"config_{tag}.json"), 'w') as f:
                json.dump({"seed": seed, "split_mode": split_mode, "fp_types": FP_TYPES,
                           "model_hparams": FIXED_HPARAMS,
                           "mod_dims": {k: int(v) for k, v in mod_dims.items()},
                           "test_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                            for k, v in metrics.items()}}, f, indent=2)
            if scaler is not None:
                with open(os.path.join(OUT_DIR, 'scalers', f"scaler_{tag}.pkl"), 'wb') as f:
                    pickle.dump(scaler, f)

            # 즉시 저장
            pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)
            print(f"  roc_auc={metrics['roc_auc']:.4f} | mcc={metrics['mcc']:.4f} | f1={metrics['f1']:.4f}")

            del model, optimizer, loss_fn
            torch.cuda.empty_cache()

    # 요약
    results_df = pd.read_csv(RESULTS_CSV)
    print("\n" + "="*80)
    print("FINAL SUMMARY — 8피쳐 HPO params × 6피쳐 조합 (mean ± std, 10 seeds)")
    print("="*80)
    for split_mode in SPLIT_MODES:
        sub = results_df[results_df['split_mode'] == split_mode]
        print(f"\n[{split_mode}]")
        for metric in ['roc_auc', 'mcc', 'f1', 'auprc', 'accuracy']:
            m, s = sub[metric].mean(), sub[metric].std()
            print(f"  {metric:<12}: {m:.4f} ± {s:.4f}")
    print("="*80)
    print(f"\n[Done] 결과: {RESULTS_CSV}")


if __name__ == '__main__':
    main()
