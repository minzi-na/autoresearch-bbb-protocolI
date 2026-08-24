"""
Protocol I (combo1 = maccs+avalon+rdkit+mole) dataset-size learning curve.

Counterpart to BBB_paper/learning_curve_protocolII_datasize.py, but in Protocol I's
NATIVE 8:1:1 setup using this repo's own machinery (bbb_prepare / bbb_train).

Design (mirrors gmlp_ext_augment.py, with two changes):
  - Architecture/training = the PRE-AutoResearch BASE model (commit a489d67):
    original gMLP (no DropPath), Adam, val_loss early stopping, pos_weight=auto.
    Loaded from bbb_train_base_a489d67.py (git show a489d67:bbb_train.py). NOT the
    Phase-2 optimized HPARAMS.
  - External (GM-BBB) data is appended to the TRAIN set only, in a sweep of
    fractions 0% -> 100%. Internal 8:1:1 (train/val/test) is FIXED (scaffold split,
    seed-independent); val/test stay pure internal. Only the amount of external
    training data varies.

Evaluation (per seed, then per-seed mean +/- std):
  - roc_s        : internal scaffold test set (in-distribution / MoleculeNet)
  - holdout_888  : common simfilter09 holdout total subset (n=888)  [matches Protocol II curve]
  - holdout_329  : common simfilter09 holdout nn05 subset  (n=329)  [matches Protocol II curve]

f=0% (no external) == the faithful Protocol I combo1 base point.

Run (GPU env):
    conda run -n rapids-25.02 python learning_curve_extaug_combo1_base.py
"""
import os
import sys
import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger

from bbb_prepare import (
    split_then_normalize, set_seed,
    LABEL_PATH, EMBED_PATHS, FP_TYPES,
)
from bbb_train import (
    load_cached_dataset,
    EXT_LABEL_PATH, EXT_EMBED_PATHS,
    HOLDOUT_LABEL_PATH, HOLDOUT_EMBED_PATHS,
    SEEDS, SPLIT_MODE,
)
import bbb_train_base_a489d67 as base_train  # pre-AutoResearch base model + train_model + BASE_CONFIG

RDLogger.DisableLog("rdApp.*")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]  # external fraction appended to TRAIN
OUT_DIR = "/home/minji/BBB_paper/datasize_learning_curve"
SUBSET_PATHS = {
    "holdout_888": "/home/minji/holdout_subset/merged_holdout_10pct_seed42_simfilter09_total/label_holdout.csv",
    "holdout_329": "/home/minji/holdout_subset/merged_holdout_10pct_seed42_simfilter09_nn05/label_holdout.csv",
    "holdout_int179": "/home/minji/holdout_subset/merged_holdout_10pct_seed42_simfilter09_internal/label_holdout.csv",
    "holdout_ext709": "/home/minji/holdout_subset/merged_holdout_10pct_seed42_simfilter09_external/label_holdout.csv",
}


def canon(smi):
    m = Chem.MolFromSmiles(str(smi))
    return Chem.MolToSmiles(m, canonical=True) if m else None


def get_rd_slice(fp_types, expected_dims):
    offset = 0
    for t in fp_types:
        if t == "rdkit":
            return offset, offset + expected_dims[t]
        offset += expected_dims[t]
    return None, None


def sample_external(n_ext, labels, fraction, seed):
    """Stratified-by-label external subsample. Returns sorted local indices into the
    external array. fraction>=1 -> all; fraction<=0 -> none."""
    idx = np.arange(n_ext)
    if fraction >= 1.0:
        return idx
    if fraction <= 0.0:
        return np.array([], dtype=np.int64)
    rng = random.Random(int(round(seed * 1000 + fraction * 100)))
    sel = []
    for lab_val in (0, 1):
        grp = idx[labels == lab_val].tolist()
        sel += rng.sample(grp, int(round(fraction * len(grp))))
    return np.array(sorted(sel), dtype=np.int64)


def roc_on(model, X, y, rd_start, rd_end, scaler):
    Xs = X.copy()
    if rd_start is not None and scaler is not None:
        Xs[:, rd_start:rd_end] = scaler.transform(Xs[:, rd_start:rd_end])
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(Xs), 256):
            xb = torch.tensor(Xs[i:i + 256], dtype=torch.float32).to(DEVICE)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    p = np.concatenate(probs)
    return float(roc_auc_score(y.astype(int), p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (conda env rapids-25.02 on a GPU node).")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[combo1] FP_TYPES={FP_TYPES} SPLIT_MODE={SPLIT_MODE}", flush=True)
    int_ds = load_cached_dataset("internal", LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    expected_dims = int_ds.expected_dims
    mod_dims = OrderedDict((t, expected_dims[t]) for t in FP_TYPES)
    ext_ds = load_cached_dataset("external", EXT_LABEL_PATH, EXT_EMBED_PATHS,
                                 fp_types=FP_TYPES, expected_dims=expected_dims)
    hold_ds = load_cached_dataset("holdout", HOLDOUT_LABEL_PATH, HOLDOUT_EMBED_PATHS,
                                  fp_types=FP_TYPES, expected_dims=expected_dims)

    rd_start, rd_end = get_rd_slice(FP_TYPES, expected_dims)
    X_int, y_int = int_ds.features.numpy(), int_ds.labels.numpy()
    X_ext, y_ext = ext_ds.features.numpy(), ext_ds.labels.numpy()
    X_hold, y_hold = hold_ds.features.numpy(), hold_ds.labels.numpy()
    print(f"Internal={len(X_int)} External={len(X_ext)} Holdout={len(X_hold)} rd_slice=({rd_start},{rd_end})", flush=True)

    # build common-holdout subset masks (888 / 329) by canonical-SMILES membership
    hold_smiles = [canon(s) for s in hold_ds.df["smiles"].tolist()]
    masks = {}
    for name, path in SUBSET_PATHS.items():
        sub = set(canon(s) for s in pd.read_csv(path)["smiles"].tolist())
        mask = np.array([s in sub for s in hold_smiles])
        masks[name] = mask
        print(f"[subset] {name}: matched {int(mask.sum())} of holdout (csv had {len(sub)})", flush=True)

    rows = []
    for f in args.fractions:
        for seed in args.seeds:
            set_seed(seed)
            out = split_then_normalize(int_ds, split_mode=SPLIT_MODE,
                                       train_ratio=0.8, val_ratio=0.1, seed=seed)
            train_idx, val_idx, test_idx = out[5]

            ext_sel = sample_external(len(X_ext), y_ext, f, seed)
            X_train = np.concatenate([X_int[train_idx], X_ext[ext_sel]], axis=0) if len(ext_sel) \
                else X_int[train_idx].copy()
            y_train = np.concatenate([y_int[train_idx], y_ext[ext_sel]], axis=0) if len(ext_sel) \
                else y_int[train_idx].copy()
            X_val, y_val = X_int[val_idx].copy(), y_int[val_idx].copy()
            X_test, y_test = X_int[test_idx].copy(), y_int[test_idx].copy()

            # scaler fit on COMBINED train (rdkit slice only)
            scaler = None
            if rd_start is not None:
                scaler = StandardScaler()
                X_train = X_train.copy()
                X_train[:, rd_start:rd_end] = scaler.fit_transform(X_train[:, rd_start:rd_end])
                X_val = X_val.copy(); X_val[:, rd_start:rd_end] = scaler.transform(X_val[:, rd_start:rd_end])

            bs = base_train.BASE_CONFIG["batch_size"]
            to_ds = lambda x, y: data.TensorDataset(torch.tensor(x, dtype=torch.float32),
                                                    torch.tensor(y, dtype=torch.float32))
            train_loader = data.DataLoader(to_ds(X_train, y_train), batch_size=bs, shuffle=True, num_workers=4)
            val_loader = data.DataLoader(to_ds(X_val, y_val), batch_size=bs, shuffle=False, num_workers=4)

            set_seed(seed)
            model = base_train.MultiModalGMLPFromFlat(
                mod_dims=mod_dims,
                d_model=base_train.BASE_CONFIG["d_model"],
                d_ffn=base_train.BASE_CONFIG["d_ffn"],
                depth=base_train.BASE_CONFIG["depth"],
                dropout=base_train.BASE_CONFIG["dropout"],
                use_gated_pool=True,
            ).to(DEVICE)
            optimizer = optim.Adam(model.parameters(),
                                   lr=base_train.BASE_CONFIG["lr"],
                                   weight_decay=base_train.BASE_CONFIG["weight_decay"])
            # §3.1 screening base = train_all_combos.py = PLAIN BCE (no pos_weight).
            # (a489d67/pos_weight=auto is the AutoResearch start, NOT the screening base.)
            loss_fn = nn.BCEWithLogitsLoss()

            model = base_train.train_model(model, optimizer, train_loader, val_loader, loss_fn,
                                           num_epochs=base_train.BASE_CONFIG["num_epochs"],
                                           patience=base_train.BASE_CONFIG["patience"])

            roc_s = roc_on(model, X_test, y_test, rd_start, rd_end, scaler)
            rec = {"combo": "maccs+avalon+rdkit+mole", "arch": "base_a489d67",
                   "ext_fraction": f, "seed": seed,
                   "train_n": int(len(X_train)), "internal_train_n": int(len(train_idx)),
                   "ext_n": int(len(ext_sel)), "roc_s": roc_s}
            for name, mask in masks.items():
                rec[name] = roc_on(model, X_hold[mask], y_hold[mask], rd_start, rd_end, scaler)
            rows.append(rec)
            print(f"[base] ext_f={f:.2f} seed={seed} train_n={rec['train_n']} "
                  f"roc_s={roc_s:.4f} h888={rec['holdout_888']:.4f} h329={rec['holdout_329']:.4f}", flush=True)

            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "per_seed_protocolI_combo1_base.csv"), index=False)

    df = pd.DataFrame(rows)
    g = df.groupby("ext_fraction")
    print("\n=== Protocol I combo1 BASE per-fraction (per-seed mean ± std) ===", flush=True)
    for f, sub in g:
        print(f"ext={f:.2f} (n={len(sub)}, train={int(sub.train_n.iloc[0])}): "
              f"roc_s={sub.roc_s.mean():.4f}±{sub.roc_s.std():.4f}  "
              f"h888={sub.holdout_888.mean():.4f}±{sub.holdout_888.std():.4f}  "
              f"h329={sub.holdout_329.mean():.4f}±{sub.holdout_329.std():.4f}", flush=True)
    agg = {c: ["mean", "std"] for c in ["roc_s", "holdout_888", "holdout_329"]}
    agg["train_n"] = ["first"]
    summ = g.agg(agg).reset_index()
    summ.columns = ["_".join([c for c in col if c]) for col in summ.columns.to_flat_index()]
    summ.to_csv(os.path.join(OUT_DIR, "summary_protocolI_combo1_base.csv"), index=False)


if __name__ == "__main__":
    main()
