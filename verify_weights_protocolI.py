#!/usr/bin/env python3
"""Load verification for the Protocol I released weights (phase2_best).

Loads each saved model.pth (NO retraining) and reproduces the per-seed metrics
recorded in bbb_artifacts/phase2_best/seed_*/config.json, plus the ensemble
numbers in summary.json.

Critical detail: the architecture is imported from the 88fdf45 SNAPSHOT, not
from the working-tree bbb_train.py. The working tree drifted after 88fdf45
(iter52-75: RMSNorm -> DropPath, gMLPBlock signature change), so importing the
live file would build a different model and the load would fail or mismatch.

Usage:
    conda run -n rapids-25.02 python verify_weights_protocolI.py
    conda run -n rapids-25.02 python verify_weights_protocolI.py --variant phase2_best_auto
"""

import argparse
import importlib.util
import json
import os
import pickle
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data as data

REPO = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = ("/home/minji/BBB_paper/model_snapshots/protocolI/"
            "07_phase2_hpo_final__88fdf45__bbb_train.py")

TOL = 1e-4          # config.json stores metrics rounded to 4 decimals
KEYS = [("roc_auc_scaffold", "internal test"),
        ("roc_auc_external", "external"),
        ("roc_auc_holdout", "holdout")]


def load_snapshot_module():
    """Import the 88fdf45 bbb_train as module 'p1_snapshot'."""
    if not os.path.exists(SNAPSHOT):
        sys.exit(f"[FAIL] snapshot not found: {SNAPSHOT}")
    sys.path.insert(0, REPO)          # so 'from bbb_prepare import *' resolves
    spec = importlib.util.spec_from_file_location("p1_snapshot", SNAPSHOT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p1_snapshot"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_best",
                    choices=["phase2_best", "phase2_best_auto"])
    args = ap.parse_args()

    save_dir = os.path.join(REPO, "bbb_artifacts", args.variant)
    summary = json.load(open(os.path.join(save_dir, "summary.json")))

    bt = load_snapshot_module()
    print(f"[Arch] imported from snapshot 88fdf45 "
          f"({os.path.basename(SNAPSHOT)})")
    print(f"[Arch] has RMSNorm={hasattr(bt, 'RMSNorm')}  "
          f"has DropPath={hasattr(bt, 'DropPath')}  (expect True / False)")
    print(f"[Variant] {args.variant}")

    print("\n[Data] loading cached datasets ...")
    dataset = bt.load_cached_dataset("internal", bt.LABEL_PATH, bt.EMBED_PATHS,
                                     fp_types=bt.FP_TYPES)
    expected_dims = dataset.expected_dims
    mod_dims = OrderedDict((t, expected_dims[t]) for t in bt.FP_TYPES)
    ext_ds = bt.load_cached_dataset("external", bt.EXT_LABEL_PATH,
                                    bt.EXT_EMBED_PATHS, fp_types=bt.FP_TYPES,
                                    expected_dims=expected_dims)
    hold_ds = bt.load_cached_dataset("holdout", bt.HOLDOUT_LABEL_PATH,
                                     bt.HOLDOUT_EMBED_PATHS, fp_types=bt.FP_TYPES,
                                     expected_dims=expected_dims)
    print(f"[Data] internal={len(dataset)}  external={len(ext_ds)}  "
          f"holdout={len(hold_ds)}  mod_dims={dict(mod_dims)}")

    failures = []
    n_checked = 0
    int_aucs, ext_stacks, hold_stacks = [], [], []

    for seed in bt.SEEDS:
        seed_dir = os.path.join(save_dir, f"seed_{seed}")
        cfg = json.load(open(os.path.join(seed_dir, "config.json")))
        hp = cfg["hparams"]

        # mirror retrain_phase2_best.py exactly: set_seed then split
        bt.set_seed(seed)
        train_ds, val_ds, test_ds, scaler, (rd_start, rd_end), _ = \
            bt.split_then_normalize(dataset, split_mode=bt.SPLIT_MODE,
                                    train_ratio=0.8, val_ratio=0.1, seed=seed)

        # the refit scaler must match the one saved next to the weights
        saved_scaler_path = os.path.join(seed_dir, "rdkit_scaler.pkl")
        if os.path.exists(saved_scaler_path) and scaler is not None:
            saved = pickle.load(open(saved_scaler_path, "rb"))
            n_checked += 2
            if not np.allclose(saved.mean_, scaler.mean_, atol=1e-8):
                failures.append(f"seed {seed} scaler.mean_ mismatch")
            if not np.allclose(saved.scale_, scaler.scale_, atol=1e-8):
                failures.append(f"seed {seed} scaler.scale_ mismatch")

        ext_X = bt.apply_scaler(ext_ds.features, scaler, rd_start, rd_end)
        hold_X = bt.apply_scaler(hold_ds.features, scaler, rd_start, rd_end)

        bs = hp["batch_size"]
        test_loader = data.DataLoader(test_ds, batch_size=bs, shuffle=False)
        ext_loader = data.DataLoader(data.TensorDataset(ext_X, ext_ds.labels),
                                     batch_size=bs, shuffle=False)
        hold_loader = data.DataLoader(data.TensorDataset(hold_X, hold_ds.labels),
                                      batch_size=bs, shuffle=False)

        model = bt.MultiModalGMLPFromFlat(
            mod_dims=mod_dims,
            d_model=hp["d_model"], d_ffn=hp["d_ffn"], depth=hp["depth"],
            dropout=hp["dropout"], use_gated_pool=True,
            stochastic_depth_rate=hp["stochastic_depth_rate"],
        ).to(bt.device)
        state = torch.load(os.path.join(seed_dir, "model.pth"),
                           map_location=bt.device, weights_only=False)
        model.load_state_dict(state, strict=True)

        got = {
            "roc_auc_scaffold": bt.eval_model(model, test_loader)["roc_auc"],
        }
        ext_probs = bt.predict_probs(model, ext_loader)
        hold_probs = bt.predict_probs(model, hold_loader)
        got["roc_auc_external"] = bt.roc_auc_from_probs(ext_ds.labels, ext_probs)
        got["roc_auc_holdout"] = bt.roc_auc_from_probs(hold_ds.labels, hold_probs)

        int_aucs.append(got["roc_auc_scaffold"])
        ext_stacks.append(np.asarray(ext_probs, dtype=np.float64))
        hold_stacks.append(np.asarray(hold_probs, dtype=np.float64))

        marks = []
        for k, _label in KEYS:
            rec = cfg["metrics"].get(k)
            n_checked += 1
            ok = rec is not None and abs(float(got[k]) - float(rec)) <= TOL
            if not ok:
                failures.append(
                    f"seed {seed} {k}: {got[k]:.6f} != recorded "
                    f"{'n/a' if rec is None else format(rec, '.4f')}")
            marks.append(f"{k.replace('roc_auc_', '')}={got[k]:.4f}"
                         f"{'' if ok else '(MISMATCH)'}")
        print(f"  seed={seed:>3}  " + "  ".join(marks))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    int_mean = float(np.mean(int_aucs))
    ext_ens = bt.roc_auc_from_probs(
        ext_ds.labels, np.mean(np.stack(ext_stacks, axis=0), axis=0))
    hold_ens = bt.roc_auc_from_probs(
        hold_ds.labels, np.mean(np.stack(hold_stacks, axis=0), axis=0))

    print("\n-- aggregate --")
    for label, got_v, rec_v in [
        ("internal mean", int_mean, summary["roc_auc_scaffold_mean"]),
        ("external ensemble", ext_ens, summary["roc_auc_external_ensemble"]),
        ("holdout ensemble", hold_ens, summary["roc_auc_holdout_ensemble"]),
    ]:
        n_checked += 1
        ok = abs(got_v - float(rec_v)) <= TOL
        if not ok:
            failures.append(f"{label}: {got_v:.6f} != recorded {rec_v}")
        print(f"  {label:<19} {got_v:.5f}  (recorded {rec_v})  "
              f"{'ok' if ok else 'MISMATCH'}")

    print("\n" + "=" * 64)
    print(f"comparisons checked = {n_checked}   mismatches = {len(failures)}")
    print("=" * 64)

    report = {
        "variant": args.variant,
        "arch_source": f"snapshot 88fdf45 ({os.path.basename(SNAPSHOT)})",
        "tolerance": TOL,
        "comparisons_checked": n_checked,
        "internal_per_seed_mean": round(int_mean, 6),
        "external_ensemble": round(ext_ens, 6),
        "holdout_ensemble": round(hold_ens, 6),
        "passed": not failures,
        "failures": failures,
    }
    out = os.path.join(save_dir, "verify_load.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"[Output] report -> {out}")

    if failures:
        print("\n[FAIL] mismatches:")
        for f in failures[:40]:
            print(f"  - {f}")
        sys.exit(1)
    print("\n[PASS] released weights reproduce every recorded metric")


if __name__ == "__main__":
    main()
