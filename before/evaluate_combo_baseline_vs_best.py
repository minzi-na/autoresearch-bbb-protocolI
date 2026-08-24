#!/usr/bin/env python3
"""
Evaluate combo baseline vs best models with the same policy as BBB/evaluate_hpo_vs_baseline.py.

Policy:
  - roc_s: mean 10-seed internal scaffold test ROC-AUC from saved per-seed metrics
  - roc_ext: soft-voting ensemble ROC-AUC across 10 saved seed models on external set
  - roc_holdout: soft-voting ensemble ROC-AUC across 10 saved seed models on holdout set
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


ROOT = "/home/minji/autoresearch"
BBB_ROOT = "/home/minji/BBB"
FINAL_SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]

COMBOS = {
    "combo1": {
        "combo_str": "maccs+avalon+rdkit+mole",
        "worktree": "/home/minji/bbb-combo1",
    },
    "combo2": {
        "combo_str": "ecfp+maccs+avalon+tt+rdkit+scage1+mole",
        "worktree": "/home/minji/bbb-combo2",
    },
    "combo3": {
        "combo_str": "ecfp+maccs+avalon+tt+rdkit+mole",
        "worktree": "/home/minji/bbb-combo3",
    },
}


@dataclass
class LoadedCombo:
    name: str
    combo_str: str
    worktree: str
    train_mod: object
    prepare_mod: object


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combo",
        action="append",
        choices=sorted(COMBOS.keys()),
        help="Evaluate only selected combo(s). Default: all combo1/2/3.",
    )
    parser.add_argument(
        "--best-subdir",
        default="best",
        help="Best model subdir inside bbb_artifacts. Default: best",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "combo_eval_outputs"),
        help="Directory to write evaluation outputs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader workers for inference.",
    )
    return parser.parse_args()


def load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_combo_modules(combo_name: str) -> LoadedCombo:
    info = COMBOS[combo_name]
    worktree = info["worktree"]
    train_path = os.path.join(worktree, "bbb_train.py")
    prepare_path = os.path.join(worktree, "bbb_prepare.py")

    sys.modules.pop("bbb_prepare", None)
    train_mod_name = f"{combo_name}_bbb_train"
    prepare_mod_name = f"{combo_name}_bbb_prepare"
    prepare_mod = load_module(prepare_mod_name, prepare_path)
    sys.modules["bbb_prepare"] = prepare_mod
    train_mod = load_module(train_mod_name, train_path)
    sys.modules.pop("bbb_prepare", None)

    return LoadedCombo(
        name=combo_name,
        combo_str=info["combo_str"],
        worktree=worktree,
        train_mod=train_mod,
        prepare_mod=prepare_mod,
    )


def load_tac_module():
    sys.modules.pop("train_all_combos", None)
    return load_module("bbb_train_all_combos", os.path.join(BBB_ROOT, "train_all_combos.py"))


def predict_probs(model, x_tensor, batch_size: int, num_workers: int) -> np.ndarray:
    loader = data.DataLoader(
        data.TensorDataset(x_tensor, torch.zeros(len(x_tensor), dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    probs: List[float] = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(next(model.parameters()).device)
            probs.extend(torch.sigmoid(model(x)).cpu().numpy().tolist())
    return np.asarray(probs, dtype=np.float32)


def metric_dict_from_probs(labels: np.ndarray, probs: np.ndarray, metrics_mod) -> Dict[str, float]:
    y_true = np.asarray(labels)
    y_prob = np.asarray(probs)
    y_pred = (y_prob > 0.5).astype(int)
    cm = metrics_mod.confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
    has_both = len(set(y_true.tolist())) > 1
    return {
        "accuracy": round(float(metrics_mod.accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(metrics_mod.precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(metrics_mod.recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(metrics_mod.f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(metrics_mod.roc_auc_score(y_true, y_prob) if has_both else 0.0), 4),
        "mcc": round(float(metrics_mod.matthews_corrcoef(y_true, y_pred)), 4),
        "sensitivity": round(float(metrics_mod.recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(specificity), 4),
    }


def summarize_internal(rows: List[Dict[str, object]]) -> Dict[str, float]:
    values = [float(row["internal_roc_auc"]) for row in rows]
    return {
        "roc_auc_mean": round(float(np.mean(values)), 4),
        "roc_auc_std": round(float(np.std(values, ddof=0)), 4),
    }


def summarize_ensemble(rows: List[Dict[str, object]], prob_key: str, label_key: str, metrics_mod):
    probs = np.stack([row[prob_key] for row in rows], axis=0)
    mean_probs = probs.mean(axis=0)
    labels = np.asarray(rows[0][label_key])
    metrics = metric_dict_from_probs(labels, mean_probs, metrics_mod)
    return metrics, mean_probs, labels


def save_predictions(path: str, labels: np.ndarray, probs: np.ndarray):
    df = pd.DataFrame(
        {
            "label": labels,
            "prob": probs,
            "pred": (np.asarray(probs) > 0.5).astype(int),
        }
    )
    df.to_csv(path, index=False)


def load_combo_datasets(combo: LoadedCombo):
    tm = combo.train_mod
    dataset = tm.load_cached_dataset("internal", tm.LABEL_PATH, tm.EMBED_PATHS, fp_types=tm.FP_TYPES)
    expected_dims = dataset.expected_dims
    ext_dataset = tm.load_cached_dataset(
        "external",
        tm.EXT_LABEL_PATH,
        tm.EXT_EMBED_PATHS,
        fp_types=tm.FP_TYPES,
        expected_dims=expected_dims,
    )
    holdout_dataset = tm.load_cached_dataset(
        "holdout",
        tm.HOLDOUT_LABEL_PATH,
        tm.HOLDOUT_EMBED_PATHS,
        fp_types=tm.FP_TYPES,
        expected_dims=expected_dims,
    )
    mod_dims = tm.OrderedDict((t, expected_dims[t]) for t in tm.FP_TYPES)
    return dataset, ext_dataset, holdout_dataset, mod_dims


def apply_scaler_for_combo(tm, dataset, scaler, x):
    x = x.clone()
    offset = 0
    rd_start = rd_end = None
    for t in dataset.fp_types:
        dim = dataset.expected_dims[t]
        if t == "rdkit":
            rd_start, rd_end = offset, offset + dim
            break
        offset += dim
    if scaler is not None and rd_start is not None:
        x[:, rd_start:rd_end] = torch.tensor(
            scaler.transform(x[:, rd_start:rd_end]),
            dtype=torch.float32,
        )
    return x


def load_best_seed_model(
    combo: LoadedCombo,
    artifact_root: str,
    datasets,
    mod_dims,
    seed: int,
    num_workers: int,
) -> Dict[str, object]:
    tm = combo.train_mod
    dataset, ext_dataset, holdout_dataset, _ = datasets
    seed_dir = os.path.join(artifact_root, f"seed_{seed}")
    config_path = os.path.join(seed_dir, "config.json")
    model_path = os.path.join(seed_dir, "model.pth")
    scaler_path = os.path.join(seed_dir, "rdkit_scaler.pkl")

    with open(config_path) as f:
        config = json.load(f)

    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    ext_x = apply_scaler_for_combo(tm, dataset, scaler, ext_dataset.features)
    hold_x = apply_scaler_for_combo(tm, dataset, scaler, holdout_dataset.features)

    tm.set_seed(seed)
    model = tm.MultiModalGMLPFromFlat(
        mod_dims=mod_dims,
        d_model=config["base_config"]["d_model"],
        d_ffn=config["base_config"]["d_ffn"],
        depth=config["base_config"]["depth"],
        dropout=config["base_config"]["dropout"],
        use_gated_pool=True,
    ).to(tm.device)
    state = torch.load(model_path, map_location=tm.device)
    model.load_state_dict(state)

    batch_size = int(config["base_config"]["batch_size"])
    ext_probs = predict_probs(model, ext_x, batch_size, num_workers)
    hold_probs = predict_probs(model, hold_x, batch_size, num_workers)

    return {
        "seed": seed,
        "internal_roc_auc": float(config["metrics"]["roc_auc_scaffold"]),
        "external_probs": ext_probs,
        "external_labels": ext_dataset.labels.cpu().numpy(),
        "holdout_probs": hold_probs,
        "holdout_labels": holdout_dataset.labels.cpu().numpy(),
    }


def apply_rdkit_scaler_if_needed(tac, combo_tuple: Tuple[str, ...], scaler, x: np.ndarray) -> np.ndarray:
    x = x.copy()
    if scaler is None or "rdkit" not in combo_tuple:
        return x
    offset = 0
    rd_start = rd_end = None
    for fp_type in combo_tuple:
        dim = tac.FP_DIM[fp_type]
        if fp_type == "rdkit":
            rd_start, rd_end = offset, offset + dim
            break
        offset += dim
    x[:, rd_start:rd_end] = scaler.transform(x[:, rd_start:rd_end])
    return x


def load_baseline_datasets(tac, combo_tuple: Tuple[str, ...]):
    _, int_labels, int_feats = tac.precompute_all_features(
        tac.INTERNAL_LABEL,
        tac.INTERNAL_SCAGE1,
        tac.INTERNAL_SCAGE2,
        tac.INTERNAL_MOLE,
    )
    _, ext_labels, ext_feats = tac.precompute_all_features(
        tac.EXTERNAL_LABEL,
        tac.EXTERNAL_SCAGE1,
        tac.EXTERNAL_SCAGE2,
        tac.EXTERNAL_MOLE,
    )
    _, hold_labels, hold_feats = tac.precompute_all_features(
        os.path.join(BBB_ROOT, "holdout_splits/merged_holdout_10pct_seed42/label_holdout.csv"),
        os.path.join(BBB_ROOT, "holdout_splits/merged_holdout_10pct_seed42/scage1_holdout.csv"),
        os.path.join(BBB_ROOT, "holdout_splits/merged_holdout_10pct_seed42/scage2_holdout.csv"),
        os.path.join(BBB_ROOT, "holdout_splits/merged_holdout_10pct_seed42/mole_holdout.csv"),
    )
    return {
        "internal_labels": int_labels,
        "external_labels": ext_labels,
        "holdout_labels": hold_labels,
        "external_x": tac.build_feature_matrix(combo_tuple, ext_feats),
        "holdout_x": tac.build_feature_matrix(combo_tuple, hold_feats),
    }


def load_baseline_seed_model(
    tac,
    combo_str: str,
    combo_tuple: Tuple[str, ...],
    datasets: Dict[str, np.ndarray],
    seed: int,
    num_workers: int,
) -> Dict[str, object]:
    run_dir = os.path.join(BBB_ROOT, "combos", combo_str, "scaffold", f"seed_{seed}")
    model_path = os.path.join(run_dir, "model.pth")
    scaler_path = os.path.join(run_dir, "rdkit_scaler.pkl")
    metrics_path = os.path.join(run_dir, "metrics.json")

    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    ext_x = apply_rdkit_scaler_if_needed(tac, combo_tuple, scaler, datasets["external_x"])
    hold_x = apply_rdkit_scaler_if_needed(tac, combo_tuple, scaler, datasets["holdout_x"])

    with open(metrics_path) as f:
        metrics = json.load(f)

    model = tac.MultiModalGMLPFromFlat(
        mod_dims=tac.OrderedDict((fp, tac.FP_DIM[fp]) for fp in combo_tuple),
        d_model=tac.BASE_CONFIG["d_model"],
        d_ffn=tac.BASE_CONFIG["d_ffn"],
        depth=tac.BASE_CONFIG["depth"],
        dropout=tac.BASE_CONFIG["dropout"],
        use_gated_pool=tac.BASE_CONFIG["use_gated_pool"],
    ).to(tac.device)
    state = torch.load(model_path, map_location=tac.device)
    model.load_state_dict(state)

    batch_size = int(tac.BASE_CONFIG["batch_size"])
    ext_probs = predict_probs(model, torch.tensor(ext_x, dtype=torch.float32), batch_size, num_workers)
    hold_probs = predict_probs(model, torch.tensor(hold_x, dtype=torch.float32), batch_size, num_workers)

    return {
        "seed": seed,
        "internal_roc_auc": float(metrics["internal"]["roc_auc"]),
        "external_probs": ext_probs,
        "external_labels": datasets["external_labels"],
        "holdout_probs": hold_probs,
        "holdout_labels": datasets["holdout_labels"],
    }


def evaluate_combo(combo_name: str, best_subdir: str, out_dir: str, num_workers: int):
    combo = load_combo_modules(combo_name)
    tac = load_tac_module()

    os.makedirs(out_dir, exist_ok=True)
    combo_out_dir = os.path.join(out_dir, combo_name)
    os.makedirs(combo_out_dir, exist_ok=True)

    combo_tuple = tuple(combo.prepare_mod.FP_TYPES)
    baseline_dataset_bundle = load_baseline_datasets(tac, combo_tuple)
    combo_dataset_bundle = load_combo_datasets(combo)
    artifact_root = os.path.join(combo.worktree, "bbb_artifacts", best_subdir)

    baseline_rows = []
    best_rows = []
    for seed in FINAL_SEEDS:
        baseline_rows.append(
            load_baseline_seed_model(tac, combo.combo_str, combo_tuple, baseline_dataset_bundle, seed, num_workers)
        )
        best_rows.append(
            load_best_seed_model(combo, artifact_root, combo_dataset_bundle, combo_dataset_bundle[3], seed, num_workers)
        )

    baseline_internal = summarize_internal(baseline_rows)
    best_internal = summarize_internal(best_rows)
    baseline_ext_metrics, baseline_ext_probs, baseline_ext_labels = summarize_ensemble(
        baseline_rows, "external_probs", "external_labels", tac
    )
    baseline_hold_metrics, baseline_hold_probs, baseline_hold_labels = summarize_ensemble(
        baseline_rows, "holdout_probs", "holdout_labels", tac
    )
    best_ext_metrics, best_ext_probs, best_ext_labels = summarize_ensemble(
        best_rows, "external_probs", "external_labels", combo.prepare_mod
    )
    best_hold_metrics, best_hold_probs, best_hold_labels = summarize_ensemble(
        best_rows, "holdout_probs", "holdout_labels", combo.prepare_mod
    )

    summary = {
        "combo": combo_name,
        "combo_str": combo.combo_str,
        "best_subdir": best_subdir,
        "baseline": {
            "roc_s": baseline_internal["roc_auc_mean"],
            "roc_ext": baseline_ext_metrics["roc_auc"],
            "roc_holdout": baseline_hold_metrics["roc_auc"],
            "internal_test": baseline_internal,
            "external_ensemble": baseline_ext_metrics,
            "holdout_ensemble": baseline_hold_metrics,
        },
        "best": {
            "roc_s": best_internal["roc_auc_mean"],
            "roc_ext": best_ext_metrics["roc_auc"],
            "roc_holdout": best_hold_metrics["roc_auc"],
            "internal_test": best_internal,
            "external_ensemble": best_ext_metrics,
            "holdout_ensemble": best_hold_metrics,
        },
        "delta_best_minus_baseline": {
            "roc_s": round(best_internal["roc_auc_mean"] - baseline_internal["roc_auc_mean"], 4),
            "roc_ext": round(best_ext_metrics["roc_auc"] - baseline_ext_metrics["roc_auc"], 4),
            "roc_holdout": round(best_hold_metrics["roc_auc"] - baseline_hold_metrics["roc_auc"], 4),
        },
    }

    with open(os.path.join(combo_out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([summary["baseline"] | {"model_type": "baseline"}]).to_csv(
        os.path.join(combo_out_dir, "baseline_summary.csv"),
        index=False,
    )
    pd.DataFrame([summary["best"] | {"model_type": "best"}]).to_csv(
        os.path.join(combo_out_dir, "best_summary.csv"),
        index=False,
    )
    save_predictions(
        os.path.join(combo_out_dir, "baseline_external_ensemble_predictions.csv"),
        baseline_ext_labels,
        baseline_ext_probs,
    )
    save_predictions(
        os.path.join(combo_out_dir, "baseline_holdout_ensemble_predictions.csv"),
        baseline_hold_labels,
        baseline_hold_probs,
    )
    save_predictions(
        os.path.join(combo_out_dir, "best_external_ensemble_predictions.csv"),
        best_ext_labels,
        best_ext_probs,
    )
    save_predictions(
        os.path.join(combo_out_dir, "best_holdout_ensemble_predictions.csv"),
        best_hold_labels,
        best_hold_probs,
    )

    return summary


def main():
    args = parse_args()
    combo_names = args.combo or list(COMBOS.keys())
    all_rows = []
    for combo_name in combo_names:
        summary = evaluate_combo(combo_name, args.best_subdir, args.output_dir, args.num_workers)
        all_rows.append(
            {
                "combo": combo_name,
                "combo_str": summary["combo_str"],
                "best_subdir": summary["best_subdir"],
                "baseline_roc_s": summary["baseline"]["roc_s"],
                "baseline_roc_ext": summary["baseline"]["roc_ext"],
                "baseline_roc_holdout": summary["baseline"]["roc_holdout"],
                "best_roc_s": summary["best"]["roc_s"],
                "best_roc_ext": summary["best"]["roc_ext"],
                "best_roc_holdout": summary["best"]["roc_holdout"],
                "delta_roc_s": summary["delta_best_minus_baseline"]["roc_s"],
                "delta_roc_ext": summary["delta_best_minus_baseline"]["roc_ext"],
                "delta_roc_holdout": summary["delta_best_minus_baseline"]["roc_holdout"],
            }
        )
        print(
            f"{combo_name}: baseline(s={summary['baseline']['roc_s']:.4f}, ext={summary['baseline']['roc_ext']:.4f}, hold={summary['baseline']['roc_holdout']:.4f}) "
            f"| best(s={summary['best']['roc_s']:.4f}, ext={summary['best']['roc_ext']:.4f}, hold={summary['best']['roc_holdout']:.4f})"
        )

    pd.DataFrame(all_rows).to_csv(os.path.join(args.output_dir, "all_combo_summary.csv"), index=False)


if __name__ == "__main__":
    main()
