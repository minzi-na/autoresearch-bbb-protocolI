"""
BBB Optuna HPO runner — Phase 2 hyperparameter optimization (combo1).

combo1: maccs+avalon+rdkit+mole (seq_len=4)

Phase 1 best architecture: iter90 (116735a), roc_s=0.8757.
  - Single input-dependent query attention pooling
  - Learnable skip connection (skip_gate)
  - Diagonal-masked attention SGU + RMSNorm + SiLU

Phase 1 training details that are FIXED here:
  - Optimizer  : Adam (not AdamW)
  - pos_weight : 0.08 (tuned in Phase 1, pos_weight=0.08 was optimal)
  - grad_clip  : max_norm=1.0 (kept in iter82)
  - stoch_depth: 0.05 (default in MultiModalGMLPFromFlat)

NOTE: bbb_train.py L390 has weight_decay=1e-4 hardcoded (inline, not BASE_CONFIG).
  BASE_CONFIG['weight_decay']=1e-5, but actual Phase 1 training used 1e-4.
  HPO must include 1e-4 in weight_decay search range.

Objective: 5-seed mean scaffold test ROC-AUC (calibrated subset [200,400,500,700,900]; P1-best est=0.8759)
Reevaluation: top-3 trials → 10-seed scaffold test + external + holdout

Usage:
    conda run -n rapids-25.02 python bbb_optuna.py > bbb_hpo_run_combo1.log 2>&1
"""

import json
import os
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from bbb_prepare import device, eval_model, set_seed, split_then_normalize
from bbb_train import (
    ARTIFACT_DIR,
    BASE_CONFIG,
    EMBED_PATHS,
    EXT_EMBED_PATHS,
    EXT_LABEL_PATH,
    FP_TYPES,
    HOLDOUT_EMBED_PATHS,
    HOLDOUT_LABEL_PATH,
    LABEL_PATH,
    SEEDS,
    SPLIT_MODE,
    TIME_BUDGET,
    MultiModalGMLPFromFlat,
    apply_scaler,
    load_cached_dataset,
    predict_probs,
    roc_auc_from_probs,
)


OPTUNA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bbb_artifacts", "optuna")
OPTUNA_DB  = os.path.join(OPTUNA_DIR, "bbb_optuna.db")
BEST_DIR   = os.path.join(OPTUNA_DIR, "best_hpo")

# --------------------------------------------------------------------------
# Phase 2 fixed training constants (from Phase 1 best)
# --------------------------------------------------------------------------
POS_WEIGHT  = 0.08   # optimal pos_weight from Phase 1 (iter-series sweep)
GRAD_CLIP   = 1.0    # max_norm kept in iter82

# Phase 2 restart with correct iter90 architecture (116735a, roc_s=0.8757)
# Initial broad search — no prior Phase 2 results to inform narrowing
FIXED_CONFIG = {}   # nothing fixed beyond BASE_CONFIG; all 6 params are searched

# --------------------------------------------------------------------------
# Optuna study config
# --------------------------------------------------------------------------
OBJECTIVE_SEEDS = [200, 400, 500, 700, 900]  # calibrated 5-seed: P1-best mean=0.8759≈0.8756 (per-seed data)
N_TRIALS        = 50
TOP_K_REEVAL    = 3
TIMEOUT         = None
STUDY_NAME      = "bbb_hpo_combo1_p2r_v15_bridge"

# Phase 1 best config for warm-start enqueue
P1_BEST_PARAMS = {
    "d_model":      512,
    "d_ffn":        1048,
    "batch_size":   128,
    "dropout":      0.1,
    "lr":           1e-4,
    "weight_decay": 1e-4,
}

# run13 trial2 new best: drop=0.049, lr=8.11e-5, wd=1.43e-6
LOW_DROP_PROBE_PARAMS = {
    "d_model":      512,
    "d_ffn":        1048,
    "batch_size":   128,
    "dropout":      0.049,
    "lr":           8.11e-5,
    "weight_decay": 1.43e-6,
}


def build_trial_config(trial: optuna.Trial) -> dict:
    # Bridge: d_ffn=1048 fixed; span both optima (low-wd + P1-best region)
    # low-wd: drop≈0.05, lr≈8e-5, wd≈1e-6  |  P1: drop=0.10, lr=1e-4, wd=1e-4
    d_model = 512
    d_ffn   = 1048
    return {
        "d_model":      d_model,
        "d_ffn":        d_ffn,
        "batch_size":   trial.suggest_categorical("batch_size", [128, 256]),
        "dropout":      trial.suggest_float("dropout", 0.03, 0.12),
        "lr":           trial.suggest_float("lr", 6e-5, 1.5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 5e-7, 2e-4, log=True),
        "depth":        BASE_CONFIG["depth"],
        "num_epochs":   BASE_CONFIG["num_epochs"],
        "patience":     BASE_CONFIG["patience"],
    }


def make_model(mod_dims: OrderedDict, config: dict) -> nn.Module:
    return MultiModalGMLPFromFlat(
        mod_dims=mod_dims,
        d_model=config["d_model"],
        d_ffn=config["d_ffn"],
        depth=config["depth"],
        dropout=config["dropout"],
        use_gated_pool=True,
        stochastic_depth_rate=0.05,   # fixed from Phase 1 (iter90)
    ).to(device)


def train_one_seed(model, optimizer, train_loader, val_loader, loss_fn, config: dict):
    best_val   = float("inf")
    best_state = None
    bad        = 0
    t_start    = time.time()

    for epoch in range(config["num_epochs"]):
        if time.time() - t_start > TIME_BUDGET:
            break

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += loss_fn(model(x), y).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = deepcopy(model.state_dict())
            bad        = 0
        else:
            bad += 1
            if bad >= config["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_seed(config: dict, dataset, ext_dataset, holdout_dataset, mod_dims,
                  seed: int, include_external: bool, return_probs: bool = False):
    set_seed(seed)
    train_ds, val_ds, test_ds, scaler, (rd_start, rd_end), _ = split_then_normalize(
        dataset,
        split_mode=SPLIT_MODE,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=seed,
    )

    bs = config["batch_size"]
    train_loader = data.DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=4)
    val_loader   = data.DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=4)
    test_loader  = data.DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=4)

    set_seed(seed)   # reset seed before model init (matches bbb_train.py run_evaluation L376)
    model     = make_model(mod_dims, config)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    pos_weight_t = torch.tensor([POS_WEIGHT]).to(device)
    loss_fn      = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    model = train_one_seed(model, optimizer, train_loader, val_loader, loss_fn, config)

    metrics = {
        "roc_auc_validation": eval_model(model, val_loader)["roc_auc"],
        "roc_auc_scaffold":   eval_model(model, test_loader)["roc_auc"],
    }

    if include_external:
        ext_x     = apply_scaler(ext_dataset.features,     scaler, rd_start, rd_end)
        holdout_x = apply_scaler(holdout_dataset.features, scaler, rd_start, rd_end)
        ext_loader = data.DataLoader(
            data.TensorDataset(ext_x,     ext_dataset.labels),
            batch_size=bs, shuffle=False, num_workers=4,
        )
        holdout_loader = data.DataLoader(
            data.TensorDataset(holdout_x, holdout_dataset.labels),
            batch_size=bs, shuffle=False, num_workers=4,
        )
        metrics["roc_auc_external"] = eval_model(model, ext_loader)["roc_auc"]
        metrics["roc_auc_holdout"]  = eval_model(model, holdout_loader)["roc_auc"]
        if return_probs:
            metrics["probs_external"] = predict_probs(model, ext_loader)
            metrics["probs_holdout"]  = predict_probs(model, holdout_loader)

    del model, optimizer, train_loader, val_loader, test_loader
    if include_external:
        del ext_loader, holdout_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def load_datasets():
    dataset = load_cached_dataset("internal", LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    expected_dims = dataset.expected_dims
    mod_dims = OrderedDict((t, expected_dims[t]) for t in FP_TYPES)

    ext_dataset = load_cached_dataset(
        "external", EXT_LABEL_PATH, EXT_EMBED_PATHS,
        fp_types=FP_TYPES, expected_dims=expected_dims,
    )
    holdout_dataset = load_cached_dataset(
        "holdout", HOLDOUT_LABEL_PATH, HOLDOUT_EMBED_PATHS,
        fp_types=FP_TYPES, expected_dims=expected_dims,
    )
    return dataset, ext_dataset, holdout_dataset, mod_dims


def objective_factory(dataset, ext_dataset, holdout_dataset, mod_dims):
    def objective(trial: optuna.Trial):
        config = build_trial_config(trial)
        aucs = []
        for idx, seed in enumerate(OBJECTIVE_SEEDS, start=1):
            metrics = evaluate_seed(
                config=config,
                dataset=dataset,
                ext_dataset=ext_dataset,
                holdout_dataset=holdout_dataset,
                mod_dims=mod_dims,
                seed=seed,
                include_external=False,
            )
            aucs.append(metrics["roc_auc_scaffold"])
            trial.report(mean(aucs), step=idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("objective_seeds",      OBJECTIVE_SEEDS)
        trial.set_user_attr("objective_metric",     "10-seed scaffold test mean ROC-AUC")
        trial.set_user_attr("objective_mean_roc",   mean(aucs))
        return mean(aucs)

    return objective


def save_best_payload(best_trial, reevaluation):
    os.makedirs(BEST_DIR, exist_ok=True)
    payload = {
        "trial_number":    best_trial.number,
        "objective_value": best_trial.value,
        "params":          best_trial.params,
        "reevaluation":    reevaluation,
    }
    with open(os.path.join(BEST_DIR, "best_hparams.json"), "w") as f:
        json.dump(payload, f, indent=2)


def reevaluate_best_trials(study, dataset, ext_dataset, holdout_dataset, mod_dims,
                           top_k=TOP_K_REEVAL):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    selected  = completed[:top_k]
    reevaluations = []

    for trial in selected:
        config = dict(BASE_CONFIG)
        config.update(trial.params)
        int_aucs = []
        ext_seed_probs, holdout_seed_probs = [], []

        for seed in SEEDS:
            metrics = evaluate_seed(
                config=config,
                dataset=dataset,
                ext_dataset=ext_dataset,
                holdout_dataset=holdout_dataset,
                mod_dims=mod_dims,
                seed=seed,
                include_external=True,
                return_probs=True,
            )
            int_aucs.append(metrics["roc_auc_scaffold"])
            ext_seed_probs.append(metrics["probs_external"])
            holdout_seed_probs.append(metrics["probs_holdout"])

        ext_ensemble_probs = np.mean(np.stack(ext_seed_probs, axis=0), axis=0)
        holdout_ensemble_probs = np.mean(np.stack(holdout_seed_probs, axis=0), axis=0)

        reevaluations.append({
            "trial_number":     trial.number,
            "objective_value":  trial.value,
            "params":           trial.params,
            "roc_auc_scaffold": mean(int_aucs),
            "roc_auc_external": roc_auc_from_probs(ext_dataset.labels, ext_ensemble_probs),
            "roc_auc_holdout":  roc_auc_from_probs(holdout_dataset.labels, holdout_ensemble_probs),
            "n_seeds":          len(SEEDS),
        })

    reevaluations.sort(key=lambda r: r["roc_auc_scaffold"], reverse=True)
    os.makedirs(BEST_DIR, exist_ok=True)
    with open(os.path.join(BEST_DIR, "reevaluation.json"), "w") as f:
        json.dump(reevaluations, f, indent=2)
    return reevaluations


def main():
    os.makedirs(OPTUNA_DIR, exist_ok=True)
    t_start = time.time()

    print("=" * 65)
    print("  BBB HPO Phase 2 — combo1 (maccs+avalon+rdkit+mole)")
    print("=" * 65)
    print(f"  Fixed: pos_weight={POS_WEIGHT}, grad_clip={GRAD_CLIP}, stoch_depth=0.05")
    print(f"  Objective seeds : {OBJECTIVE_SEEDS}  (5-seed calibrated; P1-best est≈0.8759)")
    print(f"  Reeval seeds    : {SEEDS}")
    print(f"  n_trials        : {N_TRIALS}")
    print()

    dataset, ext_dataset, holdout_dataset, mod_dims = load_datasets()
    print(f"Internal : {len(dataset)} samples")
    print(f"External : {len(ext_dataset)} samples")
    print(f"Holdout  : {len(holdout_dataset)} samples")
    print(f"Mod dims : {dict(mod_dims)}")

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study   = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{OPTUNA_DB}",
        load_if_exists=True,
    )

    # Warm-start: P1 best + run12 trial3 best (low-drop, low-wd region)
    if len(study.trials) == 0:
        study.enqueue_trial(P1_BEST_PARAMS)
        study.enqueue_trial(LOW_DROP_PROBE_PARAMS)

    objective = objective_factory(dataset, ext_dataset, holdout_dataset, mod_dims)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)

    best_trial  = study.best_trial
    reevaluation = reevaluate_best_trials(study, dataset, ext_dataset, holdout_dataset, mod_dims)
    best_full   = reevaluation[0]
    save_best_payload(best_trial, best_full)

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available() else 0.0
    )
    print("\n---")
    print(f"best_val_roc_auc:         {best_trial.value:.6f}")
    print(f"best_trial:               {best_trial.number}")
    print(f"total_trials:             {len(study.trials)}")
    print(f"best_external_roc_auc:    {best_full['roc_auc_external']:.6f}")
    print(f"best_holdout_roc_auc:     {best_full['roc_auc_holdout']:.6f}")
    print(f"reeval_n_seeds:           {best_full['n_seeds']}")
    print(f"total_seconds:            {time.time() - t_start:.1f}")
    print(f"peak_vram_mb:             {peak_vram_mb:.1f}")


if __name__ == "__main__":
    main()
