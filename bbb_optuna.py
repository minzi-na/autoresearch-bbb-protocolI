"""
BBB Optuna HPO runner — Phase 2 hyperparameter optimization.

Uses the fixed best architecture from bbb_train.py and searches only
hyperparameters. The objective is the 3-seed mean validation scaffold
ROC-AUC; the best trials are then reevaluated with the full 10-seed scaffold
test + external/holdout protocol.

Usage:
    conda run -n rapids-25.02 python bbb_optuna.py
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
from sklearn.metrics import roc_auc_score

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
)


OPTUNA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bbb_artifacts", "optuna")
OPTUNA_DB = os.path.join(OPTUNA_DIR, "bbb_optuna.db")
BEST_DIR = os.path.join(OPTUNA_DIR, "best_hpo")

OBJECTIVE_SEEDS = [42, 100, 200]
N_TRIALS = 40
TOP_K_REEVAL = 3
TIMEOUT = None
STUDY_NAME = "bbb_hpo"


def build_trial_config(trial: optuna.Trial) -> dict:
    return {
        "d_model": trial.suggest_categorical("d_model", [256, 384, 512, 768]),
        "d_ffn": trial.suggest_categorical("d_ffn", [512, 768, 1048, 1536, 2048]),
        "depth": BASE_CONFIG["depth"],
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "num_epochs": BASE_CONFIG["num_epochs"],
        "patience": BASE_CONFIG["patience"],
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
    }


def make_model(mod_dims: OrderedDict, config: dict) -> nn.Module:
    return MultiModalGMLPFromFlat(
        mod_dims=mod_dims,
        d_model=config["d_model"],
        d_ffn=config["d_ffn"],
        depth=config["depth"],
        dropout=config["dropout"],
        use_gated_pool=True,
    ).to(device)


def train_one_seed(model, optimizer, train_loader, val_loader, loss_fn, config: dict):
    best_val = float("inf")
    best_state = None
    bad = 0
    t_start = time.time()

    for epoch in range(config["num_epochs"]):
        elapsed = time.time() - t_start
        if elapsed > TIME_BUDGET:
            break

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += loss_fn(model(x), y).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= config["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_probs(model, loader) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            probs.extend(torch.sigmoid(model(x)).cpu().numpy())
    return np.asarray(probs, dtype=np.float32)


def roc_auc_from_probs(y_true: torch.Tensor, probs: np.ndarray) -> float:
    y_true_np = y_true.cpu().numpy()
    if len(set(y_true_np.tolist())) <= 1:
        return 0.0
    return float(roc_auc_score(y_true_np, probs))


def evaluate_seed(config: dict, dataset, ext_dataset, holdout_dataset, mod_dims, seed: int, include_external: bool):
    set_seed(seed)
    train_ds, val_ds, test_ds, scaler, (rd_start, rd_end), _ = split_then_normalize(
        dataset,
        split_mode=SPLIT_MODE,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=seed,
    )

    bs = config["batch_size"]
    train_loader = data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4)

    model = make_model(mod_dims, config)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = nn.BCEWithLogitsLoss()
    model = train_one_seed(model, optimizer, train_loader, val_loader, loss_fn, config)

    metrics = {
        "roc_auc_validation": eval_model(model, val_loader)["roc_auc"],
        "roc_auc_scaffold": eval_model(model, test_loader)["roc_auc"],
    }

    if include_external:
        ext_x = apply_scaler(ext_dataset.features, scaler, rd_start, rd_end)
        holdout_x = apply_scaler(holdout_dataset.features, scaler, rd_start, rd_end)
        ext_loader = data.DataLoader(
            data.TensorDataset(ext_x, ext_dataset.labels),
            batch_size=bs,
            shuffle=False,
            num_workers=4,
        )
        holdout_loader = data.DataLoader(
            data.TensorDataset(holdout_x, holdout_dataset.labels),
            batch_size=bs,
            shuffle=False,
            num_workers=4,
        )
        ext_probs = predict_probs(model, ext_loader)
        holdout_probs = predict_probs(model, holdout_loader)
        metrics["external_probs"] = ext_probs
        metrics["holdout_probs"] = holdout_probs
        metrics["roc_auc_external"] = roc_auc_from_probs(ext_dataset.labels, ext_probs)
        metrics["roc_auc_holdout"] = roc_auc_from_probs(holdout_dataset.labels, holdout_probs)

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
        "external",
        EXT_LABEL_PATH,
        EXT_EMBED_PATHS,
        fp_types=FP_TYPES,
        expected_dims=expected_dims,
    )
    holdout_dataset = load_cached_dataset(
        "holdout",
        HOLDOUT_LABEL_PATH,
        HOLDOUT_EMBED_PATHS,
        fp_types=FP_TYPES,
        expected_dims=expected_dims,
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
            aucs.append(metrics["roc_auc_validation"])
            running = mean(aucs)
            trial.report(running, step=idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("objective_seeds", OBJECTIVE_SEEDS)
        trial.set_user_attr("objective_metric", "3-seed validation scaffold mean ROC-AUC")
        trial.set_user_attr("objective_mean_roc_auc", mean(aucs))
        return mean(aucs)

    return objective


def save_best_payload(best_trial, reevaluation):
    os.makedirs(BEST_DIR, exist_ok=True)
    payload = {
        "trial_number": best_trial.number,
        "objective_value": best_trial.value,
        "params": best_trial.params,
        "reevaluation": reevaluation,
    }
    with open(os.path.join(BEST_DIR, "best_hparams.json"), "w") as f:
        json.dump(payload, f, indent=2)


def reevaluate_best_trials(study, dataset, ext_dataset, holdout_dataset, mod_dims, top_k=TOP_K_REEVAL):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    selected = completed[:top_k]
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
            )
            int_aucs.append(metrics["roc_auc_scaffold"])
            ext_seed_probs.append(metrics["external_probs"])
            holdout_seed_probs.append(metrics["holdout_probs"])

        ext_ensemble_probs = np.mean(np.stack(ext_seed_probs, axis=0), axis=0)
        holdout_ensemble_probs = np.mean(np.stack(holdout_seed_probs, axis=0), axis=0)
        roc_auc_external = roc_auc_from_probs(ext_dataset.labels, ext_ensemble_probs)
        roc_auc_holdout = roc_auc_from_probs(holdout_dataset.labels, holdout_ensemble_probs)

        reevaluations.append(
            {
                "trial_number": trial.number,
                "objective_value": trial.value,
                "params": trial.params,
                "roc_auc_scaffold": mean(int_aucs),
                "roc_auc_external": roc_auc_external,
                "roc_auc_holdout": roc_auc_holdout,
                "n_seeds": len(SEEDS),
            }
        )

    reevaluations.sort(key=lambda row: row["roc_auc_scaffold"], reverse=True)
    with open(os.path.join(BEST_DIR, "reevaluation.json"), "w") as f:
        json.dump(reevaluations, f, indent=2)
    return reevaluations


def main():
    os.makedirs(OPTUNA_DIR, exist_ok=True)
    t_start = time.time()

    print("Loading datasets...")
    dataset, ext_dataset, holdout_dataset, mod_dims = load_datasets()
    print(f"Internal : {len(dataset)} samples, feature_dim={dataset.features.shape[1]}")
    print(f"External : {len(ext_dataset)} samples")
    print(f"Holdout  : {len(holdout_dataset)} samples")
    print(f"Mod dims : {dict(mod_dims)}")
    print(f"Objective seeds: {OBJECTIVE_SEEDS}")
    print("Objective metric: 3-seed validation scaffold mean ROC-AUC")
    print(f"Full reeval seeds: {SEEDS}")

    sampler = optuna.samplers.TPESampler(seed=BASE_CONFIG["batch_size"])
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{OPTUNA_DB}",
        load_if_exists=True,
    )

    objective = objective_factory(dataset, ext_dataset, holdout_dataset, mod_dims)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)

    best_trial = study.best_trial
    reevaluation = reevaluate_best_trials(
        study=study,
        dataset=dataset,
        ext_dataset=ext_dataset,
        holdout_dataset=holdout_dataset,
        mod_dims=mod_dims,
    )
    best_full = reevaluation[0]
    save_best_payload(best_trial, best_full)

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
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
