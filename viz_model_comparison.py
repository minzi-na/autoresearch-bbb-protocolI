"""Generate slide-ready visualizations for 5_model_comparison_summary.md."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# All labels are ASCII; keep default sans family
mpl.rcParams["axes.unicode_minus"] = False

ROOT = Path("/home/minji/bbb-combo1")
OUT = ROOT / "viz_outputs"
OUT.mkdir(exist_ok=True)

with open(ROOT / "eval_all_models_results.json") as f:
    rows = json.load(f)

# TabPFN baseline (single inference, no std)
tabpfn = {
    "model": "tabpfn_full",
    "int_mean": {"roc_auc": 0.8499, "mcc": 0.3492, "f1": 0.9240, "accuracy": np.nan},
    "int_std": {"roc_auc": 0.0, "mcc": 0.0, "f1": 0.0, "accuracy": 0.0},
    "ext": {"roc_auc": 0.7683, "mcc": 0.3966, "f1": 0.7960, "accuracy": np.nan},
    "holdout": {"roc_auc": 0.8055, "mcc": 0.4508, "f1": 0.8261, "accuracy": np.nan},
}
all_rows = rows + [tabpfn]

MODEL_ORDER = [
    "iter90_pw008",
    "iter90_p2_auto",
    "iter90_base_auto",
    "iter55_base_auto",
    "tabpfn_full",
]
MODEL_LABELS = {
    "iter90_pw008": "iter90\n(pw=0.08)",
    "iter90_p2_auto": "iter90 P2\n(pw=auto)",
    "iter90_base_auto": "iter90 BASE\n(pw=auto)",
    "iter55_base_auto": "iter55 BASE\n(pw=auto)",
    "tabpfn_full": "TabPFN-2.5\n(baseline)",
}
COLORS = {
    "iter90_pw008": "#d9534f",       # red — bad calibration
    "iter90_p2_auto": "#5bc0de",     # cyan
    "iter90_base_auto": "#0275d8",   # blue
    "iter55_base_auto": "#5cb85c",   # green
    "tabpfn_full": "#6f42c1",        # purple — baseline
}
METRICS = ["roc_auc", "mcc", "f1", "accuracy"]
METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "mcc": "MCC",
    "f1": "F1",
    "accuracy": "ACC",
}
DATASETS = [("int_mean", "Internal Test (10-seed mean)"),
            ("ext", "External Dataset (soft-vote ensemble)"),
            ("holdout", "Holdout Set (soft-vote ensemble)")]


def lookup(model_id):
    for r in all_rows:
        if r["model"] == model_id:
            return r
    raise KeyError(model_id)


# ---------------------------------------------------------------------------
# Figure 1: per-dataset grouped bars across all 4 metrics
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(13, 12.5), constrained_layout=True)
x = np.arange(len(METRICS))
bar_w = 0.16

for ax, (key, title) in zip(axes, DATASETS):
    for i, m in enumerate(MODEL_ORDER):
        rec = lookup(m)
        vals = [rec[key].get(metric, np.nan) for metric in METRICS]
        if key == "int_mean":
            errs = [rec["int_std"].get(metric, 0.0) for metric in METRICS]
        else:
            errs = [0.0] * len(METRICS)
        offset = (i - (len(MODEL_ORDER) - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, yerr=errs, capsize=3,
                      color=COLORS[m], label=MODEL_LABELS[m],
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.32),
               ncol=5, fontsize=10, frameon=False)
fig.suptitle("Combo1 model comparison across three evaluation sets",
             fontsize=15, fontweight="bold", y=1.03)
fig.savefig(OUT / "model_comparison_bars.png", dpi=170, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: generalization gap — Internal -> External -> Holdout per metric
# ---------------------------------------------------------------------------
gap_metrics = ["roc_auc", "mcc", "f1"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
ds_keys = ["int_mean", "ext", "holdout"]
ds_short = ["Internal", "External", "Holdout"]
for ax, metric in zip(axes, gap_metrics):
    for m in MODEL_ORDER:
        rec = lookup(m)
        ys = [rec[k].get(metric, np.nan) for k in ds_keys]
        ax.plot(ds_short, ys, marker="o", linewidth=2.2, markersize=8,
                color=COLORS[m], label=MODEL_LABELS[m].replace("\n", " "))
    ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.4, linestyle="--")
    ax.set_ylim(0.20 if metric == "mcc" else 0.45, 1.0 if metric != "mcc" else 0.55)

axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=5, fontsize=9, frameon=False)
fig.suptitle("Generalization gap across datasets",
             fontsize=14, fontweight="bold")
fig.savefig(OUT / "model_comparison_generalization.png",
            dpi=170, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: pos_weight effect — iter90 same arch, pw=0.08 vs pw=auto
# ---------------------------------------------------------------------------
pw_pair = ["iter90_pw008", "iter90_p2_auto"]
pw_colors = ["#d9534f", "#0275d8"]
pw_labels = ["pw=0.08 (fixed)", "pw=auto (n_neg/n_pos)"]
metrics_pw = ["roc_auc", "mcc", "f1", "accuracy"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
x = np.arange(len(metrics_pw))
bar_w = 0.34
for ax, (key, title) in zip(axes, DATASETS):
    for i, (m, c, lab) in enumerate(zip(pw_pair, pw_colors, pw_labels)):
        rec = lookup(m)
        vals = [rec[key].get(metric, np.nan) for metric in metrics_pw]
        offset = (i - 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, color=c, label=lab,
                      edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    # annotate delta on MCC/F1/ACC
    rec0, rec1 = lookup(pw_pair[0]), lookup(pw_pair[1])
    for j, metric in enumerate(metrics_pw):
        v0 = rec0[key].get(metric, np.nan)
        v1 = rec1[key].get(metric, np.nan)
        if np.isnan(v0) or np.isnan(v1):
            continue
        delta = v1 - v0
        if metric == "roc_auc":
            continue
        ax.annotate(f"Δ={delta:+.3f}", xy=(x[j], max(v0, v1) + 0.07),
                    ha="center", fontsize=8, color="black",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="#fff3cd" if delta > 0 else "#f8d7da",
                              ec="none", alpha=0.85))
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics_pw], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title(title.split(" (")[0], fontsize=12, fontweight="bold")
    ax.set_ylabel("Score")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

axes[0].legend(loc="upper center", bbox_to_anchor=(1.65, 1.22),
               ncol=2, fontsize=10, frameon=False)
fig.suptitle("Effect of pos_weight (same iter90 architecture, identical hyperparams)",
             fontsize=13.5, fontweight="bold", y=1.05)
fig.savefig(OUT / "model_comparison_posweight.png",
            dpi=170, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: combined heatmap — rows = models, cols = (dataset, metric)
# ---------------------------------------------------------------------------
heat_metrics = ["roc_auc", "mcc", "f1"]
col_labels = []
matrix = []
for m in MODEL_ORDER:
    rec = lookup(m)
    row = []
    for k, _ in DATASETS:
        for metric in heat_metrics:
            row.append(rec[k].get(metric, np.nan))
    matrix.append(row)
matrix = np.array(matrix)
for k, _ in DATASETS:
    short = {"int_mean": "Internal", "ext": "External", "holdout": "Holdout"}[k]
    for metric in heat_metrics:
        col_labels.append(f"{short}\n{METRIC_LABELS[metric]}")

# per-column rank shading: best=darkest within column
fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
norm_matrix = np.zeros_like(matrix)
for j in range(matrix.shape[1]):
    col = matrix[:, j]
    cmin, cmax = np.nanmin(col), np.nanmax(col)
    rng = cmax - cmin if cmax > cmin else 1.0
    norm_matrix[:, j] = (col - cmin) / rng

im = ax.imshow(norm_matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(np.arange(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=9)
ax.set_yticks(np.arange(len(MODEL_ORDER)))
ax.set_yticklabels([MODEL_LABELS[m].replace("\n", " ") for m in MODEL_ORDER],
                   fontsize=10)

# vertical separators between datasets
for sep in [3, 6]:
    ax.axvline(sep - 0.5, color="white", linewidth=2.5)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        v = matrix[i, j]
        if np.isnan(v):
            continue
        text_color = "white" if norm_matrix[i, j] > 0.55 else "black"
        # mark column-best with bold
        col = matrix[:, j]
        is_best = np.isclose(v, np.nanmax(col))
        weight = "bold" if is_best else "normal"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                color=text_color, fontsize=9.5, fontweight=weight)

ax.set_title("Per-column normalized score (darker = best in column, bold = column max)",
             fontsize=12, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
             label="Column-normalized score")
fig.savefig(OUT / "model_comparison_heatmap.png",
            dpi=170, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: radar / spider chart on External + Holdout (the generalization view)
# ---------------------------------------------------------------------------
radar_axes = ["Ext\nROC-AUC", "Ext\nMCC", "Ext\nF1",
              "Hold\nROC-AUC", "Hold\nMCC", "Hold\nF1"]
metric_keys = [("ext", "roc_auc"), ("ext", "mcc"), ("ext", "f1"),
               ("holdout", "roc_auc"), ("holdout", "mcc"), ("holdout", "f1")]

# rescale each axis to [0,1] across all models for visibility
raw = np.array([[lookup(m)[k][met] for k, met in metric_keys]
                for m in MODEL_ORDER])
mins, maxs = raw.min(axis=0), raw.max(axis=0)
rng = np.where(maxs > mins, maxs - mins, 1.0)
scaled = (raw - mins) / rng

angles = np.linspace(0, 2 * np.pi, len(radar_axes), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8.5, 7.5),
                       subplot_kw=dict(polar=True), constrained_layout=True)
for i, m in enumerate(MODEL_ORDER):
    vals = scaled[i].tolist() + [scaled[i][0]]
    ax.plot(angles, vals, color=COLORS[m], linewidth=2.2,
            label=MODEL_LABELS[m].replace("\n", " "))
    ax.fill(angles, vals, color=COLORS[m], alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_axes, fontsize=10)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25%", "50%", "75%", "best"], fontsize=8, color="gray")
ax.set_ylim(0, 1.05)
ax.set_title("External + Holdout generalization profile\n(per-axis min-max scaling)",
             fontsize=12.5, fontweight="bold", pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=9)
fig.savefig(OUT / "model_comparison_radar.png",
            dpi=170, bbox_inches="tight")
plt.close(fig)

print("Saved figures to", OUT)
for p in sorted(OUT.glob("model_comparison_*.png")):
    print(" -", p.name)
