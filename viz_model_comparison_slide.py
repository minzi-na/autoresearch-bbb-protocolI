"""Single-slide composite figure summarizing combo1 model comparison."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import numpy as np

mpl.rcParams["axes.unicode_minus"] = False

ROOT = Path("/home/minji/bbb-combo1")
OUT = ROOT / "viz_outputs"
OUT.mkdir(exist_ok=True)

with open(ROOT / "eval_all_models_results.json") as f:
    rows = json.load(f)

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
    "iter90_pw008": "iter90 (pw=0.08)",
    "iter90_p2_auto": "iter90 P2 (pw=auto)",
    "iter90_base_auto": "iter90 BASE (pw=auto)",
    "iter55_base_auto": "iter55 BASE (pw=auto)",
    "tabpfn_full": "TabPFN-2.5 baseline",
}
COLORS = {
    "iter90_pw008": "#d9534f",
    "iter90_p2_auto": "#5bc0de",
    "iter90_base_auto": "#0275d8",
    "iter55_base_auto": "#5cb85c",
    "tabpfn_full": "#6f42c1",
}


def lookup(model_id):
    for r in all_rows:
        if r["model"] == model_id:
            return r
    raise KeyError(model_id)


# ---------------------------------------------------------------------------
# Composite figure: 16:9 single slide
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 11.25))
gs = GridSpec(
    nrows=2, ncols=3,
    width_ratios=[1.55, 1.0, 1.0],
    height_ratios=[1.0, 0.95],
    hspace=0.42, wspace=0.32,
    left=0.045, right=0.985, top=0.90, bottom=0.06,
)

fig.suptitle(
    "Combo1 BBB model comparison — 4 multimodal variants vs TabPFN-2.5 baseline",
    fontsize=20, fontweight="bold", y=0.965,
)
fig.text(
    0.5, 0.928,
    "10-seed soft-vote ensemble · evaluated on Internal Test, External, Holdout",
    ha="center", fontsize=12.5, color="#444444",
)

# ---- Panel A: heatmap (left, spans both rows) -----------------------------
axA = fig.add_subplot(gs[:, 0])
heat_metrics = ["roc_auc", "mcc", "f1"]
metric_labels = {"roc_auc": "ROC-AUC", "mcc": "MCC", "f1": "F1"}
ds_short = {"int_mean": "Internal", "ext": "External", "holdout": "Holdout"}
DATASETS = ["int_mean", "ext", "holdout"]

matrix = np.array([
    [lookup(m)[k][met] for k in DATASETS for met in heat_metrics]
    for m in MODEL_ORDER
])
col_labels = [f"{ds_short[k]}\n{metric_labels[met]}"
              for k in DATASETS for met in heat_metrics]

norm = np.zeros_like(matrix)
for j in range(matrix.shape[1]):
    col = matrix[:, j]
    cmin, cmax = np.nanmin(col), np.nanmax(col)
    rng = cmax - cmin if cmax > cmin else 1.0
    norm[:, j] = (col - cmin) / rng

im = axA.imshow(norm, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
axA.set_xticks(np.arange(len(col_labels)))
axA.set_xticklabels(col_labels, fontsize=10)
axA.set_yticks(np.arange(len(MODEL_ORDER)))
axA.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=10.5)
for sep in [3, 6]:
    axA.axvline(sep - 0.5, color="white", linewidth=2.5)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        v = matrix[i, j]
        if np.isnan(v):
            continue
        text_color = "white" if norm[i, j] > 0.55 else "black"
        is_best = np.isclose(v, np.nanmax(matrix[:, j]))
        weight = "bold" if is_best else "normal"
        axA.text(j, i, f"{v:.3f}", ha="center", va="center",
                 color=text_color, fontsize=10.5, fontweight=weight)
axA.set_title("A. Per-column normalized scores (bold = column max)",
              fontsize=13, fontweight="bold", loc="left", pad=10)
cbar = fig.colorbar(im, ax=axA, fraction=0.025, pad=0.015)
cbar.set_label("column-normalized", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# ---- Panel B: generalization gap (top right, spans cols 2-3) --------------
axB = fig.add_subplot(gs[0, 1:])
ds_x = ["Internal", "External", "Holdout"]
for m in MODEL_ORDER:
    rec = lookup(m)
    ys = [rec[k]["mcc"] for k in DATASETS]
    axB.plot(ds_x, ys, marker="o", linewidth=2.4, markersize=9,
             color=COLORS[m], label=MODEL_LABELS[m])
    for x_i, y_i in zip(ds_x, ys):
        axB.annotate(f"{y_i:.3f}", xy=(x_i, y_i), xytext=(0, 8),
                     textcoords="offset points", ha="center",
                     fontsize=8.5, color=COLORS[m])
axB.set_title("B. Generalization profile — MCC across datasets",
              fontsize=13, fontweight="bold", loc="left", pad=8)
axB.set_ylabel("MCC", fontsize=11)
axB.grid(alpha=0.4, linestyle="--")
axB.set_axisbelow(True)
axB.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.9)
axB.set_ylim(0.20, 0.55)

# ---- Panel C: pos_weight effect (bottom-middle) ---------------------------
axC = fig.add_subplot(gs[1, 1])
pw_pair = ["iter90_pw008", "iter90_p2_auto"]
pw_colors = ["#d9534f", "#0275d8"]
pw_labels = ["pw=0.08 (fixed)", "pw=auto"]
metrics_pw = ["roc_auc", "mcc", "f1"]
ds_for_pw = ["ext", "holdout"]
ds_for_pw_labels = ["External", "Holdout"]

x_groups = []
group_centers = []
labels_x = []
bw = 0.35
gap = 0.25  # space between dataset groups
group_w = bw * 2 * len(metrics_pw) + bw * (len(metrics_pw) - 1) * 0.4

for gi, (k, kl) in enumerate(zip(ds_for_pw, ds_for_pw_labels)):
    base = gi * (len(metrics_pw) + 1.0)
    rec0 = lookup(pw_pair[0])
    rec1 = lookup(pw_pair[1])
    for mi, metric in enumerate(metrics_pw):
        center = base + mi
        v0 = rec0[k][metric]
        v1 = rec1[k][metric]
        axC.bar(center - bw / 2, v0, bw, color=pw_colors[0],
                edgecolor="black", linewidth=0.4,
                label=pw_labels[0] if (gi == 0 and mi == 0) else None)
        axC.bar(center + bw / 2, v1, bw, color=pw_colors[1],
                edgecolor="black", linewidth=0.4,
                label=pw_labels[1] if (gi == 0 and mi == 0) else None)
        delta = v1 - v0
        if metric != "roc_auc":
            axC.annotate(f"Δ{delta:+.2f}", xy=(center, max(v0, v1) + 0.06),
                         ha="center", fontsize=8.5,
                         bbox=dict(boxstyle="round,pad=0.2",
                                   fc="#fff3cd", ec="none", alpha=0.9))
        x_groups.append(center)
        labels_x.append(metric_labels[metric])
    group_centers.append(base + 1)

axC.set_xticks(x_groups)
axC.set_xticklabels(labels_x, fontsize=9)
for c, lab in zip(group_centers, ds_for_pw_labels):
    axC.text(c, -0.15, lab, ha="center", fontsize=11, fontweight="bold",
             transform=axC.get_xaxis_transform())
axC.set_ylim(0, 1.05)
axC.set_ylabel("Score", fontsize=10)
axC.set_title("C. pos_weight effect (iter90 same arch)",
              fontsize=12.5, fontweight="bold", loc="left", pad=8)
axC.grid(axis="y", linestyle="--", alpha=0.4)
axC.set_axisbelow(True)
axC.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9)

# ---- Panel D: takeaways (bottom-right) ------------------------------------
axD = fig.add_subplot(gs[1, 2])
axD.axis("off")
axD.set_title("D. Key takeaways", fontsize=13, fontweight="bold",
              loc="left", pad=8)

bullets = [
    ("pw=0.08 fails to generalize",
     "Internal AUC top, but External MCC 0.27 / F1 0.51\n— threshold calibration breaks under shift."),
    ("pw=auto restores MCC & F1",
     "Same arch + auto pos_weight: External MCC +0.07,\nHoldout F1 +0.16. ROC-AUC unchanged."),
    ("Phase 2 tuning ≈ BASE under pw=auto",
     "iter90 P2 vs BASE within ±0.01 on every metric.\nGains were entangled with pw, not the HPs."),
    ("iter55 vs iter90 trade-off",
     "iter55 BASE leads Internal MCC/F1; iter90 leads\nExternal MCC. Holdout AUC tied (~0.817)."),
    ("TabPFN owns calibration, loses ranking",
     "Highest MCC/F1 on Ext+Hold, lowest ROC-AUC\neverywhere → ensemble candidate."),
]
y0 = 0.96
dy = 0.185
for i, (head, body) in enumerate(bullets):
    y = y0 - i * dy
    axD.text(0.0, y, f"{i+1}. {head}",
             transform=axD.transAxes, fontsize=11, fontweight="bold",
             color="#1f3b66", va="top")
    axD.text(0.025, y - 0.045, body,
             transform=axD.transAxes, fontsize=9.5, color="#222222", va="top")

# Footer note
fig.text(
    0.045, 0.018,
    "Models: iter90/iter55 (multimodal mol-encoder, soft-vote 10 seeds)  ·  "
    "TabPFN-2.5 with maccs+avalon+rdkit+mole 1663-d feats (single inference).",
    fontsize=9, color="#555555",
)

out_path = OUT / "model_comparison_slide.png"
fig.savefig(out_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print("Saved:", out_path)
