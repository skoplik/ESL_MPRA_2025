#!/usr/bin/env python3
"""
Bar plot + scatter plots for model comparison (no retraining benchmarking here,
just inference). Models in order: MMSplice, HAL, Pangolin, SpliceAI, AlphaGenome.
MMSplice retrained scatter also saved with its own color.

Outputs saved to: /ESL/ESL_MPRA/Figure_3/plots/merged_output/
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

MEGA_FILE = "/ESL/ESL_MPRA/Figure_3/model_comparison/mega_pred_file_filtered.csv"
OUTDIR    = "/ESL/ESL_MPRA/Figure_3/plots/merged_output"
os.makedirs(OUTDIR, exist_ok=True)

TRUE_COL = "avg_delta_logit_pooled"

# ── Colors ─────────────────────────────────────────────────────────────────────
# 5 inference models: sample plasma colormap in range [0.1, 0.8] evenly
# Order: MMSplice, HAL, Pangolin, SpliceAI, AlphaGenome
_plasma = mpl.colormaps["plasma"]
_positions = np.linspace(0.25, 0.9, 5)  # 5 evenly-spaced samples in 0.25–0.9
INFERENCE_MODELS = [
    ("baseline_mmsplice_delta_logit", "MMSplice",    _plasma(_positions[0])),
    ("hal_delta_logit",               "HAL",          _plasma(_positions[1])),
    ("pangolin_delta_logit",          "Pangolin",     _plasma(_positions[2])),
    ("spliceai_delta_logit",          "SpliceAI",     _plasma(_positions[3])),
    ("alphagenome_delta_logit",       "AlphaGenome",  _plasma(_positions[4])),
]

# MMSplice retrained: blue
RETRAINED_COLOR = "#377EB8"

# ── Helpers ────────────────────────────────────────────────────────────────────
def r_n(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return float("nan"), 0
    r, _ = pearsonr(x, y)
    return round(float(r), 4), int(len(x))


def scatter_ax(ax, x, y, color, label, alpha=0.10, s=8):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    lim = max(np.abs(np.concatenate([x, y])).max() * 1.05, 1.0)
    ax.plot([-lim, lim], [-lim, lim], color="grey", linestyle="--",
            linewidth=1, zorder=1)
    ax.scatter(x, y, s=s, color=color, alpha=alpha, linewidths=0,
               rasterized=True, zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted Δlogit(PSI)", fontsize=9)
    ax.set_ylabel("Measured Δlogit(PSI)", fontsize=9)
    ax.set_title(label, fontsize=9, pad=4)
    r, n = r_n(x, y)
    ax.text(0.05, 0.95, f"r={r:.2f}\nn={n:,}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8)


def save_fig(fig, stem):
    for ext in ("pdf", "png"):
        path = os.path.join(OUTDIR, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")


# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading mega pred file...")
mega = pd.read_csv(MEGA_FILE, low_memory=False)
print(f"  {len(mega):,} rows")

# Aggregate mask: rows where MMSplice has a prediction (n=70,544)
# All models are evaluated on this shared set so n is comparable
AGG_MASK = mega["baseline_mmsplice_delta_logit"].notna()
mega_agg = mega.loc[AGG_MASK].copy()
print(f"  Aggregate mask (MMSplice coverage): {AGG_MASK.sum():,} rows")

# ── Bar plot: inference models only ───────────────────────────────────────────
print("\n── Bar plot ──")
bar_labels = []
bar_colors = []
bar_rs     = []
bar_ns     = []

for pred_col, label, color in INFERENCE_MODELS:
    sub = mega[[pred_col, TRUE_COL]].dropna()
    r, n = r_n(sub[pred_col].values, sub[TRUE_COL].values)
    bar_labels.append(label)
    bar_colors.append(color)
    bar_rs.append(r)
    bar_ns.append(n)
    print(f"  {label:15s}  r={r:.4f}  n={n:,}")

BAR_WIDTH = 0.4   # skinny bars
BAR_SPACING = 0.55  # distance between bar centers (< 1 compresses spacing)

x = np.arange(len(INFERENCE_MODELS)) * BAR_SPACING
fig, ax = plt.subplots(figsize=(4.5, 5), dpi=300)
bars = ax.bar(x, bar_rs, color=bar_colors, edgecolor="black", linewidth=1.0,
              width=BAR_WIDTH)
for bar, r_val, n_val in zip(bars, bar_rs, bar_ns):
    ax.text(bar.get_x() + bar.get_width() / 2,
            r_val + 0.012,
            f"r={r_val:.2f}\nn={n_val:,}",
            ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, 1)
ax.set_ylabel("Pearson r", fontsize=13)
ax.set_title("Model Correlations with Measured Δlogit(PSI)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(bar_labels, fontsize=11, rotation=45, ha="right")
ax.tick_params(labelsize=10)
ax.axhline(0, color="black", linewidth=1.0)
plt.tight_layout()
save_fig(fig, "bar_inference_model_comparison")
plt.close(fig)

# ── Wide horizontal version ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
bars = ax.bar(x, bar_rs, color=bar_colors, edgecolor="black", linewidth=1.0,
              width=BAR_WIDTH)
for bar, r_val, n_val in zip(bars, bar_rs, bar_ns):
    ax.text(bar.get_x() + bar.get_width() / 2,
            r_val + 0.012,
            f"r={r_val:.2f}\nn={n_val:,}",
            ha="center", va="bottom", fontsize=11)
ax.set_ylim(0, 1)
ax.set_ylabel("Pearson r", fontsize=13)
ax.set_title("Model Correlations with Measured Δlogit(PSI)", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(bar_labels, fontsize=12, rotation=0, ha="center")
ax.tick_params(labelsize=11)
ax.axhline(0, color="black", linewidth=1.0)
plt.tight_layout()
save_fig(fig, "bar_inference_model_comparison_wide")
plt.close(fig)

# ── Bar plot: retrained benchmarking (Agg / Train / Test) ─────────────────────
# Models: MMSplice Baseline, MMSplice Retrained, HAL, Pangolin, SpliceAI, AlphaGenome
# For MMSplice (baseline/retrained), Agg = all rows with predictions;
# Train = mmsplice_is_test == False; Test = mmsplice_is_test == True
# For other models, Train/Test subsets use the same mmsplice_is_test split.
print("\n── Bar plot: retrained benchmarking (Agg / Train / Test) ──")

RETRAIN_MODELS = [
    ("baseline_mmsplice_delta_logit",  "MMSplice\nBaseline",  INFERENCE_MODELS[0][2]),
    ("retrained_mmsplice_delta_logit", "MMSplice\nRetrained", RETRAINED_COLOR),
    ("hal_delta_logit",                "HAL",                  INFERENCE_MODELS[1][2]),
    ("pangolin_delta_logit",           "Pangolin",             INFERENCE_MODELS[2][2]),
    ("spliceai_delta_logit",           "SpliceAI",             INFERENCE_MODELS[3][2]),
    ("alphagenome_delta_logit",        "AlphaGenome",          INFERENCE_MODELS[4][2]),
]

SUBSETS = [
    ("Aggregate", None),   # MMSplice-covered rows (mega_agg), no fill
    ("Test",      True),   # mmsplice_is_test == True, filled
]

# Build r matrix: shape (n_models, n_subsets)
rs_matrix = np.full((len(RETRAIN_MODELS), len(SUBSETS)), np.nan)
ns_matrix = np.zeros((len(RETRAIN_MODELS), len(SUBSETS)), dtype=int)

for i, (pred_col, lbl, _) in enumerate(RETRAIN_MODELS):
    for j, (subset_name, is_test_val) in enumerate(SUBSETS):
        if is_test_val is None:
            sub = mega_agg[[pred_col, TRUE_COL]].dropna()
        else:
            sub = mega_agg[mega_agg["mmsplice_is_test"] == is_test_val][[pred_col, TRUE_COL]].dropna()
        r, n = r_n(sub[pred_col].values, sub[TRUE_COL].values)
        rs_matrix[i, j] = r
        ns_matrix[i, j] = n
        print(f"  {lbl.replace(chr(10),' '):20s}  {subset_name:10s}  r={r:.4f}  n={n:,}")

n_models  = len(RETRAIN_MODELS)
n_subsets = len(SUBSETS)

# Blend a color toward white by fraction t (0=original, 1=white)
def lighten(color, t):
    c = np.array(mpl.colors.to_rgb(color))
    return tuple(c + (1 - c) * t)

# Aggregate=full color solid, Test=full color + hatch
HATCH = ["", "////"]

# Each model group: n_subsets tightly packed bars, with a small gap between groups
GROUP_W   = 0.6    # total width allocated per group (controls inter-group gap)
BAR_W     = GROUP_W / n_subsets

group_centers = np.arange(n_models)
offsets = np.linspace(-(GROUP_W - BAR_W) / 2,
                       (GROUP_W - BAR_W) / 2,
                       n_subsets)

fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

for j, (subset_name, _) in enumerate(SUBSETS):
    for i, (_, lbl, color) in enumerate(RETRAIN_MODELS):
        xpos  = group_centers[i] + offsets[j]
        r_val = rs_matrix[i, j]
        ax.bar(xpos, r_val, width=BAR_W, color=color,
               edgecolor="black", linewidth=0.8,
               hatch=HATCH[j],
               label=subset_name if i == 0 else "_nolegend_")
        n_val = ns_matrix[i, j]
        ax.text(xpos, r_val + 0.012,
                f"r={r_val:.2f}\nn={n_val:,}",
                ha="center", va="bottom", fontsize=8, rotation=90)

ax.set_ylim(0, 1.15)
ax.set_ylabel("Pearson r", fontsize=13)
ax.set_title("Model Correlations with Measured Δlogit(PSI)\n(Aggregate / Test)", fontsize=13)
ax.set_xticks(group_centers)
ax.set_xticklabels([m[1] for m in RETRAIN_MODELS], fontsize=11, rotation=45, ha="right")
ax.axhline(0, color="black", linewidth=0.5)
ax.legend(title="Subset", fontsize=11, title_fontsize=11,
          handles=[
              mpl.patches.Patch(facecolor="grey", edgecolor="black", hatch="",     label="Aggregate"),
              mpl.patches.Patch(facecolor="grey", edgecolor="black", hatch="////", label="Test"),
          ],
          bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
save_fig(fig, "bar_retrained_benchmarking_agg_train_test")
plt.close(fig)

# ── Individual scatter plots: 5 inference models ──────────────────────────────
print("\n── Individual scatter plots (inference models) ──")
for pred_col, label, color in INFERENCE_MODELS:
    sub = mega[[pred_col, TRUE_COL]].dropna()
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
    scatter_ax(ax, sub[pred_col].values, sub[TRUE_COL].values, color, label)
    plt.tight_layout()
    safe = label.lower().replace(" ", "_")
    save_fig(fig, f"scatter_{safe}")
    plt.close(fig)

# ── 5-panel scatter ────────────────────────────────────────────────────────────
print("\n── 5-panel scatter (inference models) ──")
fig, axes = plt.subplots(1, 5, figsize=(21, 4.2), dpi=300)
for ax, (pred_col, label, color) in zip(axes, INFERENCE_MODELS):
    sub = mega[[pred_col, TRUE_COL]].dropna()
    scatter_ax(ax, sub[pred_col].values, sub[TRUE_COL].values, color, label)
plt.tight_layout()
save_fig(fig, "scatter_5panel_inference")
plt.close(fig)

# ── MMSplice retrained scatter ────────────────────────────────────────────────
print("\n── MMSplice retrained scatter ──")
sub_ret = mega[["retrained_mmsplice_delta_logit", TRUE_COL]].dropna()
fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
scatter_ax(ax, sub_ret["retrained_mmsplice_delta_logit"].values,
           sub_ret[TRUE_COL].values, RETRAINED_COLOR, "MMSplice Retrained")
plt.tight_layout()
save_fig(fig, "scatter_mmsplice_retrained")
plt.close(fig)

# ── MMSplice baseline + retrained side-by-side ────────────────────────────────
print("\n── MMSplice baseline vs retrained 2-panel ──")
fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), dpi=300)

# baseline
sub_base = mega[["baseline_mmsplice_delta_logit", TRUE_COL]].dropna()
scatter_ax(axes[0], sub_base["baseline_mmsplice_delta_logit"].values,
           sub_base[TRUE_COL].values, INFERENCE_MODELS[0][2], "MMSplice (Baseline)")

# retrained
scatter_ax(axes[1], sub_ret["retrained_mmsplice_delta_logit"].values,
           sub_ret[TRUE_COL].values, RETRAINED_COLOR, "MMSplice Retrained")

plt.tight_layout()
save_fig(fig, "scatter_mmsplice_baseline_vs_retrained")
plt.close(fig)

# ── 6-panel scatter with train/test overlay ───────────────────────────────────
# Panels: 5 inference models + MMSplice Retrained
# Each panel: all points in model color (train), test set points in black on top
print("\n── 6-panel scatter with test set overlay ──")

SIX_PANEL_MODELS = INFERENCE_MODELS + [
    ("retrained_mmsplice_delta_logit", "MMSplice Retrained", RETRAINED_COLOR),
]

is_test  = mega_agg["mmsplice_is_test"] == True

fig, axes = plt.subplots(1, 6, figsize=(28, 5.5), dpi=300)
for ax_idx, (ax, (pred_col, label, color)) in enumerate(zip(axes, SIX_PANEL_MODELS)):
    sub_all  = mega_agg[[pred_col, TRUE_COL]].dropna()
    x_all    = sub_all[pred_col].values
    y_all    = sub_all[TRUE_COL].values

    sub_test = mega_agg.loc[is_test, [pred_col, TRUE_COL]].dropna()
    x_test   = sub_test[pred_col].values
    y_test   = sub_test[TRUE_COL].values

    valid_all  = np.isfinite(x_all)  & np.isfinite(y_all)
    valid_test = np.isfinite(x_test) & np.isfinite(y_test)
    x_all,  y_all  = x_all[valid_all],   y_all[valid_all]
    x_test, y_test = x_test[valid_test], y_test[valid_test]

    lim = max(np.abs(np.concatenate([x_all, y_all])).max() * 1.05, 1.0)
    ax.plot([-lim, lim], [-lim, lim], color="grey", linestyle="--",
            linewidth=1.5, zorder=1)

    ax.scatter(x_all, y_all, s=8, color=color, alpha=0.10,
               linewidths=0, rasterized=True, zorder=2)
    ax.scatter(x_test, y_test, s=8, color="black", alpha=0.45,
               linewidths=0, rasterized=True, zorder=3)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted Δlogit(PSI)", fontsize=12)
    ax.set_ylabel("Measured Δlogit(PSI)", fontsize=12)
    ax.set_title(label, fontsize=12, pad=5)
    ax.tick_params(labelsize=10)

    r_all,  n_all  = r_n(x_all,  y_all)
    r_test, n_test = r_n(x_test, y_test)
    ax.text(0.05, 0.95,
            f"r={r_all:.2f} (n={n_all:,})\nTest: r={r_test:.2f} (n={n_test:,})",
            transform=ax.transAxes, va="top", ha="left", fontsize=10)

    # per-panel legend bottom right
    handles = [
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                         markersize=8, label='Aggregate'),
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                         markersize=8, label='Test set'),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right", borderaxespad=0.5)

plt.tight_layout()
save_fig(fig, "scatter_6panel_with_testset")
plt.close(fig)

print("\nDone.")
