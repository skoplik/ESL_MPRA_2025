#!/usr/bin/env python3
"""
Retrained MMSplice benchmarking plots.

4 models: Baseline MMSplice, Retrained MMSplice, SpliceAI, AlphaGenome.
All aggregate comparisons use the same 70,544 MMSplice-covered variants.
Test set uses the same 3,526 held-out variants for all 4 models.

Scatter plots: aggregate (model color) with test set overlaid in black.
Bar plot: aggregate (left group) + test set (right group).

Reads from mega_pred_file_filtered.csv (built by build_mega_pred_file.py).
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

MEGA_FILE = "/ESL/ESL_MPRA/Figure_3/model_comparison/mega_pred_file_filtered.csv"
PLOTDIR   = "/ESL/ESL_MPRA/Figure_3/plots"

OUTDIR = {
    "SpliceAI":      os.path.join(PLOTDIR, "SpliceAI"),
    "AlphaGenome":   os.path.join(PLOTDIR, "AlphaGenome"),
    "MMSplice":      os.path.join(PLOTDIR, "MMSplice"),
    "merged_output": os.path.join(PLOTDIR, "merged_output"),
}
for d in OUTDIR.values():
    os.makedirs(d, exist_ok=True)

MODEL_OUTDIR = {
    "Baseline MMSplice":  OUTDIR["MMSplice"],
    "Retrained MMSplice": OUTDIR["MMSplice"],
    "SpliceAI":           OUTDIR["SpliceAI"],
    "AlphaGenome":        OUTDIR["AlphaGenome"],
}

TRUE_COL = "avg_delta_logit_pooled"

# Colors: Plasma colormap shades; Retrained MMSplice stays blue
COLOR = {
    "Baseline MMSplice":  "#7d51f9",  # bright purple (lum≈0.45)
    "Retrained MMSplice": "#377EB8",  # blue (fixed)
    "SpliceAI":           "#a219d1",  # mid magenta (lum≈0.34)
    "AlphaGenome":        "#ed7a52",  # plasma 0.67 — orange-red
}

MODELS = [
    ("baseline_mmsplice_delta_logit",  "Baseline MMSplice",  COLOR["Baseline MMSplice"]),
    ("retrained_mmsplice_delta_logit", "Retrained MMSplice", COLOR["Retrained MMSplice"]),
    ("spliceai_delta_logit",           "SpliceAI",           COLOR["SpliceAI"]),
    ("alphagenome_delta_logit",        "AlphaGenome",        COLOR["AlphaGenome"]),
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def r_n(series_pred, series_true):
    valid = series_pred.notna() & series_true.notna() & np.isfinite(series_pred) & np.isfinite(series_true)
    x, y = series_pred[valid].values, series_true[valid].values
    if len(x) < 2:
        return float("nan"), 0
    r, _ = pearsonr(x, y)
    return round(float(r), 4), int(len(x))


def scatter_overlay(ax, x_agg, y_agg, x_test, y_test, color, label):
    """Aggregate in model color, test set in black on top."""
    all_vals = np.concatenate([x_agg, y_agg, x_test, y_test])
    lim = max(np.abs(all_vals).max() * 1.05, 1.0)

    ax.plot([-lim, lim], [-lim, lim], color="grey", linestyle="--", linewidth=1, zorder=1)
    ax.scatter(x_agg, y_agg, s=8, color=color, alpha=0.10, linewidths=0,
               rasterized=True, zorder=2)
    ax.scatter(x_test, y_test, s=8, color="black", alpha=0.20, linewidths=0,
               rasterized=True, zorder=3)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted Δlogit(PSI)", fontsize=9)
    ax.set_ylabel("Measured Δlogit(PSI)", fontsize=9)
    ax.set_title(label, fontsize=9, pad=4)

    r_agg,  n_agg  = pearsonr(x_agg,  y_agg)[0],  len(x_agg)
    r_test, n_test = pearsonr(x_test, y_test)[0],  len(x_test)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6,
               label=f"Aggregate  r={r_agg:.2f}, n={n_agg:,}"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="black", markersize=6,
               label=f"Test set   r={r_test:.2f}, n={n_test:,}"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=7, loc="upper left")


def save_fig(fig, dirpath, stem):
    for ext in ("pdf", "png"):
        path = os.path.join(dirpath, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")


def save_overlay(x_agg, y_agg, x_test, y_test, color, title, fname, label):
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
    scatter_overlay(ax, x_agg, y_agg, x_test, y_test, color, title)
    plt.tight_layout()
    stem = fname.replace(".pdf", "")
    save_fig(fig, MODEL_OUTDIR[label], stem)
    plt.close(fig)


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading filtered mega pred file...")
mega = pd.read_csv(MEGA_FILE, low_memory=False)
agg  = mega[mega["retrained_mmsplice_delta_logit"].notna()].copy()
test = mega[mega["mmsplice_is_test"] == True].copy()
print(f"  Aggregate (MMSplice-covered): {len(agg):,}   Test set: {len(test):,}")

# ── Scatter plots: aggregate + test overlay ────────────────────────────────────
print("\n── Scatter plots (agg + test overlay) ──")
for pred_col, label, color in MODELS:
    sub_agg  = agg[[pred_col, TRUE_COL]].dropna()
    sub_test = test[[pred_col, TRUE_COL]].dropna()
    safe = label.lower().replace(" ", "_")
    save_overlay(
        sub_agg[pred_col].values,  sub_agg[TRUE_COL].values,
        sub_test[pred_col].values, sub_test[TRUE_COL].values,
        color, label, f"aggregate_testoverlay_{safe}.pdf", label
    )

# 4-panel overlay
fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.2), dpi=300)
for ax, (pred_col, label, color) in zip(axes, MODELS):
    sub_agg  = agg[[pred_col, TRUE_COL]].dropna()
    sub_test = test[[pred_col, TRUE_COL]].dropna()
    scatter_overlay(ax,
        sub_agg[pred_col].values,  sub_agg[TRUE_COL].values,
        sub_test[pred_col].values, sub_test[TRUE_COL].values,
        color, label)
plt.tight_layout()
save_fig(fig, OUTDIR["merged_output"], "retrained_bench_4panel_overlay")
plt.close(fig)

# ── Stats ──────────────────────────────────────────────────────────────────────
print("\n── Pearson r summary ──")
stats = {}
for pred_col, label, color in MODELS:
    r_a, n_a = r_n(agg[pred_col],  agg[TRUE_COL])
    r_t, n_t = r_n(test[pred_col], test[TRUE_COL])
    stats[(label, "Aggregate")] = (r_a, n_a, color)
    stats[(label, "Test set")]  = (r_t, n_t, color)
    print(f"  {label:25s}  Agg r={r_a:.4f} n={n_a:,}   Test r={r_t:.4f} n={n_t:,}")

# ── Bar plot: Aggregate (left) + Test set (right) ─────────────────────────────
print("\n── Bar plot: retrained benchmarking ──")

n_models = len(MODELS)
gap      = 1.5
x_agg_pos  = np.arange(n_models)
x_test_pos = np.arange(n_models) + n_models + gap

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

for x_pos, group in [(x_agg_pos, "Aggregate"), (x_test_pos, "Test set")]:
    for xi, (pred_col, label, color) in zip(x_pos, MODELS):
        r, n = stats[(label, group)][:2]
        ax.bar(xi, r, color=color, edgecolor="black", width=0.7)
        ax.text(xi, r + 0.012, f"r={r:.2f}\nn={n:,}",
                ha="center", va="bottom", fontsize=7.5)
    ax.text(x_pos.mean(), -0.07, group, ha="center", va="top",
            fontsize=11, fontweight="bold", transform=ax.get_xaxis_transform())

ax.set_ylim(0, 1)
ax.set_ylabel("Pearson r", fontsize=12)
ax.set_title("Retrained MMSplice Benchmarking\n(avg Δlogit across cell lines)", fontsize=11)
ax.set_xticks(np.concatenate([x_agg_pos, x_test_pos]))
ax.set_xticklabels([m[1] for m in MODELS] * 2, fontsize=9, rotation=45, ha="right")
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(n_models + gap / 2 - 0.5, color="grey", linewidth=0.8, linestyle="--")
plt.tight_layout()
save_fig(fig, OUTDIR["merged_output"], "bar_retrained_benchmarking")
plt.close(fig)

print("\nDone.")
