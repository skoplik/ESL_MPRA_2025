#!/usr/bin/env python3
"""
Compare model predictions against COMPASS experimental avg_delta_logit_pooled.

Test set  (3,526 variants, WT-PSI-filtered): Baseline MMSplice, Retrained MMSplice,
                                              SpliceAI, AlphaGenome
Aggregate (all filtered variants):            Baseline MMSplice, Retrained MMSplice,
                                              SpliceAI, AlphaGenome, Pangolin, HAL

Reads from mega_pred_file_filtered.csv (built by build_mega_pred_file.py).
Outputs scatter PDFs + bar chart to mmsplice_retrain_plots/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

MEGA_FILE = "/ESL/ESL_MPRA/Figure_2/model_comparison/mega_pred_file_filtered.csv"
PLOTDIR   = "/ESL/ESL_MPRA/Figure_2/plots"

OUTDIR = {
    "SpliceAI":      os.path.join(PLOTDIR, "SpliceAI"),
    "AlphaGenome":   os.path.join(PLOTDIR, "AlphaGenome"),
    "Pangolin":      os.path.join(PLOTDIR, "Pangolin"),
    "HAL":           os.path.join(PLOTDIR, "HAL"),
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
    "Pangolin":           OUTDIR["Pangolin"],
    "HAL":                OUTDIR["HAL"],
}

TRUE_COL = "avg_delta_logit_pooled"

# Colors: Plasma colormap shades; Retrained MMSplice stays blue
# plasma positions: Baseline=0.05, SpliceAI=0.25, AlphaGenome=0.45, Pangolin=0.65, HAL=0.85
COLOR = {
    "Baseline MMSplice":  "#2a0593",  # plasma 0.05 — deep purple
    "Retrained MMSplice": "#377EB8",  # blue (fixed)
    "SpliceAI":           "#7e03a8",  # plasma 0.25 — purple
    "AlphaGenome":        "#bf3984",  # plasma 0.45 — magenta-pink
    "Pangolin":           "#ea7457",  # plasma 0.65 — orange-red
    "HAL":                "#feba2c",  # plasma 0.85 — amber
}

# Models shown in test set (must be in mmsplice test set)
TEST_MODELS = [
    ("baseline_mmsplice_delta_logit",  "Baseline MMSplice",  COLOR["Baseline MMSplice"]),
    ("retrained_mmsplice_delta_logit", "Retrained MMSplice", COLOR["Retrained MMSplice"]),
    ("spliceai_delta_logit",           "SpliceAI",           COLOR["SpliceAI"]),
    ("alphagenome_delta_logit",        "AlphaGenome",        COLOR["AlphaGenome"]),
]

# Models shown in aggregate (full filtered set)
AGG_MODELS = [
    ("baseline_mmsplice_delta_logit",  "Baseline MMSplice",  COLOR["Baseline MMSplice"]),
    ("retrained_mmsplice_delta_logit", "Retrained MMSplice", COLOR["Retrained MMSplice"]),
    ("spliceai_delta_logit",           "SpliceAI",           COLOR["SpliceAI"]),
    ("alphagenome_delta_logit",        "AlphaGenome",        COLOR["AlphaGenome"]),
    ("pangolin_delta_logit",           "Pangolin",           COLOR["Pangolin"]),
    ("hal_delta_logit",                "HAL",                COLOR["HAL"]),
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def pearson_str(x, y):
    if len(x) < 2:
        return "r=NA\nn=0"
    r, _ = pearsonr(x, y)
    return f"r={r:.2f}\nn={len(x):,}"


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
    ax.text(0.05, 0.95, pearson_str(x, y),
            transform=ax.transAxes, va="top", ha="left", fontsize=8)


def save_single(x, y, color, title, fname, label):
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
    scatter_ax(ax, x, y, color, title)
    plt.tight_layout()
    path = os.path.join(MODEL_OUTDIR[label], fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def r_n(series_pred, series_true):
    valid = series_pred.notna() & series_true.notna() & np.isfinite(series_pred) & np.isfinite(series_true)
    x = series_pred[valid].values
    y = series_true[valid].values
    if len(x) < 2:
        return float("nan"), 0
    r, _ = pearsonr(x, y)
    return round(float(r), 4), int(len(x))


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading filtered mega pred file...")
mega = pd.read_csv(MEGA_FILE, low_memory=False)
test = mega[mega["mmsplice_is_test"] == True].copy()
print(f"  Aggregate: {len(mega):,}   Test set: {len(test):,}")

# ── Test set scatter plots ─────────────────────────────────────────────────────
print("\n── Test set scatter plots ──")
for pred_col, label, color in TEST_MODELS:
    sub = test[[pred_col, TRUE_COL]].dropna()
    safe = label.lower().replace(" ", "_")
    save_single(sub[pred_col].values, sub[TRUE_COL].values,
                color, f"{label} — Test set",
                f"testset_{safe}_vs_true.pdf", label)

# 4-panel test set
fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.2), dpi=300)
for ax, (pred_col, label, color) in zip(axes, TEST_MODELS):
    sub = test[[pred_col, TRUE_COL]].dropna()
    scatter_ax(ax, sub[pred_col].values, sub[TRUE_COL].values, color, f"{label} — Test set")
plt.tight_layout()
path = os.path.join(OUTDIR["merged_output"], "testset_all_four_comparison.pdf")
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

# ── Aggregate scatter plots ────────────────────────────────────────────────────
print("\n── Aggregate scatter plots ──")
for pred_col, label, color in AGG_MODELS:
    sub = mega[[pred_col, TRUE_COL]].dropna()
    safe = label.lower().replace(" ", "_")
    save_single(sub[pred_col].values, sub[TRUE_COL].values,
                color, f"{label} — Aggregate",
                f"aggregate_{safe}_vs_true.pdf", label)

# 6-panel aggregate (2×3)
fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.4), dpi=300)
for ax, (pred_col, label, color) in zip(axes.flatten(), AGG_MODELS):
    sub = mega[[pred_col, TRUE_COL]].dropna()
    scatter_ax(ax, sub[pred_col].values, sub[TRUE_COL].values, color, f"{label} — Aggregate")
plt.tight_layout()
path = os.path.join(OUTDIR["merged_output"], "aggregate_all_six_comparison.pdf")
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

# ── Stats ──────────────────────────────────────────────────────────────────────
print("\n── Pearson r summary ──")
stats = {}
print("  Test set:")
for pred_col, label, color in TEST_MODELS:
    r, n = r_n(test[pred_col], test[TRUE_COL])
    stats[("Test set", label)] = (r, n, color)
    print(f"    {label:25s}  r={r:.4f}  n={n:,}")

print("  Aggregate:")
for pred_col, label, color in AGG_MODELS:
    r, n = r_n(mega[pred_col], mega[TRUE_COL])
    stats[("Aggregate", label)] = (r, n, color)
    print(f"    {label:25s}  r={r:.4f}  n={n:,}")

# ── Bar plot 2: Retrained benchmarking — Aggregate (left) + Test set (right) ───
print("\n── Bar plot: retrained benchmarking ──")

# Models in display order (same for both groups)
BENCH_MODELS = [
    ("baseline_mmsplice_delta_logit",  "Baseline\nMMSplice",  COLOR["Baseline MMSplice"]),
    ("retrained_mmsplice_delta_logit", "Retrained\nMMSplice", COLOR["Retrained MMSplice"]),
    ("alphagenome_delta_logit",        "AlphaGenome",         COLOR["AlphaGenome"]),
    ("spliceai_delta_logit",           "SpliceAI",            COLOR["SpliceAI"]),
]

n_models = len(BENCH_MODELS)
gap = 1.5  # space between the two groups
x_agg  = np.arange(n_models)
x_test = np.arange(n_models) + n_models + gap

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

for x_pos, group, df in [(x_agg, "Aggregate", mega), (x_test, "Test set", test)]:
    for xi, (pred_col, label, color) in zip(x_pos, BENCH_MODELS):
        r, n = r_n(df[pred_col], df[TRUE_COL])
        bar = ax.bar(xi, r, color=color, edgecolor="black", width=0.7)
        ax.text(xi, r + 0.012, f"r={r:.2f}\nn={n:,}",
                ha="center", va="bottom", fontsize=7.5)
    # Group label below x-axis
    ax.text(x_pos.mean(), -0.07, group, ha="center", va="top",
            fontsize=11, fontweight="bold", transform=ax.get_xaxis_transform())

ax.set_ylim(0, 1)
ax.set_ylabel("Pearson r", fontsize=12)
ax.set_title("Retrained MMSplice Benchmarking\n(Δlogit avg across cell lines)", fontsize=11)
ax.set_xticks(np.concatenate([x_agg, x_test]))
ax.set_xticklabels([m[1] for m in BENCH_MODELS] * 2, fontsize=9)
ax.axhline(0, color="black", linewidth=0.5)
# Vertical separator between groups
ax.axvline(n_models + gap / 2 - 0.5, color="grey", linewidth=0.8, linestyle="--")
plt.tight_layout()
path = os.path.join(OUTDIR["merged_output"], "bar_retrained_benchmarking.pdf")
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

print("\nDone.")
