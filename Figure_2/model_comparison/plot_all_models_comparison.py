#!/usr/bin/env python3
"""
Plot scatter + bar chart comparing all splicing model predictions
against COMPASS experimental avg_delta_logit_pooled.

Models: SpliceAI, AlphaGenome, Pangolin, HAL,
        Baseline MMSplice, Retrained MMSplice

All comparisons use the full aggregate (no test-set split).
Outputs saved to: /ESL/ESL_MPRA/Figure_2/mmsplice_retrain_plots/
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
    "SpliceAI":            OUTDIR["SpliceAI"],
    "AlphaGenome":         OUTDIR["AlphaGenome"],
    "Pangolin":            OUTDIR["Pangolin"],
    "HAL":                 OUTDIR["HAL"],
    "Baseline MMSplice":   OUTDIR["MMSplice"],
    "Retrained MMSplice":  OUTDIR["MMSplice"],
}

TRUE_COL = "avg_delta_logit_pooled"

# Colors: Plasma colormap shades; Retrained MMSplice stays blue
# plasma positions: Baseline=0.05, SpliceAI=0.25, AlphaGenome=0.45, Pangolin=0.65, HAL=0.85
COLOR = {
    "Baseline MMSplice":  "#7d51f9",  # bright purple (lum≈0.45)
    "Retrained MMSplice": "#377EB8",  # blue (fixed)
    "SpliceAI":           "#a219d1",  # mid magenta (lum≈0.34)
    "AlphaGenome":        "#ed7a52",  # plasma 0.67 — orange-red
    "Pangolin":           "#fa9c3c",  # plasma 0.77 — amber-orange
    "HAL":                "#fdc627",  # plasma 0.88 — yellow-amber
}

# 5 models for general benchmarking (no retrained MMSplice — that has its own script)
# Each model uses its own full coverage (n varies)
MODELS = [
    ("baseline_mmsplice_delta_logit",  "Baseline MMSplice",  COLOR["Baseline MMSplice"]),
    ("hal_delta_logit",                "HAL",                COLOR["HAL"]),
    ("pangolin_delta_logit",           "Pangolin",           COLOR["Pangolin"]),
    ("spliceai_delta_logit",           "SpliceAI",           COLOR["SpliceAI"]),
    ("alphagenome_delta_logit",        "AlphaGenome",        COLOR["AlphaGenome"]),
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


def save_fig(fig, dirpath, stem):
    for ext in ("pdf", "png"):
        path = os.path.join(dirpath, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")


def save_single(x, y, color, title, fname, label):
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
    scatter_ax(ax, x, y, color, title)
    plt.tight_layout()
    stem = fname.replace(".pdf", "")
    save_fig(fig, MODEL_OUTDIR[label], stem)
    plt.close(fig)


def r_n(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return float("nan"), 0
    r, _ = pearsonr(x, y)
    return round(float(r), 4), int(len(x))


# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading mega pred file...")
mega = pd.read_csv(MEGA_FILE, low_memory=False)
print(f"  {len(mega):,} rows")

# ── Individual scatter plots ───────────────────────────────────────────────────
print("\n── Individual scatter plots ──")
stats = {}
for pred_col, label, color in MODELS:
    sub = mega[[pred_col, TRUE_COL]].dropna()
    x = sub[pred_col].values
    y = sub[TRUE_COL].values
    r, n = r_n(x, y)
    stats[(label,)] = (r, n)
    print(f"  {label:25s}  r={r:.4f}  n={n:,}")
    safe_name = label.lower().replace(" ", "_")
    save_single(x, y, color, f"{label} — Aggregate", f"general_bench_{safe_name}.pdf", label)

# ── 5-panel scatter (1 row × 5) ───────────────────────────────────────────────
print("\n── 5-panel scatter ──")
fig, axes = plt.subplots(1, 5, figsize=(21, 4.2), dpi=300)
for ax, (pred_col, label, color) in zip(axes, MODELS):
    sub = mega[[pred_col, TRUE_COL]].dropna()
    scatter_ax(ax, sub[pred_col].values, sub[TRUE_COL].values, color, label)
plt.tight_layout()
save_fig(fig, OUTDIR["merged_output"], "general_bench_5panel_scatter")
plt.close(fig)

# ── Bar plot 1: Full model comparison (no retrained MMSplice), sorted low→high ──
print("\n── Bar plot: full model comparison ──")

# Fixed display order (low→high by aggregate r): Baseline MMSplice, HAL, Pangolin, SpliceAI, AlphaGenome
BAR1_MODELS = [
    ("baseline_mmsplice_delta_logit",  "Baseline\nMMSplice", COLOR["Baseline MMSplice"]),
    ("hal_delta_logit",                "HAL",                COLOR["HAL"]),
    ("pangolin_delta_logit",           "Pangolin",           COLOR["Pangolin"]),
    ("spliceai_delta_logit",           "SpliceAI",           COLOR["SpliceAI"]),
    ("alphagenome_delta_logit",        "AlphaGenome",        COLOR["AlphaGenome"]),
]

bar1_labels = [m[1] for m in BAR1_MODELS]
bar1_colors = [m[2] for m in BAR1_MODELS]
bar1_rs = []
bar1_ns = []
for pred_col, lbl, _ in BAR1_MODELS:
    sub = mega[[pred_col, TRUE_COL]].dropna()
    r, n = r_n(sub[pred_col].values, sub[TRUE_COL].values)
    bar1_rs.append(r)
    bar1_ns.append(n)

x = np.arange(len(BAR1_MODELS))
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
bars = ax.bar(x, bar1_rs, color=bar1_colors, edgecolor="black", width=0.6)
for bar, r_val, n_val in zip(bars, bar1_rs, bar1_ns):
    ax.text(bar.get_x() + bar.get_width() / 2,
            r_val + 0.012,
            f"r={r_val:.2f}\nn={n_val:,}",
            ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, 1)
ax.set_ylabel("Pearson r", fontsize=12)
ax.set_title("Splicing Model Performance\n(Aggregate, avg Δlogit across cell lines)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(bar1_labels, fontsize=10, rotation=45, ha="right")
ax.axhline(0, color="black", linewidth=0.5)
plt.tight_layout()
save_fig(fig, OUTDIR["merged_output"], "bar_full_model_comparison")
plt.close(fig)

print("\nDone.")
