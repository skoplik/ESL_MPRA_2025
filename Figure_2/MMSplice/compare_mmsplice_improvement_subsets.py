#!/usr/bin/env python3
"""
Compare MSE and Pearson r (baseline vs. retrained MMSplice) on held-out test set.

Produces one figure per subset dimension, each with two rows:
  Row 1: absolute Pearson r  (baseline vs retrained paired bars)
  Row 2: absolute MSE        (baseline vs retrained paired bars)
  Inset annotation: Δr and ΔMSE

Dimensions: region, pos_bin, source, seq_type, psi_bin
Also prints a summary table.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

MEGA_FILE = "/ESL/ESL_MPRA/Figure_2/model_comparison/mega_pred_file_filtered.csv"
DATA_PATH = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz"
OUTDIR = "/ESL/ESL_MPRA/Figure_2/plots/MMSplice"
os.makedirs(OUTDIR, exist_ok=True)

_plasma = mpl.colormaps["plasma"]
COLOR_BASE     = _plasma(np.linspace(0.25, 0.9, 5)[0])  # plasma pos 0 — baseline MMSplice
COLOR_RETRAIN  = "#377EB8"                               # blue — retrained MMSplice
IMPROVE_COLOR  = "#2ca02c"   # green — improvement arrow/label
DEGRADE_COLOR  = "#d62728"   # red   — degradation arrow/label


# ── Load & prepare ─────────────────────────────────────────────────────────────
print("Loading data...")
mega = pd.read_csv(MEGA_FILE, low_memory=False)
test = mega[mega["mmsplice_is_test"] == True].copy()
test = test.dropna(subset=["baseline_mmsplice_delta_logit",
                             "retrained_mmsplice_delta_logit",
                             "avg_delta_logit_pooled"])
print(f"  Test set: {len(test):,}")

seqs = pd.read_csv(DATA_PATH, low_memory=False,
                   usecols=["Reference", "intron1", "exon", "intron2"])
seqs["intron1_len"] = seqs["intron1"].str.len()
seqs["exon_len"]    = seqs["exon"].str.len()
seqs["intron2_len"] = seqs["intron2"].str.len()
test = test.merge(seqs[["Reference", "intron1_len", "exon_len", "intron2_len"]],
                  on="Reference", how="left")

# Region
test["var_pos"] = test["snp"].str.extract(r"^(\d+)").astype(float)
test["region"] = "intron1"
test.loc[test["var_pos"] >= test["intron1_len"], "region"] = "exon"
test.loc[test["var_pos"] >= test["intron1_len"] + test["exon_len"], "region"] = "intron2"

# Position bin
test["pos_rel_5ss"] = test["var_pos"] - test["intron1_len"]
test["pos_rel_3ss"] = test["var_pos"] - (test["intron1_len"] + test["exon_len"])
test["dist_nearest_ss"] = test[["pos_rel_5ss", "pos_rel_3ss"]].abs().min(axis=1)

def pos_bin(d):
    if d <= 4:   return "0–4 nt\n(core)"
    if d <= 20:  return "5–20 nt\n(proximal)"
    if d <= 80:  return "21–80 nt\n(distal)"
    return "81+ nt\n(distal)"

POS_ORDER = ["0–4 nt\n(core)", "5–20 nt\n(proximal)", "21–80 nt\n(distal)", "81+ nt\n(distal)"]
test["pos_bin"] = test["dist_nearest_ss"].apply(pos_bin)

# PSI bin
test["psi_bin"] = pd.cut(test["HEK_wt_pooled_psi_raw"],
                          bins=[0, 0.2, 0.8, 1.0],
                          labels=["Near-excluded\n(0–0.2)",
                                  "Intermediate\n(0.2–0.8)",
                                  "Near-included\n(0.8–1.0)"],
                          include_lowest=True).astype(str)
PSI_ORDER = ["Near-excluded\n(0–0.2)", "Intermediate\n(0.2–0.8)", "Near-included\n(0.8–1.0)"]

# SDV strength bin (by |avg_delta_logit_pooled|)
test["sdv_bin"] = pd.cut(test["avg_delta_logit_pooled"].abs(),
                          bins=[0, 0.5, 1.0, 2.0, np.inf],
                          labels=["<0.5\n(sub-threshold)",
                                  "0.5–1\n(moderate)",
                                  "1–2\n(SDV)",
                                  ">2\n(strong SDV)"],
                          include_lowest=True).astype(str)
SDV_ORDER = ["<0.5\n(sub-threshold)", "0.5–1\n(moderate)", "1–2\n(SDV)", ">2\n(strong SDV)"]


# ── Metrics helper ─────────────────────────────────────────────────────────────
def compute_metrics(df):
    y = df["avg_delta_logit_pooled"].values
    yb = df["baseline_mmsplice_delta_logit"].values
    yr = df["retrained_mmsplice_delta_logit"].values
    mse_b = float(np.mean((yb - y) ** 2))
    mse_r = float(np.mean((yr - y) ** 2))
    r_b, _ = stats.pearsonr(y, yb)
    r_r, _ = stats.pearsonr(y, yr)
    return {
        "n": len(df),
        "r_baseline":   round(float(r_b), 4),
        "r_retrained":  round(float(r_r), 4),
        "delta_r":      round(float(r_r - r_b), 4),
        "MSE_baseline":  round(mse_b, 4),
        "MSE_retrained": round(mse_r, 4),
        "delta_MSE":     round(mse_r - mse_b, 4),
    }

def build_table(col, order=None):
    groups = order if order else sorted(test[col].dropna().unique())
    rows = []
    for g in groups:
        sub = test[test[col] == g]
        if len(sub) < 5:
            continue
        m = compute_metrics(sub)
        m["subset"] = g
        rows.append(m)
    return pd.DataFrame(rows)


# ── Build all tables ───────────────────────────────────────────────────────────
DIMENSIONS = [
    ("region",   ["intron1", "exon", "intron2"],
     "Genomic region",     "Region"),
    ("pos_bin",  POS_ORDER,
     "Distance to nearest splice site", "Position bin"),
    ("source",   None,
     "Variant source",     "Source"),
    ("seq_type", None,
     "Sequence type",      "Seq type"),
    ("psi_bin",  PSI_ORDER,
     "Reference PSI (HEK)", "PSI bin"),
    ("sdv_bin",  SDV_ORDER,
     "Effect size (|Δlogit|)", "SDV strength"),
]

tables = {}
for col, order, title, short in DIMENSIONS:
    tables[col] = build_table(col, order)

# Overall row
overall = compute_metrics(test)
overall["subset"] = "ALL"

# Print summary
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 120)

print("\n── Overall ──")
print(pd.DataFrame([overall]).to_string(index=False))
for col, _, title, _ in DIMENSIONS:
    print(f"\n── By {title} ──")
    print(tables[col].to_string(index=False))


# ── Plot helper ────────────────────────────────────────────────────────────────
def make_dimension_figure(tbl, dim_title, figsize=None):
    """
    Two-row figure for one dimension.
    Row 0: Pearson r  — paired bars (baseline grey, retrained blue) + Δr annotation
    Row 1: MSE        — paired bars + ΔMSE annotation
    """
    subsets = tbl["subset"].tolist()
    ns      = tbl["n"].tolist()
    n_grps  = len(subsets)

    if figsize is None:
        figsize = (max(5, n_grps * 1.6 + 1.5), 7)

    fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=300)

    BW = 0.32
    x  = np.arange(n_grps)

    xtick_labels = [f"{s}\n(n={n:,})" for s, n in zip(subsets, ns)]

    ROWS = [
        ("r_baseline",   "r_retrained",   "delta_r",   "Pearson r", "positive", axes[0]),
        ("MSE_baseline", "MSE_retrained", "delta_MSE", "MSE",       "negative", axes[1]),
    ]

    for row_idx, (metric_b, metric_r, delta_col, ylabel, better_dir, ax) in enumerate(ROWS):
        vals_b = tbl[metric_b].values
        vals_r = tbl[metric_r].values
        deltas = tbl[delta_col].values

        ax.bar(x - BW/2, vals_b, BW, color=COLOR_BASE,
               edgecolor="black", linewidth=0.6, label="Baseline")
        ax.bar(x + BW/2, vals_r, BW, color=COLOR_RETRAIN,
               edgecolor="black", linewidth=0.6, label="Retrained")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=11)

        if row_idx == 0:
            ylim_top = min(1.0, max(vals_b.max(), vals_r.max()) * 1.3)
            note = "Δ > 0 = retrained better"
        else:
            ylim_top = max(vals_b.max(), vals_r.max()) * 1.35
            note = "Δ < 0 = retrained better"
        ax.set_ylim(0, ylim_top)

        ax.text(0.99, 0.97, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color="grey", style="italic")

        # Δ annotation — placed after ylim is set
        for xi, (vb, vr, d) in enumerate(zip(vals_b, vals_r, deltas)):
            improved = (d > 0) if better_dir == "positive" else (d < 0)
            color = IMPROVE_COLOR if improved else DEGRADE_COLOR
            sign  = "+" if d > 0 else ""
            ax.text(xi, max(vb, vr) + ylim_top * 0.02,
                    f"Δ={sign}{d:.3f}",
                    ha="center", va="bottom", fontsize=7.5,
                    color=color, fontweight="bold")

    # Shared legend above figure, outside plot area
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_BASE,    edgecolor="black", linewidth=0.6, label="Baseline"),
        mpatches.Patch(facecolor=COLOR_RETRAIN, edgecolor="black", linewidth=0.6, label="Retrained"),
    ]
    fig.legend(handles=legend_handles, fontsize=10, frameon=False,
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"MMSplice retraining improvement — by {dim_title}\n(held-out test set, n={len(test):,} total)",
                 fontsize=11, y=1.05)
    plt.tight_layout()
    return fig


# ── Generate and save one figure per dimension ─────────────────────────────────
for col, _, dim_title, short in DIMENSIONS:
    tbl = tables[col]
    if tbl.empty:
        continue
    fig = make_dimension_figure(tbl, dim_title)
    stem = f"mmsplice_improvement_{col}"
    for ext in ("pdf", "png"):
        path = os.path.join(OUTDIR, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ── Combined overview: all dimensions in one tall figure ───────────────────────
print("\nBuilding combined overview figure...")

n_dims = len(DIMENSIONS)
fig, axes = plt.subplots(n_dims, 2, figsize=(13, n_dims * 3.2), dpi=300)

BW = 0.32

for row_idx, (col, _, dim_title, short) in enumerate(DIMENSIONS):
    tbl     = tables[col]
    subsets = tbl["subset"].tolist()
    ns      = tbl["n"].tolist()
    x       = np.arange(len(subsets))
    xtick_labels = [f"{s}\n(n={n:,})" for s, n in zip(subsets, ns)]

    for col_idx, (metric_b, metric_r, delta_col, ylabel, better_dir) in enumerate([
        ("r_baseline",   "r_retrained",   "delta_r",   "Pearson r", "positive"),
        ("MSE_baseline", "MSE_retrained", "delta_MSE", "MSE",       "negative"),
    ]):
        ax = axes[row_idx, col_idx]
        vals_b = tbl[metric_b].values
        vals_r = tbl[metric_r].values
        deltas = tbl[delta_col].values

        ax.bar(x - BW/2, vals_b, BW, color=COLOR_BASE,
               edgecolor="black", linewidth=0.5, label="Baseline")
        ax.bar(x + BW/2, vals_r, BW, color=COLOR_RETRAIN,
               edgecolor="black", linewidth=0.5, label="Retrained")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        if row_idx == 0:
            ax.set_title(f"{'Pearson r' if col_idx==0 else 'MSE'}", fontsize=10, pad=4)

        ylim_top = (min(1.0, max(vals_b.max(), vals_r.max()) * 1.3)
                    if col_idx == 0
                    else max(vals_b.max(), vals_r.max()) * 1.3)
        ax.set_ylim(0, ylim_top)

        for xi, (vb, vr, d) in enumerate(zip(vals_b, vals_r, deltas)):
            improved = (d > 0) if better_dir == "positive" else (d < 0)
            color = IMPROVE_COLOR if improved else DEGRADE_COLOR
            sign  = "+" if d > 0 else ""
            ax.text(xi, max(vb, vr) + ylim_top * 0.025,
                    f"Δ={sign}{d:.3f}",
                    ha="center", va="bottom", fontsize=6.5,
                    color=color, fontweight="bold")

        # Row label on left axis only
        if col_idx == 0:
            ax.set_ylabel(f"{short}\n{ylabel}", fontsize=8.5)

legend_handles = [
    mpatches.Patch(facecolor=COLOR_BASE,    edgecolor="black", linewidth=0.6, label="Baseline"),
    mpatches.Patch(facecolor=COLOR_RETRAIN, edgecolor="black", linewidth=0.6, label="Retrained"),
]
fig.legend(handles=legend_handles, fontsize=10, frameon=False,
           loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.suptitle(
    "MMSplice retraining: Pearson r and MSE by subset\n(held-out test set, n=3,526)",
    fontsize=12, y=1.04
)
plt.tight_layout()

stem = "mmsplice_improvement_all_subsets_overview"
for ext in ("pdf", "png"):
    path = os.path.join(OUTDIR, f"{stem}.{ext}")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)


# ── Save CSV ───────────────────────────────────────────────────────────────────
rows = [{"dimension": "overall", **overall}]
for col, _, dim_title, _ in DIMENSIONS:
    for _, row in tables[col].iterrows():
        rows.append({"dimension": col, **row.to_dict()})

summary = pd.DataFrame(rows)
csv_path = os.path.join(OUTDIR, "mmsplice_improvement_by_subset.csv")
summary.to_csv(csv_path, index=False, float_format="%.6f")
print(f"Saved: {csv_path}")

print("\nDone.")
