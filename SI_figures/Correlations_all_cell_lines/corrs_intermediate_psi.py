"""
Replicate-vs-replicate correlations restricted to intermediate PSI values.
Produces plots for PSI, dPSI, and delta logit(PSI) at two PSI filter ranges (0.1–0.9, 0.2–0.8).
Addresses reviewer concern about reproducibility in the dynamic range of the assay.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 9

# === Inputs ===
input_file = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz"
output_dir = "/ESL/ESL_MPRA/SI_figures/Correlations_all_cell_lines/outputs/intermediate_psi"
os.makedirs(output_dir, exist_ok=True)

# === Cell line definitions: (rep1, rep2, pooled_psi_filter, rep1_metric, rep2_metric, axis_label) per metric
cell_line_names = ["HEK293", "HeLa", "K562", "MCF7", "HMC3"]
cell_prefixes   = ["HEK",    "HeLa", "K562", "MCF7", "HMC3"]

METRICS = {
    "psi": {
        "rep1":   "{p}_rep1_psi_raw",
        "rep2":   "{p}_rep2_psi_raw",
        "xlabel": "Rep 1 PSI",
        "ylabel": "Rep 2 PSI",
        "lim":    None,          # set dynamically from psi_low/psi_high
    },
    "dpsi": {
        "rep1":   "{p}_sd_dpsi",   # sd col not available per-rep; use rep logit - wt below
        "rep2":   None,
        "xlabel": "Rep 1 ΔPSI",
        "ylabel": "Rep 2 ΔPSI",
        "lim":    (-1, 1),
    },
    "delta_logit": {
        "rep1":   None,
        "rep2":   None,
        "xlabel": "Rep 1 Δlogit(PSI)",
        "ylabel": "Rep 2 Δlogit(PSI)",
        "lim":    (-8, 8),
    },
}

# === Load data ===
df = pd.read_csv(input_file, low_memory=False)
vars_df = df[df["snp"] != "none"].copy()
wt_df   = df[df["snp"] == "none"].copy()

# Pre-compute per-replicate dPSI and delta logit by subtracting matched WT
for p in cell_prefixes:
    reps_psi   = [c for c in [f"{p}_rep1_psi_raw",  f"{p}_rep2_psi_raw",
                               f"{p}_rep3_psi_raw",  f"{p}_rep4_psi_raw"] if c in df.columns]
    reps_logit = [c for c in [f"{p}_rep1_logit",    f"{p}_rep2_logit",
                               f"{p}_rep3_logit",    f"{p}_rep4_logit"]    if c in df.columns]

    wt_psi_by_event   = wt_df.set_index("event_id")[[c for c in reps_psi   if c in wt_df.columns]]
    wt_logit_by_event = wt_df.set_index("event_id")[[c for c in reps_logit if c in wt_df.columns]]

    for col in reps_psi:
        wt_col = col
        if wt_col in wt_psi_by_event.columns:
            vars_df[f"{col}_dpsi"] = (
                vars_df[col].astype(float).values
                - vars_df["event_id"].map(wt_psi_by_event[wt_col]).astype(float).values
            )

    for col in reps_logit:
        wt_col = col
        if wt_col in wt_logit_by_event.columns:
            vars_df[f"{col}_dlogit"] = (
                vars_df[col].astype(float).values
                - vars_df["event_id"].map(wt_logit_by_event[wt_col]).astype(float).values
            )

def make_plot(psi_low, psi_high, metric):
    """
    metric: "psi" | "dpsi" | "delta_logit"
    Filter is always on pooled PSI; x/y axes depend on metric.
    """
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharey=(metric != "psi"))
    fig.subplots_adjust(wspace=0.08)

    for ax, (cl, p) in zip(axes, zip(cell_line_names, cell_prefixes)):
        pooled_col = f"{p}_pooled_psi_raw"

        if metric == "psi":
            rep1_col = f"{p}_rep1_psi_raw"
            rep2_col = f"{p}_rep2_psi_raw"
            xlabel, ylabel = "Rep 1 PSI", "Rep 2 PSI"
            lim = (psi_low, psi_high)
        elif metric == "dpsi":
            rep1_col = f"{p}_rep1_psi_raw_dpsi"
            rep2_col = f"{p}_rep2_psi_raw_dpsi"
            xlabel, ylabel = "Rep 1 ΔPSI", "Rep 2 ΔPSI"
            lim = (-1, 1)
        else:  # delta_logit
            rep1_col = f"{p}_rep1_logit_dlogit"
            rep2_col = f"{p}_rep2_logit_dlogit"
            xlabel, ylabel = "Rep 1 Δlogit(PSI)", "Rep 2 Δlogit(PSI)"
            lim = (-8, 8)

        needed = [rep1_col, rep2_col, pooled_col]
        if any(c not in vars_df.columns for c in needed):
            ax.set_visible(False)
            continue

        sub = vars_df[needed].apply(pd.to_numeric, errors="coerce").dropna()
        sub = sub[(sub[pooled_col] >= psi_low) & (sub[pooled_col] <= psi_high)]

        x = sub[rep1_col].values
        y = sub[rep2_col].values
        if len(x) < 2:
            ax.set_visible(False)
            continue

        r, _   = pearsonr(x, y)
        rho, _ = spearmanr(x, y)
        n = len(sub)

        ax.scatter(x, y, s=3, alpha=0.2, color="steelblue", rasterized=True, linewidths=0)
        ax.plot([lim[0], lim[1]], [lim[0], lim[1]], "--", color="#888888", linewidth=0.8, zorder=5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(cl, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel(xlabel, fontsize=9)

        ax.text(
            0.05, 0.95,
            f"r = {r:.2f}  r² = {r**2:.2f}\nρ = {rho:.2f}  n = {n:,}",
            transform=ax.transAxes,
            fontsize=7.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8)
        )
        sns.despine(ax=ax)

    axes[0].set_ylabel(ylabel, fontsize=9)
    metric_label = {"psi": "PSI", "dpsi": "ΔPSI", "delta_logit": "Δlogit(PSI)"}[metric]
    fig.suptitle(
        f"Replicate {metric_label} Correlation — Intermediate PSI filter ({psi_low}–{psi_high})",
        fontsize=11, y=1.02
    )

    for ext in ("pdf", "png"):
        out = os.path.join(output_dir, f"replicate_corr_{metric}_{psi_low}_{psi_high}.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()

for psi_range in [(0.1, 0.9), (0.2, 0.8)]:
    for metric in ["psi", "dpsi", "delta_logit"]:
        make_plot(*psi_range, metric)

# === Combined 3-row figure for 0.2–0.8 ===
def make_combined_plot(psi_low, psi_high):
    metrics = [
        ("psi",         "Rep 1 PSI",          "Rep 2 PSI",          (psi_low, psi_high)),
        ("dpsi",        "Rep 1 ΔPSI",          "Rep 2 ΔPSI",         (-1, 1)),
        ("delta_logit", "Rep 1 Δlogit(PSI)",   "Rep 2 Δlogit(PSI)",  (-8, 8)),
    ]

    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    fig.subplots_adjust(hspace=0.35, wspace=0.08)

    for row, (metric, xlabel, ylabel, lim) in enumerate(metrics):
        for col, (cl, p) in enumerate(zip(cell_line_names, cell_prefixes)):
            ax = axes[row, col]
            pooled_col = f"{p}_pooled_psi_raw"

            if metric == "psi":
                rep1_col = f"{p}_rep1_psi_raw"
                rep2_col = f"{p}_rep2_psi_raw"
            elif metric == "dpsi":
                rep1_col = f"{p}_rep1_psi_raw_dpsi"
                rep2_col = f"{p}_rep2_psi_raw_dpsi"
            else:
                rep1_col = f"{p}_rep1_logit_dlogit"
                rep2_col = f"{p}_rep2_logit_dlogit"

            needed = [rep1_col, rep2_col, pooled_col]
            if any(c not in vars_df.columns for c in needed):
                ax.set_visible(False)
                continue

            sub = vars_df[needed].apply(pd.to_numeric, errors="coerce").dropna()
            sub = sub[(sub[pooled_col] >= psi_low) & (sub[pooled_col] <= psi_high)]
            x = sub[rep1_col].values
            y = sub[rep2_col].values
            if len(x) < 2:
                ax.set_visible(False)
                continue

            r, _   = pearsonr(x, y)
            rho, _ = spearmanr(x, y)
            n = len(sub)

            ax.scatter(x, y, s=3, alpha=0.2, color="steelblue", rasterized=True, linewidths=0)
            ax.plot([lim[0], lim[1]], [lim[0], lim[1]], "--", color="#888888", linewidth=0.8, zorder=5)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal")

            # Column titles only on top row
            if row == 0:
                ax.set_title(cl, fontsize=10, fontweight="bold", pad=6)

            # X labels only on bottom row
            if row == 2:
                ax.set_xlabel(xlabel, fontsize=9)
            else:
                ax.set_xlabel("")

            # Y labels only on leftmost column
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)

            ax.text(
                0.05, 0.95,
                f"r = {r:.2f}  r² = {r**2:.2f}\nρ = {rho:.2f}  n = {n:,}",
                transform=ax.transAxes,
                fontsize=7, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8)
            )
            sns.despine(ax=ax)

    fig.suptitle(
        f"Replicate Correlations — Intermediate PSI filter ({psi_low}–{psi_high})",
        fontsize=12, y=1.01
    )

    for ext in ("pdf", "png"):
        out = os.path.join(output_dir, f"replicate_corr_combined_{psi_low}_{psi_high}.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()

make_combined_plot(0.2, 0.8)
