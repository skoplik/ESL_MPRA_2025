#!/usr/bin/env python3
"""SI Figure 1A - combined replicate matrix, PSI upper (blue) / dPSI lower (orange).

Ported 2026-10-14 from /ESL/Figures_SK/supplement/scripts/corrs_si_f1.py, which is
the only copy that draws the combined upper/lower matrix but predates HMC3 and
reads July 2025 data. The repo copy (corrs_si_f1.py) has HMC3 and reads May data
but only emits the two SEPARATE matrices, so neither one produced the 10x10
combined panel in the manuscript. Only the config below differs from the original;
the plotting code is unchanged.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import matplotlib as mpl

# Fonts: keep text editable in Illustrator
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# Inputs
input_file = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz"
output_dir = "/ESL/ESL_MPRA/SI_figures/Correlations_all_cell_lines/outputs/"
os.makedirs(output_dir, exist_ok=True)

# Replicates to include
replicate_cols = {
    "HEK293": ["HEK_rep1", "HEK_rep2"],
    "HeLa": ["HeLa_rep1", "HeLa_rep2"],
    "K562": ["K562_rep1", "K562_rep2"],
    "MCF7": ["MCF7_rep1", "MCF7_rep2"],
    # HMC3 postdates this script; without it the matrix is 8x8, not 10x10.
    "HMC3": ["HMC3_rep1", "HMC3_rep2"],
}

# Load input
df = pd.read_csv(input_file, low_memory=False)

# Helpers
def get_all_replicates(df, suffix):
    reps = []
    for cl, rlist in replicate_cols.items():
        for r in rlist:
            col = f"{r}_{suffix}"
            if col in df.columns:
                reps.append((cl, r, col))
    return reps

def compute_delta(df, suffix):
    wt_rows = df[df["snp"] == "none"]
    wt_by_event = wt_rows.set_index("event_id")
    deltas = {}
    for cl, rlist in replicate_cols.items():
        for r in rlist:
            col = f"{r}_{suffix}"
            if col not in df.columns:
                continue
            v = df[["event_id", col]].copy()
            if col not in wt_by_event.columns:
                print(f"[WARN] Missing WT column: {col}")
                continue
            v["wt"] = v["event_id"].map(wt_by_event[col])
            v["delta"] = v[col] - v["wt"]
            deltas[f"{cl} {r}"] = v["delta"]
    out_df = pd.DataFrame(deltas)
    out_df.index = df["event_id"]
    return out_df

def format_label(colname):
    for cl in replicate_cols:
        if colname.startswith(cl):
            for rep in replicate_cols[cl]:
                if rep in colname:
                    repnum = rep.split('_')[-1].replace('rep', 'Rep')
                    return f"{cl} {repnum}"
    return colname

def plot_upper_lower(psi_data, dpsi_data, title, filename):
    reps = psi_data.columns.tolist()
    n = len(reps)
    if n < 2:
        print("[WARN] Not enough replicate columns to plot")
        return

    fig, axes = plt.subplots(n, n, figsize=(2.2*n, 2.2*n), constrained_layout=False)
    if n == 1:
        axes = np.array([[axes]])

    # Color-blind safe palette
    color_upper = "#0072B2"  # dark blue for PSI
    color_lower = "#E69F00"  # orange for dPSI

    psi_lim = (0, 1)
    psi_ticks = [0, 0.5, 1]
    dpsi_lim = (-1, 1)
    dpsi_ticks = [-1, 0, 1]

    # Draw panels
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)

            if i == j:
                ax.annotate(format_label(reps[i]), xy=(0.5, 0.5), xycoords="axes fraction",
                            ha="center", va="center", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            if j > i:
                # Upper triangle: PSI
                x = psi_data[reps[j]]
                y = psi_data[reps[i]]
                lim = psi_lim
                ticks = psi_ticks
                color = color_upper
                is_psi = True
            else:
                # Lower triangle: dPSI
                x = dpsi_data[reps[j]]
                y = dpsi_data[reps[i]]
                lim = dpsi_lim
                ticks = dpsi_ticks
                color = color_lower
                is_psi = False

            valid = x.notna() & y.notna()
            x = x[valid]
            y = y[valid]
            if len(x) < 2:
                ax.annotate("n < 2", xy=(0.5, 0.5), xycoords="axes fraction",
                            ha="center", va="center", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            ax.scatter(x, y, s=3, alpha=0.08, color=color, rasterized=True)
            r, _ = pearsonr(x, y)
            ax.text(0.04, 0.96, f"r={r:.2f}", transform=ax.transAxes,
                    fontsize=10, va="top", ha="left")
            ax.text(0.04, 0.82, f"n={len(x)}", transform=ax.transAxes,
                    fontsize=9, va="top", ha="left")

            ax.set_xlim(*lim)
            ax.set_ylim(*lim)

            # Edge tick logic
            # PSI edges should be top and right only
            # dPSI edges should be bottom and left only
            if is_psi:
                # Top edge for PSI
                if i == 0 and j > i:
                    ax.set_xticks(psi_ticks)
                    ax.set_xticklabels([str(t) for t in psi_ticks], fontsize=9)
                else:
                    ax.set_xticks([])
                # Right edge for PSI
                if j == n - 1 and i < j:
                    ax.set_yticks(psi_ticks)
                    ax.set_yticklabels([str(t) for t in psi_ticks], fontsize=9)
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(format_label(reps[i]), fontsize=10, rotation=270, labelpad=8)
                else:
                    # Do not show left ticks for PSI
                    if j != n - 1:
                        ax.set_yticks([])
            else:
                # dPSI on bottom edge
                if i == n - 1 and j < i:
                    ax.set_xticks(dpsi_ticks)
                    ax.set_xticklabels([str(t) for t in dpsi_ticks], fontsize=9)
                    ax.set_xlabel(format_label(reps[j]), fontsize=10)
                else:
                    ax.set_xticks([])
                # dPSI on left edge
                if j == 0 and i > j:
                    ax.set_yticks(dpsi_ticks)
                    ax.set_yticklabels([str(t) for t in dpsi_ticks], fontsize=9)
                    ax.set_ylabel(format_label(reps[i]), fontsize=10, rotation=90)
                else:
                    # Do not show right ticks for dPSI
                    if j != 0:
                        ax.set_yticks([])

    # Global axis titles
    fig.text(0.5, 0.985, "PSI (0 to 1)", ha="center", va="top", fontsize=14)
    fig.text(0.985, 0.5, "PSI (0 to 1)", ha="right", va="center", rotation=270, fontsize=14)
    fig.text(0.5, 0.015, "ΔPSI (-1 to +1)", ha="center", va="bottom", fontsize=14)
    fig.text(0.015, 0.5, "ΔPSI (-1 to +1)", ha="left", va="center", rotation=90, fontsize=14)

    # Title and layout
    fig.suptitle(title, fontsize=16, y=0.96)
    # Make room so nothing is cut off
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08, wspace=0.08, hspace=0.08)

    out_base = os.path.join(output_dir, filename.replace(".pdf", ""))
    fig.savefig(f"{out_base}.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# PSI replicate matrix
psi_meta = get_all_replicates(df, "psi_raw")
psi_cols = [x[2] for x in psi_meta]
psi_df = df[psi_cols].copy()
psi_df.columns = [f"{x[0]} {x[1]}" for x in psi_meta]

# dPSI matrix
dpsi_df = compute_delta(df, "psi_raw")
dpsi_df = dpsi_df[[col for col in dpsi_df.columns if col in psi_df.columns]]

# Debug
print("[DEBUG] dpsi_df shape:", dpsi_df.shape)
print(dpsi_df.notna().sum())

# Combined PSI upper and dPSI lower plot
plot_upper_lower(
    psi_df,
    dpsi_df,
    "Replicate vs Replicate PSI (upper, blue) and ΔPSI (lower, orange)",
    "supplement_fig1_PSI_upper_dPSI_lower.pdf"
)
