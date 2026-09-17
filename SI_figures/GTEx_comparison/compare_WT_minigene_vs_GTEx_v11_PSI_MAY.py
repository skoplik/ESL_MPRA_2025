"""
compare_WT_minigene_vs_GTEx_v11_PSI.py

Compares MPRA WT minigene PSI (per cell line and mean) vs GTEx v11 pan-tissue PSI.
Identical logic to compare_WT_minigene_vs_GTEx_PSI.py, updated for v11.

Produces:
  - output_GTEx_v11_WT_comparison/Fig_WT_vs_GTEx_v11_panTissue_logit.pdf
  - output_GTEx_v11_WT_comparison/Fig_WT_vs_GTEx_v11_panTissue_logit.png
  - output_GTEx_v11_WT_comparison/Fig_WT_vs_GTEx_v11_panTissue_PSI.pdf
  - output_GTEx_v11_WT_comparison/Fig_WT_vs_GTEx_v11_panTissue_PSI.png
  - output_GTEx_v11_WT_comparison/summary_stats.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--mincov", type=int, default=20)
args = parser.parse_args()
MINCOV = args.mincov

# ── Paths ────────────────────────────────────────────────────────────────────
MPRA_CSV   = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz"  # MAY 2026 reprocessed
OUTPUT_DIR = f"/ESL/Analysis/Native/output_GTEx_v11_WT_comparison_mincov{MINCOV}_MAY"
GTEX_PSI   = "/ESL/Analysis/Native/output_GTEx_v11_WT_comparison_mincov10/GTEx_v11_allTissue_avg_STAR_ref_PSIs.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CELL_LINES = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data ...", flush=True)
mpra = pd.read_csv(MPRA_CSV, low_memory=False)
gtex = pd.read_csv(GTEX_PSI, sep="\t")

wt = mpra[mpra["snp"] == "none"].copy()

wt = wt.merge(
    gtex.rename(columns={"STAR_ref": "Reference", "avg_PSI": "gtex_psi", "logit_PSI": "gtex_logit"}),
    on="Reference",
    how="left"
)
print(f"  WT rows total: {len(wt)}, with GTEx PSI: {wt['gtex_psi'].notna().sum()}", flush=True)

def logit_clip(p, lo=0.01, hi=0.99):
    p = np.clip(p, lo, hi)
    return np.log(p / (1 - p))

logit_cols = [f"{cl}_wt_pooled_logit" for cl in CELL_LINES]
psi_cols   = [f"{cl}_wt_pooled_psi_raw" for cl in CELL_LINES]

wt["mean_wt_logit"] = wt[logit_cols].mean(axis=1)
wt["mean_wt_psi"]   = wt[psi_cols].mean(axis=1)

# ── Compute statistics ───────────────────────────────────────────────────────
def compute_stats(x, y, label):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return {"label": label, "N": n, "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_rho": np.nan, "spearman_p": np.nan}
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {"label": label, "N": n,
            "pearson_r": round(pr, 4), "pearson_p": f"{pp:.2e}",
            "spearman_rho": round(sr, 4), "spearman_p": f"{sp:.2e}"}

all_panels = CELL_LINES + ["Mean"]

# ── Logit-scale stats ─────────────────────────────────────────────────────────
logit_stat_rows  = []
logit_panel_data = {}
for cl in CELL_LINES:
    x_col = f"{cl}_wt_pooled_logit"
    logit_panel_data[cl] = (wt[x_col].values, wt["gtex_logit"].values)
    row = compute_stats(wt[x_col].values, wt["gtex_logit"].values, cl)
    row["scale"] = "logit"
    logit_stat_rows.append(row)

logit_panel_data["Mean"] = (wt["mean_wt_logit"].values, wt["gtex_logit"].values)
row = compute_stats(wt["mean_wt_logit"].values, wt["gtex_logit"].values, "Mean")
row["scale"] = "logit"
logit_stat_rows.append(row)

# ── PSI-scale stats ───────────────────────────────────────────────────────────
psi_stat_rows  = []
psi_panel_data = {}
for cl in CELL_LINES:
    x_col = f"{cl}_wt_pooled_psi_raw"
    psi_panel_data[cl] = (wt[x_col].values, wt["gtex_psi"].values)
    row = compute_stats(wt[x_col].values, wt["gtex_psi"].values, cl)
    row["scale"] = "PSI"
    psi_stat_rows.append(row)

psi_panel_data["Mean"] = (wt["mean_wt_psi"].values, wt["gtex_psi"].values)
row = compute_stats(wt["mean_wt_psi"].values, wt["gtex_psi"].values, "Mean")
row["scale"] = "PSI"
psi_stat_rows.append(row)

# ── Save combined stats ───────────────────────────────────────────────────────
stats_df   = pd.DataFrame(logit_stat_rows + psi_stat_rows)
stats_path = os.path.join(OUTPUT_DIR, "summary_stats.csv")
stats_df.to_csv(stats_path, index=False)
print(f"Saved {stats_path}", flush=True)
print("\n--- Logit scale ---")
print(pd.DataFrame(logit_stat_rows).drop(columns="scale").to_string(index=False))
print("\n--- PSI scale ---")
print(pd.DataFrame(psi_stat_rows).drop(columns="scale").to_string(index=False))

POINT_COLOR = "#4472C4"
LINE_COLOR  = "#C0392B"
ALPHA       = 0.35
POINT_SIZE  = 8

def make_scatter_fig(panel_data, stat_rows, xlabel_fn, ylabel, suptitle):
    stat_lookup = {r["label"]: r for r in stat_rows}
    fig, axes   = plt.subplots(2, 3, figsize=(12, 8))
    axes_flat   = axes.flatten()
    for ax, panel_label in zip(axes_flat, all_panels):
        x_vals, y_vals = panel_data[panel_label]
        mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        xm, ym = x_vals[mask], y_vals[mask]

        ax.scatter(xm, ym, s=POINT_SIZE, alpha=ALPHA, color=POINT_COLOR, linewidths=0)

        row = stat_lookup[panel_label]
        pr, sr, n = row["pearson_r"], row["spearman_rho"], row["N"]
        ax.text(0.05, 0.95,
                f"r = {pr:.2f}\nρ = {sr:.2f}\nn = {n:,}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_xlabel(xlabel_fn(panel_label), fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(panel_label, fontsize=10, fontweight="bold")

    for ax in axes_flat[len(all_panels):]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig

# ── Logit figure ──────────────────────────────────────────────────────────────
fig_logit = make_scatter_fig(
    logit_panel_data,
    logit_stat_rows,
    xlabel_fn=lambda cl: f"Minigene WT logit PSI ({cl})",
    ylabel="GTEx v11 pan-tissue logit PSI",
    suptitle=f"Minigene WT PSI vs GTEx v11 Pan-Tissue PSI (logit scale, mincov={MINCOV})",
)
for ext, dpi in [("pdf", None), ("png", 150)]:
    path = os.path.join(OUTPUT_DIR, f"Fig_WT_vs_GTEx_v11_panTissue_logit.{ext}")
    kw = dict(bbox_inches="tight") if dpi is None else dict(bbox_inches="tight", dpi=dpi)
    fig_logit.savefig(path, **kw)
    print(f"Saved {path}", flush=True)
plt.close(fig_logit)

# ── PSI figure ────────────────────────────────────────────────────────────────
fig_psi = make_scatter_fig(
    psi_panel_data,
    psi_stat_rows,
    xlabel_fn=lambda cl: f"Minigene WT PSI ({cl})",
    ylabel="GTEx v11 pan-tissue PSI",
    suptitle=f"Minigene WT PSI vs GTEx v11 Pan-Tissue PSI (linear scale, mincov={MINCOV})",
)
for ax in fig_psi.axes:
    if ax.get_visible():
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

for ext, dpi in [("pdf", None), ("png", 150)]:
    path = os.path.join(OUTPUT_DIR, f"Fig_WT_vs_GTEx_v11_panTissue_PSI.{ext}")
    kw = dict(bbox_inches="tight") if dpi is None else dict(bbox_inches="tight", dpi=dpi)
    fig_psi.savefig(path, **kw)
    print(f"Saved {path}", flush=True)
plt.close(fig_psi)

print("Done.", flush=True)
