#!/usr/bin/env python3
"""Figure S4C: COMPASS reference/variant PSI vs MFASS SMN1 index for shared sequences.

Rebuild of the Sep 2026 mfass_vs_compass_FIXED figure, whose generating script was
lost. Reproduces its published numbers exactly (AllCells n=157, r=0.687; HEK293
n=153, r=0.713; HeLa 98/0.669; K562 133/0.692; MCF7 130/0.665; HMC3 148/0.642).

Inputs
  * MFASS overlap files (supertable_seq <-> MFASS sequence + MFASS index), one per
    sample, written by find_supertable_overlap_MFASS.py into outputs/reprocessed_runs.
  * May 2026 PSI: 1e-2_ALL_WITH_WT.csv.gz.

COMPASS PSI is looked up by the matched SEQUENCE, using the highest-coverage row per
sequence, never by Reference id: a Reference join misaligns mutant constructs and
crosses the 0-/1-index boundary, which once produced a spurious r~0.37.
For "All cells (avg)" the PSI is the mean of the five cell lines' pooled PSI.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

REPO = "/ESL/ESL_MPRA"
PSI_FILE = f"{REPO}/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz"
OVERLAP = f"{REPO}/SI_figures/MFASS_overlap/outputs/reprocessed_runs/MFASS_highFilter_%s_reprocessed_MFASS_overlap_PSI_Rep1.txt"
OUT_DIR = f"{REPO}/SI_figures/MFASS_overlap/outputs/S4C_MAY"
ALL5 = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]
SAMPLES = [("AllCells_avg", "All cells (avg)", ALL5), ("HEK293_avg", "HEK293", ["HEK"]),
           ("HeLa_avg", "HeLa", ["HeLa"]), ("K562_avg", "K562", ["K562"]),
           ("MCF7_avg", "MCF7", ["MCF7"]), ("HMC3_avg", "HMC3", ["HMC3"])]

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

cols = ["full_seq"] + [f"{c}_pooled_psi_raw" for c in ALL5] + [f"{c}_total_pooled" for c in ALL5]
m = pd.read_csv(PSI_FILE, usecols=cols, low_memory=False)
m["cov"] = m[[f"{c}_total_pooled" for c in ALL5]].sum(axis=1)
canon = m.sort_values("cov", ascending=False).drop_duplicates("full_seq").set_index("full_seq")

os.makedirs(OUT_DIR, exist_ok=True)
fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
rows, points = [], []
for ax, (key, title, cells) in zip(axes.flat, SAMPLES):
    ov = pd.read_csv(OVERLAP % key, sep="\t")
    psi = canon[[f"{c}_pooled_psi_raw" for c in cells]].mean(axis=1, skipna=True)
    d = ov.assign(compass=ov.supertable_seq.map(psi)).dropna(subset=["compass", "MFASS_index"])
    r, _ = stats.pearsonr(d.compass, d.MFASS_index)
    rho, _ = stats.spearmanr(d.compass, d.MFASS_index)
    rows.append({"sample": key, "n": len(d), "pearson_r": r, "r2": r * r, "spearman": rho})
    points.append(d.assign(sample=key)[["sample", "supertable_seq", "MFASS_seq", "compass", "MFASS_index"]])
    ax.scatter(d.compass, d.MFASS_index, s=14, alpha=0.6, color="#3b6ea5", edgecolor="none")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(f"{title} PSI (COMPASS)"); ax.set_ylabel("MFASS SMN1 index")
    ax.set_title(f"{title}\nr={r:.2f}, r²={r*r:.2f}, ρ={rho:.2f}, n={len(d)}", fontsize=10)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT_DIR}/mfass_vs_compass_MAY.{ext}", dpi=300, bbox_inches="tight")
summary = pd.DataFrame(rows)
summary.to_csv(f"{OUT_DIR}/mfass_correlation_summary_MAY.tsv", sep="\t", index=False)
pd.concat(points).to_csv(f"{OUT_DIR}/mfass_vs_compass_points_MAY.tsv", sep="\t", index=False)
print(summary.round(4).to_string(index=False))
