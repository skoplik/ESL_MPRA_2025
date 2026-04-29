#!/usr/bin/env python3
"""
Characterize which variants benefit from MMSplice retraining.
Uses held-out test set (n=3,526) only.

Outputs:
  plots/MMSplice/improvement_by_position_region.pdf/png
  plots/MMSplice/baseline_vs_retrained_error_scatter.pdf/png
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

MEGA_FILE = "/ESL/ESL_MPRA/Figure_2/model_comparison/mega_pred_file_filtered.csv"
DATA_PATH = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz"
OUTDIR = "/ESL/ESL_MPRA/Figure_2/plots/MMSplice"
os.makedirs(OUTDIR, exist_ok=True)


# ── Load & join ────────────────────────────────────────────────────────────────
print("Loading data...")
mega = pd.read_csv(MEGA_FILE, low_memory=False)
test = mega[mega["mmsplice_is_test"] == True].copy()
print(f"  Test set: {len(test):,} variants")

seqs = pd.read_csv(DATA_PATH, low_memory=False, usecols=["Reference", "intron1", "exon", "intron2"])
seqs["intron1_len"] = seqs["intron1"].str.len()
seqs["exon_len"]    = seqs["exon"].str.len()
seqs["intron2_len"] = seqs["intron2"].str.len()
test = test.merge(seqs[["Reference", "intron1_len", "exon_len", "intron2_len"]],
                  on="Reference", how="left")

missing_seq = test["intron1_len"].isna().sum()
if missing_seq:
    print(f"  Warning: {missing_seq} test variants missing sequence data after join")


# ── Compute per-variant metrics ────────────────────────────────────────────────
# Variant position within construct (0-indexed from construct start)
test["var_pos"] = test["snp"].str.extract(r"^(\d+)").astype(float)

# Region
test["region"] = "intron1"
test.loc[test["var_pos"] >= test["intron1_len"], "region"] = "exon"
test.loc[test["var_pos"] >= test["intron1_len"] + test["exon_len"], "region"] = "intron2"

# Distance from nearest splice site (exon boundary)
test["pos_rel_5ss"] = test["var_pos"] - test["intron1_len"]          # neg = intronic 5' side
test["pos_rel_3ss"] = test["var_pos"] - (test["intron1_len"] + test["exon_len"])  # neg = exonic
test["dist_nearest_ss"] = test[["pos_rel_5ss", "pos_rel_3ss"]].abs().min(axis=1)

# Position bins
def pos_bin(d):
    if d <= 4:   return "0–4 nt\n(core)"
    if d <= 20:  return "5–20 nt\n(proximal)"
    if d <= 80:  return "21–80 nt\n(intermediate)"
    return "81+ nt\n(distal)"

test["pos_bin"] = test["dist_nearest_ss"].apply(pos_bin)
POS_ORDER = ["0–4 nt\n(core)", "5–20 nt\n(proximal)", "21–80 nt\n(intermediate)", "81+ nt\n(distal)"]

# Reference PSI bins (use HEK)
psi = test["HEK_wt_pooled_psi_raw"]
test["psi_bin"] = pd.cut(psi, bins=[0, 0.2, 0.8, 1.0],
                          labels=["Near-excluded\n(0–0.2)", "Intermediate\n(0.2–0.8)", "Near-included\n(0.8–1.0)"],
                          include_lowest=True)

# Drop rows missing predictions
test = test.dropna(subset=["baseline_mmsplice_delta_logit", "retrained_mmsplice_delta_logit",
                             "avg_delta_logit_pooled"])
print(f"  After dropping missing predictions: {len(test):,}")

# Error and improvement
test["err_baseline"]  = (test["baseline_mmsplice_delta_logit"]  - test["avg_delta_logit_pooled"]).abs()
test["err_retrained"] = (test["retrained_mmsplice_delta_logit"] - test["avg_delta_logit_pooled"]).abs()
test["improvement"]   = test["err_baseline"] - test["err_retrained"]  # positive = retrained better

test["outcome"] = "unchanged"
test.loc[test["improvement"] >  0.05, "outcome"] = "improved"
test.loc[test["improvement"] < -0.05, "outcome"] = "degraded"


# ── Summary stats ──────────────────────────────────────────────────────────────
print("\n── Outcome summary ──")
print(test["outcome"].value_counts())
print(f"\nOverall mean improvement: {test['improvement'].mean():.4f}")

print("\n── Mean improvement by region ──")
print(test.groupby("region")["improvement"].agg(["mean","sem","count"]).round(4))

print("\n── Mean improvement by position bin ──")
print(test.groupby("pos_bin")["improvement"].agg(["mean","sem","count"]).round(4))

print("\n── Mean improvement by source ──")
print(test.groupby("source")["improvement"].agg(["mean","sem","count"]).round(4))


# ── Plot A: Bar chart — mean improvement by position bin × region ─────────────
print("\n── Plot A: improvement by position bin × region ──")

REGION_COLOR = {"intron1": "#e07b39", "exon": "#377EB8", "intron2": "#4daf4a"}
regions = ["intron1", "exon", "intron2"]
n_regions = len(regions)
n_bins    = len(POS_ORDER)
bar_width = 0.22
x = np.arange(n_bins)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

for i, region in enumerate(regions):
    sub = test[test["region"] == region]
    grp = sub.groupby("pos_bin")["improvement"]
    means = [grp.get_group(b).mean() if b in grp.groups else np.nan for b in POS_ORDER]
    sems  = [grp.get_group(b).sem()  if b in grp.groups else np.nan for b in POS_ORDER]
    ns    = [grp.get_group(b).count() if b in grp.groups else 0      for b in POS_ORDER]

    offset = (i - n_regions / 2 + 0.5) * bar_width
    bars = ax.bar(x + offset, means, bar_width,
                  color=REGION_COLOR[region], edgecolor="black", linewidth=0.5,
                  label=region, yerr=sems, capsize=3, error_kw={"linewidth": 0.8})
    for xi, (m, n) in enumerate(zip(means, ns)):
        if n > 0 and not np.isnan(m):
            ax.text(x[xi] + offset, max(m, 0) + 0.005, f"n={n}",
                    ha="center", va="bottom", fontsize=5.5, rotation=90)

ax.axhline(0, color="black", linewidth=0.7)
ax.set_xticks(x)
ax.set_xticklabels(POS_ORDER, fontsize=9)
ax.set_ylabel("Mean improvement in absolute error\n(baseline − retrained, Δlogit units)", fontsize=9)
ax.set_title("MMSplice retraining benefit by position and region\n(held-out test set, n=3,526)", fontsize=10)
ax.legend(title="Region", frameon=False, fontsize=8)
plt.tight_layout()

for ext in ("pdf", "png"):
    path = os.path.join(OUTDIR, f"improvement_by_position_region.{ext}")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {path}")
plt.close(fig)


# ── Plot B: Scatter — baseline error vs. retrained error, colored by PSI bin ──
print("\n── Plot B: baseline vs retrained error scatter ──")

PSI_COLOR = {
    "Near-excluded\n(0–0.2)":   "#e41a1c",
    "Intermediate\n(0.2–0.8)":  "#984ea3",
    "Near-included\n(0.8–1.0)": "#377EB8",
}
PSI_LABEL = {k: k.replace("\n", " ") for k in PSI_COLOR}

fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

lim = max(test["err_baseline"].max(), test["err_retrained"].max()) * 1.05

for psi_bin, color in PSI_COLOR.items():
    sub = test[test["psi_bin"] == psi_bin]
    ax.scatter(sub["err_baseline"], sub["err_retrained"],
               s=6, color=color, alpha=0.15, linewidths=0,
               rasterized=True, label=PSI_LABEL[psi_bin])

ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=1, zorder=5)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Baseline MMSplice |error| (Δlogit)", fontsize=10)
ax.set_ylabel("Retrained MMSplice |error| (Δlogit)", fontsize=10)
ax.set_title("Baseline vs. retrained prediction error\n(held-out test set; below diagonal = retrained better)", fontsize=9)

frac_improved = (test["outcome"] == "improved").mean()
frac_degraded = (test["outcome"] == "degraded").mean()
ax.text(0.97, 0.05,
        f"Improved: {frac_improved:.1%}\nDegraded: {frac_degraded:.1%}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

ax.legend(title="Reference PSI (HEK)", frameon=False, fontsize=8, loc="upper left")
plt.tight_layout()

for ext in ("pdf", "png"):
    path = os.path.join(OUTDIR, f"baseline_vs_retrained_error_scatter.{ext}")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {path}")
plt.close(fig)

print("\nDone.")
