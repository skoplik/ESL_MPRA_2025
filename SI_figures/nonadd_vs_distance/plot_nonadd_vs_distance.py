"""
Analyses whether the distance between two SNVs in a double variant predicts
the probability of non-additive effects on splicing (Δlogit PSI).

Input: full_results TSV from analyze_dnv_snv.py
Output: P(non-additive) vs inter-variant distance plot
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

# === Paths ===
input_tsv  = "/ESL/Figures_SK/compare_snv_dnv_additive/7_13_2025/7_13_2025_full_results.tsv"
output_dir = "/ESL/Figures_SK/compare_snv_dnv_additive/distance_analysis"
os.makedirs(output_dir, exist_ok=True)

THRESHOLD = 1.0   # |deviation| > threshold → non-additive

# === Load ===
df = pd.read_csv(input_tsv, sep="\t")

# === Parse position from "pos:ref>alt" ===
def parse_pos(snp_str):
    try:
        return int(snp_str.strip().split(":")[0])
    except Exception:
        return np.nan

df["pos1"] = df["snp_snv1"].apply(parse_pos)
df["pos2"] = df["snp_snv2"].apply(parse_pos)
df["distance"] = (df["pos2"] - df["pos1"]).abs()
df["non_additive"] = (df["deviation"].abs() > THRESHOLD).astype(int)

df = df.dropna(subset=["distance", "deviation"])
df["distance"] = df["distance"].astype(int)

print(f"Total pairs: {len(df)}")
print(f"Non-additive (|dev|>{THRESHOLD}): {df['non_additive'].sum()} ({100*df['non_additive'].mean():.1f}%)")
print(f"Distance range: {df['distance'].min()} – {df['distance'].max()} nt")

# === Bin by distance ===
bins   = [0, 5, 10, 20, 40, 80, 161]
labels = ["1–5", "6–10", "11–20", "21–40", "41–80", "81–161"]
df["dist_bin"] = pd.cut(df["distance"], bins=bins, labels=labels, right=True)

binned = (
    df.groupby("dist_bin", observed=True)
    .agg(
        n=("non_additive", "count"),
        n_nonadd=("non_additive", "sum"),
        p_nonadd=("non_additive", "mean"),
    )
    .reset_index()
)
binned["se"] = np.sqrt(binned["p_nonadd"] * (1 - binned["p_nonadd"]) / binned["n"])
binned["ci95"] = 1.96 * binned["se"]

print("\nBinned summary:")
print(binned.to_string(index=False))

# === Spearman on raw distance ===
r_p, p_p = pearsonr(df["distance"], df["non_additive"])
r_s, p_s = spearmanr(df["distance"], df["non_additive"])
print(f"\nPearson r = {r_p:.3f} (p={p_p:.2e})")
print(f"Spearman ρ = {r_s:.3f} (p={p_s:.2e})")

# === Plot ===
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.subplots_adjust(wspace=0.35)

# --- Panel A: P(non-additive) per distance bin ---
ax = axes[0]
ax.bar(
    range(len(binned)),
    binned["p_nonadd"],
    yerr=binned["ci95"],
    color="steelblue", alpha=0.8,
    capsize=4, edgecolor="white", linewidth=0.5,
    error_kw=dict(elinewidth=1, ecolor="black")
)
ax.set_xticks(range(len(binned)))
ax.set_xticklabels(
    [f"{l}\n(n={n:,})" for l, n in zip(binned["dist_bin"], binned["n"])],
    fontsize=8
)
ax.set_xlabel("Distance between SNVs (nt)", fontsize=10)
ax.set_ylabel("P(non-additive)", fontsize=10)
ax.set_title("Non-additivity vs. inter-variant distance", fontsize=10)
ax.text(
    0.97, 0.03,
    f"Non-additive: |Δlogit(double) −\n(Δlogit(SNV1) + Δlogit(SNV2))| > {THRESHOLD}",
    transform=ax.transAxes, fontsize=7.5,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=1.0)
)
sns.despine(ax=ax)

# --- Panel B: Distribution of distances (all vs non-additive) ---
ax2 = axes[1]
bins_hist = np.arange(0, 162, 5)
ax2.hist(df["distance"], bins=bins_hist, color="lightgrey",
         edgecolor="none", label="All pairs", density=True, alpha=0.8, rasterized=True)
ax2.hist(df[df["non_additive"] == 1]["distance"], bins=bins_hist,
         color="tomato", edgecolor="none", label="Non-additive", density=True, alpha=0.7, rasterized=True)
ax2.set_xlabel("Distance between SNVs (nt)", fontsize=10)
ax2.set_ylabel("Density", fontsize=10)
ax2.set_title("Distance distribution: all vs. non-additive pairs", fontsize=10)
ax2.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
           frameon=True, framealpha=1.0, facecolor="white", edgecolor="grey")
ax2.text(
    0.97, 0.03,
    f"8.1% of pairs non-additive overall\n(2,243 / 27,840)",
    transform=ax2.transAxes, fontsize=7.5,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=1.0)
)
sns.despine(ax=ax2)

plt.suptitle("Impact of inter-variant distance on non-additive splicing effects", fontsize=11, y=1.02)

for ext in ("pdf", "png"):
    out = os.path.join(output_dir, f"nonadd_vs_distance.{ext}")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close()

# === Save summary table ===
binned.to_csv(os.path.join(output_dir, "nonadd_vs_distance_binned.tsv"), sep="\t", index=False)
