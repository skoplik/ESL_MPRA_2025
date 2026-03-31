"""
SDV Threshold Sensitivity Analysis for COMPASS MPRA
====================================================
Empirical justification for the |Δlogit(PSI)| >= 1 SDV threshold via:
  1. Null distribution: Δlogit for intronic variants >50 bp from any splice site
     (pooled across all 5 cell lines)
  2. ROC curve: extended canonical SS variants (WT PSI >= 0.5) as positives vs.
     deep intronic variants (WT PSI >= 0.5) as negatives

Output: /ESL/Analysis/sensitivity_analysis_SDVs/output_03_29_2026/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_PATH = (
    "/ESL/Figures_SK/General_preprocessing/output_03_16_2026/"
    "03_16_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"
)
OUT_DIR   = "/ESL/Analysis/sensitivity_analysis_SDVs/output_03_29_2026"
os.makedirs(OUT_DIR, exist_ok=True)

CELL_LINES = ["HeLa", "K562", "MCF7", "HMC3", "HEK"]
COV_MIN    = 20
THRESHOLD  = 1.0

delta_cols = {c: f"{c}_delta_logit_pooled" for c in CELL_LINES}
cov_cols   = {c: f"{c}_total_pooled"       for c in CELL_LINES}
wt_cols    = {c: f"{c}_wt_pooled_psi_raw"  for c in CELL_LINES}

DPI = 300

# ---------------------------------------------------------------------------
# Load and classify variants
# ---------------------------------------------------------------------------
print("Loading data...")
df     = pd.read_csv(DATA_PATH, low_memory=False)
single = df[(df["snp"] != "none") & (~df["snp"].str.contains(";", na=False))].copy()
print(f"  Single variants: {len(single)}")

single["intron1_len"] = single["intron1"].str.len()
single["exon_len"]    = single["exon"].str.len()
single["snp_pos"]     = single["snp"].apply(lambda s: int(s.split(":")[0]))


def classify_variant(row):
    p, il, el = row["snp_pos"], row["intron1_len"], row["exon_len"]
    if p < il and (il - 1 - p) > 50:  return "intronic_distal"
    return "other"


def is_extended_ss(row):
    """Acceptor: intronic side il-4 to il-1 (-4 to -1);
       Donor:    both sides il+el-2 to il+el+3 (-2 exonic, +4 intronic)."""
    p, il, el = row["snp_pos"], row["intron1_len"], row["exon_len"]
    if p in {il - 4, il - 3, il - 2, il - 1}:  return True  # acceptor: -4 to -1
    if il + el - 2 <= p <= il + el + 3:          return True  # donor: -2 to +4
    return False


single["variant_class"] = single.apply(classify_variant, axis=1)
single["is_ext_ss"]     = single.apply(is_extended_ss, axis=1)
print("  Intronic distal:", (single["variant_class"] == "intronic_distal").sum())
print("  Extended SS:    ", single["is_ext_ss"].sum())

# ---------------------------------------------------------------------------
# Fig 1: Null distribution — deep intronic variants, WT PSI >= 0.5, pooled
# ---------------------------------------------------------------------------
print("\nFig 1: Null distribution (WT PSI >= 0.5)...")

null_rows = single[single["variant_class"] == "intronic_distal"]

null_vals = []
for cell in CELL_LINES:
    dcol, ccol, wcol = delta_cols[cell], cov_cols[cell], wt_cols[cell]
    sub = null_rows[
        (null_rows[ccol] >= COV_MIN) &
        null_rows[dcol].notna() &
        null_rows[wcol].notna() &
        (null_rows[wcol] >= 0.5)
    ][dcol]
    null_vals.append(sub.values)

null_pooled = np.concatenate(null_vals)
abs_null    = np.abs(null_pooled)

n_null    = len(null_pooled)
p95       = np.percentile(abs_null, 95)
p99       = np.percentile(abs_null, 99)
fpr_at_1  = float((abs_null >= THRESHOLD).mean()) * 100
pct_below = 100 - fpr_at_1

print(f"  n={n_null}, p95={p95:.3f}, p99={p99:.3f}, "
      f"pct(|Δlogit|<1)={pct_below:.1f}%, FPR@1={fpr_at_1:.1f}%")

null_df = pd.DataFrame([{
    "n_pooled": n_null, "p95": round(p95, 4), "p99": round(p99, 4),
    "pct_below_1": round(pct_below, 2), "fpr_at_1_pct": round(fpr_at_1, 2)
}])
null_df.to_csv(os.path.join(OUT_DIR, "null_distribution_summary.csv"), index=False)

fig1, ax1 = plt.subplots(figsize=(6, 4))
kde = stats.gaussian_kde(null_pooled, bw_method="scott")
x   = np.linspace(min(null_pooled.min(), -5), max(null_pooled.max(), 5), 600)
ax1.fill_between(x, kde(x), alpha=0.35, color="#4878CF")
ax1.plot(x, kde(x), color="#4878CF", lw=1.8)
ax1.axvline( THRESHOLD, color="black", lw=1.5, ls="--", zorder=3)
ax1.axvline(-THRESHOLD, color="black", lw=1.5, ls="--", zorder=3)
ax1.text(0.97, 0.97,
         f"n = {n_null:,}\np95 = {p95:.2f}\np99 = {p99:.2f}\n"
         f"{pct_below:.1f}% below |Δlogit| = 1",
         transform=ax1.transAxes, ha="right", va="top", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))
ax1.set_xlabel("Δlogit(PSI)", fontsize=11)
ax1.set_ylabel("Density", fontsize=11)
ax1.set_title(
    "Δlogit(PSI) — intronic variants >50 bp from splice site, WT PSI ≥ 0.5\n"
    "(pooled across HeLa, K562, MCF7, HMC3, HEK)",
    fontsize=10, fontweight="bold"
)
ax1.spines[["top", "right"]].set_visible(False)
fig1.tight_layout()
fig1_path = os.path.join(OUT_DIR, "fig1_null_distribution.png")
fig1.savefig(fig1_path, bbox_inches="tight", dpi=DPI)
plt.close(fig1)
print(f"  Saved: {fig1_path}")

# ---------------------------------------------------------------------------
# Fig 2: ROC — extended canonical SS + WT PSI>=0.5 (pos)
#             vs. deep intronic + WT PSI>=0.5 (neg)
# ---------------------------------------------------------------------------
print("\nFig 2: ROC curve...")

pool_ss  = single[single["is_ext_ss"]]
pool_neg = single[single["variant_class"] == "intronic_distal"]

yt, ys = [], []
for cell in CELL_LINES:
    dcol, ccol, wcol = delta_cols[cell], cov_cols[cell], wt_cols[cell]
    ps = pool_ss[
        (pool_ss[ccol] >= COV_MIN) &
        pool_ss[dcol].notna() &
        pool_ss[wcol].notna() &
        (pool_ss[wcol] >= 0.5)
    ]
    ns = pool_neg[
        (pool_neg[ccol] >= COV_MIN) &
        pool_neg[dcol].notna() &
        pool_neg[wcol].notna() &
        (pool_neg[wcol] >= 0.5)
    ]
    yt += [np.ones(len(ps)),  np.zeros(len(ns))]
    ys += [ps[dcol].abs().values, ns[dcol].abs().values]

y_true  = np.concatenate(yt)
y_score = np.concatenate(ys)
n_pos, n_neg = int(y_true.sum()), int((y_true == 0).sum())

fpr, tpr, thr_roc = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)

youden_j = tpr - fpr
opt_idx  = np.argmax(youden_j[1:]) + 1
opt_thr  = thr_roc[opt_idx]
opt_sens = tpr[opt_idx]
opt_spec = 1 - fpr[opt_idx]

print(f"  n_pos={n_pos}, n_neg={n_neg}, AUC={auc:.3f}")
print(f"  Youden-optimal: {opt_thr:.3f}  sens={opt_sens:.3f}, spec={opt_spec:.3f}")

fixed_thresholds = {}
for tv in [0.5, 1.0, 1.5]:
    idx = np.argmin(np.abs(thr_roc - tv))
    fixed_thresholds[tv] = {"fpr": fpr[idx], "tpr": tpr[idx],
                             "sens": tpr[idx], "spec": 1 - fpr[idx]}
    print(f"  |Δlogit|={tv}: sens={tpr[idx]:.3f}, spec={1-fpr[idx]:.3f}")

# Save outputs
pd.DataFrame({
    "threshold": thr_roc[1:], "sensitivity": tpr[1:],
    "specificity": 1 - fpr[1:], "youden_j": youden_j[1:],
}).to_csv(os.path.join(OUT_DIR, "roc_sweep.csv"), index=False)

pd.DataFrame([{
    "n_pos": n_pos, "n_neg": n_neg, "AUC": round(auc, 4),
    "youden_optimal_threshold": round(opt_thr, 4),
    "youden_optimal_sens": round(opt_sens, 4),
    "youden_optimal_spec": round(opt_spec, 4),
    **{f"sensitivity_at_{tv}": round(m["sens"], 4) for tv, m in fixed_thresholds.items()},
    **{f"specificity_at_{tv}": round(m["spec"], 4) for tv, m in fixed_thresholds.items()},
}]).to_csv(os.path.join(OUT_DIR, "roc_summary.csv"), index=False)

# Plot
thr_colors = {0.5: "#2196F3", 1.0: "#E53935", 1.5: "#FF9800"}
thr_shapes = {0.5: "^",       1.0: "o",       1.5: "s"}

fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.plot(fpr, tpr, color="#333333", lw=2)
ax2.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
ax2.scatter([fpr[opt_idx]], [tpr[opt_idx]], s=120, color="#7B1FA2", zorder=6, marker="D",
            label=f"Youden opt: {opt_thr:.2f}  sens={opt_sens:.2f}, spec={opt_spec:.2f}")
for tv, m in fixed_thresholds.items():
    ax2.scatter([m["fpr"]], [m["tpr"]], s=90, color=thr_colors[tv],
                marker=thr_shapes[tv], zorder=5,
                label=f"|Δlogit| = {tv}  sens={m['sens']:.2f}, spec={m['spec']:.2f}")

ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
ax2.set_xlabel("1 − Specificity (FPR)", fontsize=11)
ax2.set_ylabel("Sensitivity (TPR)", fontsize=11)
ax2.set_title(
    "ROC: canonical SS variants (WT PSI ≥ 0.5)\nvs. deep intronic variants (WT PSI ≥ 0.5)",
    fontsize=10, fontweight="bold"
)
ax2.text(0.97, 0.97,
         f"Positives: {n_pos} extended SS\nNegatives: {n_neg} intronic distal",
         transform=ax2.transAxes, ha="right", va="top", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

legend_handles = [Line2D([0], [0], color="#333333", lw=2, label=f"AUC = {auc:.3f}")] + [
    plt.scatter([], [], s=120, color="#7B1FA2", marker="D",
                label=f"Youden opt: {opt_thr:.2f}  sens={opt_sens:.2f}, spec={opt_spec:.2f}"),
] + [
    plt.scatter([], [], s=90, color=thr_colors[tv], marker=thr_shapes[tv],
                label=f"|Δlogit| = {tv}  sens={m['sens']:.2f}, spec={m['spec']:.2f}")
    for tv, m in fixed_thresholds.items()
]
ax2.legend(handles=legend_handles, fontsize=8, loc="lower right",
           handletextpad=0.4, borderpad=0.6)
ax2.spines[["top", "right"]].set_visible(False)
fig2.tight_layout()
fig2_path = os.path.join(OUT_DIR, "fig2_roc.png")
fig2.savefig(fig2_path, bbox_inches="tight", dpi=DPI)
plt.close(fig2)
print(f"  Saved: {fig2_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\nNull distribution (intronic >50 bp from SS, WT PSI >= 0.5, pooled):")
print(null_df.to_string(index=False))
print("\nROC (extended canonical SS vs. deep intronic, WT PSI >= 0.5):")
print(pd.read_csv(os.path.join(OUT_DIR, "roc_summary.csv")).to_string(index=False))
print(f"\nOutputs saved to: {OUT_DIR}")
