"""
SDV Threshold Justification — Extended Analysis v7
====================================================
Produces three supplementary figures:

  Fig A  FDR vs. |Δlogit| threshold curve
         Null = intronic variants >=75 bp from BOTH splice sites,
                WT PSI in [0, 0.1] or [0.9, 1] in ALL covered cell lines

  Fig B  Replicate-level noise
         Panel 1: violin of sd_delta_logit per cell line
         Panel 2: scatter of avg rep1 Δlogit vs avg rep2 Δlogit per variant
                  (averaged across cell lines — one point per variant)

  Fig C  ROC curve
         Positives: extended canonical SS variants (WT PSI >= 0.5)
         Negatives: intronic >=50 bp from SS, WT PSI ∈ [0,0.1]∪[0.9,1] all cells
         Thresholds marked at 0.5, 1.0, 1.5, 2.0

Output: /ESL/Analysis/sensitivity_analysis_SDVs/output_04_01_2026/
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
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_PATH = (
    "/ESL/Figures_SK/General_preprocessing/output_03_16_2026/"
    "03_16_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"
)
OUT_DIR    = "/ESL/Analysis/sensitivity_analysis_SDVs/output_04_01_2026"
os.makedirs(OUT_DIR, exist_ok=True)

CELL_LINES  = ["HeLa", "K562", "MCF7", "HMC3", "HEK"]
COV_MIN     = 20
THRESHOLDS  = [0.5, 1.0, 1.5, 2.0]
DPI         = 300

delta_cols    = {c: f"{c}_delta_logit_pooled" for c in CELL_LINES}
cov_cols      = {c: f"{c}_total_pooled"        for c in CELL_LINES}
wt_cols       = {c: f"{c}_wt_pooled_psi_raw"   for c in CELL_LINES}
sd_cols       = {c: f"{c}_sd_delta_logit"      for c in CELL_LINES}
wt_logit_cols = {c: f"{c}_wt_pooled_logit"     for c in CELL_LINES}
REP1          = {c: f"{c}_rep1_logit"           for c in CELL_LINES}
REP2          = {c: f"{c}_rep2_logit"           for c in CELL_LINES}

# ---------------------------------------------------------------------------
# Load & classify variants
# ---------------------------------------------------------------------------
print("Loading data...")
df     = pd.read_csv(DATA_PATH, low_memory=False)
single = df[(df["snp"] != "none") & (~df["snp"].str.contains(";", na=False))].copy()
print(f"  Single variants: {len(single):,}")

single["intron1_len"] = single["intron1"].str.len()
single["exon_len"]    = single["exon"].str.len()
single["snp_pos"]     = single["snp"].apply(lambda s: int(s.split(":")[0]))


PSI_LOW   = 0.1   # near-floor upper bound
PSI_HIGH  = 0.9   # near-ceiling lower bound
DISTAL_BP = 75    # minimum distance from either splice site


def is_intronic_distal(row):
    """Intronic and >= DISTAL_BP nt from both splice sites.
    intron1: distance from acceptor = il-1-p >= DISTAL_BP
    intron2: distance from donor    = p-(il+el) >= DISTAL_BP
    """
    p, il, el = row["snp_pos"], row["intron1_len"], row["exon_len"]
    if (p < il) and ((il - 1 - p) >= DISTAL_BP):
        return True
    if (p >= il + el) and ((p - (il + el)) >= DISTAL_BP):
        return True
    return False


def has_wt_psi_near_binary_all_cells(row):
    """WT PSI is in [0, PSI_LOW] or [PSI_HIGH, 1] in every cell line with
    coverage >= COV_MIN. Returns False if no cell line has sufficient coverage."""
    any_covered = False
    for cell in CELL_LINES:
        cov = row[cov_cols[cell]]
        wt  = row[wt_cols[cell]]
        if pd.notna(cov) and cov >= COV_MIN:
            any_covered = True
            if pd.isna(wt) or (PSI_LOW < wt < PSI_HIGH):
                return False
    return any_covered


def is_extended_ss(row):
    """Acceptor: il-4 to il-1; Donor: il+el-2 to il+el+3."""
    p, il, el = row["snp_pos"], row["intron1_len"], row["exon_len"]
    if p in {il - 4, il - 3, il - 2, il - 1}:
        return True
    if il + el - 2 <= p <= il + el + 3:
        return True
    return False


single["intronic_distal"]    = single.apply(is_intronic_distal,              axis=1)
single["wt_psi_near_binary"] = single.apply(has_wt_psi_near_binary_all_cells, axis=1)
single["is_ext_ss"]          = single.apply(is_extended_ss,                   axis=1)

# Null: intronic >=50 bp from both SS AND near-binary PSI in all cells
null_mask = single["intronic_distal"] & single["wt_psi_near_binary"]

print(f"  Intronic (>={DISTAL_BP} bp from both SS):              {single['intronic_distal'].sum():,}")
print(f"  WT PSI in [0,{PSI_LOW}]∪[{PSI_HIGH},1] all cells:     {single['wt_psi_near_binary'].sum():,}")
print(f"  Extended SS:                                  {single['is_ext_ss'].sum():,}")
print(f"  Null (intronic distal + near-binary PSI):     {null_mask.sum():,}")

# ---------------------------------------------------------------------------
# Helper: pool observations across cell lines
# ---------------------------------------------------------------------------
def pool_abs_delta(rows_df, require_wt_psi_gte=None):
    vals = []
    for cell in CELL_LINES:
        dcol, ccol, wcol = delta_cols[cell], cov_cols[cell], wt_cols[cell]
        mask = (rows_df[ccol] >= COV_MIN) & rows_df[dcol].notna()
        if require_wt_psi_gte is not None:
            mask = mask & rows_df[wcol].notna() & (rows_df[wcol] >= require_wt_psi_gte)
        vals.append(np.abs(rows_df.loc[mask, dcol].values))
    return np.concatenate(vals)


null_rows = single[null_mask]
ss_rows   = single[single["is_ext_ss"]]

# Null pre-filtered at row level — no additional PSI filter needed
null_pool = pool_abs_delta(null_rows, require_wt_psi_gte=None)
ss_pool   = pool_abs_delta(ss_rows,   require_wt_psi_gte=0.5)

NULL_LABEL = f"intronic ≥{DISTAL_BP} bp from SS, WT PSI ∈ [0,{PSI_LOW}]∪[{PSI_HIGH},1] all cells"

print(f"\nNull ({NULL_LABEL}): n={len(null_pool):,}")
print(f"Canonical SS (WT PSI>=0.5):                         n={len(ss_pool):,}")
print(f"Null p95={np.percentile(null_pool,95):.3f}, p99={np.percentile(null_pool,99):.3f}")

# ---------------------------------------------------------------------------
# Pre-compute Youden-optimal threshold from ROC (used in all figures)
# ---------------------------------------------------------------------------
ss_abs_roc = pool_abs_delta(ss_rows, require_wt_psi_gte=0.5)
_y_true  = np.concatenate([np.ones(len(ss_abs_roc)), np.zeros(len(null_pool))])
_y_score = np.concatenate([ss_abs_roc, null_pool])
_fpr, _tpr, _thr = roc_curve(_y_true, _y_score)
_j = _tpr - _fpr
_opt_idx = np.argmax(_j[1:]) + 1
opt_thr  = float(_thr[_opt_idx])
print(f"Youden-optimal threshold (pre-computed): {opt_thr:.3f}")

# ---------------------------------------------------------------------------
# Fig A: Empirical FDR
# ---------------------------------------------------------------------------
print("\nFig A: Empirical FDR...")

null_sorted = np.sort(null_pool)

def empirical_pval(x_arr, null_sorted):
    idx = np.searchsorted(null_sorted, x_arr, side="left")
    return np.clip((len(null_sorted) - idx) / len(null_sorted),
                   1.0 / len(null_sorted), 1.0)

p_vals = empirical_pval(ss_pool, null_sorted)
_, q_vals, _, _ = multipletests(p_vals, method="fdr_bh")

fdr_results = {}
for t in THRESHOLDS:
    called = ss_pool >= t
    fdr_t  = float(q_vals[called].max()) * 100 if called.any() else np.nan
    fdr_results[t] = {"n_called": int(called.sum()), "fdr_pct": round(fdr_t, 3)}
    print(f"  |Δlogit| >= {t}: n={called.sum():,}, FDR={fdr_t:.2f}%")

pd.DataFrame([
    {"threshold": t, "n_called": v["n_called"], "fdr_pct": v["fdr_pct"]}
    for t, v in fdr_results.items()
]).to_csv(os.path.join(OUT_DIR, "fdr_summary.csv"), index=False)

sweep_t   = np.linspace(0, 5, 500)
fdr_sweep = []
for t in sweep_t:
    called = ss_pool >= t
    fdr_sweep.append(float(q_vals[called].max()) * 100 if called.any() else np.nan)
fdr_sweep = np.array(fdr_sweep)
pd.DataFrame({"threshold": sweep_t, "fdr_pct": fdr_sweep}).to_csv(
    os.path.join(OUT_DIR, "fdr_sweep.csv"), index=False)

mark_colors = {0.5: "#FF9800", 1.0: "#E53935", opt_thr: "#1565C0", 1.5: "#7B1FA2", 2.0: "#2E7D32"}

# FDR at Youden-optimal threshold
youden_fdr_idx = np.argmin(np.abs(sweep_t - opt_thr))
youden_fdr_pct = fdr_sweep[youden_fdr_idx]

fig_a, ax = plt.subplots(figsize=(5.5, 4))
valid = ~np.isnan(fdr_sweep)
ax.plot(sweep_t[valid], fdr_sweep[valid], color="#333333", lw=2)
ax.axhline(5, color="#2196F3", lw=1, ls=":", alpha=0.8, label="5% FDR")
for t, v in fdr_results.items():
    if not np.isnan(v["fdr_pct"]):
        ax.scatter([t], [v["fdr_pct"]], s=70, color=mark_colors[t], zorder=5)
        ax.axvline(t, color=mark_colors[t], lw=1.2, ls="--", alpha=0.6,
                   label=f"|Δlogit| = {t}  →  FDR = {v['fdr_pct']:.1f}%")
# Add Youden-optimal marker
ax.scatter([opt_thr], [youden_fdr_pct], s=90, color=mark_colors[opt_thr],
           marker="*", zorder=6)
ax.axvline(opt_thr, color=mark_colors[opt_thr], lw=1.2, ls="--", alpha=0.6,
           label=f"|Δlogit| = {opt_thr:.2f} (Youden)  →  FDR = {youden_fdr_pct:.1f}%")
ax.set_xlabel("|Δlogit(PSI)|", fontsize=11)
ax.set_ylabel("Empirical BH-FDR (%)", fontsize=11)
ax.set_title(f"Empirical FDR vs. |Δlogit| threshold\n"
             f"(canonical SS variants; null = {NULL_LABEL})",
             fontsize=10, fontweight="bold")
ax.set_xlim(0, 5); ax.set_ylim(0, 105)
ax.legend(fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig_a.tight_layout()
for ext in ["png", "pdf"]:
    fig_a.savefig(os.path.join(OUT_DIR, f"figA_fdr_curve.{ext}"),
                  dpi=DPI, bbox_inches="tight")
plt.close(fig_a)
print("  Saved figA_fdr_curve")

# ---------------------------------------------------------------------------
# FDR exploration: test different observed pools
# ---------------------------------------------------------------------------
print("\nFDR exploration across test sets...")

def compute_fdr_at_thresholds(obs_arr, null_sorted, thresholds):
    """BH-FDR at each threshold for a given observed pool."""
    p  = empirical_pval(obs_arr, null_sorted)
    _, q, _, _ = multipletests(p, method="fdr_bh")
    out = {}
    for t in thresholds:
        called = obs_arr >= t
        out[t] = {
            "n_tested": len(obs_arr),
            "n_called": int(called.sum()),
            "fdr_pct":  round(float(q[called].max()) * 100, 2) if called.any() else np.nan,
        }
    return out

# Define test sets
test_sets = {
    "All single variants (WT PSI>=0.5)":
        pool_abs_delta(single, require_wt_psi_gte=0.5),
    "Canonical SS variants only":
        pool_abs_delta(ss_rows, require_wt_psi_gte=0.5),
    "ExAC variants only":
        pool_abs_delta(single[single["source"] == "exac"], require_wt_psi_gte=0.5),
    "ClinVar single variants only":
        pool_abs_delta(single[single["source"] == "clinvar_single"], require_wt_psi_gte=0.5),
    f"Intronic ≥{DISTAL_BP} bp from SS, PSI ∈ [0,{PSI_LOW}]∪[{PSI_HIGH},1] all cells (self-test)":
        null_pool,
}

fdr_explore_rows = []
for label, obs_arr in test_sets.items():
    fdrs = compute_fdr_at_thresholds(obs_arr, null_sorted, THRESHOLDS)
    for t, v in fdrs.items():
        fdr_explore_rows.append({"test_set": label, "threshold": t, **v})
    print(f"  {label} (n={len(obs_arr):,}):")
    for t, v in fdrs.items():
        print(f"    |Δlogit|>={t}: n_called={v['n_called']:,}, FDR={v['fdr_pct']:.1f}%")

fdr_explore_df = pd.DataFrame(fdr_explore_rows)
fdr_explore_df.to_csv(os.path.join(OUT_DIR, "fdr_exploration.csv"), index=False)
print("  Saved fdr_exploration.csv")

# ---------------------------------------------------------------------------
# Fig B: Replicate-level noise
# ---------------------------------------------------------------------------
print("\nFig B: Replicate-level noise...")

# Panel 1: violin of sd_delta_logit per cell line
# HEK: pre-computed sd requires all 4 reps; compute from rep1+rep2 instead
sd_data = {}
for cell in CELL_LINES:
    if cell == "HEK":
        d1   = single[REP1[cell]] - single[wt_logit_cols[cell]]
        d2   = single[REP2[cell]] - single[wt_logit_cols[cell]]
        mask = (single[cov_cols[cell]] >= COV_MIN) & d1.notna() & d2.notna()
        vals = (np.abs(d1 - d2) / np.sqrt(2))[mask].values
    else:
        mask = (single[cov_cols[cell]] >= COV_MIN) & single[sd_cols[cell]].notna()
        vals = single.loc[mask, sd_cols[cell]].values
    sd_data[cell] = vals
    print(f"  {cell}: n={len(vals):,}  median={np.median(vals):.3f}  "
          f"p95={np.percentile(vals,95):.3f}  p99={np.percentile(vals,99):.3f}")

pd.DataFrame([
    {"cell_line": c, "n": len(sd_data[c]),
     "median_sd": round(float(np.median(sd_data[c])), 4),
     "p95_sd":    round(float(np.percentile(sd_data[c], 95)), 4),
     "p99_sd":    round(float(np.percentile(sd_data[c], 99)), 4)}
    for c in CELL_LINES
]).to_csv(os.path.join(OUT_DIR, "replicate_noise_summary.csv"), index=False)

# — figB_violin: standalone IQR violin —
fig_b_violin, ax_vln = plt.subplots(figsize=(6, 4.5))

vp = ax_vln.violinplot(
    [sd_data[c] for c in CELL_LINES],
    positions=range(len(CELL_LINES)),
    showmedians=True, showextrema=False, widths=0.7
)
for pc in vp["bodies"]:
    pc.set_facecolor("#4878CF"); pc.set_alpha(0.55)
    pc.set_edgecolor("#2c5099"); pc.set_linewidth(0.8)
vp["cmedians"].set_color("black")

CAP_W = 0.12
for i, cell in enumerate(CELL_LINES):
    q1, q3 = np.percentile(sd_data[cell], [25, 75])
    ax_vln.vlines(i, q1, q3, color="#444", lw=1.5, zorder=2)
    ax_vln.hlines([q1, q3], i - CAP_W, i + CAP_W, color="#444", lw=1.5, zorder=2)

ax_vln.axhline(1.0,     color="#E53935", lw=1.4, ls="--", label="|Δlogit| = 1.0")
ax_vln.axhline(opt_thr, color="#1565C0", lw=1.4, ls="--", label=f"|Δlogit| = {opt_thr:.2f} (Youden)")
ax_vln.axhline(1.5,     color="#7B1FA2", lw=1.4, ls="--", label="|Δlogit| = 1.5")
ax_vln.set_xticks(range(len(CELL_LINES)))
ax_vln.set_xticklabels(CELL_LINES, fontsize=10)
ax_vln.set_ylabel("SD of Δlogit(PSI) across replicates", fontsize=10)
ax_vln.set_title("Replicate-level measurement noise", fontsize=10, fontweight="bold")
ax_vln.legend(fontsize=8)
ax_vln.spines[["top", "right"]].set_visible(False)
for i, cell in enumerate(CELL_LINES):
    med = np.median(sd_data[cell])
    ax_vln.text(i + 0.15, med + 0.04, f"{med:.2f}",
                ha="left", va="bottom", fontsize=8, color="black")

fig_b_violin.tight_layout()
for ext in ["png", "pdf"]:
    fig_b_violin.savefig(os.path.join(OUT_DIR, f"figB_violin.{ext}"),
                         dpi=DPI, bbox_inches="tight")
plt.close(fig_b_violin)
print("  Saved figB_violin")

# — figB_scatter: per-cell-line rep1 vs rep2 Δlogit, 5 panels —
# Tiers ordered by ascending threshold so Youden (lowest) is plotted first,
# then ≥1.0 on top, then ≥1.5 on top of that.
tier_specs = [
    (0, "#AAAAAA", 0.15, f"|Δlogit| < {opt_thr:.2f}"),
    (1, "#FFAB40", 0.55, f"≥ {opt_thr:.2f} (Youden)"),
    (2, "#1565C0", 0.65, "≥ 1.0"),
    (3, "#7B1FA2", 0.75, "≥ 1.5"),
]
CLIP = 8
rng  = np.random.default_rng(42)

fig_b_scatter, axes_s = plt.subplots(1, 5, figsize=(17, 3.5), sharey=True)

for ax_s, cell in zip(axes_s, CELL_LINES):
    mask = (
        (single[cov_cols[cell]] >= COV_MIN) &
        single[REP1[cell]].notna() & single[REP2[cell]].notna() &
        single[wt_logit_cols[cell]].notna() &
        single[delta_cols[cell]].notna()
    )
    sub = single[mask].copy()
    r1 = (sub[REP1[cell]] - sub[wt_logit_cols[cell]]).values
    r2 = (sub[REP2[cell]] - sub[wt_logit_cols[cell]]).values
    adl = sub[delta_cols[cell]].abs().values

    cell_tier = np.where(adl >= 1.5, 3, np.where(adl >= 1.0, 2, np.where(adl >= opt_thr, 1, 0)))

    # downsample tier-0 for display
    t0_idx  = np.where(cell_tier == 0)[0]
    sdv_idx = np.where(cell_tier > 0)[0]
    keep_c  = np.concatenate([
        rng.choice(t0_idx, min(10_000, len(t0_idx)), replace=False),
        sdv_idx
    ])
    keep_c = rng.permutation(keep_c)

    r1_c = np.clip(r1, -CLIP, CLIP)
    r2_c = np.clip(r2, -CLIP, CLIP)

    for t_val, col, alpha, _ in tier_specs:
        idx = keep_c[cell_tier[keep_c] == t_val]
        ax_s.scatter(r1_c[idx], r2_c[idx], color=col, alpha=alpha, s=3, linewidths=0)

    r_c, _ = stats.pearsonr(r1, r2)
    ax_s.set_xlim(-CLIP, CLIP); ax_s.set_ylim(-CLIP, CLIP)
    ax_s.axhline(0, color="#CCCCCC", lw=0.5); ax_s.axvline(0, color="#CCCCCC", lw=0.5)
    ax_s.set_title(cell, fontsize=10, fontweight="bold")
    ax_s.set_xlabel("Rep 1 Δlogit(PSI)", fontsize=9)
    ax_s.text(0.05, 0.97, f"n={len(r1):,}\nr={r_c:.3f}",
              transform=ax_s.transAxes, ha="left", va="top", fontsize=7.5,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    ax_s.spines[["top", "right"]].set_visible(False)

axes_s[0].set_ylabel("Rep 2 Δlogit(PSI)", fontsize=9)

fig_b_scatter.legend(
    handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor=col,
               markersize=7, label=lbl)
        for _, col, _, lbl in tier_specs
    ],
    title="|Δlogit| (pooled)", title_fontsize=8,
    fontsize=8, loc="lower center", ncol=4,
    bbox_to_anchor=(0.5, -0.18)
)
fig_b_scatter.suptitle("Replicate concordance per cell line", fontsize=11, fontweight="bold", y=1.02)
fig_b_scatter.tight_layout()
for ext in ["png", "pdf"]:
    fig_b_scatter.savefig(os.path.join(OUT_DIR, f"figB_scatter_per_cellline.{ext}"),
                          dpi=DPI, bbox_inches="tight")
plt.close(fig_b_scatter)
print("  Saved figB_scatter_per_cellline")

# ---------------------------------------------------------------------------
# Fig C: ROC curve
# ---------------------------------------------------------------------------
print("\nFig C: ROC curve...")

ss_abs   = pool_abs_delta(ss_rows,   require_wt_psi_gte=0.5)
n_pos    = len(ss_abs)
n_neg    = len(null_pool)
y_true   = np.concatenate([np.ones(n_pos),  np.zeros(n_neg)])
y_score  = np.concatenate([ss_abs,           null_pool])

fpr, tpr, thr_roc = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)

youden_j = tpr - fpr
opt_idx  = np.argmax(youden_j[1:]) + 1
opt_thr  = thr_roc[opt_idx]
opt_sens = tpr[opt_idx]
opt_spec = 1 - fpr[opt_idx]

print(f"  n_pos={n_pos:,}, n_neg={n_neg:,}, AUC={auc:.3f}")
print(f"  Youden-optimal: thr={opt_thr:.3f}, sens={opt_sens:.3f}, spec={opt_spec:.3f}")

fixed = {}
for tv in THRESHOLDS:
    idx = np.argmin(np.abs(thr_roc - tv))
    fixed[tv] = {"fpr": fpr[idx], "tpr": tpr[idx],
                 "sens": tpr[idx], "spec": 1 - fpr[idx]}
    print(f"  |Δlogit|={tv}: sens={tpr[idx]:.3f}, spec={1-fpr[idx]:.3f}")

pd.DataFrame({
    "threshold": thr_roc[1:], "sensitivity": tpr[1:],
    "specificity": 1 - fpr[1:], "youden_j": youden_j[1:]
}).to_csv(os.path.join(OUT_DIR, "roc_sweep.csv"), index=False)

pd.DataFrame([{
    "n_pos": n_pos, "n_neg": n_neg, "AUC": round(auc, 4),
    "youden_optimal_threshold": round(opt_thr, 4),
    "youden_optimal_sens": round(opt_sens, 4),
    "youden_optimal_spec": round(opt_spec, 4),
    **{f"sens_at_{tv}": round(fixed[tv]["sens"], 4) for tv in THRESHOLDS},
    **{f"spec_at_{tv}": round(fixed[tv]["spec"], 4) for tv in THRESHOLDS},
}]).to_csv(os.path.join(OUT_DIR, "roc_summary.csv"), index=False)

thr_colors = {0.5: "#FF9800", 1.0: "#E53935", 1.5: "#7B1FA2", 2.0: "#2E7D32"}
thr_shapes = {0.5: "^",       1.0: "o",       1.5: "s",       2.0: "D"}

fig_c, ax2 = plt.subplots(figsize=(5.5, 5.5))
ax2.plot(fpr, tpr, color="#333333", lw=2)
ax2.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
ax2.scatter([fpr[opt_idx]], [tpr[opt_idx]], s=120, color="#1565C0", zorder=6,
            marker="*", label=f"Youden opt: {opt_thr:.2f}  "
                              f"sens={opt_sens:.2f}, spec={opt_spec:.2f}")
for tv, m in fixed.items():
    ax2.scatter([m["fpr"]], [m["tpr"]], s=90, color=thr_colors[tv],
                marker=thr_shapes[tv], zorder=5,
                label=f"|Δlogit| = {tv}  sens={m['sens']:.2f}, spec={m['spec']:.2f}")

ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
ax2.set_xlabel("1 − Specificity (FPR)", fontsize=11)
ax2.set_ylabel("Sensitivity (TPR)", fontsize=11)
ax2.set_title(
    f"ROC: canonical SS variants (WT PSI ≥ 0.5)\nvs. {NULL_LABEL}",
    fontsize=10, fontweight="bold"
)
legend_handles = (
    [Line2D([0],[0], color="#333333", lw=2,
             label=f"AUC = {auc:.3f}  (pos={n_pos:,} SS, neg={n_neg:,} null)"),
     plt.scatter([], [], s=100, color="#1565C0", marker="*",
                 label=f"Youden: {opt_thr:.2f}  sens={opt_sens:.2f} spec={opt_spec:.2f}")]
    + [plt.scatter([], [], s=80, color=thr_colors[tv], marker=thr_shapes[tv],
                   label=f"|Δlogit|={tv}  sens={fixed[tv]['sens']:.2f} spec={fixed[tv]['spec']:.2f}")
       for tv in THRESHOLDS]
)
ax2.legend(handles=legend_handles, fontsize=7.5,
           bbox_to_anchor=(0.5, -0.18), loc="upper center",
           ncol=2, handletextpad=0.3, columnspacing=0.8, borderpad=0.5)
ax2.spines[["top", "right"]].set_visible(False)
fig_c.tight_layout()
fig_c.subplots_adjust(bottom=0.3)
for ext in ["png", "pdf"]:
    fig_c.savefig(os.path.join(OUT_DIR, f"figC_roc.{ext}"),
                  dpi=DPI, bbox_inches="tight")
plt.close(fig_c)
print(f"  Saved figC_roc  (AUC={auc:.3f})")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nNull definition: {NULL_LABEL}")
print(f"  n_variants in null: {len(null_rows):,}")
print(f"  n_observations pooled: {len(null_pool):,}")
print("\nFDR (new null):")
for t, v in fdr_results.items():
    print(f"  |Δlogit| >= {t}: n={v['n_called']:,}, FDR={v['fdr_pct']:.2f}%")
print(f"\nReplicate scatter: per-cell-line panels (see figB_scatter_per_cellline)")
print(f"\nROC: AUC={auc:.3f}, Youden thr={opt_thr:.3f} (sens={opt_sens:.3f}, spec={opt_spec:.3f})")
print("\nPerformance at fixed thresholds:")
for tv in THRESHOLDS:
    print(f"  {tv}: sens={fixed[tv]['sens']:.3f}, spec={fixed[tv]['spec']:.3f}")
print(f"\nOutputs: {OUT_DIR}")
