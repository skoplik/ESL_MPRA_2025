#!/usr/bin/env python3
"""
Full figure panel for MMSplice retraining analysis (reviewer response).

Layout (2 rows x 3 cols):
  A: SpliceAI vs Baseline MMSplice (aggregate)
  B: SpliceAI vs Retrained MMSplice (aggregate)
  C: Improvement vs experiment by effect size bin (test set)
  D: SpliceAI vs Baseline MMSplice (test set)
  E: SpliceAI vs Retrained MMSplice (test set)
  F: imp_vs_exp vs imp_vs_sai scatter (aggregate, colored by effect size)
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

MEGA_FILE = "/ESL/ESL_MPRA/Figure_3/model_comparison/mega_pred_file_filtered.csv"
OUTDIR    = "/ESL/ESL_MPRA/Figure_3/plots/MMSplice/aggregate_analysis"
os.makedirs(OUTDIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
mega = pd.read_csv(MEGA_FILE, low_memory=False)
PRED_COLS = ["baseline_mmsplice_delta_logit", "retrained_mmsplice_delta_logit", "avg_delta_logit_pooled"]
mega_mm = mega.dropna(subset=PRED_COLS).copy()

test = mega_mm[mega_mm["mmsplice_is_test"] == True].copy()
agg  = mega_mm.copy()

# Improvement metrics
for df in [test, agg]:
    df["err_base_exp"] = (df["baseline_mmsplice_delta_logit"]  - df["avg_delta_logit_pooled"]).abs()
    df["err_ret_exp"]  = (df["retrained_mmsplice_delta_logit"] - df["avg_delta_logit_pooled"]).abs()
    df["imp_vs_exp"]   = df["err_base_exp"] - df["err_ret_exp"]

# SpliceAI improvement metrics (aggregate)
sai = agg.dropna(subset=["spliceai_delta_logit"]).copy()
sai["err_base_sai"] = (sai["baseline_mmsplice_delta_logit"]  - sai["spliceai_delta_logit"]).abs()
sai["err_ret_sai"]  = (sai["retrained_mmsplice_delta_logit"] - sai["spliceai_delta_logit"]).abs()
sai["imp_vs_sai"]   = sai["err_base_sai"] - sai["err_ret_sai"]
sai["effect_bin"]   = pd.cut(sai["avg_delta_logit_pooled"].abs(),
                              bins=[0, 0.5, 1, 2, np.inf], labels=["<0.5", "0.5–1", "1–2", ">2"])

sai_test = test.dropna(subset=["spliceai_delta_logit"]).copy()

print(f"  test={len(test):,}  agg={len(agg):,}  sai={len(sai):,}  sai_test={len(sai_test):,}")


# ── Colors ────────────────────────────────────────────────────────────────────
C_BASE = "#6baed6"   # medium blue (was too light before)
C_RET  = "#08519c"   # dark blue
EFF_COLORS = {"<0.5": "#d9d9d9", "0.5–1": "#fdae6b", "1–2": "#f16913", ">2": "#7f2704"}
EFF_ORDER  = ["<0.5", "0.5–1", "1–2", ">2"]

# Shared axis limits for SpliceAI scatter panels
_sai_vals = pd.concat([sai["spliceai_delta_logit"],
                        sai["baseline_mmsplice_delta_logit"],
                        sai["retrained_mmsplice_delta_logit"]]).abs()
LIM_SAI = _sai_vals.quantile(0.999) * 1.05   # clip extreme outliers for readability


# ── Helper: SpliceAI vs MMSplice scatter ─────────────────────────────────────
def sai_scatter(ax, df, col, color, title):
    sub = df.dropna(subset=["spliceai_delta_logit", col])
    r, _ = stats.pearsonr(sub["spliceai_delta_logit"], sub[col])
    ax.scatter(sub["spliceai_delta_logit"], sub[col],
               s=2, alpha=0.15, color=color, linewidths=0, rasterized=True)
    ax.plot([-LIM_SAI, LIM_SAI], [-LIM_SAI, LIM_SAI],
            color="grey", linestyle="--", linewidth=0.8, zorder=5)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.4)
    ax.set_xlim(-LIM_SAI, LIM_SAI)
    ax.set_ylim(-LIM_SAI, LIM_SAI)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("SpliceAI Δlogit", fontsize=9)
    ylabel = ("Baseline" if "baseline" in col else "Retrained") + " MMSplice Δlogit"
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
    ax.text(0.97, 0.03, f"r={r:.3f}\nn={len(sub):,}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
    ax.grid(False)


# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), dpi=300)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[0, 2])
ax_D = fig.add_subplot(gs[1, 0])
ax_E = fig.add_subplot(gs[1, 1])
ax_F = fig.add_subplot(gs[1, 2])


# ── A: SpliceAI vs Baseline MMSplice (aggregate) ──────────────────────────────
sai_scatter(ax_A, sai, "baseline_mmsplice_delta_logit", C_BASE,
            "A  SpliceAI vs Baseline MMSplice\n(aggregate)")

# ── B: SpliceAI vs Retrained MMSplice (aggregate) ─────────────────────────────
sai_scatter(ax_B, sai, "retrained_mmsplice_delta_logit", C_RET,
            "B  SpliceAI vs Retrained MMSplice\n(aggregate)")

# ── C: Improvement by effect size (test set) ──────────────────────────────────
BIN_ORDER  = ["<0.5\n(sub-thresh.)", "0.5–1\n(moderate)", "1–2\n(SDV)", ">2\n(strong SDV)"]
BIN_COLORS = [EFF_COLORS[k] for k in EFF_ORDER]

test["effect_bin_plot"] = pd.cut(
    test["avg_delta_logit_pooled"].abs(),
    bins=[0, 0.5, 1, 2, np.inf], labels=BIN_ORDER
)
grp   = test.groupby("effect_bin_plot")["imp_vs_exp"]
means = [grp.get_group(b).mean()  if b in grp.groups else np.nan for b in BIN_ORDER]
sems  = [grp.get_group(b).sem()   if b in grp.groups else np.nan for b in BIN_ORDER]
ns    = [grp.get_group(b).count() if b in grp.groups else 0      for b in BIN_ORDER]

xc = np.arange(len(BIN_ORDER))
ax_C.bar(xc, means, color=BIN_COLORS, edgecolor="black", linewidth=0.6,
         yerr=sems, capsize=4, error_kw={"linewidth": 0.8})
for xi, (m, s, n) in enumerate(zip(means, sems, ns)):
    if not np.isnan(m):
        top = (m + s) if not np.isnan(s) else m
        ax_C.text(xi, max(top, 0) + 0.005, f"n={n:,}",
                  ha="center", va="bottom", fontsize=6.5)
ax_C.axhline(0, color="black", linewidth=0.8)
ax_C.set_xticks(xc)
ax_C.set_xticklabels(BIN_ORDER, fontsize=8)
ax_C.set_xlabel("|Measured Δlogit(PSI)|", fontsize=9)
ax_C.set_ylabel("Mean improvement\n(baseline − retrained |error|)", fontsize=8)
ax_C.set_title("C  Retraining benefit by effect size\n(held-out test set)", fontsize=9, fontweight="bold", loc="left")
ax_C.grid(False)

# ── D: SpliceAI vs Baseline MMSplice (test set) ───────────────────────────────
sai_scatter(ax_D, sai_test, "baseline_mmsplice_delta_logit", C_BASE,
            "D  SpliceAI vs Baseline MMSplice\n(held-out test set)")

# ── E: SpliceAI vs Retrained MMSplice (test set) ──────────────────────────────
sai_scatter(ax_E, sai_test, "retrained_mmsplice_delta_logit", C_RET,
            "E  SpliceAI vs Retrained MMSplice\n(held-out test set)")

# ── F: imp_vs_exp vs imp_vs_sai scatter ───────────────────────────────────────
r_ee, _ = stats.pearsonr(sai["imp_vs_exp"], sai["imp_vs_sai"])

for bin_label, color in EFF_COLORS.items():
    s2 = sai[sai["effect_bin"] == bin_label]
    ax_F.scatter(s2["imp_vs_exp"], s2["imp_vs_sai"],
                 s=2, alpha=0.2, color=color, linewidths=0, rasterized=True,
                 label=bin_label)

# Clip axes at 99th percentile to avoid extreme outliers distorting layout
lim_f = max(sai["imp_vs_exp"].abs().quantile(0.99),
            sai["imp_vs_sai"].abs().quantile(0.99)) * 1.1
ax_F.plot([-lim_f, lim_f], [-lim_f, lim_f], color="grey", linestyle="--", linewidth=0.8, zorder=5)
ax_F.axhline(0, color="black", linewidth=0.4)
ax_F.axvline(0, color="black", linewidth=0.4)
ax_F.set_xlim(-lim_f, lim_f)
ax_F.set_ylim(-lim_f, lim_f)
ax_F.set_aspect("equal", adjustable="box")
ax_F.set_xlabel("Improvement vs. COMPASS\n(baseline − retrained |error|)", fontsize=8)
ax_F.set_ylabel("Improvement vs. SpliceAI\n(baseline − retrained |error|)", fontsize=8)
ax_F.set_title("F  Improvement vs. experiment vs. SpliceAI\n(aggregate)", fontsize=9, fontweight="bold", loc="left")
ax_F.text(0.97, 0.03, f"r={r_ee:.3f}\nn={len(sai):,}",
          transform=ax_F.transAxes, ha="right", va="bottom", fontsize=8,
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
legend_handles = [Line2D([0],[0], marker="o", color="w", markerfacecolor=EFF_COLORS[b],
                          markersize=6, label=b) for b in EFF_ORDER]
ax_F.legend(handles=legend_handles, title="|Δlogit|", frameon=False, fontsize=7, loc="upper left")
ax_F.grid(False)


# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    path = os.path.join(OUTDIR, f"mmsplice_retraining_panel.{ext}")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
print("Done.")
