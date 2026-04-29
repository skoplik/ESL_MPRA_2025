"""
Cryptic Splice Site Figures
============================
Produces 4 publication-ready figures:

  fig1  — Cryptic fraction vs. WT PSI: boxplot by decile + LOWESS trend ribbon
  fig2  — Cryptic fraction by variant position class (bar chart)
  fig3  — Cryptic SS position map with variable-region annotation
  fig4  — Top 20 exons by cryptic fraction (WT sequences)

Inputs:
  - old/cryptic_junctions_per_sequence.csv   (pre-computed)
  - old/variant_vs_wt_cryptic_comparison.csv (pre-computed)
  - Main COMPASS PSI CSV                      (for WT PSI values)
  - Raw STAR splicing count files             (for position map; cached after first run)
"""

import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
SEQ_CSV   = SCRIPT_DIR / "old" / "cryptic_junctions_per_sequence.csv"
VAR_CSV   = SCRIPT_DIR / "old" / "variant_vs_wt_cryptic_comparison.csv"
PSI_CSV   = Path(
    "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/"
    "03_16_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"
)
POS_CACHE = SCRIPT_DIR / "cryptic_pos_cache.csv"

CELL_LINES = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]
AMB_SJS    = Path("/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/ambiguous_sjs/SI_alt_transcript_psi.csv")
SS_WINDOW  = 4   # nt either side of intron/exon boundary for splice-site class

# Variable-region canonical boundaries (relative to upstream constitutive 5'SS)
# Derived from data across all constructs
REL_CONST_END    = 260   # end of constant SMN2 upstream intron
REL_EXON_3SS_Q1  = 320   # 25th %ile of exon 3'SS
REL_EXON_3SS_MED = 329   # median exon 3'SS
REL_EXON_3SS_Q3  = 350   # 75th %ile of exon 3'SS
REL_EXON_5SS     = 402   # exon 5'SS (fixed for all constructs)
REL_VAR_END      = 422   # end of variable intron2 / start of Citrine (fixed)


# ============================================================================
# Utility
# ============================================================================

def avg_cryptic(df):
    cols = [f"{cl}_cryptic_fraction" for cl in CELL_LINES
            if f"{cl}_cryptic_fraction" in df.columns]
    return df[cols].mean(axis=1)


def classify_variant_position(snp, intron1_len, exon_len):
    if pd.isna(snp):
        return "unknown"
    parts = str(snp).split(";")
    if len(parts) > 1:
        return "double"
    try:
        pos = int(parts[0].split(":")[0])
    except (ValueError, IndexError):
        return "unknown"
    exon_start = intron1_len
    exon_end   = intron1_len + exon_len
    if max(0, exon_start - SS_WINDOW) <= pos < exon_start + SS_WINDOW:
        return "3SS"
    if max(0, exon_end - SS_WINDOW) <= pos < exon_end + SS_WINDOW:
        return "5SS"
    if exon_start <= pos < exon_end:
        return "exonic"
    if pos < exon_start:
        return "intronic_up"
    return "intronic_dn"


# ============================================================================
# Load data
# ============================================================================

print("Loading pre-computed cryptic junction data...")
seq_df = pd.read_csv(SEQ_CSV)

# Exclude sequences whose Parent_ref appears in the ambiguous splice junctions list
amb_refs = set(pd.read_csv(AMB_SJS, usecols=["Reference"])["Reference"].dropna().astype(int))
before = len(seq_df)
seq_df = seq_df[~seq_df["Parent_ref"].isin(amb_refs)].copy()
print(f"  Excluded {before - len(seq_df):,} sequences matching ambiguous SJ list "
      f"({len(amb_refs):,} ambiguous References)")

seq_df["avg_cryptic_fraction"] = avg_cryptic(seq_df)
print(f"  {len(seq_df):,} sequences after exclusion")

print("Loading WT PSI and exon coordinates from main COMPASS table...")
wt_psi_cols = (["Reference", "event_id", "exon_start_hg38", "exon_end_hg38"]
               + [f"{cl}_wt_pooled_psi_raw" for cl in CELL_LINES])
main_df = pd.read_csv(PSI_CSV, usecols=lambda c: c in wt_psi_cols, low_memory=False)
psi_cols = [c for c in main_df.columns if "wt_pooled_psi_raw" in c]
wt_psi_event = (
    main_df.dropna(subset=["event_id"])
    .groupby("event_id")[psi_cols + ["exon_start_hg38", "exon_end_hg38"]]
    .first()
    .reset_index()
)
wt_psi_event["wt_psi_avg"] = wt_psi_event[psi_cols].mean(axis=1)

var_seq = seq_df[seq_df["seq_type"] != "none"].copy()
var_seq = var_seq.merge(wt_psi_event[["event_id", "wt_psi_avg"]], on="event_id", how="left")
var_seq = var_seq.dropna(subset=["wt_psi_avg", "avg_cryptic_fraction"])
print(f"  {len(var_seq):,} variants with WT PSI")


# ============================================================================
# Position data (cached after first run)
# ============================================================================

def _load_junctions_and_positions():
    sys.path.insert(0, str(SCRIPT_DIR / "old"))
    from analyze_cryptic_splice_sites import (
        load_all_junctions, pool_junctions_per_cellline,
        compute_cryptic_relative_positions,
    )
    print("  Loading raw junction files (this may take ~1 min)...")
    jxn_raw = load_all_junctions()
    pooled  = pool_junctions_per_cellline(jxn_raw)
    cpos    = compute_cryptic_relative_positions(pooled, jxn_raw)
    cpos.to_csv(POS_CACHE, index=False)
    return cpos

if POS_CACHE.exists():
    print(f"Loading cached position data...")
    cpos_df = pd.read_csv(POS_CACHE)
else:
    print("Computing cryptic splice site positions from raw junction files...")
    cpos_df = _load_junctions_and_positions()

before_pos = len(cpos_df)
cpos_df = cpos_df[~cpos_df["Parent_ref"].isin(amb_refs)].copy()
print(f"  {len(cpos_df):,} cryptic junction rows after exclusion "
      f"({before_pos - len(cpos_df):,} removed)")


# ============================================================================
# Figure 1 — Cryptic fraction vs. WT PSI
# Left:  boxplot by PSI decile
# Right: LOWESS median + IQR ribbon (no raw scatter, no hexbin)
# ============================================================================
print("\n[Figure 1] Cryptic fraction vs. WT PSI...")

x_all = var_seq["wt_psi_avg"].values
y_all = var_seq["avg_cryptic_fraction"].values
mask  = np.isfinite(x_all) & np.isfinite(y_all)
x_all, y_all = x_all[mask], y_all[mask]

var_seq_clean = var_seq.dropna(subset=["wt_psi_avg", "avg_cryptic_fraction"]).copy()
var_seq_clean["wt_psi_bin"] = pd.qcut(var_seq_clean["wt_psi_avg"], q=10,
                                       labels=False, duplicates="drop")
bin_medpsi = var_seq_clean.groupby("wt_psi_bin")["wt_psi_avg"].median()

fig1, (ax_box, ax_trend) = plt.subplots(1, 2, figsize=(12, 4.5))

# ── Left: boxplot by decile ──────────────────────────────────────────────────
bin_ids   = sorted(var_seq_clean["wt_psi_bin"].dropna().unique())
box_data  = [var_seq_clean.loc[var_seq_clean["wt_psi_bin"] == b,
                                "avg_cryptic_fraction"].dropna().values
             for b in bin_ids]
cmap = plt.cm.RdYlGn
bp = ax_box.boxplot(box_data, positions=range(len(bin_ids)), widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.5))
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(cmap(i / len(bp["boxes"])))
    patch.set_alpha(0.8)
ax_box.set_xticks(range(len(bin_ids)))
ax_box.set_xticklabels([f"{v:.2f}" for v in bin_medpsi.values],
                        rotation=45, fontsize=7)
ax_box.set_xlabel("Median WT PSI (decile bin)", fontsize=10)
ax_box.set_ylabel("Avg cryptic fraction (variant)", fontsize=10)
ax_box.set_title("Distribution by WT PSI decile", fontsize=10)

# ── Right: LOWESS median + IQR ribbon ────────────────────────────────────────
sort_idx = np.argsort(x_all)
xs, ys   = x_all[sort_idx], y_all[sort_idx]

# LOWESS on median, 25th, 75th, 90th quantiles (rolling window then smoothed)
window = max(1, len(xs) // 60)
def smooth_quantile(q):
    raw = pd.Series(ys).rolling(window, center=True, min_periods=1).quantile(q).values
    sm  = lowess(raw, xs, frac=0.25, it=1, return_sorted=True)
    return sm[:, 0], sm[:, 1]

x50, y50 = smooth_quantile(0.50)
x25, y25 = smooth_quantile(0.25)
x75, y75 = smooth_quantile(0.75)

ax_trend.fill_between(x75, y25, y75, color="#4878cf", alpha=0.20, label="IQR (25–75th %ile)")
ax_trend.plot(x50, y50, color="#4878cf", lw=2.2, label="Median (LOWESS)")

r_sp, p_sp = stats.spearmanr(x_all, y_all, nan_policy="omit")
ax_trend.text(0.04, 0.96,
              f"Spearman r = {r_sp:.3f}\np = {p_sp:.1e}\nn = {len(x_all):,}",
              transform=ax_trend.transAxes, fontsize=9, va="top",
              bbox=dict(fc="white", ec="gray", alpha=0.85, pad=3))
ax_trend.set_xlabel("Reference (WT) PSI", fontsize=10)
ax_trend.set_ylabel("Avg cryptic fraction (variant)", fontsize=10)
ax_trend.set_title("Trend: LOWESS median + IQR ribbon", fontsize=10)
ax_trend.set_xlim(-0.03, 1.03)
ax_trend.set_ylim(-0.003, None)
ax_trend.legend(fontsize=8)

fig1.suptitle("Cryptic junction usage vs. exon inclusion rate (WT)", fontsize=11)
plt.tight_layout()
fig1.savefig(SCRIPT_DIR / "fig1_cryptic_vs_wt_psi.png", bbox_inches="tight", dpi=150)
fig1.savefig(SCRIPT_DIR / "fig1_cryptic_vs_wt_psi.pdf", bbox_inches="tight")
plt.close(fig1)
print("  Saved fig1_cryptic_vs_wt_psi.png/.pdf")


# ============================================================================
# Figure 2 — Cryptic fraction by variant position class (bar chart, no stats)
# ============================================================================
print("\n[Figure 2] Cryptic fraction by variant position class...")

single = seq_df[seq_df["seq_type"] == "single"].copy()
single = single.dropna(subset=["snp", "intron1", "exon"])
single["i1_len"]   = single["intron1"].str.len()
single["ex_len"]   = single["exon"].str.len()
single["var_class"] = single.apply(
    lambda r: classify_variant_position(r["snp"], r["i1_len"], r["ex_len"]), axis=1
)
single = single.dropna(subset=["avg_cryptic_fraction"])

CLASS_ORDER  = ["3SS", "5SS", "exonic", "intronic_up", "intronic_dn"]
CLASS_LABELS = {"3SS": "3′SS", "5SS": "5′SS", "exonic": "Exonic",
                "intronic_up": "Intronic\n(upstream)", "intronic_dn": "Intronic\n(downstream)"}
BAR_COLOR    = "#4878cf"

present = [c for c in CLASS_ORDER if c in single["var_class"].unique()]
means   = [single.loc[single["var_class"] == c, "avg_cryptic_fraction"].mean() for c in present]
sems    = [stats.sem(single.loc[single["var_class"] == c, "avg_cryptic_fraction"].dropna())
           for c in present]
ns      = [int((single["var_class"] == c).sum()) for c in present]

fig2, ax = plt.subplots(figsize=(6, 4.5))
x_pos = np.arange(len(present))
ax.bar(x_pos, means, color=BAR_COLOR, alpha=0.85, edgecolor="white",
       yerr=sems, capsize=4, error_kw=dict(lw=1.2, ecolor="black"))

# n= inside bars
for xi, (n, mean) in enumerate(zip(ns, means)):
    ax.text(xi, mean * 0.45, f"n={n:,}", ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels([CLASS_LABELS[c] for c in present], fontsize=10)
ax.set_ylabel("Mean cryptic fraction ± SEM", fontsize=10)
ax.set_title("Cryptic splice site usage by variant position class\n(single variants only)", fontsize=10)
ax.set_ylim(0, max(m + s for m, s in zip(means, sems)) * 1.25)

plt.tight_layout()
fig2.savefig(SCRIPT_DIR / "fig2_cryptic_by_variant_class.png", bbox_inches="tight", dpi=150)
fig2.savefig(SCRIPT_DIR / "fig2_cryptic_by_variant_class.pdf", bbox_inches="tight")
plt.close(fig2)
print("  Saved fig2_cryptic_by_variant_class.png/.pdf")


# ============================================================================
# Figure 3 — Cryptic SS distance from canonical SA (top) and SD (bottom)
#
# Top panel:    cryptic 3'SS positions as distance from per-construct canonical SA
#               SA = 0;  x-range: -100 to +600
# Bottom panel: cryptic 5'SS positions as distance from canonical SD
#               SD = 0;  x-range: -430 to +80
# Each panel has a colour region-track immediately below it.
# ============================================================================
print("\n[Figure 3] Cryptic SS distance from canonical SD / SA...")

REL_CITRINE2_3SS = 845
CITRINE1_LEN     = 25   # upstream_5ss = 26, so Citrine exon 1 = 25 nt

# Print region lengths
print("\nConstruct region lengths:")
print(f"  Citrine exon 1:              {CITRINE1_LEN} nt  (fixed)")
print(f"  SMN2 intron 6 (constant):    {REL_CONST_END} nt  (fixed)")
print(f"  Variable 5′ intron:          51–129 nt  (median {REL_EXON_3SS_MED - REL_CONST_END})")
print(f"  Variable exon:               12–90 nt   (median ~{REL_EXON_5SS - REL_EXON_3SS_MED})")
print(f"  Variable 3′ intron:          {REL_VAR_END - REL_EXON_5SS} nt  (fixed)")
print(f"  SMN2 intron 7 (constant):    {REL_CITRINE2_3SS - REL_VAR_END} nt  (fixed)")
print(f"  Citrine exon 2:              starts at rel. pos. {REL_CITRINE2_3SS}")

# Distance calculations (per-construct SA, fixed SD)
cpos_df["rel_exon_3ss"] = cpos_df["exon_3ss"] - cpos_df["upstream_5ss"]
cpos_df["dist_from_SA"] = cpos_df["rel_3ss"] - cpos_df["rel_exon_3ss"]
cpos_df["dist_from_SD"] = cpos_df["rel_5ss"] - REL_EXON_5SS   # always 402

# Panel x-ranges
SA_MIN, SA_MAX = -100, 600
SD_MIN, SD_MAX = -430,  80

# Helper: convert absolute rel-pos to distance-from-SA or -SD
# (use median SA = REL_EXON_3SS_MED for region boundary annotations)
def rel_to_dSA(x): return x - REL_EXON_3SS_MED
def rel_to_dSD(x): return x - REL_EXON_5SS

# Region definitions: (start, end) in distance-from-SS space, fill colour, label
REGIONS_SA = [
    (rel_to_dSA(-CITRINE1_LEN),     rel_to_dSA(0),                 "#98df8a", "Citrine\nexon 1"),
    (rel_to_dSA(0),                  rel_to_dSA(REL_CONST_END),     "#e0e0e0", "SMN2\nintron 6"),
    (rel_to_dSA(REL_CONST_END),      rel_to_dSA(REL_EXON_3SS_MED), "#aec6e8", "Var.\n5′ intron"),
    (rel_to_dSA(REL_EXON_3SS_MED),   rel_to_dSA(REL_EXON_5SS),     "#ffc48c", "Variable\nexon"),
    (rel_to_dSA(REL_EXON_5SS),       rel_to_dSA(REL_VAR_END),       "#c5b0d5", "Var.\n3′ intron"),
    (rel_to_dSA(REL_VAR_END),        rel_to_dSA(REL_CITRINE2_3SS),  "#e0e0e0", "SMN2\nintron 7"),
    (rel_to_dSA(REL_CITRINE2_3SS),   SA_MAX + 50,                   "#98df8a", "Citrine\nexon 2"),
]
# SA-panel boundaries (nt from SA):
#   Citrine exon 1: -354 to -329 | SMN2 intron 6: -329 to -69 | Var. 5' intron: -69 to 0
#   Variable exon: 0 to +73 | Var. 3' intron: +73 to +93 | SMN2 intron 7: +93 to +516
#   Citrine exon 2: +516 onwards

REGIONS_SD = [
    (rel_to_dSD(-CITRINE1_LEN),      rel_to_dSD(0),                 "#98df8a", "Citrine\nexon 1"),
    (rel_to_dSD(0),                   rel_to_dSD(REL_CONST_END),    "#e0e0e0", "SMN2\nintron 6"),
    (rel_to_dSD(REL_CONST_END),       rel_to_dSD(REL_EXON_3SS_MED), "#aec6e8", "Var.\n5′ intron"),
    (rel_to_dSD(REL_EXON_3SS_MED),    rel_to_dSD(REL_EXON_5SS),     "#ffc48c", "Variable\nexon"),
    (rel_to_dSD(REL_EXON_5SS),        rel_to_dSD(REL_VAR_END),       "#c5b0d5", "Var.\n3′ intron"),
    (rel_to_dSD(REL_VAR_END),         rel_to_dSD(REL_CITRINE2_3SS),  "#e0e0e0", "SMN2\nintron 7"),
    (rel_to_dSD(REL_CITRINE2_3SS),    SD_MAX + 50,                   "#98df8a", "Citrine\nexon 2"),
]
# SD-panel boundaries (nt from SD):
#   Citrine exon 1: -427 to -402 | SMN2 intron 6: -402 to -142 | Var. 5' intron: -142 to -73
#   Variable exon: -73 to 0 | Var. 3' intron: 0 to +20 | SMN2 intron 7: +20 to +443 (clipped)


def draw_track(ax_tr, regions, x_min, x_max, xlabel=None, min_label_width=30):
    """Draw a coloured region-track bar below a histogram panel."""
    for x0, x1, fc, lbl in regions:
        x0c = max(x0, x_min)
        x1c = min(x1, x_max)
        if x0c >= x1c:
            continue
        ax_tr.barh(0, x1c - x0c, left=x0c, height=1,
                   color=fc, edgecolor="white", linewidth=0.5)
        if (x1c - x0c) >= min_label_width:
            ax_tr.text((x0c + x1c) / 2, 0, lbl,
                       ha="center", va="center", fontsize=7,
                       color="#222222", fontweight="bold", linespacing=1.1)
    ax_tr.set_xlim(x_min, x_max)
    ax_tr.set_ylim(-0.5, 0.5)
    ax_tr.set_yticks([])
    for spine in ["left", "right", "top"]:
        ax_tr.spines[spine].set_visible(False)
    if xlabel:
        ax_tr.spines["bottom"].set_visible(True)
        ax_tr.tick_params(axis="x", bottom=True, labelbottom=True)
        ax_tr.set_xlabel(xlabel, fontsize=10)
    else:
        ax_tr.spines["bottom"].set_visible(False)
        ax_tr.tick_params(axis="x", bottom=False, labelbottom=False)


def shade_regions(ax, regions, x_min, x_max):
    for x0, x1, fc, _ in regions:
        x0c = max(x0, x_min)
        x1c = min(x1, x_max)
        if x0c < x1c:
            ax.axvspan(x0c, x1c, facecolor=fc, alpha=0.25, zorder=0)


# ── Build figure with nested GridSpec ─────────────────────────────────────────
fig3 = plt.figure(figsize=(11, 9))
outer_gs = fig3.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.38)
gs_sa    = outer_gs[0].subgridspec(2, 1, height_ratios=[5, 0.55], hspace=0.0)
gs_sd    = outer_gs[1].subgridspec(2, 1, height_ratios=[5, 0.55], hspace=0.0)

ax_sa    = fig3.add_subplot(gs_sa[0])
ax_sa_tr = fig3.add_subplot(gs_sa[1], sharex=ax_sa)
ax_sd    = fig3.add_subplot(gs_sd[0])
ax_sd_tr = fig3.add_subplot(gs_sd[1], sharex=ax_sd)

# ── SA panel: cryptic 3'SS positions ──────────────────────────────────────────
cryptic = cpos_df[cpos_df["is_cryptic"].astype(bool)]

sa_vals = cryptic["dist_from_SA"].dropna()
sa_wts  = cryptic.loc[sa_vals.index, "Count_pooled"].fillna(1)
sa_mask = (sa_vals >= SA_MIN) & (sa_vals <= SA_MAX)
counts_sa, edges_sa = np.histogram(sa_vals[sa_mask], bins=140,
                                    weights=sa_wts[sa_mask],
                                    range=(SA_MIN, SA_MAX))
centers_sa = (edges_sa[:-1] + edges_sa[1:]) / 2
ax_sa.bar(centers_sa, counts_sa, width=np.diff(edges_sa),
          color="#d62728", edgecolor="none", alpha=0.75, zorder=3,
          label="Cryptic 3′SS")
shade_regions(ax_sa, REGIONS_SA, SA_MIN, SA_MAX)

ax_sa.axvline(0, color="#1f77b4", lw=1.5, linestyle="--", zorder=5,
              label="Canonical SA (0)")
ax_sa.axvline(rel_to_dSA(REL_CITRINE2_3SS), color="#2ca02c", lw=1.2,
              linestyle=":", zorder=5,
              label=f"Citrine exon 2 SA (+{rel_to_dSA(REL_CITRINE2_3SS)})")

ax_sa.set_yscale("log")
ax_sa.set_ylabel("Read count (log scale)", fontsize=10)
ax_sa.set_title("Cryptic 3′SS: distance from canonical splice acceptor (SA)",
                fontsize=10)
ax_sa.legend(fontsize=8, loc="upper right", framealpha=0.9)
ymax_sa = ax_sa.get_ylim()[1]
ax_sa.set_ylim(1, ymax_sa * 3)
ax_sa.set_xlim(SA_MIN, SA_MAX)
plt.setp(ax_sa.get_xticklabels(), visible=False)
ax_sa.tick_params(axis="x", bottom=False)

draw_track(ax_sa_tr, REGIONS_SA, SA_MIN, SA_MAX,
           xlabel="Distance from canonical SA (nt)")

# ── SD panel: cryptic 5'SS positions ──────────────────────────────────────────
sd_vals = cryptic["dist_from_SD"].dropna()
sd_wts  = cryptic.loc[sd_vals.index, "Count_pooled"].fillna(1)
sd_mask = (sd_vals >= SD_MIN) & (sd_vals <= SD_MAX)
counts_sd, edges_sd = np.histogram(sd_vals[sd_mask], bins=102,
                                    weights=sd_wts[sd_mask],
                                    range=(SD_MIN, SD_MAX))
centers_sd = (edges_sd[:-1] + edges_sd[1:]) / 2
ax_sd.bar(centers_sd, counts_sd, width=np.diff(edges_sd),
          color="#4878cf", edgecolor="none", alpha=0.75, zorder=3,
          label="Cryptic 5′SS")
shade_regions(ax_sd, REGIONS_SD, SD_MIN, SD_MAX)

ax_sd.axvline(0, color="#e55c00", lw=1.5, linestyle="--", zorder=5,
              label="Canonical SD (0)")
ax_sd.axvline(rel_to_dSD(0), color="#555555", lw=1.0, linestyle=":", zorder=5,
              label=f"SMN2 intron 6 5′SS ({rel_to_dSD(0)})")

ax_sd.set_yscale("log")
ax_sd.set_ylabel("Read count (log scale)", fontsize=10)
ax_sd.set_title("Cryptic 5′SS: distance from canonical splice donor (SD)",
                fontsize=10)
ax_sd.legend(fontsize=8, loc="upper left", framealpha=0.9)
ymax_sd = ax_sd.get_ylim()[1]
ax_sd.set_ylim(1, ymax_sd * 3)
ax_sd.set_xlim(SD_MIN, SD_MAX)
plt.setp(ax_sd.get_xticklabels(), visible=False)
ax_sd.tick_params(axis="x", bottom=False)

draw_track(ax_sd_tr, REGIONS_SD, SD_MIN, SD_MAX,
           xlabel="Distance from canonical SD (nt)")

fig3.suptitle("Cryptic splice site positions relative to canonical splice sites",
              fontsize=11)
fig3.savefig(SCRIPT_DIR / "fig3_cryptic_position_map.png", bbox_inches="tight", dpi=150)
fig3.savefig(SCRIPT_DIR / "fig3_cryptic_position_map.pdf", bbox_inches="tight")
plt.close(fig3)
print("  Saved fig3_cryptic_position_map.png/.pdf")


# ============================================================================
# Figure 4 — Top 20 exons by cryptic fraction (WT sequences)
# ============================================================================
print("\n[Figure 4] Top 20 exons by cryptic fraction (WT)...")

cf_cols = [f"{cl}_cryptic_fraction" for cl in CELL_LINES
           if f"{cl}_cryptic_fraction" in seq_df.columns]
wt_df = seq_df[seq_df["seq_type"] == "none"].copy()
wt_df["avg_cryptic"] = wt_df[cf_cols].mean(axis=1)

# One row per WT exon (event_id)
exon_avg = (
    wt_df.groupby(["event_id", "gene_name"])["avg_cryptic"]
    .agg(mean="mean", sem="sem", count="count")
    .reset_index()
    .sort_values("mean", ascending=False)
    .head(10)
    .iloc[::-1]
)

# Attach hg38 exon start/end coordinates for labels
coord_map = (wt_psi_event[["event_id", "exon_start_hg38", "exon_end_hg38"]]
             .dropna(subset=["exon_start_hg38", "exon_end_hg38"])
             .drop_duplicates("event_id"))
exon_avg = exon_avg.merge(coord_map, on="event_id", how="left")
chrom = exon_avg["event_id"].str.extract(r"(chr[^:]+)", expand=False)
exon_labels = (exon_avg["gene_name"] + "  "
               + chrom + ":"
               + exon_avg["exon_start_hg38"].astype("Int64").astype(str) + "-"
               + exon_avg["exon_end_hg38"].astype("Int64").astype(str))

fig4, ax = plt.subplots(figsize=(7.5, 7))
y_pos = np.arange(len(exon_avg))
ax.barh(y_pos, exon_avg["mean"], xerr=exon_avg["sem"],
        color="#4878cf", alpha=0.85, edgecolor="white",
        error_kw=dict(lw=1.2, ecolor="black", capsize=3))
for yi, (_, row) in enumerate(exon_avg.iterrows()):
    ax.text(row["mean"] + row["sem"] + 0.005, yi,
            f"n={int(row['count'])}", va="center", fontsize=7.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(exon_labels, fontsize=8)
ax.axvline(0.1, color="#d62728", lw=1, linestyle="--", label="10% threshold")
ax.set_xlabel("Mean cryptic fraction (WT sequences, ± SEM)", fontsize=10)
ax.set_title("Top 10 exons by cryptic splice site usage\n(reference/WT sequences only)", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
fig4.savefig(SCRIPT_DIR / "fig4_top_exons.png", bbox_inches="tight", dpi=150)
fig4.savefig(SCRIPT_DIR / "fig4_top_exons.pdf", bbox_inches="tight")
plt.close(fig4)
print("  Saved fig4_top_exons.png/.pdf")


# ============================================================================
# Summary stats
# ============================================================================
print("\n=== Summary ===")
low  = var_seq[var_seq["wt_psi_avg"] < 0.2]["avg_cryptic_fraction"]
high = var_seq[var_seq["wt_psi_avg"] > 0.8]["avg_cryptic_fraction"]
_, pmw = stats.mannwhitneyu(low.dropna(), high.dropna(), alternative="two-sided")
print(f"Fig 1  Spearman r={r_sp:.3f}, p={p_sp:.1e}; "
      f"low-PSI median={low.median():.4f} vs high-PSI={high.median():.4f}, MW p={pmw:.1e}")
print("Fig 2  Mean cryptic fraction by class:")
for cls in present:
    print(f"       {CLASS_LABELS[cls]:22s}  "
          f"{single[single['var_class']==cls]['avg_cryptic_fraction'].mean():.4f}")
print("Done.")
