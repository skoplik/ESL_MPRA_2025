"""
Cryptic Junction Usage — Three Targeted Analyses
=================================================

1. Cryptic fraction vs. reference (WT) PSI
   Do variants at low-PSI exons generate more cryptic junction reads?

2. Cryptic fraction by variant position class
   Splice-site (SD/SA) variants vs. exonic vs. deep-intronic variants.

3. Cryptic fraction vs. Δlogit(PSI)
   Do strongly skipping variants (large negative Δlogit) show higher cryptic usage?

Inputs (pre-computed in this directory):
  - cryptic_junctions_per_sequence.csv
  - variant_vs_wt_cryptic_comparison.csv

Also loads the main COMPASS PSI table for WT PSI values.

Outputs:
  - fig1_cryptic_vs_wt_psi.png/.pdf
  - fig2_cryptic_by_variant_class.png/.pdf
  - fig3_cryptic_vs_delta_logit.png/.pdf
"""

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
SEQ_CSV = SCRIPT_DIR / "cryptic_junctions_per_sequence.csv"
VAR_CSV = SCRIPT_DIR / "variant_vs_wt_cryptic_comparison.csv"
PSI_CSV = Path(
    "/ESL/Figures_SK/General_preprocessing/output_03_16_2026/"
    "03_16_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"
)

CELL_LINES = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]

# Splice-site window (nt from intron/exon boundary on each side)
SS_WINDOW = 4  # last 4 nt of intron + first 4 nt of exon (or vice versa for 5'SS)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def logit_clip(psi, eps=0.01):
    psi = np.clip(psi, eps, 1 - eps)
    return np.log(psi / (1 - psi))


def avg_cryptic_fraction(df, cell_lines=CELL_LINES):
    """Mean cryptic fraction across all available cell lines (per row)."""
    cols = [f"{cl}_cryptic_fraction" for cl in cell_lines if f"{cl}_cryptic_fraction" in df.columns]
    return df[cols].mean(axis=1)


# ---------------------------------------------------------------------------
# Variant position classifier
# ---------------------------------------------------------------------------

def classify_variant_position(snp, intron1_len, exon_len):
    """
    Return one of: '3SS', '5SS', 'exonic', 'intronic_up', 'intronic_dn', 'double'
    for a single snp field value like '42:G>T' or '42:G>T;88:A>C'.

    Coordinates are 0-indexed from the start of intron1:
      [0 .. intron1_len-1]  = intron1
      [intron1_len .. intron1_len+exon_len-1] = exon
      [intron1_len+exon_len .. ] = intron2
    """
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
    exon_end = intron1_len + exon_len  # exclusive

    # 3'SS window: end of intron1 & start of exon
    three_ss_start = max(0, exon_start - SS_WINDOW)
    three_ss_end = exon_start + SS_WINDOW  # exclusive

    # 5'SS window: end of exon & start of intron2
    five_ss_start = max(0, exon_end - SS_WINDOW)
    five_ss_end = exon_end + SS_WINDOW

    if three_ss_start <= pos < three_ss_end:
        return "3SS"
    elif five_ss_start <= pos < five_ss_end:
        return "5SS"
    elif exon_start <= pos < exon_end:
        return "exonic"
    elif pos < exon_start:
        return "intronic_up"
    else:
        return "intronic_dn"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading cryptic junction data...")
seq_df = pd.read_csv(SEQ_CSV)
var_df = pd.read_csv(VAR_CSV)

print(f"  seq_df: {len(seq_df):,} sequences")
print(f"  var_df: {len(var_df):,} variant-cellline rows")

# Average cryptic fraction for seq_df
seq_df["avg_cryptic_fraction"] = avg_cryptic_fraction(seq_df)

# ---------------------------------------------------------------------------
# Load WT PSI from main CSV
# ---------------------------------------------------------------------------
print("Loading WT PSI from main COMPASS table...")
wt_psi_cols = ["Reference", "event_id"] + [f"{cl}_wt_pooled_psi_raw" for cl in CELL_LINES]
main_df = pd.read_csv(PSI_CSV, usecols=lambda c: c in wt_psi_cols, low_memory=False)

# One WT PSI row per event_id (take first, they should all be the same)
wt_psi = main_df.dropna(subset=["event_id"])
# Build an event-level WT PSI table (average across all sequences sharing an event_id)
wt_psi_event = (
    wt_psi.groupby("event_id")[
        [c for c in wt_psi.columns if "wt_pooled_psi_raw" in c]
    ]
    .first()
    .reset_index()
)
# Average WT PSI across cell lines
wt_psi_event["wt_psi_avg"] = wt_psi_event[
    [c for c in wt_psi_event.columns if "wt_pooled_psi_raw" in c]
].mean(axis=1)

print(f"  WT PSI events: {len(wt_psi_event):,}")

# ---------------------------------------------------------------------------
# Figure 1 — Cryptic fraction vs. reference (WT) PSI
# ---------------------------------------------------------------------------
print("\n[Figure 1] Cryptic fraction vs. reference PSI...")

# Merge per-sequence cryptic data with WT PSI
# seq_df has event_id for variant and WT sequences; for the relationship we want
# variants' cryptic fraction vs. the WT PSI of their exon.
# Use wt_cryptic_fraction (from var_df) and compute the WT PSI separately.

# Build per-event WT cryptic fraction from seq_df (seq_type == 'none')
wt_seq = seq_df[seq_df["seq_type"] == "none"][["event_id", "avg_cryptic_fraction"]].rename(
    columns={"avg_cryptic_fraction": "wt_avg_cryptic"}
)

# For variants, get their average cryptic fraction
var_seq = seq_df[seq_df["seq_type"] != "none"][["Parent_ref", "event_id", "avg_cryptic_fraction", "snp",
                                                   "exon", "intron1", "intron2", "seq_type", "source"]].copy()

# Merge with WT PSI and WT cryptic fraction
var_seq = var_seq.merge(wt_psi_event[["event_id", "wt_psi_avg"]], on="event_id", how="left")
var_seq = var_seq.merge(wt_seq, on="event_id", how="left")

# Drop rows without WT PSI
var_seq = var_seq.dropna(subset=["wt_psi_avg", "avg_cryptic_fraction"])
print(f"  Variants with PSI data: {len(var_seq):,}")

# Bin WT PSI into deciles
var_seq["wt_psi_bin"] = pd.qcut(var_seq["wt_psi_avg"], q=10, labels=False, duplicates="drop")
bin_labels = var_seq.groupby("wt_psi_bin")["wt_psi_avg"].median()

fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))

# Left: boxplot of cryptic fraction by PSI decile
ax = axes1[0]
bin_data = [
    var_seq.loc[var_seq["wt_psi_bin"] == b, "avg_cryptic_fraction"].dropna().values
    for b in sorted(var_seq["wt_psi_bin"].dropna().unique())
]
bp = ax.boxplot(
    bin_data,
    positions=range(len(bin_data)),
    widths=0.6,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", lw=1.5),
)
cmap = plt.cm.RdYlGn
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(cmap(i / len(bp["boxes"])))
    patch.set_alpha(0.8)
ax.set_xticks(range(len(bin_data)))
ax.set_xticklabels([f"{v:.2f}" for v in bin_labels.values], rotation=45, fontsize=7)
ax.set_xlabel("Median WT PSI (decile bin)")
ax.set_ylabel("Avg cryptic fraction (variant)")
ax.set_title("Cryptic usage by WT PSI decile\n(variant sequences)", fontsize=10)

# Right: median cryptic fraction per decile + spearman r
decile_median = var_seq.groupby("wt_psi_bin")["avg_cryptic_fraction"].median()
ax2 = axes1[1]
ax2.scatter(bin_labels.values, decile_median.values, s=60, color="#4878cf", zorder=3)
ax2.plot(bin_labels.values, decile_median.values, color="#4878cf", lw=1.5, alpha=0.7)

# Overall Spearman
r_sp, p_sp = stats.spearmanr(var_seq["wt_psi_avg"], var_seq["avg_cryptic_fraction"],
                               nan_policy="omit")
ax2.text(0.05, 0.95, f"Spearman r = {r_sp:.3f}\np = {p_sp:.2e}",
         transform=ax2.transAxes, fontsize=9, va="top")
ax2.set_xlabel("WT PSI (median per decile)")
ax2.set_ylabel("Median cryptic fraction (variant)")
ax2.set_title("Trend: cryptic usage vs. reference PSI", fontsize=10)
ax2.axhline(var_seq["avg_cryptic_fraction"].median(), color="gray", lw=0.8, linestyle="--",
            label="Overall median")
ax2.legend(fontsize=8)

fig1.suptitle("Do variants at low-PSI exons show more cryptic junction reads?", fontsize=11, y=1.02)
plt.tight_layout()
fig1.savefig(SCRIPT_DIR / "fig1_cryptic_vs_wt_psi.png", bbox_inches="tight", dpi=150)
fig1.savefig(SCRIPT_DIR / "fig1_cryptic_vs_wt_psi.pdf", bbox_inches="tight")
plt.close(fig1)
print("  Saved fig1_cryptic_vs_wt_psi.png/.pdf")

# ---------------------------------------------------------------------------
# Figure 2 — Cryptic fraction by variant position class
# ---------------------------------------------------------------------------
print("\n[Figure 2] Cryptic fraction by variant position class...")

# Classify position for single-variant sequences
single = seq_df[seq_df["seq_type"] == "single"].copy()
single = single.dropna(subset=["snp", "intron1", "exon"])

single["intron1_len"] = single["intron1"].str.len()
single["exon_len"] = single["exon"].str.len()
single["var_class"] = single.apply(
    lambda r: classify_variant_position(r["snp"], r["intron1_len"], r["exon_len"]), axis=1
)

# Average cryptic fraction already on single
single = single.dropna(subset=["avg_cryptic_fraction"])
print(f"  Single-variant class counts:\n{single['var_class'].value_counts()}")

CLASS_ORDER = ["3SS", "5SS", "exonic", "intronic_up", "intronic_dn"]
CLASS_LABELS = {
    "3SS": "3'SS",
    "5SS": "5'SS",
    "exonic": "Exonic",
    "intronic_up": "Intronic\n(upstream)",
    "intronic_dn": "Intronic\n(downstream)",
}
CLASS_COLORS = {
    "3SS":          "#d62728",
    "5SS":          "#ff7f0e",
    "exonic":       "#2ca02c",
    "intronic_up":  "#1f77b4",
    "intronic_dn":  "#9467bd",
}

present_classes = [c for c in CLASS_ORDER if c in single["var_class"].unique()]

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# Left: violin + median scatter
ax = axes2[0]
plot_data_list = [
    single.loc[single["var_class"] == c, "avg_cryptic_fraction"].values
    for c in present_classes
]
vp = ax.violinplot(plot_data_list, positions=range(len(present_classes)),
                   showmedians=True, showextrema=False, widths=0.65)
for i, (body, cls) in enumerate(zip(vp["bodies"], present_classes)):
    body.set_facecolor(CLASS_COLORS[cls])
    body.set_alpha(0.7)
    body.set_rasterized(True)
vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(1.5)

ax.set_xticks(range(len(present_classes)))
ax.set_xticklabels([CLASS_LABELS[c] for c in present_classes], fontsize=9)
ax.set_ylabel("Avg cryptic fraction")
ax.set_title("Cryptic usage by variant position class\n(single variants)", fontsize=10)
ax.set_yscale("log")

# Right: median + CI with sample sizes annotated
medians = [np.median(d) for d in plot_data_list]
sems = [stats.sem(d) for d in plot_data_list]
ns = [len(d) for d in plot_data_list]

ax2 = axes2[1]
x = range(len(present_classes))
ax2.bar(x, medians, color=[CLASS_COLORS[c] for c in present_classes],
        alpha=0.8, edgecolor="white", yerr=sems, capsize=4, error_kw=dict(lw=1))
for i, (n, med) in enumerate(zip(ns, medians)):
    ax2.text(i, med + sems[i] * 1.2, f"n={n:,}", ha="center", va="bottom", fontsize=7)
ax2.set_xticks(list(x))
ax2.set_xticklabels([CLASS_LABELS[c] for c in present_classes], fontsize=9)
ax2.set_ylabel("Median cryptic fraction ± SEM")
ax2.set_title("Median cryptic fraction by variant class", fontsize=10)

# Pairwise Mann-Whitney vs 5SS (the most activating)
ref_cls = "5SS" if "5SS" in present_classes else present_classes[0]
ref_data = single.loc[single["var_class"] == ref_cls, "avg_cryptic_fraction"].values
print(f"  Mann-Whitney vs '{ref_cls}':")
for cls in present_classes:
    if cls == ref_cls:
        continue
    cls_data = single.loc[single["var_class"] == cls, "avg_cryptic_fraction"].values
    u, p = stats.mannwhitneyu(ref_data, cls_data, alternative="greater")
    print(f"    {ref_cls} vs {cls}: U={u:.0f}, p={p:.3e}")

fig2.suptitle("Are splice-site variants more likely to generate cryptic junctions?", fontsize=11, y=1.02)
plt.tight_layout()
fig2.savefig(SCRIPT_DIR / "fig2_cryptic_by_variant_class.png", bbox_inches="tight", dpi=150)
fig2.savefig(SCRIPT_DIR / "fig2_cryptic_by_variant_class.pdf", bbox_inches="tight")
plt.close(fig2)
print("  Saved fig2_cryptic_by_variant_class.png/.pdf")

# ---------------------------------------------------------------------------
# Figure 3 — Cryptic fraction vs. Δlogit(PSI)
# ---------------------------------------------------------------------------
print("\n[Figure 3] Cryptic fraction vs. Δlogit(PSI)...")

# var_df has one row per (Parent_ref, cell_line) with cryptic_fraction and
# per-cell-line delta_logit in wide format. Melt to long for per-CL analysis.
# Build long format: match each row's cell_line to the correct delta_logit column.

rows = []
for cl in CELL_LINES:
    dl_col = f"{cl}_delta_logit_pooled"
    if dl_col not in var_df.columns:
        continue
    sub = var_df[var_df["cell_line"] == cl][
        ["Parent_ref", "cell_line", "cryptic_fraction", "source", dl_col]
    ].dropna(subset=["cryptic_fraction", dl_col])
    sub = sub.rename(columns={dl_col: "delta_logit"})
    rows.append(sub)

long_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
print(f"  Usable variant-cellline rows: {len(long_df):,}")

ncols = min(len(CELL_LINES), 3)
nrows = int(np.ceil(len(CELL_LINES) / ncols))
fig3, axes3 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
axes3_flat = np.array(axes3).flatten()

SOURCE_COLOR = {
    "clinvar_single": "#d62728",
    "clinvar_double": "#9467bd",
    "exac":           "#1f77b4",
    "geuvadis":       "#ff7f0e",
}

for i, cl in enumerate(CELL_LINES):
    ax = axes3_flat[i]
    sub = long_df[long_df["cell_line"] == cl]
    if len(sub) == 0:
        ax.set_visible(False)
        continue

    colors = sub["source"].map(lambda s: SOURCE_COLOR.get(str(s).lower(), "#7f7f7f"))

    ax.scatter(sub["delta_logit"], sub["cryptic_fraction"],
               c=colors, s=5, alpha=0.25, linewidths=0, rasterized=True)

    # Spearman r (only negative Δlogit = skipping)
    skipping = sub[sub["delta_logit"] < 0]
    if len(skipping) > 10:
        r_neg, p_neg = stats.spearmanr(skipping["delta_logit"], skipping["cryptic_fraction"],
                                        nan_policy="omit")
        ax.text(0.05, 0.95,
                f"Skipping only\nSpearman r = {r_neg:.3f}\np = {p_neg:.2e}\nn = {len(skipping):,}",
                transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.8))

    # Highlight large Δlogit skippers with elevated cryptic fraction
    top = sub[(sub["delta_logit"] < -1) & (sub["cryptic_fraction"] > 0.1)]
    frac_elevated = len(top) / max(len(sub[sub["delta_logit"] < -1]), 1)
    ax.text(0.98, 0.95,
            f"SDV skippers with\ncryptic > 10%: {frac_elevated:.1%}",
            transform=ax.transAxes, fontsize=7, va="top", ha="right",
            color="#d62728")

    ax.axvline(-1, color="#d62728", lw=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0,  color="black",   lw=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel(f"Δlogit(PSI) [{cl}]", fontsize=9)
    ax.set_ylabel("Cryptic fraction", fontsize=9)
    ax.set_title(cl, fontsize=10)

# Remove unused axes
for ax in axes3_flat[len(CELL_LINES):]:
    ax.set_visible(False)

# Shared legend
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=v, markersize=7, label=k.replace("_", " "))
    for k, v in SOURCE_COLOR.items()
]
fig3.legend(handles=legend_elems, loc="lower center", ncol=4, fontsize=8,
            bbox_to_anchor=(0.5, -0.04))

fig3.suptitle("Do strongly skipping variants show more cryptic junction reads?", fontsize=11, y=1.01)
plt.tight_layout()
fig3.savefig(SCRIPT_DIR / "fig3_cryptic_vs_delta_logit.png", bbox_inches="tight", dpi=150)
fig3.savefig(SCRIPT_DIR / "fig3_cryptic_vs_delta_logit.pdf", bbox_inches="tight")
plt.close(fig3)
print("  Saved fig3_cryptic_vs_delta_logit.png/.pdf")

# ---------------------------------------------------------------------------
# Summary statistics printout
# ---------------------------------------------------------------------------
print("\n=== Summary Statistics ===")

print("\n--- Figure 1: Cryptic fraction vs. WT PSI ---")
low_psi = var_seq[var_seq["wt_psi_avg"] < 0.2]["avg_cryptic_fraction"].dropna()
high_psi = var_seq[var_seq["wt_psi_avg"] > 0.8]["avg_cryptic_fraction"].dropna()
print(f"  Low PSI (<0.2)  median cryptic fraction: {low_psi.median():.4f}  (n={len(low_psi):,})")
print(f"  High PSI (>0.8) median cryptic fraction: {high_psi.median():.4f}  (n={len(high_psi):,})")
u_stat, p_mw = stats.mannwhitneyu(low_psi, high_psi, alternative="two-sided")
print(f"  Mann-Whitney p (low vs high PSI): {p_mw:.3e}")
print(f"  Overall Spearman r (PSI vs cryptic): {r_sp:.3f}, p={p_sp:.2e}")

print("\n--- Figure 2: Cryptic fraction by variant class ---")
for cls in present_classes:
    d = single.loc[single["var_class"] == cls, "avg_cryptic_fraction"]
    print(f"  {CLASS_LABELS[cls]:20s}  median={d.median():.4f}  n={len(d):,}")

print("\n--- Figure 3: Cryptic usage in large-Δlogit skippers ---")
for cl in CELL_LINES:
    sub = long_df[long_df["cell_line"] == cl]
    if len(sub) == 0:
        continue
    sdv = sub[sub["delta_logit"] < -1]
    elev = sdv[sdv["cryptic_fraction"] > 0.1]
    if len(sdv) > 0:
        print(f"  {cl}: {len(elev):,}/{len(sdv):,} SDV skippers have cryptic > 10% ({100*len(elev)/len(sdv):.1f}%)")

print("\nDone.")
