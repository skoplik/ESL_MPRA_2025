import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib as mpl
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# === Fonts: keep text editable in Illustrator
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# === Config
cell_lines = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]
replicate_cols = {
    "HEK": ["HEK_rep1_psi_raw", "HEK_rep2_psi_raw", "HEK_rep3_psi_raw", "HEK_rep4_psi_raw"],
    "HeLa": ["HeLa_rep1_psi_raw", "HeLa_rep2_psi_raw"],
    "K562": ["K562_rep1_psi_raw", "K562_rep2_psi_raw"],
    "MCF7": ["MCF7_rep1_psi_raw", "MCF7_rep2_psi_raw"],
    "HMC3": ["HMC3_rep1_psi_raw", "HMC3_rep2_psi_raw"]
}

MINGAP_THRESHOLD = 0.25  # primary threshold
SWEEP_STEP       = 0.05  # step size for sensitivity sweep
N_BASELINE       = 200   # bottom-mingap variants for negative control figure

# === Paths
psi_table_path  = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz"
meta_table_path = "/ESL/ESL_MPRA/Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25.csv.gz"
output_dir      = "/ESL/ESL_MPRA/Figure_5/outputs/mingap"
os.makedirs(output_dir, exist_ok=True)

# === Load
psi_df  = pd.read_csv(psi_table_path)
meta_df = pd.read_csv(meta_table_path)
psi_df["Reference"]  = pd.to_numeric(psi_df["Reference"],  errors="coerce")
meta_df["Reference"] = pd.to_numeric(meta_df["Reference"], errors="coerce") + 1

# === Compute avg PSI
avg_psi_df = psi_df[["Reference"]].copy()
for cl in cell_lines:
    rep_cols_existing = [c for c in replicate_cols[cl] if c in psi_df.columns]
    avg_psi_df[cl] = psi_df[rep_cols_existing].mean(axis=1, skipna=True)

# === Merge metadata
annot_df = avg_psi_df.merge(
    meta_df[["Reference", "gene_exon", "variant_hg38", "snp", "event_id"]],
    on="Reference", how="left"
)
annot_df["event_id_str"] = annot_df["event_id"].astype(str).str.strip()
annot_df = annot_df.dropna(subset=cell_lines).copy()
annot_df[cell_lines] = annot_df[cell_lines].clip(1e-2, 1 - 1e-2)

# =============================================================================
# === STEP 1: ONE-WAY ANOVA on ALL variants (raw p stored only)
#
# For each variant, groups = per-cell-line replicates (raw PSI, NaN-dropped).
# Replicate count per cell line varies per variant; scipy handles unbalanced
# groups natively. Cell lines with < 2 valid replicates for a given variant
# are excluded from that variant's ANOVA. Variants with < 2 qualifying groups
# are untestable and receive NaN.
#
# BH FDR correction is applied later, scoped to the mingap-passing set only.
# This avoids penalizing the correction pool with the majority of variants
# that show no cell-type-specific behavior by design.
# =============================================================================

def run_per_variant_anova(psi_df, replicate_cols, cell_lines, ref_col="Reference"):
    """
    One-way ANOVA (cell line as factor) for every row in psi_df.
    Returns raw F and p only -- significance is determined downstream
    after BH correction within the mingap-passing set.

    Columns returned:
        Reference, anova_F, anova_pval,
        between_cl_variance, mean_within_replicate_variance
    """
    records = []
    for _, row in tqdm(psi_df.iterrows(), total=len(psi_df), desc="  ANOVA per variant"):
        groups = []
        for cl in cell_lines:
            cols = [c for c in replicate_cols[cl] if c in psi_df.columns]
            vals = row[cols].dropna().values.astype(float)
            if len(vals) >= 2:
                groups.append(vals)

        if len(groups) < 2:
            records.append({
                ref_col: row[ref_col],
                "anova_F": np.nan,
                "anova_pval": np.nan,
                "between_cl_variance": np.nan,
                "mean_within_replicate_variance": np.nan,
            })
            continue

        f_stat, p_val = stats.f_oneway(*groups)

        group_means     = np.array([g.mean() for g in groups])
        between_var     = np.var(group_means, ddof=1)
        within_vars     = [np.var(g, ddof=1) for g in groups if len(g) > 1]
        mean_within_var = np.mean(within_vars) if within_vars else np.nan

        records.append({
            ref_col: row[ref_col],
            "anova_F": f_stat,
            "anova_pval": p_val,
            "between_cl_variance": between_var,
            "mean_within_replicate_variance": mean_within_var,
        })

    return pd.DataFrame(records)


# Replicate count distribution -- for methods section
print("=" * 60)
print("Replicate count distribution per cell line (across all variants):")
for cl in cell_lines:
    cols   = [c for c in replicate_cols[cl] if c in psi_df.columns]
    counts = psi_df[cols].notna().sum(axis=1).value_counts().sort_index()
    print(f"  {cl:6s}: {dict(counts)}  (n valid replicates -> n variants)")

print("=" * 60)
print(f"[STEP 1] Running per-variant one-way ANOVA on {len(psi_df)} variants...")
anova_results = run_per_variant_anova(psi_df, replicate_cols, cell_lines)

n_tested = anova_results["anova_pval"].notna().sum()
print(f"  Done. {n_tested} variants testable, {len(anova_results) - n_tested} skipped (< 2 groups).")

annot_df = annot_df.merge(anova_results, on="Reference", how="left")

# =============================================================================
# === STEP 2: Compute mingap
# =============================================================================

print("=" * 60)
print("[STEP 2] Computing mingap...")
psi_vals   = annot_df[cell_lines].to_numpy()
psi_sorted = np.sort(psi_vals, axis=1)
gap_high   = psi_sorted[:, -1] - psi_sorted[:, -2]
gap_low    = psi_sorted[:, 1]  - psi_sorted[:, 0]
mingap     = np.maximum(gap_high, gap_low)
annot_df["mingap"]   = mingap
annot_df["gap_high"] = gap_high
annot_df["gap_low"]  = gap_low
annot_df["max_cl"]   = pd.Series(psi_vals.argmax(axis=1)).map(lambda i: cell_lines[i]).values
annot_df["min_cl"]   = pd.Series(psi_vals.argmin(axis=1)).map(lambda i: cell_lines[i]).values
print(f"  Mingap range: {mingap.min():.4f} - {mingap.max():.4f}  |  median: {np.median(mingap):.4f}")

# =============================================================================
# === STEP 3: BH FDR correction within the mingap-passing set
#
# Scoping BH to variants that already pass mingap >= MINGAP_THRESHOLD keeps
# the correction pool relevant -- we are asking "of variants with a meaningful
# PSI gap, which are statistically significant?" rather than correcting across
# the full universe of mostly-flat variants.
# =============================================================================

print("=" * 60)
print(f"[STEP 3] Applying BH FDR correction within mingap >= {MINGAP_THRESHOLD} set...")

pass_mask = annot_df["mingap"] >= MINGAP_THRESHOLD
pass_idx  = annot_df.index[pass_mask & annot_df["anova_pval"].notna()]

pvals_pass          = annot_df.loc[pass_idx, "anova_pval"].values
reject, padj, _, _  = multipletests(pvals_pass, method="fdr_bh")

annot_df["anova_padj"]        = np.nan
annot_df["anova_significant"]  = False
annot_df.loc[pass_idx, "anova_padj"]       = padj
annot_df.loc[pass_idx, "anova_significant"] = reject

n_pass_mingap = pass_mask.sum()
n_cts         = annot_df["anova_significant"].sum()
print(f"  Mingap >= {MINGAP_THRESHOLD}:                   {n_pass_mingap}")
print(f"  Mingap >= {MINGAP_THRESHOLD} AND ANOVA BH sig:  {n_cts}")
print(f"  Mingap >= {MINGAP_THRESHOLD} but NOT sig:       {n_pass_mingap - n_cts}  (excluded)")

annot_df["cell_type_specific"] = annot_df["anova_significant"].astype(bool)

# =============================================================================
# === STEP 4: Sensitivity sweep -- proportion ANOVA-significant vs. mingap threshold
#
# For each threshold from 0 to max_mingap by SWEEP_STEP:
#   - Select variants with mingap >= threshold
#   - Apply BH within that set
#   - Record: n variants, n significant, proportion significant
# Produces a line plot showing how significance enrichment changes with threshold.
# The primary threshold (0.25) is highlighted.
# =============================================================================

print("=" * 60)
print("[STEP 4] Running sensitivity sweep...")

max_mingap   = annot_df["mingap"].max()
thresholds   = np.arange(0.0, max_mingap + SWEEP_STEP, SWEEP_STEP)
sweep_records = []

for thresh in tqdm(thresholds, desc="  Sweep"):
    sub_idx = annot_df.index[
        (annot_df["mingap"] >= thresh) & annot_df["anova_pval"].notna()
    ]
    n_total = len(sub_idx)
    if n_total == 0:
        sweep_records.append({"threshold": thresh, "n_total": 0, "n_sig": 0, "prop_sig": np.nan})
        continue

    pvals_sub          = annot_df.loc[sub_idx, "anova_pval"].values
    reject_sub, _, _, _ = multipletests(pvals_sub, method="fdr_bh")
    n_sig = reject_sub.sum()

    sweep_records.append({
        "threshold": thresh,
        "n_total":   n_total,
        "n_sig":     n_sig,
        "prop_sig":  n_sig / n_total,
    })

sweep_df = pd.DataFrame(sweep_records)
sweep_df.to_csv(os.path.join(output_dir, "mingap_sensitivity_sweep.tsv"), sep="\t", index=False)
print("  Exported: mingap_sensitivity_sweep.tsv")

# --- Sensitivity sweep figure ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: proportion significant
ax = axes[0]
ax.plot(sweep_df["threshold"], sweep_df["prop_sig"], color="#4C72B0", lw=2)
ax.axvline(MINGAP_THRESHOLD, color="red", linestyle="--", lw=1.5,
           label=f"Primary threshold ({MINGAP_THRESHOLD})")
ax.set_xlabel("Mingap threshold", fontsize=12)
ax.set_ylabel("Proportion ANOVA significant (BH padj < 0.05)", fontsize=12)
ax.set_title("Sensitivity: proportion significant\nvs. mingap threshold", fontsize=11)
ax.set_xlim(0, max_mingap)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)

# Panel B: absolute counts (n total and n significant)
ax2 = axes[1]
ax2.plot(sweep_df["threshold"], sweep_df["n_total"], color="#AAAAAA", lw=2, label="N passing mingap")
ax2.plot(sweep_df["threshold"], sweep_df["n_sig"],   color="#E84B23", lw=2, label="N ANOVA significant")
ax2.axvline(MINGAP_THRESHOLD, color="red", linestyle="--", lw=1.5,
            label=f"Primary threshold ({MINGAP_THRESHOLD})")
ax2.set_xlabel("Mingap threshold", fontsize=12)
ax2.set_ylabel("Number of variants", fontsize=12)
ax2.set_title("Sensitivity: variant counts\nvs. mingap threshold", fontsize=11)
ax2.set_xlim(0, max_mingap)
ax2.legend(fontsize=9)

plt.tight_layout()
basepath = os.path.join(output_dir, "mingap_sensitivity_sweep")
fig.savefig(basepath + ".pdf", dpi=300, format="pdf")
fig.savefig(basepath + ".svg", dpi=300, format="svg")
plt.close()
print(f"  Saved: {basepath}.pdf and .svg")

# =============================================================================
# === STEP 5: Select top 20 per cell line from the dual-filtered set
# =============================================================================

print("=" * 60)
print("[STEP 5] Selecting top 20 per cell line...")
top_rows = []
for cl in cell_lines:
    df_high = annot_df[
        (annot_df["max_cl"] == cl) &
        (annot_df["gap_high"] >= MINGAP_THRESHOLD) &
        annot_df["anova_significant"]
    ].copy()
    df_high["gap_type"]     = "high"
    df_high["exclusive_cl"] = cl
    df_high["mingap"]       = df_high["gap_high"]
    top_rows.append(df_high.sort_values("mingap", ascending=False).head(20))

    df_low = annot_df[
        (annot_df["min_cl"] == cl) &
        (annot_df["gap_low"] >= MINGAP_THRESHOLD) &
        annot_df["anova_significant"]
    ].copy()
    df_low["gap_type"]     = "low"
    df_low["exclusive_cl"] = cl
    df_low["mingap"]       = df_low["gap_low"]
    top_rows.append(df_low.sort_values("mingap", ascending=False).head(20))

combined_df = pd.concat(top_rows, ignore_index=True)
combined_df["psi_pattern"]  = combined_df["gap_type"].map({"high": "Exclusive High", "low": "Exclusive Low"})
combined_df["label"]        = combined_df["gene_exon"] + "\n" + combined_df["snp"].fillna("NA")
combined_df["exclusive_cl"] = pd.Categorical(combined_df["exclusive_cl"], categories=cell_lines, ordered=True)
combined_df = combined_df.sort_values(["exclusive_cl", "gap_type", "mingap"], ascending=[True, True, False])

high_df = combined_df[combined_df["gap_type"] == "high"].copy()
low_df  = combined_df[combined_df["gap_type"] == "low"].copy()

print("  Cell-type-specific calls (top 20 per cell line):")
for cl in cell_lines:
    sub = combined_df[combined_df["exclusive_cl"] == cl]
    print(f"    {cl:6s}: n={len(sub)}  median F={sub['anova_F'].median():.2f}  "
          f"median padj={sub['anova_padj'].median():.2e}")

# =============================================================================
# === STEP 6: Diagnostic figure -- passing set vs. low-mingap baseline
# =============================================================================

baseline_df = (
    annot_df[annot_df["mingap"] < MINGAP_THRESHOLD]
    .sort_values("mingap")
    .head(N_BASELINE)
    .copy()
)
print(f"\n  Baseline: bottom {N_BASELINE} by mingap  |  "
      f"median mingap={baseline_df['mingap'].median():.4f}  |  "
      f"ANOVA sig in baseline: {baseline_df['anova_significant'].sum()} / {len(baseline_df)}")

def plot_anova_summary(passing_df, baseline_df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: variance components scatter
    ax = axes[0]
    for df_sub, color, label in [
        (baseline_df, "#AAAAAA", f"Low mingap baseline (n={len(baseline_df)})"),
        (passing_df,  "#E84B23", f"Mingap >= {MINGAP_THRESHOLD} (n={len(passing_df)})"),
    ]:
        dp = df_sub.dropna(subset=["between_cl_variance", "mean_within_replicate_variance"])
        ax.scatter(dp["mean_within_replicate_variance"], dp["between_cl_variance"],
                   c=color, alpha=0.5, s=15, linewidths=0, label=label)
    max_val = max(
        passing_df[["between_cl_variance", "mean_within_replicate_variance"]].max().max(),
        baseline_df[["between_cl_variance", "mean_within_replicate_variance"]].max().max(),
    )
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="y = x")
    ax.set_xlabel("Mean within-replicate variance", fontsize=12)
    ax.set_ylabel("Between-cell-line variance", fontsize=12)
    ax.set_title("Variance components:\ncell-type-specific vs. low-mingap baseline", fontsize=11)
    ax.legend(fontsize=8)

    # Panel B: -log10(padj) for passing set, -log10(pval) for baseline
    # (baseline has no padj since BH was not applied to it)
    ax2 = axes[1]
    baseline_vals = -np.log10(baseline_df["anova_pval"].dropna().clip(lower=1e-300))
    passing_vals  = -np.log10(passing_df["anova_padj"].dropna().clip(lower=1e-300))
    ax2.hist(baseline_vals, bins=40, color="#AAAAAA", alpha=0.6, edgecolor="white",
             linewidth=0.3, label=f"Low mingap baseline: -log10(p) (n={len(baseline_df)})", density=True)
    ax2.hist(passing_vals, bins=40, color="#4C72B0", alpha=0.6, edgecolor="white",
             linewidth=0.3, label=f"Mingap >= {MINGAP_THRESHOLD}: -log10(padj) (n={len(passing_df)})", density=True)
    ax2.axvline(-np.log10(0.05), color="red", linestyle="--", lw=1.5,
                label="0.05 cutoff")
    ax2.set_xlabel("-log10(p or BH padj)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("ANOVA significance:\ncell-type-specific vs. low-mingap baseline", fontsize=11)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    basepath = os.path.join(output_dir, "anova_summary_diagnostics")
    fig.savefig(basepath + ".pdf", dpi=300, format="pdf")
    fig.savefig(basepath + ".svg", dpi=300, format="svg")
    plt.close()
    print(f"  Saved: {basepath}.pdf and .svg")


print("=" * 60)
print("[STEP 6] Generating ANOVA diagnostic figure...")
passing_all_df = annot_df[annot_df["mingap"] >= MINGAP_THRESHOLD].copy()
plot_anova_summary(passing_all_df, baseline_df, output_dir)

# =============================================================================
# === STEP 7: Heatmap plotting
# =============================================================================

def plot_exclusive_heatmap(df, title, filename, show_gene_labels=False):
    x_labels = ["HEK293" if x == "HEK" else x for x in cell_lines]
    grouped_labels = []
    for cl in cell_lines:
        group = df[df["exclusive_cl"] == cl].copy()
        if cl in group.columns:
            group = group.sort_values(by=cl, ascending="high" in title.lower())
        grouped_labels.extend(group["label"].tolist())

    matrix = df.set_index("label").loc[grouped_labels][cell_lines].dropna()
    matrix = matrix[::-1]

    cl_colors = {"HEK": "#E984B6", "HeLa": "#7FBE7E", "K562": "#F9AE33",
                 "MCF7": "#807CB9", "HMC3": "#EF4025"}
    exclusive_cl_map = df.set_index("label")["exclusive_cl"].to_dict()
    row_colors = pd.Series(grouped_labels[::-1]).map(
        lambda lbl: cl_colors.get(exclusive_cl_map.get(lbl), "grey")
    )
    row_colors.index = matrix.index

    fig_height = min(max(3, 0.3 * len(matrix)), 40)
    fig = plt.figure(figsize=(9, fig_height))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 6, 0.4])

    ax_color = plt.subplot(gs[0])
    for i, color in enumerate(reversed(row_colors.tolist())):
        ax_color.add_patch(patches.Rectangle((0, i), 1, 1, color=color))
    ax_color.set_xlim(0, 1)
    ax_color.set_ylim(0, len(row_colors))
    ax_color.axis("off")

    ax_main = plt.subplot(gs[1])
    sns.heatmap(
        matrix, cmap="YlGnBu", vmin=0, vmax=1, cbar=False,
        xticklabels=x_labels,
        yticklabels=matrix.index if show_gene_labels else False,
        ax=ax_main
    )
    ax_main.set_title(title, fontsize=12)
    ax_main.set_xlabel("Cell Line", fontsize=12)
    ax_main.tick_params(axis='x', labelrotation=0, labelsize=15)
    if show_gene_labels:
        ax_main.tick_params(axis='y', labelsize=8)

    df_for_rect = df.set_index("label").loc[pd.Series(grouped_labels[::-1])].copy()
    df_for_rect["row_idx"] = range(len(df_for_rect))
    for cl in cell_lines:
        cl_rows = df_for_rect[df_for_rect["exclusive_cl"] == cl]
        if cl_rows.empty:
            continue
        col_idx = cell_lines.index(cl)
        ax_main.add_patch(patches.Rectangle(
            (col_idx, cl_rows["row_idx"].min()), 1,
            cl_rows["row_idx"].max() - cl_rows["row_idx"].min() + 1,
            linewidth=1.5, edgecolor='black', facecolor='none',
            clip_on=False, zorder=10
        ))

    cbar_ax = plt.subplot(gs[2])
    sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Average PSI", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    basepath = os.path.join(output_dir, filename.replace(".pdf", ""))
    fig.savefig(basepath + ".pdf", dpi=300, format="pdf")
    fig.savefig(basepath + ".svg", dpi=300, format="svg")
    plt.close()
    print("  Saved:", basepath + ".pdf and .svg")


print("=" * 60)
print("[STEP 7] Plotting heatmaps...")
plot_exclusive_heatmap(high_df, "Top High Mingap Sequences by Cell Type",
                       "top20_gap_high_per_cell_filtered.pdf", show_gene_labels=True)
plot_exclusive_heatmap(low_df,  "Top Low Mingap Sequences by Cell Type",
                       "top20_gap_low_per_cell_filtered.pdf", show_gene_labels=True)

# =============================================================================
# === STEP 8: Exports
# =============================================================================

print("=" * 60)
print("[STEP 8] Exporting tables...")

combined_df.to_csv(
    os.path.join(output_dir, "top20_mingap_variants_filtered.tsv"), sep="\t", index=False
)
print("  Exported: top20_mingap_variants_filtered.tsv")

all_cts_df = annot_df[annot_df["cell_type_specific"]].copy()
all_cts_df["gap_type"]     = np.where(all_cts_df["gap_high"] >= all_cts_df["gap_low"], "high", "low")
all_cts_df["exclusive_cl"] = np.where(all_cts_df["gap_type"] == "high", all_cts_df["max_cl"], all_cts_df["min_cl"])
all_cts_df["psi_pattern"]  = all_cts_df["gap_type"].map({"high": "Exclusive High", "low": "Exclusive Low"})
all_cts_df["label"]        = all_cts_df["gene_exon"] + "\n" + all_cts_df["snp"].fillna("NA")
all_cts_df.to_csv(
    os.path.join(output_dir, "mingap_variants_all_filtered.tsv"), sep="\t", index=False
)
print("  Exported: mingap_variants_all_filtered.tsv")

anova_results.merge(
    meta_df[["Reference", "gene_exon", "variant_hg38", "snp"]], on="Reference", how="left"
).sort_values("anova_pval").to_csv(
    os.path.join(output_dir, "anova_stats_all_variants.tsv"), sep="\t", index=False
)
print("  Exported: anova_stats_all_variants.tsv")

baseline_df.to_csv(
    os.path.join(output_dir, "anova_baseline_low_mingap.tsv"), sep="\t", index=False
)
print("  Exported: anova_baseline_low_mingap.tsv")

plot_exclusive_heatmap(
    all_cts_df[all_cts_df["gap_type"] == "high"],
    "All High Mingap Variants (mingap >= 0.25, ANOVA sig)",
    "all_gap_high_filtered.pdf", show_gene_labels=False
)
plot_exclusive_heatmap(
    all_cts_df[all_cts_df["gap_type"] == "low"],
    "All Low Mingap Variants (mingap >= 0.25, ANOVA sig)",
    "all_gap_low_filtered.pdf", show_gene_labels=False
)

print("=" * 60)
print("Done.")