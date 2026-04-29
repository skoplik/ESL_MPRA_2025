import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from matplotlib import gridspec
import matplotlib.patches as patches
from statsmodels.stats.multitest import multipletests

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# === Config
cell_lines = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]
replicate_cols = {
    "HEK":  ["HEK_rep1_psi_raw", "HEK_rep2_psi_raw", "HEK_rep3_psi_raw", "HEK_rep4_psi_raw"],
    "HeLa": ["HeLa_rep1_psi_raw", "HeLa_rep2_psi_raw"],
    "K562": ["K562_rep1_psi_raw", "K562_rep2_psi_raw"],
    "MCF7": ["MCF7_rep1_psi_raw", "MCF7_rep2_psi_raw"],
    "HMC3": ["HMC3_rep1_psi_raw", "HMC3_rep2_psi_raw"],
}

SWEEP_STEP           = 0.05
HIGHLIGHT_THRESHOLD  = 0.25  # original threshold, shown on sweep for reference

# === Paths
anova_path      = "/ESL/ESL_MPRA/Figure_7/outputs/mingap/anova_stats_all_variants.tsv"
psi_table_path  = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz"
meta_table_path = "/ESL/ESL_MPRA/Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25.csv.gz"
output_dir      = "/ESL/ESL_MPRA/Figure_7/outputs/mingap_redo"
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# === STEP 1: Load precomputed ANOVA stats + recompute mingap
# =============================================================================

print("=" * 60)
print("[STEP 1] Loading precomputed ANOVA stats...")
anova_df = pd.read_csv(anova_path, sep="\t")
anova_df["Reference"] = pd.to_numeric(anova_df["Reference"], errors="coerce")
print(f"  {len(anova_df)} variants, {anova_df['anova_pval'].notna().sum()} testable")

print("  Recomputing avg PSI and mingap...")
psi_df  = pd.read_csv(psi_table_path)
meta_df = pd.read_csv(meta_table_path)
psi_df["Reference"]  = pd.to_numeric(psi_df["Reference"],  errors="coerce")
meta_df["Reference"] = pd.to_numeric(meta_df["Reference"], errors="coerce") + 1

avg_psi_df = psi_df[["Reference"]].copy()
for cl in cell_lines:
    cols = [c for c in replicate_cols[cl] if c in psi_df.columns]
    avg_psi_df[cl] = psi_df[cols].mean(axis=1, skipna=True)

avg_psi_df = avg_psi_df.merge(
    meta_df[["Reference", "gene_exon", "snp"]], on="Reference", how="left"
).dropna(subset=cell_lines)
avg_psi_df[cell_lines] = avg_psi_df[cell_lines].clip(1e-2, 1 - 1e-2)

psi_vals   = avg_psi_df[cell_lines].to_numpy()
psi_sorted = np.sort(psi_vals, axis=1)
gap_high   = psi_sorted[:, -1] - psi_sorted[:, -2]
gap_low    = psi_sorted[:, 1]  - psi_sorted[:, 0]
avg_psi_df["mingap"]   = np.maximum(gap_high, gap_low)
avg_psi_df["gap_high"] = gap_high
avg_psi_df["gap_low"]  = gap_low
avg_psi_df["max_cl"]   = pd.Series(psi_vals.argmax(axis=1)).map(lambda i: cell_lines[i]).values
avg_psi_df["min_cl"]   = pd.Series(psi_vals.argmin(axis=1)).map(lambda i: cell_lines[i]).values

# Merge ANOVA pvals in
merged_df = avg_psi_df.merge(
    anova_df[["Reference", "anova_F", "anova_pval"]], on="Reference", how="left"
)
print(f"  Merged. {len(merged_df)} variants with mingap computed.")
print(f"  Mingap range: {merged_df['mingap'].min():.4f} - {merged_df['mingap'].max():.4f}  "
      f"|  median: {merged_df['mingap'].median():.4f}")

# =============================================================================
# === STEP 2: Sensitivity sweep
# For each threshold, apply BH within the passing set and record stats.
# =============================================================================

print("=" * 60)
print("[STEP 2] Running sensitivity sweep...")

max_mingap    = merged_df["mingap"].max()
thresholds    = np.arange(0.0, max_mingap + SWEEP_STEP, SWEEP_STEP)
sweep_records = []

for thresh in thresholds:
    sub_idx = merged_df.index[
        (merged_df["mingap"] >= thresh) & merged_df["anova_pval"].notna()
    ]
    n_total = len(sub_idx)
    if n_total == 0:
        sweep_records.append({"threshold": thresh, "n_total": 0, "n_sig": 0, "prop_sig": np.nan})
        continue

    pvals_sub            = merged_df.loc[sub_idx, "anova_pval"].values
    reject_sub, _, _, _  = multipletests(pvals_sub, method="fdr_bh")
    n_sig = int(reject_sub.sum())

    sweep_records.append({
        "threshold": round(float(thresh), 4),
        "n_total":   n_total,
        "n_sig":     n_sig,
        "prop_sig":  n_sig / n_total,
    })

sweep_df = pd.DataFrame(sweep_records)
sweep_df.to_csv(os.path.join(output_dir, "mingap_sensitivity_sweep.tsv"), sep="\t", index=False)
print("  Exported: mingap_sensitivity_sweep.tsv")
print(sweep_df.to_string(index=False))

# === Find threshold where n_sig first equals n_total (all passing variants significant)
converge_row = sweep_df[sweep_df["n_sig"] == sweep_df["n_total"]]
if not converge_row.empty:
    converge_thresh = converge_row.iloc[0]["threshold"]
    converge_n      = int(converge_row.iloc[0]["n_total"])
    print(f"\n  Convergence (n_sig == n_total) first reached at mingap >= {converge_thresh:.2f}  "
          f"(n = {converge_n})")
else:
    converge_thresh = None
    print("\n  Lines do not fully converge within the swept range.")

# =============================================================================
# === STEP 3: Sensitivity sweep figure
# =============================================================================

print("=" * 60)
print("[STEP 3] Plotting sensitivity sweep...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: proportion significant
ax = axes[0]
ax.plot(sweep_df["threshold"], sweep_df["prop_sig"], color="#4C72B0", lw=2)
ax.axvline(HIGHLIGHT_THRESHOLD, color="red", linestyle="--", lw=1.5,
           label=f"Original threshold ({HIGHLIGHT_THRESHOLD})")
if converge_thresh is not None:
    ax.axvline(converge_thresh, color="green", linestyle=":", lw=1.5,
               label=f"Convergence threshold ({converge_thresh:.2f})")
ax.set_xlabel("Mingap threshold", fontsize=12)
ax.set_ylabel("Proportion ANOVA significant\n(BH padj < 0.05, within passing set)", fontsize=11)
ax.set_title("Sensitivity: proportion significant\nvs. mingap threshold", fontsize=11)
ax.set_xlim(0, max_mingap)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)

# Panel B: absolute counts
ax2 = axes[1]
ax2.plot(sweep_df["threshold"], sweep_df["n_total"], color="#AAAAAA", lw=2, label="N passing mingap")
ax2.plot(sweep_df["threshold"], sweep_df["n_sig"],   color="#E84B23", lw=2, label="N ANOVA significant")
ax2.axvline(HIGHLIGHT_THRESHOLD, color="red", linestyle="--", lw=1.5,
            label=f"Original threshold ({HIGHLIGHT_THRESHOLD})")
if converge_thresh is not None:
    ax2.axvline(converge_thresh, color="green", linestyle=":", lw=1.5,
                label=f"Convergence ({converge_thresh:.2f}, n={converge_n})")
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
# === STEP 4: Ask user to confirm threshold, then apply BH and generate heatmaps
#
# Default: use convergence threshold if found, else fall back to HIGHLIGHT_THRESHOLD.
# Override by setting FINAL_THRESHOLD manually below.
# =============================================================================

FINAL_THRESHOLD = converge_thresh if converge_thresh is not None else HIGHLIGHT_THRESHOLD
print("=" * 60)
print(f"[STEP 4] Using final mingap threshold: {FINAL_THRESHOLD}")
print(f"  (Edit FINAL_THRESHOLD in the script to override.)")

# Apply BH within the final passing set
final_idx = merged_df.index[
    (merged_df["mingap"] >= FINAL_THRESHOLD) & merged_df["anova_pval"].notna()
]
pvals_final          = merged_df.loc[final_idx, "anova_pval"].values
reject_final, padj_final, _, _ = multipletests(pvals_final, method="fdr_bh")

merged_df["anova_padj"]       = np.nan
merged_df["anova_significant"] = False
merged_df.loc[final_idx, "anova_padj"]       = padj_final
merged_df.loc[final_idx, "anova_significant"] = reject_final

n_pass  = (merged_df["mingap"] >= FINAL_THRESHOLD).sum()
n_sig   = merged_df["anova_significant"].sum()
print(f"  Mingap >= {FINAL_THRESHOLD}:              {n_pass}")
print(f"  Mingap >= {FINAL_THRESHOLD} AND BH sig:   {n_sig}")
print(f"  Excluded (not BH sig):          {n_pass - n_sig}")

# =============================================================================
# === STEP 5: Select top 20 per cell line and generate heatmaps
# =============================================================================

print("=" * 60)
print("[STEP 5] Selecting top 20 per cell line...")

top_rows = []
for cl in cell_lines:
    df_high = merged_df[
        (merged_df["max_cl"] == cl) &
        (merged_df["gap_high"] >= FINAL_THRESHOLD) &
        merged_df["anova_significant"]
    ].copy()
    df_high["gap_type"]     = "high"
    df_high["exclusive_cl"] = cl
    df_high["mingap"]       = df_high["gap_high"]
    top_rows.append(df_high.sort_values("mingap", ascending=False).head(20))

    df_low = merged_df[
        (merged_df["min_cl"] == cl) &
        (merged_df["gap_low"] >= FINAL_THRESHOLD) &
        merged_df["anova_significant"]
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

print("  Calls by cell line:")
for cl in cell_lines:
    sub = combined_df[combined_df["exclusive_cl"] == cl]
    print(f"    {cl:6s}: n={len(sub)}")

# =============================================================================
# === STEP 6: Heatmap plotting
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
print("[STEP 6] Plotting heatmaps...")
thresh_str = str(FINAL_THRESHOLD).replace(".", "p")
plot_exclusive_heatmap(
    high_df,
    f"Top High Mingap Sequences by Cell Type (mingap >= {FINAL_THRESHOLD})",
    f"top20_gap_high_mingap{thresh_str}.pdf",
    show_gene_labels=True
)
plot_exclusive_heatmap(
    low_df,
    f"Top Low Mingap Sequences by Cell Type (mingap >= {FINAL_THRESHOLD})",
    f"top20_gap_low_mingap{thresh_str}.pdf",
    show_gene_labels=True
)

# All-variants heatmaps
all_cts_df = merged_df[merged_df["anova_significant"]].copy()
all_cts_df["gap_type"]     = np.where(all_cts_df["gap_high"] >= all_cts_df["gap_low"], "high", "low")
all_cts_df["exclusive_cl"] = np.where(all_cts_df["gap_type"] == "high", all_cts_df["max_cl"], all_cts_df["min_cl"])
all_cts_df["psi_pattern"]  = all_cts_df["gap_type"].map({"high": "Exclusive High", "low": "Exclusive Low"})
all_cts_df["label"]        = all_cts_df["gene_exon"] + "\n" + all_cts_df["snp"].fillna("NA")

plot_exclusive_heatmap(
    all_cts_df[all_cts_df["gap_type"] == "high"],
    f"All High Mingap Variants (mingap >= {FINAL_THRESHOLD}, ANOVA sig)",
    f"all_gap_high_mingap{thresh_str}.pdf",
    show_gene_labels=False
)
plot_exclusive_heatmap(
    all_cts_df[all_cts_df["gap_type"] == "low"],
    f"All Low Mingap Variants (mingap >= {FINAL_THRESHOLD}, ANOVA sig)",
    f"all_gap_low_mingap{thresh_str}.pdf",
    show_gene_labels=False
)

# =============================================================================
# === STEP 7: Exports
# =============================================================================

print("=" * 60)
print("[STEP 7] Exporting tables...")

combined_df.to_csv(
    os.path.join(output_dir, f"top20_mingap{thresh_str}_variants.tsv"), sep="\t", index=False
)
all_cts_df.to_csv(
    os.path.join(output_dir, f"all_mingap{thresh_str}_variants.tsv"), sep="\t", index=False
)
print(f"  Exported: top20_mingap{thresh_str}_variants.tsv")
print(f"  Exported: all_mingap{thresh_str}_variants.tsv")

print("=" * 60)
print("Done.")