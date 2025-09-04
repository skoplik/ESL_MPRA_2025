import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import gridspec
import matplotlib as mpl
from scipy.stats import kruskal

# === Fonts: keep text editable in Illustrator
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# === Config
input_file = "/ESL/Figures_SK/sig_cell_types/out/exclusive_heatmaps/exclusive_psi_outliers_combined.tsv"
output_dir = "/ESL/Figures_SK/sig_cell_types/out/exclusive_heatmaps/recurrent_variant_positions_mingap"
os.makedirs(output_dir, exist_ok=True)

cell_lines = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]

# === Load data
df = pd.read_csv(input_file, sep="\t")
df["variant_hg38"] = df["variant_hg38"].fillna("WT")
df["event_id_str"] = df["event_id_str"].astype(str)

# Drop duplicate variants
df = df.drop_duplicates(subset=["Reference", "gene_exon", "snp"])

# === Format variant label using snp instead of variant_hg38
def format_variant_label(row):
    if row["snp"] in ["none", "WT", None, np.nan]:
        return row["gene_exon"] + "\nReference"
    return row["gene_exon"] + "\n" + row["snp"]

df["label"] = df.apply(format_variant_label, axis=1)

def variant_sort_key(row):
    if row["snp"] in ["none", "WT", None, np.nan]:
        return (-1, row["label"])  # WT first
    elif isinstance(row["snp"], str) and row["snp"].count(">") == 1:
        return (0, row["label"])   # SNV next
    else:
        return (1, row["label"])   # doubles last

# === Extract all individual positions from multi-mutation variant_hg38s
def extract_positions(variant_str):
    if variant_str in ["WT", "none"] or pd.isna(variant_str):
        return []
    return [x.split(">")[0] for x in variant_str.split(";") if ">" in x]

df["variant_positions"] = df["variant_hg38"].apply(extract_positions)

# === Explode to count per-position per-gene_exon usage
exploded = df.explode("variant_positions")
position_counts = exploded.groupby(["gene_exon", "variant_positions"]).size()
recurrent = position_counts[position_counts > 2].reset_index()
recurrent.columns = ["gene_exon", "variant_pos", "count"]

# === Add COLQ exon 5 manually to recurrent list
colq_variants = df[df["gene_exon"] == "COLQ exon 5"]
colq_positions = set(pos for poslist in colq_variants["variant_positions"] if isinstance(poslist, list) for pos in poslist)
for pos in colq_positions:
    recurrent = pd.concat([
        recurrent,
        pd.DataFrame([{"gene_exon": "COLQ exon 5", "variant_pos": pos, "count": 999}])
    ], ignore_index=True)

# Collect stats results
stats_records = []

# === Plot function with dynamic height (0.2 per row)
def plot_group_heatmap(group_df, title, outname, gene_exon, variant_pos):
    matrix = group_df.set_index("label")[cell_lines].dropna()

    # Compute Kruskal–Wallis p-values: each CL vs rest
    pvals = {}
    for cl in cell_lines:
        vals = matrix[cl].values
        others = matrix.drop(columns=[cl]).values.flatten()
        if len(vals) > 0 and len(others) > 0:
            _, pval = kruskal(vals, others)
        else:
            pval = np.nan
        pvals[cl] = pval
        stats_records.append({
            "gene_exon": gene_exon,
            "variant_pos": variant_pos,
            "cell_line": cl,
            "pval": pval
        })

    # Text annotation string for plot
    pval_text = ", ".join([f"{cl}: {pvals[cl]:.1e}" if not np.isnan(pvals[cl]) else f"{cl}: NA"
                           for cl in cell_lines])

    # Dynamic figure height: 0.2 per row + padding
    n_rows = len(matrix)
    fig_height = 0.2 * n_rows + 2
    fig = plt.figure(figsize=(8, fig_height))
    gs = gridspec.GridSpec(1, 2, width_ratios=[6, 0.3])

    ax = plt.subplot(gs[0])
    sns.heatmap(
        matrix,
        cmap="YlGnBu", vmin=0, vmax=1,
        cbar=False, xticklabels=True, yticklabels=True,
        ax=ax
    )
    ax.set_title(f"{title}\n{pval_text}", fontsize=9)
    ax.set_xlabel("Cell Line", fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(axis='x', labelrotation=0, labelsize=10)
    ax.tick_params(axis='y', labelsize=6)

    # Add colorbar
    cax = plt.subplot(gs[1])
    sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Average PSI", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + 0.25 * pos.height, pos.width, pos.height * 0.5])

    plt.tight_layout()

    basepath = os.path.join(output_dir, outname.replace(".pdf", ""))
    fig.savefig(basepath + ".pdf", dpi=300, format="pdf")
    fig.savefig(basepath + ".svg", dpi=300, format="svg")
    plt.close()
    print(f"Saved: {basepath}.pdf/svg with pvals {pval_text}")

# === Loop over recurrent gene_exon + position combos and plot heatmaps
for _, row in recurrent.iterrows():
    gene_exon = row["gene_exon"]
    variant_pos = row["variant_pos"]
    group = df[(df["gene_exon"] == gene_exon) &
               df["variant_positions"].apply(lambda pos_list: variant_pos in pos_list if isinstance(pos_list, list) else False)]
    wt_rows = df[(df["gene_exon"] == gene_exon) & (df["variant_hg38"].isin(["WT", "none"]))]
    combined = pd.concat([group, wt_rows], axis=0)
    combined["sort_key"] = combined.apply(variant_sort_key, axis=1)
    combined = combined.sort_values("sort_key").drop(columns="sort_key")

    safe_name = f"{gene_exon.replace(' ', '_').replace('/', '_')}_{variant_pos.replace(':', '_')}.pdf"
    title = f"{gene_exon}, {variant_pos} (n={len(group)})"
    plot_group_heatmap(combined, title, safe_name, gene_exon, variant_pos)

# Save all p-values to TSV
stats_outfile = os.path.join(output_dir, "recurrent_variant_pvals_kruskal.tsv")
pd.DataFrame(stats_records).to_csv(stats_outfile, sep="\t", index=False)
print(f"Saved all Kruskal–Wallis p-values to {stats_outfile}")
