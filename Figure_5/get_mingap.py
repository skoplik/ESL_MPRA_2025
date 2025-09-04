import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import gridspec
import matplotlib.patches as patches
import matplotlib as mpl

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

# === Paths
psi_table_path = "/ESL/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WITH_WT.csv"
meta_table_path = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv"
output_dir = "/ESL/Figures_SK/sig_cell_types/out/exclusive_heatmaps/mingap"
os.makedirs(output_dir, exist_ok=True)

# === Load
psi_df = pd.read_csv(psi_table_path)
meta_df = pd.read_csv(meta_table_path)
psi_df["Reference"] = pd.to_numeric(psi_df["Reference"], errors="coerce")
meta_df["Reference"] = pd.to_numeric(meta_df["Reference"], errors="coerce") + 1

# === Compute avg PSI
avg_psi_df = psi_df[["Reference"]].copy()
for cl in cell_lines:
    rep_cols = replicate_cols[cl]
    rep_cols_existing = [c for c in rep_cols if c in psi_df.columns]
    avg_psi_df[cl] = psi_df[rep_cols_existing].mean(axis=1, skipna=True)

# === Merge metadata
annot_df = avg_psi_df.merge(
    meta_df[["Reference", "gene_exon", "variant_hg38", "snp", "event_id"]],
    on="Reference", how="left"
)
annot_df["event_id_str"] = annot_df["event_id"].astype(str).str.strip()
annot_df = annot_df.dropna(subset=cell_lines).copy()
annot_df[cell_lines] = annot_df[cell_lines].clip(1e-2, 1 - 1e-2)

# === Compute mingap
psi_vals = annot_df[cell_lines].to_numpy()
psi_sorted = np.sort(psi_vals, axis=1)
gap_high = psi_sorted[:, -1] - psi_sorted[:, -2]
gap_low = psi_sorted[:, 1] - psi_sorted[:, 0]
mingap = np.maximum(gap_high, gap_low)
annot_df["mingap"] = mingap
annot_df["max_cl"] = psi_vals.argmax(axis=1).astype(int)
annot_df["min_cl"] = psi_vals.argmin(axis=1).astype(int)
annot_df["max_cl"] = annot_df["max_cl"].map(lambda i: cell_lines[i])
annot_df["min_cl"] = annot_df["min_cl"].map(lambda i: cell_lines[i])

# === Select top 20 gap-high and gap-low variants per cell line with threshold
top_rows = []
for cl in cell_lines:
    df_high = annot_df.copy()
    df_high["gap_high"] = psi_sorted[:, -1] - psi_sorted[:, -2]
    df_high = df_high[(df_high["max_cl"] == cl) & (df_high["gap_high"] >= 0.25)]
    df_high["gap_type"] = "high"
    df_high["exclusive_cl"] = cl
    df_high["mingap"] = df_high["gap_high"]
    top_high = df_high.sort_values("mingap", ascending=False).head(20)
    top_rows.append(top_high)

    df_low = annot_df.copy()
    df_low["gap_low"] = psi_sorted[:, 1] - psi_sorted[:, 0]
    df_low = df_low[(df_low["min_cl"] == cl) & (df_low["gap_low"] >= 0.25)]
    df_low["gap_type"] = "low"
    df_low["exclusive_cl"] = cl
    df_low["mingap"] = df_low["gap_low"]
    top_low = df_low.sort_values("mingap", ascending=False).head(20)
    top_rows.append(top_low)

# === Combine and sort
combined_df = pd.concat(top_rows, ignore_index=True)
combined_df["psi_pattern"] = combined_df["gap_type"].map({"high": "Exclusive High", "low": "Exclusive Low"})
# Use relative SNP for labels, not hg38
combined_df["label"] = combined_df["gene_exon"] + "\n" + combined_df["snp"].fillna("NA")
combined_df["exclusive_cl"] = pd.Categorical(combined_df["exclusive_cl"], categories=cell_lines, ordered=True)
combined_df = combined_df.sort_values(["exclusive_cl", "gap_type", "mingap"], ascending=[True, True, False])

high_df = combined_df[combined_df["gap_type"] == "high"].copy()
low_df = combined_df[combined_df["gap_type"] == "low"].copy()

# === Plotting
def plot_exclusive_heatmap(df, title, filename, show_gene_labels=False):
    x_labels = ["HEK293" if x == "HEK" else x for x in cell_lines]
    grouped_labels = []
    for cl in cell_lines:
        group = df[df["exclusive_cl"] == cl].copy()
        if cl in group.columns:
            group = group.sort_values(
                by=cl,
                ascending=True if title.lower().startswith("top high") or "high" in title.lower() else False
            )
        labels = group["label"].tolist()
        grouped_labels.extend(labels)

    matrix = df.set_index("label").loc[grouped_labels][cell_lines].dropna()
    matrix = matrix[::-1]

    exclusive_cl_map = df.set_index("label")["exclusive_cl"].to_dict()
    row_colors = pd.Series(grouped_labels[::-1]).map(lambda label: {
        "HEK": "#E984B6", "HeLa": "#7FBE7E", "K562": "#F9AE33",
        "MCF7": "#807CB9", "HMC3": "#EF4025"
    }.get(exclusive_cl_map.get(label), "grey"))
    row_colors.index = matrix.index

    # dynamic height based on number of rows
    n_rows = len(matrix)
    fig_height = min(max(3, 0.3 * n_rows), 40)  # scale with row count
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
        matrix,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar=False,
        xticklabels=x_labels,
        yticklabels=matrix.index if show_gene_labels else False,
        ax=ax_main
    )
    ax_main.set_title(title, fontsize=12)
    ax_main.set_xlabel("Cell Line", fontsize=12)
    ax_main.tick_params(axis='x', labelrotation=0, labelsize=15)
    if show_gene_labels:
        ax_main.tick_params(axis='y', labelsize=8)

    grouped_labels_series = pd.Series(grouped_labels[::-1], name="label")
    df_for_rect = df.set_index("label").loc[grouped_labels_series]
    df_for_rect["row_idx"] = range(len(df_for_rect))

    for cl in cell_lines:
        cl_rows = df_for_rect[df_for_rect["exclusive_cl"] == cl]
        if cl_rows.empty:
            continue
        col_idx = cell_lines.index(cl)
        top = cl_rows["row_idx"].min()
        bottom = cl_rows["row_idx"].max()
        height = bottom - top + 1
        rect = patches.Rectangle(
            (col_idx, top), 1, height,
            linewidth=1.5, edgecolor='black', facecolor='none',
            clip_on=False, zorder=10
        )
        ax_main.add_patch(rect)

    cbar_ax = plt.subplot(gs[2])
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Average PSI", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()

    basepath = os.path.join(output_dir, filename.replace(".pdf", ""))
    fig.savefig(basepath + ".pdf", dpi=300, format="pdf")
    fig.savefig(basepath + ".svg", dpi=300, format="svg")
    plt.close()
    print("Saved:", basepath + ".pdf and .svg")



# === Plot top 20 with y labels
plot_exclusive_heatmap(high_df, "Top High Mingap Sequences by Cell Type", "top20_gap_high_per_cell_filtered.pdf", show_gene_labels=True)
plot_exclusive_heatmap(low_df, "Top Low Mingap Sequences by Cell Type", "top20_gap_low_per_cell_filtered.pdf", show_gene_labels=True)

# === Export
combined_df.to_csv(os.path.join(output_dir, "top20_mingap_variants_filtered.tsv"), sep="\t", index=False)
print("Exported combined top 20 per-cell-line gap variants with mingap ≥ 0.25")

# === Export + plot all variants with mingap ≥ 0.25
all_df = annot_df.copy()
all_df["gap_high"] = psi_sorted[:, -1] - psi_sorted[:, -2]
all_df["gap_low"] = psi_sorted[:, 1] - psi_sorted[:, 0]
all_df["gap_type"] = np.where(all_df["gap_high"] >= all_df["gap_low"], "high", "low")
all_df["exclusive_cl"] = np.where(all_df["gap_type"] == "high", all_df["max_cl"], all_df["min_cl"])
all_df = all_df[all_df["mingap"] >= 0.25].copy()
all_df["psi_pattern"] = all_df["gap_type"].map({"high": "Exclusive High", "low": "Exclusive Low"})
# use relative SNP here too for labels
all_df["label"] = all_df["gene_exon"] + "\n" + all_df["snp"].fillna("NA")

all_df.to_csv(os.path.join(output_dir, "mingap_variants_all_filtered.tsv"), sep="\t", index=False)
print("Exported all variants with mingap ≥ 0.25")

# === New: plot all ≥ 0.25 high and low separately (without y labels to stay compact)
plot_exclusive_heatmap(
    all_df[all_df["gap_type"] == "high"],
    "All High Mingap Variants (mingap ≥ 0.25)",
    "all_gap_high_filtered.pdf",
    show_gene_labels=False
)
plot_exclusive_heatmap(
    all_df[all_df["gap_type"] == "low"],
    "All Low Mingap Variants (mingap ≥ 0.25)",
    "all_gap_low_filtered.pdf",
    show_gene_labels=False
)
