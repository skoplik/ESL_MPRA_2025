import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import re

# === Load CSVs ===
df = pd.read_csv("/ESL/Figures_SK/effect_size_final/out_08_10_2025/subsample.csv")
st = pd.read_csv("/ESL/Figures_SK/General_preprocessing/fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv", dtype=str)

# === Map event_id to gene_exon ===
event_to_label = st.drop_duplicates(subset="event_id")[["event_id", "gene_exon"]].set_index("event_id")["gene_exon"]

# === Define cell lines ===
cell_lines = ["HEK", "HeLa", "K562", "HMC3", "MCF7"]
effect_cols = [f"Effect Size {c} Boot 0" for c in cell_lines]

# === Melt for plotting ===
df_melted = df.melt(
    id_vars=["motif", "event_id"],
    value_vars=effect_cols,
    var_name="cell_line",
    value_name="effect_size"
)
df_melted["cell_line"] = df_melted["cell_line"].str.extract(r"Effect Size (.+) Boot 0")
df_melted = df_melted.dropna(subset=["effect_size"])
df_melted["effect_size"] = pd.to_numeric(df_melted["effect_size"], errors="coerce")

# Add gene_exon label
df_melted["gene_exon"] = df_melted["event_id"].map(event_to_label)
df_melted = df_melted.dropna(subset=["gene_exon"])

# === Color settings (keep cmap normal, centered at 0) ===
true_vmin = -2.8809031703175343
true_vmax = 1.9402510339372532
norm = mcolors.TwoSlopeNorm(vmin=true_vmin, vcenter=0, vmax=true_vmax)
cmap = plt.get_cmap("coolwarm")
tick_values = [-2, -1, 0, 1, 2]
tick_labels = ["–2", "–1", "0", "+1", "+2"]

# === Plot each cluster (all cell lines) ===
for cluster in ["cluster_023", "cluster_050", "cluster_021", "cluster_079"]:
    cluster_num = re.sub(r"^cluster_0*", "", cluster)

    df_cluster = df_melted[df_melted["motif"] == cluster].copy()

    # Compute mean per gene_exon and sort ascending
    mean_df = df_cluster.groupby("gene_exon")["effect_size"].mean().reset_index()
    mean_df = mean_df.sort_values("effect_size", ascending=False)
    mean_df["color"] = mean_df["effect_size"].map(lambda x: cmap(norm(x)))
    df_cluster["gene_exon"] = pd.Categorical(df_cluster["gene_exon"], categories=mean_df["gene_exon"], ordered=True)

    # === Start plot ===
    fig, ax = plt.subplots(figsize=(10, 7))

    # Bars
    for _, row in mean_df.iterrows():
        ax.barh(
            y=row["gene_exon"],
            width=row["effect_size"],
            color=row["color"],
            edgecolor="black",
            linewidth=1,
            zorder=1
        )

    # Dots
    sns.stripplot(
        data=df_cluster,
        y="gene_exon",
        x="effect_size",
        hue="cell_line",
        dodge=True,
        size=9,
        jitter=0.2,
        marker='o',
        linewidth=0.5,
        edgecolor='black',
        palette='tab10',
        ax=ax,
        orient='h',
        zorder=10
    )

    # Axes and labels
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlim(true_vmin, true_vmax)
    ax.set_title(f"Motif Cluster {cluster_num}", fontsize=25)
    ax.set_xlabel("Effect Size (Δlogit PSI)", fontsize=25)
    ax.set_ylabel("Exon Families", fontsize=26)
    ax.tick_params(labelsize=20)

    # Thicken plot border
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    # === Custom short colorbar with fixed ticks ===
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    # Position and size (30% height)
    box = ax.get_position()
    cbar_height = 0.3 * box.height
    cbar_bottom = box.y0 + box.height * 0.35
    cbar_ax = fig.add_axes([box.x1 + 0.03, cbar_bottom, 0.02, cbar_height])
    
    cbar = plt.colorbar(sm, cax=cbar_ax, ticks=tick_values)
    cbar.set_label("Mean Effect Size", rotation=270, labelpad=20, fontsize=14)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=12)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(1)

    # Legend
    ax.legend(
        title="Cell Line",
        bbox_to_anchor=(1.26, 1),
        loc="upper left",
        fontsize=12,
        title_fontsize=14,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(f"/ESL/Figures_SK/effect_size_final/out_08_10_2025/plots/example_concordance/example_{cluster}.pdf", bbox_inches='tight')
    plt.close()

# === Plot each cluster (HEK only) ===
for cluster in ["cluster_023", "cluster_050", "cluster_021", "cluster_079"]:
    cluster_num = re.sub(r"^cluster_0*", "", cluster)

    # Filter for this cluster and just HEK
    df_cluster = df_melted[(df_melted["motif"] == cluster) & (df_melted["cell_line"] == "HEK")].copy()

    # Compute mean per gene_exon (only HEK now) and sort
    mean_df = df_cluster.groupby("gene_exon")["effect_size"].mean().reset_index()
    mean_df = mean_df.sort_values("effect_size", ascending=True)
    mean_df["color"] = mean_df["effect_size"].map(lambda x: cmap(norm(x)))
    df_cluster["gene_exon"] = pd.Categorical(df_cluster["gene_exon"], categories=mean_df["gene_exon"], ordered=True)

    # Scale height based on number of exon families
    num_exons = len(mean_df)
    row_height = 0.5
    fig_height = max(3, row_height * num_exons)
    
    fig, ax = plt.subplots(figsize=(6, fig_height))
    
    # Bars for mean effect size
    for _, row in mean_df.iterrows():
        ax.barh(
            y=row["gene_exon"],
            width=row["effect_size"],
            color=row["color"],
            edgecolor="black",
            linewidth=1,
            zorder=1, 
            height=0.7
        )

    # Axes and labels
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlim(true_vmin, true_vmax)
    ax.set_title(f"Motif Cluster {cluster_num} – HEK", fontsize=20)
    ax.set_xlabel("Effect Size (Δlogit PSI)", fontsize=20)
    ax.set_ylabel("Exon Families", fontsize=20)
    ax.tick_params(labelsize=20)

    # Thicken plot border
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    # Colorbar (same fix)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    box = ax.get_position()
    cbar_height = 0.3 * box.height
    cbar_bottom = box.y0 + box.height * 0.35
    cbar_ax = fig.add_axes([box.x1 + 1, cbar_bottom, 0.02, cbar_height])
    cbar = plt.colorbar(sm, cax=cbar_ax, ticks=tick_values)
    cbar.set_label("Mean Effect Size", rotation=270, labelpad=20, fontsize=16)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=12)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(1)

    plt.tight_layout()
    plt.savefig(
        f"/ESL/Figures_SK/effect_size_final/out_08_10_2025/plots/example_concordance/example_{cluster}_HEK.pdf",
        bbox_inches='tight'
    )
    plt.close()
