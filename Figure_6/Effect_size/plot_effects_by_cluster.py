import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def assign_random_colormap_colors(cluster_list, cmap_name="gist_rainbow", seed=42):
    """
    Assign shuffled random colors from a continuous colormap to each cluster.
    """
    np.random.seed(seed)
    cmap = plt.get_cmap(cmap_name)
    
    # Generate N random float values in [0, 1]
    values = np.random.rand(len(cluster_list))
    
    # Map to colors from the colormap
    colors = [cmap(v) for v in values]
    
    # Assign to clusters in original order
    return {cluster: color for cluster, color in zip(cluster_list, colors)}

def plot_effect_size_per_cell_line(df, output_prefix):
    cell_lines = ["HEK", "HeLa", "K562", "HMC3", "MCF7"]
    mean_cols = [f"Mean {cell} Average" for cell in cell_lines]

    for col in mean_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["MeanAcrossCellLines"] = df[mean_cols].mean(axis=1, skipna=True)
    cluster_means = df.groupby("motif")["MeanAcrossCellLines"].mean().sort_values()
    cluster_order = cluster_means.index.tolist()

    # Assign colors in order (low to high)
    cluster_to_color = assign_random_colormap_colors(cluster_order, cmap_name="turbo")


    for cell in cell_lines:
        colname = f"Mean {cell} Average"
        df_cell = df[["motif", "event_id", colname]].dropna(subset=[colname]).copy()
        df_cell["effect"] = df_cell[colname].astype(float)
        df_cell["color"] = df_cell["motif"].map(cluster_to_color)
        df_cell = df_cell.dropna(subset=["color"])

        plt.figure(figsize=(12, max(6, len(cluster_order) * 0.2)))
        for i, cluster in enumerate(cluster_order):
            sub = df_cell[df_cell["motif"] == cluster]
            for _, row in sub.iterrows():
                plt.plot(
                    row["effect"], i,
                    "o",
                    color=row["color"],
                    markersize=10,
                    alpha=0.8
                )

        plt.yticks(range(len(cluster_order)), [c.replace("cluster_", "cl_") for c in cluster_order], fontsize=10)
        plt.xlabel(f"{cell} Effect Size", fontsize=12)
        plt.title(f"Motif Cluster Effects in {cell}", fontsize=14)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.ylim(-1,len(cluster_order))
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1.5, zorder=0)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_per_cluster_{cell}.pdf")
        plt.close()


def plot_effect_size_summary_across_cell_lines(df, output_prefix):
    cell_lines = ["HEK", "HeLa", "K562", "HMC3", "MCF7"]
    mean_cols = [f"Mean {cell} Average" for cell in cell_lines]

    for col in mean_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Mean_Effect"] = df[mean_cols].mean(axis=1, skipna=True)
    df["SD_Effect"] = df[mean_cols].std(axis=1, skipna=True)

    cluster_means = df.groupby("motif")["Mean_Effect"].mean().sort_values()
    cluster_order = cluster_means.index.tolist()

    # Assign colors in order (low to high)
    cluster_to_color = assign_random_colormap_colors(cluster_order, cmap_name="turbo")
    df["color"] = df["motif"].map(cluster_to_color)
    df = df.dropna(subset=["color"])

    plt.figure(figsize=(12, max(6, len(cluster_order) * 0.2)))
    for i, cluster in enumerate(cluster_order):
        sub = df[df["motif"] == cluster]
        for _, row in sub.iterrows():
            plt.errorbar(
                row["Mean_Effect"], i,
                xerr=row["SD_Effect"],
                fmt="o",
                color=row["color"],
                markersize=10,
                capsize=3,
                alpha=0.8
            )

    plt.yticks(range(len(cluster_order)), [c.replace("cluster_", "cl_") for c in cluster_order], fontsize=10)
    plt.xlabel("Mean Effect Size (± SD across cell lines)", fontsize=12)
    plt.title("Motif Cluster Effects (All Cell Lines)", fontsize=14)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.ylim(-1,len(cluster_order))
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1.5, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_per_cluster_ALL.pdf")
    plt.close()


# === Load data ===
effects_file = "/ESL/ESL_MPRA/Figure_6/Effect_size/outputs/out_05_19_2025_boots_0_exon.csv"
df_effects = pd.read_csv(effects_file)

# === Output prefix for the plots ===
output_prefix = "/ESL/ESL_MPRA/Figure_6/Effect_size/outputs/plots/effects_colored_by_cluster/exon"

# === Generate all plots ===
plot_effect_size_per_cell_line(df_effects, output_prefix)
plot_effect_size_summary_across_cell_lines(df_effects, output_prefix)


# === Load data ===
effects_file = "/ESL/ESL_MPRA/Figure_6/Effect_size/outputs/out_05_19_2025_boots_0_intron1.csv"
df_effects = pd.read_csv(effects_file)

# === Output prefix for the plots ===
output_prefix = "/ESL/ESL_MPRA/Figure_6/Effect_size/outputs/plots/effects_colored_by_cluster/intron1"

# === Generate all plots ===
plot_effect_size_per_cell_line(df_effects, output_prefix)
plot_effect_size_summary_across_cell_lines(df_effects, output_prefix)
