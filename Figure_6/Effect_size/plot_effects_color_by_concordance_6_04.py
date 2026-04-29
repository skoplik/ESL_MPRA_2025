import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib.colors import TwoSlopeNorm

def load_all_concordance(concordance_dir, region):
    per_cell = {}
    combo_counter = {}
    for cell in ["HEK", "HeLa", "HMC3", "K562", "MCF7"]:
        fname = Path(concordance_dir) / f"high_concordance_{cell}_{region}.csv"
        try:
            df = pd.read_csv(fname)
            pairs = set(zip(df["motif"], df["event_id"]))
            per_cell[cell] = pairs
            for p in pairs:
                combo_counter[p] = combo_counter.get(p, 0) + 1
        except Exception as e:
            print(f"[WARNING] Could not load {fname}: {e}")
            per_cell[cell] = set()
    shared = {p for p, count in combo_counter.items() if count >= 2}
    return per_cell, shared

def plot_effect_size_per_cell_line(df, output_prefix, concordant_pairs, shared_pairs, use_median=False):
    cell_lines = ["HEK", "HeLa", "K562", "HMC3", "MCF7"]
    mean_cols = [f"Mean {cell} Average" for cell in cell_lines]
    for col in mean_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["MeanAcrossCellLines"] = df[mean_cols].mean(axis=1, skipna=True)
    suffix = "_median" if use_median else "_mean"

    for cell in cell_lines:
        colname = f"Mean {cell} Average"
        df_cell = df[["motif", "event_id", colname]].dropna(subset=[colname]).copy()
        df_cell["effect"] = df_cell[colname].astype(float)
        df_cell["pair"] = list(zip(df_cell["motif"], df_cell["event_id"]))
        df_cell["color"] = df_cell["pair"].apply(lambda x: "orangered" if x in concordant_pairs.get(cell, set()) else "lightgrey")
        if use_median:
            cluster_order = df_cell.groupby("motif")["effect"].median().sort_values().index.tolist()
        else:
            cluster_order = df_cell.groupby("motif")["effect"].mean().sort_values().index.tolist()

        plt.figure(figsize=(12, min(10, len(cluster_order) * 0.6)))
        for i, cluster in enumerate(cluster_order):
            sub = df_cell[df_cell["motif"] == cluster]
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                plt.plot(row["effect"], i, "o", color=row["color"], markersize=4, alpha=0.7)
            center = sub["effect"].median() if use_median else sub["effect"].mean()
            plt.plot(center, i, "o", color="black", markersize=5.5,
                     markeredgecolor="white", markeredgewidth=0.2, zorder=10)

        red_clusters = set(df_cell["motif"])
        yticks_to_show = []
        ytick_labels = []
        for i, cluster in enumerate(cluster_order):
            if cluster in red_clusters:
                yticks_to_show.append(i)
                ytick_labels.append(cluster.replace("cluster_", "cluster "))
        method = "Median" if use_median else "Mean"
        plt.yticks(yticks_to_show, ytick_labels, fontsize=6)
        plt.ylabel(f"Clusters (n = {len(cluster_order)})", fontsize=11)
        plt.xlabel(f"{cell} Effect Size", fontsize=12)
        plt.title(f"Motif Cluster Effects in {cell} ({method})", fontsize=14)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.ylim(-2, len(cluster_order) + 2)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1.5, zorder=0)
        legend_elements = [
            Line2D([0], [0], marker='o', color='lightgrey', label='Not Concordant', linestyle='', markersize=5),
            Line2D([0], [0], marker='o', color='red', label='Concordant', linestyle='', markersize=5),
            Line2D([0], [0], marker='o', color='black', label=f'Cluster {method}', linestyle='', markersize=7,
                   markeredgecolor="white", markeredgewidth=1.2)
        ]
        plt.legend(handles=legend_elements, loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_per_cluster_{cell}{suffix}.pdf")
        plt.close()

    df_all = df[["motif", "event_id"] + mean_cols]
    df_all["MeanAcrossCellLines"] = df_all[mean_cols].mean(axis=1)
    df_all["SDAcrossCellLines"] = df_all[mean_cols].std(axis=1)
    df_all["effect"] = df_all["MeanAcrossCellLines"]
    df_all["pair"] = list(zip(df_all["motif"], df_all["event_id"]))
    df_all["color"] = df_all["pair"].apply(lambda x: "orangered" if x in shared_pairs else "lightgrey")
    if use_median:
        cluster_order = df_all.groupby("motif")["effect"].median().sort_values().index.tolist()
    else:
        cluster_order = df_all.groupby("motif")["effect"].mean().sort_values().index.tolist()

    plt.figure(figsize=(12, min(10, len(cluster_order) * 0.6)))
    for i, cluster in enumerate(cluster_order):
        sub = df_all[df_all["motif"] == cluster]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            plt.errorbar(
                row["effect"], i,
                xerr=row["SDAcrossCellLines"],
                fmt="o",
                color=row["color"],
                markersize=5,
                capthick=1.5,
                elinewidth=1.5,
                capsize=3,
                alpha=0.7,
                zorder=9
            )
        center = sub["effect"].median() if use_median else sub["effect"].mean()
        plt.plot(center, i, "o", color="black", markersize=5.5,
                 markeredgecolor="white", markeredgewidth=0.2, zorder=10)
    red_clusters_all = set(df_all["motif"])
    yticks_to_show = []
    ytick_labels = []
    for i, cluster in enumerate(cluster_order):
        if cluster in red_clusters_all:
            yticks_to_show.append(i)
            ytick_labels.append(cluster.replace("cluster_", "cluster "))
    method = "Median" if use_median else "Mean"
    plt.yticks(yticks_to_show, ytick_labels, fontsize=6)
    plt.ylabel(f"Clusters (n = {len(cluster_order)})", fontsize=11)
    plt.xlabel(f"{method} Effect Size Across Cell Lines", fontsize=12)
    plt.title(f"Motif Cluster Effects Across Cell Lines (≥2 Concordant) [{method}]", fontsize=14)
    plt.grid(axis="x", linestyle="--", alpha=1)
    plt.ylim(-2, len(cluster_order) + 2)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1.5, zorder=0)
    plt.legend(handles=legend_elements, loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_per_cluster_ALL{suffix}.pdf")
    plt.close()

def plot_effect_size_box_one_cell(df, output_prefix, concordant_pairs, cell="HEK", global_cbar_min=None, global_cbar_max=None):
    colname = f"Mean {cell} Average"
    df_cell = df[["motif", "event_id", colname]].dropna(subset=[colname]).copy()
    df_cell["effect"] = pd.to_numeric(df_cell[colname], errors="coerce")
    df_cell["pair"] = list(zip(df_cell["motif"], df_cell["event_id"]))

    # Keep only motifs with >= 3 distinct exon families
    valid_motifs = df_cell.groupby("motif")["event_id"].nunique()
    valid_motifs = valid_motifs[valid_motifs >= 3].index
    df_cell = df_cell[df_cell["motif"].isin(valid_motifs)]

    # Aggregate stats
    agg_stats = df_cell.groupby("motif")["effect"].agg(["mean", "count", "std"])
    agg_stats["sem"] = agg_stats["std"] / np.sqrt(agg_stats["count"])
    cluster_order = agg_stats["mean"].sort_values(ascending=True).index.tolist()

    # --- COLORBAR LIMITS (full data range) ---
    data_min = df_cell["effect"].min()
    data_max = df_cell["effect"].max()
    if global_cbar_min is not None and global_cbar_max is not None:
        cbar_min, cbar_max = global_cbar_min, global_cbar_max
    else:
        cbar_min, cbar_max = data_min, data_max

    # --- AXIS LIMITS (error bar range) ---
    min_with_error = (agg_stats["mean"] - agg_stats["sem"]).min() -0.3
    max_with_error = (agg_stats["mean"] + agg_stats["sem"]).max()+0.3
    axis_min, axis_max = min_with_error, max_with_error

    # --- TICK RANGES ---
    # Axis ticks: rounded inward to 0.5 within axis range
    axis_tick_min = np.ceil(axis_min * 2) / 2
    axis_tick_max = np.floor(axis_max * 2) / 2

    # Colorbar ticks: rounded inward to 0.5 within cbar range
    cbar_tick_min = np.ceil(cbar_min * 2) / 2
    cbar_tick_max = np.floor(cbar_max * 2) / 2

    print(f"Colorbar limits: vmin={cbar_min:.3f}, vmax={cbar_max:.3f}")
    print(f"Axis limits (error bars): min={axis_min:.3f}, max={axis_max:.3f}")
    print(f"Axis tick range: min={axis_tick_min:.1f}, max={axis_tick_max:.1f}")
    print(f"Colorbar tick range: min={cbar_tick_min:.1f}, max={cbar_tick_max:.1f}")

    norm = TwoSlopeNorm(vmin=cbar_min, vcenter=0, vmax=cbar_max)
    cmap = get_cmap("coolwarm")

    fig, ax = plt.subplots(figsize=(12, max(6, 0.2 * len(cluster_order))))
    ytick_labels = []

    for i, motif in enumerate(cluster_order):
        mean_val = agg_stats.loc[motif, "mean"]
        sem_val = agg_stats.loc[motif, "sem"]

        # Color from colormap
        color = cmap(norm(mean_val))

        # Marker styling
        edgecolor = "black"
        lw = 1.0

        # Label with star if concordant
        is_concordant = any(
            (motif, eid) in concordant_pairs.get(cell, set())
            for eid in df_cell[df_cell["motif"] == motif]["event_id"]
        )
        # Extract just the number from the motif label
        cluster_num = motif.replace("cluster_", "")
        label = ("★ " if is_concordant else "   ") + cluster_num
        ytick_labels.append(label)

        # Plot mean ± SEM
        ax.errorbar(
            mean_val, i,
            xerr=sem_val,
            fmt="o",
            markersize=12,
            mfc=color,
            mec=edgecolor,
            mew=lw,
            ecolor=edgecolor,
            elinewidth=1.5,
            capsize=3
        )

    # --- COLORBAR ---
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        ticks=np.arange(cbar_tick_min, cbar_tick_max + 0.5, 0.5))
    cbar.set_label("Effect Size", fontsize=25)
    cbar.ax.tick_params(labelsize=16)

    # --- AXIS SETTINGS ---
    ax.axvline(0, color="black", linestyle="--", lw=1.5, zorder=0)
    ax.set_xlim(axis_min, axis_max)
    ax.set_xticks(np.arange(axis_tick_min, axis_tick_max + 0.5, 0.5))
    ax.tick_params(axis="x", labelsize=20)  # Increased font size for x-axis ticks
    
    ax.set_yticks(range(len(cluster_order)))
    ax.set_yticklabels(ytick_labels, fontsize=14)
    ax.set_ylabel(f"Clusters (n = {len(cluster_order)})", fontsize=20)
    ax.set_xlabel(f"{cell} Effect Size", fontsize=20)
    ax.set_title(f"Motif Cluster Effect Size (Mean ± SEM) in {cell}", fontsize=22)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.set_ylim(-2, len(cluster_order) + 2)

    fig.tight_layout()
    fig.savefig(f"{output_prefix}_mean_sem_per_cluster_{cell}_min3families.pdf")
    plt.close(fig)

    return data_min, data_max  # for tracking global cbar min/max







# === Main ===
concordance_dir = "/ESL/ESL_MPRA/Figure_6/Effect_size/outputs/plots/concordance_updated_08_10_25"

for region in ["exon", "intron1", "intron2"]:
    effects_file = f"/ESL/ESL_MPRA/Figure_6/Effect_size/outputs/out_08_10_2025_boots_0_{region}.csv"
    df_effects = pd.read_csv(effects_file)

    concordant_dict, shared_concordant = load_all_concordance(concordance_dir, region)
    output_prefix = f"/ESL/ESL_MPRA/Figure_6/Effect_size/outputs/plots/effects_colored_by_cluster_concordant_fullticks_08_10_2025/{region}"

    # Original plots
    #plot_effect_size_per_cell_line(df_effects, output_prefix, concordant_dict, shared_concordant, use_median=False)
    #plot_effect_size_per_cell_line(df_effects, output_prefix, concordant_dict, shared_concordant, use_median=True)

    # New one-cell-line boxplot with coloring and highlight
    plot_effect_size_box_one_cell(df_effects, output_prefix, concordant_dict, cell="HEK")
    plot_effect_size_box_one_cell(df_effects, output_prefix, concordant_dict, cell="HeLa")
    plot_effect_size_box_one_cell(df_effects, output_prefix, concordant_dict, cell="K562")
    plot_effect_size_box_one_cell(df_effects, output_prefix, concordant_dict, cell="HMC3")
    plot_effect_size_box_one_cell(df_effects, output_prefix, concordant_dict, cell="MCF7")
     

