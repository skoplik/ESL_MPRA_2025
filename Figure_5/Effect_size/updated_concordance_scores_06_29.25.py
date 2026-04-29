import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

# Base directory and settings
base_path = "/ESL/ESL_MPRA/Figure_5/Effect_size/outputs"
regions = ["exon", "intron1", "intron2"]
cell_lines = ["HEK", "HeLa", "K562", "HMC3", "MCF7"]

# Output directory
plot_dir = f"{base_path}/plots/concordance_updated_08_10_25"
os.makedirs(plot_dir, exist_ok=True)

# Directional concordance function in [-1, 1]
def directional_concordance(x):
    signs = np.sign(x)
    weights = np.abs(x)
    return np.sum(signs * weights) / np.sum(weights)

for cell in cell_lines:
    for region in regions:
        print(f"\n=== Processing: {cell} - {region} ===")
        csv_path = f"{base_path}/out_08_10_2025_boots_0_{region}.csv"

        df = pd.read_csv(csv_path)
        colname = f"Effect Size {cell} Boot 0"
        if colname not in df.columns:
            print(f"Skipping {cell} {region} - column missing.")
            continue

        df = df.dropna(subset=[colname])

        # Compute group stats using directional concordance
        grouped = (
            df.groupby("motif")
            .agg(
                mean_effect=(colname, "mean"),
                concordance=(colname, directional_concordance),
                n_contexts=("event_id", "nunique")
            )
        )
        grouped["abs_concordance"] = grouped["concordance"].abs()

        # Filter for motifs with ≥3 contexts
        grouped_3plus = grouped[grouped["n_contexts"] >= 3].sort_values("n_contexts")

        # Identify high concordance motifs
        high_concord = grouped_3plus[grouped_3plus["abs_concordance"] >= 0.75]
        print("=== High Concordance Motifs (≥3 contexts, |concordance| ≥ 0.75) ===")
        print(high_concord)

        for motif in high_concord.index:
            rows = df[df["motif"] == motif][["event_id", colname]]
            print(f"\nMotif: {motif}")
            print(rows.to_string(index=False))

        # Plot mean effect vs. directional concordance
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            grouped_3plus["mean_effect"],
            grouped_3plus["concordance"],
            c=grouped_3plus["n_contexts"],
            cmap="viridis",
            edgecolor="black",
            alpha=0.7
        )

        plt.axhline(0.75,  color='red', linestyle='--', linewidth=1)
        plt.axhline(-0.75, color='red', linestyle='--', linewidth=1)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.axvline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel(f"Mean Effect Size ({cell} Δlogit)")
        plt.ylabel("Directional Concordance Score")
        plt.title(f"{cell} - {region} context (≥3 event IDs)")

        eqn = r"Directional Concordance = $\frac{\sum(\mathrm{sign} \cdot |\mathrm{effect}|)}{\sum|\mathrm{effect}|}$"
        note = "Score ranges from -1 (all neg) to +1 (all pos); near 0 = mixed"

        plt.text(0.05, -0.28, eqn, fontsize=11, transform=plt.gca().transAxes)
        plt.text(0.05, -0.44, note, fontsize=10, transform=plt.gca().transAxes)

        plt.colorbar(scatter, label="Number of Event Contexts")
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(f"{plot_dir}/{cell}_{region}.pdf")
        plt.close()

        # Export high concordance data
        export_df = df[df["motif"].isin(high_concord.index)].copy()
        export_df["concordance"] = export_df["motif"].map(high_concord["concordance"])
        export_df["abs_concordance"] = export_df["motif"].map(high_concord["abs_concordance"])
        export_df["n_contexts"] = export_df["motif"].map(high_concord["n_contexts"])
        export_df["mean_effect"] = export_df["motif"].map(high_concord["mean_effect"])

        export_cols = ["motif", "event_id", colname, "mean_effect", "concordance", "abs_concordance", "n_contexts"]
        export_df[export_cols].to_csv(
            f"{plot_dir}/high_concordance_{cell}_{region}.csv", index=False
        )

        # Plot high concordance motifs as boxplots with overlaid jittered dots
        filtered = export_df.copy()

        motif_order = (
            filtered.groupby("motif")[colname]
            .median()
            .sort_values(ascending=False)
            .index
        )
        filtered["motif"] = pd.Categorical(filtered["motif"], categories=motif_order, ordered=True)
        motif_to_y = {m: i for i, m in enumerate(motif_order)}
        filtered["y"] = filtered["motif"].map(motif_to_y).astype(float)
        filtered["y_jitter"] = filtered["y"] + np.random.normal(0, 0.1, size=len(filtered))

        unique_events = filtered["event_id"].unique()
        event_to_color = {eid: plt.cm.tab20(i % 20) for i, eid in enumerate(unique_events)}
        filtered["color"] = filtered["event_id"].map(event_to_color)

        plt.figure(figsize=(10, max(4, len(motif_order) * 0.5)))

        sns.boxplot(
            data=filtered,
            x=colname,
            y="motif",
            order=motif_order,
            boxprops=dict(facecolor='none', edgecolor='gray'),
            whiskerprops=dict(color='gray'),
            capprops=dict(color='gray'),
            medianprops=dict(color='gray'),
            showfliers=False,
            linewidth=1
        )

        plt.scatter(
            filtered[colname],
            filtered["y_jitter"],
            c=filtered["color"],
            s=40,
            alpha=0.9,
            linewidth=0
        )

        plt.axvline(0, color="gray", linestyle="--", linewidth=1)
        plt.yticks(range(len(motif_order)), motif_order, fontsize=10)
        plt.xlabel(f"Δlogit Effect Size ({cell})")
        plt.ylabel("Motif Cluster (sorted by median effect)")
        plt.title(f"High Concordance Motifs in {cell} - {region}")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/boxscatter_eventid_{cell}_{region}.pdf")
        plt.close()
