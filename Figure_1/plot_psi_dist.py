import argparse
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

input_file = args.input_csv
output_file = os.path.join(args.output_dir, "violin_cell_line_PSI_distributions.pdf")
os.makedirs(args.output_dir, exist_ok=True)

# === Replicate columns
replicate_cols = {
    "HEK293": ["HEK_rep1_psi_raw", "HEK_rep2_psi_raw", "HEK_rep3_psi_raw", "HEK_rep4_psi_raw"],
    "HeLa": ["HeLa_rep1_psi_raw", "HeLa_rep2_psi_raw"],
    "K562": ["K562_rep1_psi_raw", "K562_rep2_psi_raw"],
    "MCF7": ["MCF7_rep1_psi_raw", "MCF7_rep2_psi_raw"],
    "HMC3": ["HMC3_rep1_psi_raw", "HMC3_rep2_psi_raw"]
}

cell_order = ["HeLa", "MCF7", "HMC3", "K562", "HEK293"]

cell_colors = {
    "HEK293": "#E984B6",
    "HeLa": "#7FBE7E",
    "K562": "#F9AE33",
    "MCF7": "#807CB9",
    "HMC3": "#EF4025"
}

# === Load
df = pd.read_csv(input_file)

# === Prepare melted dataframe with avg PSI per cell line (allow NaNs in other cells)
all_rows = []
for cell, reps in replicate_cols.items():
    avg_psi = df[reps].mean(axis=1, skipna=True)
    valid = df[reps].notna().any(axis=1)
    subset = pd.DataFrame({
        "cell_line": cell,
        "psi": avg_psi[valid]
    })
    all_rows.append(subset)

plot_df = pd.concat(all_rows, ignore_index=True)
plot_df = plot_df[plot_df["cell_line"].isin(cell_order)]

# === Plot
plt.figure(figsize=(10, 5))
sns.violinplot(
    data=plot_df,
    x="cell_line",
    y="psi",
    palette=cell_colors,
    order=cell_order,
    cut=0,
    linewidth=1,
    inner="box"
)
ax = plt.gca()

# Redraw to ensure box/whiskers are generated
plt.draw()

# Set violin edge color and style
for violin in ax.collections:
    violin.set_edgecolor("black")

# Add sample size (n=) just above y=1
group_counts = plot_df.groupby("cell_line")["psi"].apply(lambda x: x.notna().sum())
for i, cell in enumerate(cell_order):
    n = group_counts[cell]
    ax.text(i, 1.05, f"n={n:,}", ha="center", va="bottom", fontsize=14)

# Set fixed y-limits and axis styling
ax.set_ylim(0, 1)
ax.set_ylabel("Mean PSI", fontsize=16)
ax.set_xlabel("")
ax.set_xticklabels(cell_order, fontsize=11)

yticks = ax.get_yticks()
ax.set_yticks(yticks)
ax.set_yticklabels([f"{x:.1f}" for x in yticks], fontsize=16)
ax.tick_params(axis='y', width=1.5)

# Thicken outer box
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)

# Final layout
plt.title("")
plt.tight_layout()
plt.savefig(output_file)
plt.close()
