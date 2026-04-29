import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np

# Make PDF/SVG text editable
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# === Input ===
input_file = "/ESL/Figures_SK/supplement/unexpected_sjs/cell_type_avg_unexpected_SS_usage_2024_07_26_cell_types_average_unexpected_freqs.txt"
output_pdf = "/ESL/Figures_SK/supplement/unexpected_sjs/unexpected_usage_histograms.pdf"

# === Load data ===
df = pd.read_csv(input_file, sep="\t")

# Columns to plot (skip Sequence)
cols = [
    "Average_unexpected_SS_usage",
    "HEK293_average_unexpected_SS_usage",
    "HeLa_average_unexpected_SS_usage",
    "K562_average_unexpected_SS_usage",
    "MCF7_average_unexpected_SS_usage",
    "HMC3_average_unexpected_SS_usage"
]

# Titles for plots
titles = [
    "Average",
    "HEK293",
    "HeLa",
    "K562",
    "MCF7",
    "HMC3"
]

# === Plot ===
ncols = 3
nrows = int(len(cols) / ncols) + (len(cols) % ncols > 0)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
axes = axes.flatten()

# Use consistent bins across plots (50 bins between 0 and 1)
bins = np.linspace(0, 1, 50)

for i, col in enumerate(cols):
    sns.histplot(
        df[col].dropna(),
        bins=bins,
        stat="count",  # raw counts
        kde=False,
        ax=axes[i],
        color="skyblue",
        edgecolor="black"
    )
    axes[i].set_title(titles[i])
    axes[i].set_xlabel("Unexpected SS usage")
    axes[i].set_ylabel("Number of Sequences (log scale)")
    axes[i].set_yscale("log")  # log10 y-axis

    # Only show x-axis labels 0 and 1
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(["0", "1"])

# Remove any empty subplots
for j in range(len(cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(output_pdf, dpi=300)
plt.close()
