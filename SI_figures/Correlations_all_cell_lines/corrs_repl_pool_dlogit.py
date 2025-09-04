import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import r2_score

# === Inputs ===
input_file = "/ESL/Figures_SK/General_preprocessing/output_6_09_2025/06_09_2025_1e-2_ALL_WITH_WT.csv"
output_prefix = "/ESL/Figures_SK/supplement/fig1_fig2_replicate_corrs/allcell_avg_repl_vs_pooled_dlogit_grid"
os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

# === Load data
df = pd.read_csv(input_file)

# === Cell lines and replicate/pool columns
cell_lines = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]
rep1_suffix = "_rep1_logit"
rep2_suffix = "_rep2_logit"
pooled_suffix = "_delta_logit_pooled"
plot_labels = {"HEK": "HEK293", "HeLa": "HeLa", "K562": "K562", "MCF7": "MCF7", "HMC3": "HMC3"}

# === Split WT and variant rows
wt_df = df[df["snp"] == "none"].copy()
var_df = df[df["snp"] != "none"].copy()

# === Set up plot
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)

for i, cl in enumerate(cell_lines):
    rep1 = cl + rep1_suffix
    rep2 = cl + rep2_suffix
    pooled_col = cl + pooled_suffix

    wt_cols = wt_df[["event_id", rep1, rep2]].dropna()
    wt_cols = wt_cols.rename(columns={rep1: "wt_rep1", rep2: "wt_rep2"})

    merged = pd.merge(var_df, wt_cols, on="event_id", how="left")
    merged = merged.dropna(subset=[rep1, rep2, "wt_rep1", "wt_rep2", pooled_col])

    merged["delta_rep1"] = merged[rep1] - merged["wt_rep1"]
    merged["delta_rep2"] = merged[rep2] - merged["wt_rep2"]
    merged["avg_delta_logit_repl"] = (merged["delta_rep1"] + merged["delta_rep2"]) / 2

    x = merged["avg_delta_logit_repl"]
    y = merged[pooled_col]
    r2 = r2_score(y, x)

    ax = axes[i]
    ax.scatter(x, y, alpha=0.2, s=40, color="blue", rasterized=True)
    ax.axline((0, 0), slope=1, linestyle="--", color="gray")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title(f"{plot_labels[cl]}\nR² = {r2:.3f}, n = {len(x)}")
    if i == 0:
        ax.set_ylabel("Pooled ΔLogit PSI")
    ax.set_xlabel("Avg Replicate ΔLogit PSI")

plt.tight_layout()
plt.savefig(output_prefix + ".pdf")
plt.savefig(output_prefix + ".png", dpi=300)
plt.close()
