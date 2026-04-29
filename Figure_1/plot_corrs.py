import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr
from matplotlib.ticker import FormatStrFormatter

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# === Load ===
df = pd.read_csv(args.input_csv)

# === Settings ===
cell_order = ["HeLa", "MCF7", "HMC3", "K562", "HEK"]
exclude_reps = {"HEK_rep3", "HEK_rep4"}

custom_rgb_colors = {
    "HEK": "#E984B6",
    "HeLa": "#7FBE7E",
    "MCF7": "#807CB9",
    "HMC3": "#EF4025",
    "K562": "#F9AE33"
}

def collect_replicate_columns(df, suffix):
    cols = []
    for cl in cell_order:
        reps = range(1, 3)
        if cl == "HEK":
            reps = [2, 1]  # reversed HEK rep order
        for i in reps:
            rep = f"{cl}_rep{i}"
            if rep in exclude_reps:
                continue
            col = f"{rep}_{suffix}"
            if col in df.columns:
                cols.append(col)
    return cols

def compute_replicate_dpsis(df):
    dpsi_cols = []
    for cl in cell_order:
        wt_col = f"{cl}_wt_pooled_psi_raw"
        if wt_col not in df.columns:
            continue
        for i in range(1, 5):
            rep = f"{cl}_rep{i}"
            if rep in exclude_reps:
                continue
            var_col = f"{rep}_psi_raw"
            if var_col in df.columns:
                new_col = f"{rep}_dpsi"
                df[new_col] = df[var_col] - df[wt_col]
                dpsi_cols.append(new_col)
    return dpsi_cols

def plot_corr(df, cols, title, outname, vmin=0.75):
    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    n_matrix = pd.DataFrame(index=cols, columns=cols, dtype=int)

    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                vals = df[col1].dropna().values.astype(float)
                corr_matrix.loc[col1, col2] = 1.0
                n_matrix.loc[col1, col2] = len(vals)
            else:
                pair_df = df[[col1, col2]].dropna()
                n = len(pair_df)
                if n == 0:
                    corr = np.nan
                else:
                    x = pair_df[col1].values.astype(float)
                    y = pair_df[col2].values.astype(float)
                    if np.std(x) == 0 or np.std(y) == 0:
                        corr = np.nan
                    else:
                        try:
                            r, _ = pearsonr(x, y)
                            corr = r ** 2
                        except Exception:
                            corr = np.nan
                corr_matrix.loc[col1, col2] = corr
                n_matrix.loc[col1, col2] = n

    # === Enforce order: left-right, bottom-top = HeLa → HEK, with HEK rep2 before rep1
    ordered_cols = []
    for cl in cell_order:
        reps = [1, 2]
        if cl == "HEK":
            reps = [2, 1]
        for i in reps:
            rep_col = f"{cl}_rep{i}_psi_raw"
            if rep_col in cols:
                ordered_cols.append(rep_col)
    corr_matrix = corr_matrix.loc[ordered_cols[::-1], ordered_cols]
    n_matrix = n_matrix.loc[ordered_cols[::-1], ordered_cols]

    print(f"[{outname}] Pairwise N matrix (valid sequence counts):")
    print(n_matrix)

    # === Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        corr_matrix.astype(float),
        cmap="YlGnBu",
        vmin=vmin,
        vmax=1,
        square=True,
        linewidths=0,
        cbar_kws={"label": "Pearson R²"},
        xticklabels=ordered_cols,
        yticklabels=ordered_cols[::-1],
        ax=ax
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # === Remove x-axis tick labels
    ax.set_xticks([])
    ax.set_yticks([])

    # === Draw colored bars under x-axis (bottom)
    for i, col in enumerate(ordered_cols):
        for cl in custom_rgb_colors:
            if col.startswith(cl):
                rect = Rectangle(
                    (i+0.01, 10), 1, 0.5,
                    color=custom_rgb_colors[cl],
                    transform=ax.transData,
                    clip_on=False,
                    zorder=1
                )
                ax.add_patch(rect)

    # === Draw colored bars beside y-axis (left)
    for i, col in enumerate(ordered_cols[::-1]):
        for cl in custom_rgb_colors:
            if col.startswith(cl):
                rect = Rectangle(
                    (-0.5, i-0.01), 0.5, 1,
                    color=custom_rgb_colors[cl],
                    transform=ax.transData,
                    clip_on=False,
                    zorder=1
                )
                ax.add_patch(rect)

     # === Add black outer border around entire heatmap
    border = Rectangle(
        (0, 0),
        len(ordered_cols),
        len(ordered_cols),
        linewidth=3,
        edgecolor="black",
        facecolor="none",
        zorder=2
    )
    ax.add_patch(border)

    # === Format colorbar ticks to 1 decimal place
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar_ax = cbar.ax

    # Add a border to the colorbar
    for spine in cbar_ax.spines.values():
        spine.set(visible=True, lw=1.5, edgecolor="black")

    # Title on top of colorbar
    cbar.ax.set_title("R² (Pearson)", fontsize=12, pad=8)

    # === Draw colored boxes for same-cell-type replicate pairs
    for cl in cell_order:
        reps = [1, 2]
        if cl == "HEK":
            reps = [2, 1]
        rep_cols = [f"{cl}_rep{i}_psi_raw" for i in reps if f"{cl}_rep{i}_psi_raw" in ordered_cols]
        if len(rep_cols) != 2:
            continue
        x0 = ordered_cols.index(rep_cols[0])
        x1 = ordered_cols.index(rep_cols[1])
        x_start = min(x0, x1)
        y_start = ordered_cols[::-1].index(ordered_cols[x_start])-1
        rect = Rectangle((x_start, y_start), 2, 2, fill=False,
                         edgecolor=custom_rgb_colors[cl], linewidth=3, zorder=10)
        ax.add_patch(rect)

    plt.savefig(os.path.join(output_dir, f"{outname}.pdf"))
    plt.close()

    corr_matrix.to_csv(os.path.join(output_dir, f"{outname}.csv"))
    n_matrix.to_csv(os.path.join(output_dir, f"{outname}_N.csv"))
    print(f"Saved: {outname}.pdf and N matrix")

# === 1. Replicate PSI raw ===
rep_psi_raw = collect_replicate_columns(df, "psi_raw")
plot_corr(df, rep_psi_raw, "Replicate PSI (raw)", "replicate_psi_raw_corrs", vmin=0.8)
