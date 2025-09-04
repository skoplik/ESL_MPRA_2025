import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import getopt


def make_combined_start2end_df(df):
    cell_lines = ["HeLa", "K562", "MCF7", "HMC3", "HEK"]
    logit_cols = [f"{c}_pooled_logit" for c in cell_lines]
    psi_cols = [f"{c}_pooled_psi_clipped" for c in cell_lines]
    wt_psi_cols = [f"{c}_wt_pooled_psi_raw" for c in cell_lines]

    df = df.copy()
    df["mean_logit"] = df[logit_cols].astype(float).mean(axis=1)
    df["mean_psi"] = df[psi_cols].astype(float).mean(axis=1)
    df["mean_wt_psi"] = df[wt_psi_cols].astype(float).mean(axis=1)

    wt_df = df[df["snp"] == "none"][["event_id", "mean_logit"]].rename(
        columns={"mean_logit": "start logit psi"}
    )

    var_df = df[df["snp"] != "none"][["event_id", "mean_logit", "mean_psi", "mean_wt_psi"]].rename(
        columns={
            "mean_logit": "end logit psi",
            "mean_psi": "end psi",
            "mean_wt_psi": "start psi"
        }
    )

    merged = var_df.merge(wt_df, on="event_id", how="inner")
    merged["delta logit psi"] = merged["end logit psi"] - merged["start logit psi"]
    merged["delta psi"] = merged["end psi"] - merged["start psi"]
    return merged[[
        "start psi",
        "end psi",
        "delta psi",
        "start logit psi",
        "end logit psi",
        "delta logit psi"
    ]]


def plot_swarm_start2end_general(df, out_plot_name, y_col, title):
    fig, ax = plt.subplots(figsize=(4, 3))
    df = df.copy()
    df['bins'] = pd.qcut(df['start psi'], q=10, duplicates='drop')
    df['bin_mean'] = df['bins'].apply(lambda x: round(x.mid if pd.notnull(x) else np.nan, 3))
    df['bin_mean'] = df['bin_mean'].replace(-0.0, 0.0)
    df['bin_code'] = df['bins'].cat.codes

    _plot_swarm_shared(df, ax, y_col, title, 'Start PSI (Quantiles)')
    plt.savefig(out_plot_name, dpi=300)
    plt.close()
    print("done")


def plot_swarm_start2end_fixedbins(df, out_plot_name, y_col, title):
    fig, ax = plt.subplots(figsize=(4, 3))
    df = df.copy()
    bins = np.arange(0.0, 1.05, 0.05)
    labels = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
    df['bins'] = pd.cut(df['start psi'], bins=bins, labels=labels, include_lowest=True)
    df['bin_mean'] = df['bins'].astype(float)
    df['bin_code'] = df['bins'].cat.codes

    _plot_swarm_shared(df, ax, y_col, title, 'Start PSI (Fixed Bins)')
    plt.savefig(out_plot_name, dpi=300)
    plt.close()
    print("done")


def _plot_swarm_shared(df, ax, y_col, title, xlabel):
    all_x, all_y, all_c = [], [], []
    cmap = plt.get_cmap("plasma")

    for bin_code in sorted(df['bin_code'].dropna().unique()):
        sub = df[df['bin_code'] == bin_code]
        y_vals = sub[y_col].values
        if len(y_vals) > 1:
            kde = gaussian_kde(y_vals)
            densities = kde(y_vals)
        else:
            densities = np.zeros(1)

        densities = (
            (densities - densities.min()) / (densities.ptp() + 1e-8)
            if densities.ptp() > 0 else np.zeros_like(densities)
        )

        x_jittered = bin_code + np.random.normal(0, 0.1, size=len(y_vals))
        all_x.extend(x_jittered)
        all_y.extend(y_vals)
        all_c.extend(densities)

    ax.scatter(all_x, all_y, c=[cmap(c) for c in all_c], s=1, alpha=0.3, rasterized=True, linewidths=0.1)

    bin_order = sorted(df['bin_code'].dropna().unique())
    bin_labels = [f"{abs(df[df['bin_code'] == code]['bin_mean'].iloc[0]):.3f}" for code in bin_order]
    #bin_labels = [f"{b:.3f}" for b in bin_labels]

    box_data = [df[df['bin_code'] == code][y_col].dropna() for code in bin_order]
    ax.boxplot(
        box_data,
        positions=bin_order,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor='none', color='black', linewidth=0.8),
        whiskerprops=dict(color='black', linewidth=0.8),
        capprops=dict(color='black', linewidth=0.8),
        medianprops=dict(color='black', linewidth=1)
    )

    ax.set_xticks(bin_order)
    ax.set_xticklabels(bin_labels, rotation=90, fontsize=6)
    ax.tick_params(labelsize=6)
    ax.set_xlabel(xlabel, fontsize=6)

    if y_col == "end psi":
        ax.set_ylim(0, 1)
        ax.set_ylabel('End PSI', fontsize=6)
    elif y_col == "delta psi":
        ax.set_ylim(-1, 1)
        ax.set_ylabel('ΔPSI', fontsize=6)
    elif y_col == "end logit psi":
        ax.set_ylim(-8, 8)
        ax.set_ylabel('End PSI (Logit)', fontsize=6)
    elif y_col == "delta logit psi":
        ax.set_ylim(-10, 10)
        ax.set_ylabel('ΔLogit PSI', fontsize=6)
    else:
        ax.set_ylabel(y_col, fontsize=6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, aspect=15)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Relative Density", fontsize=6)

    ax.set_title(title, fontsize=7)
    plt.tight_layout()


if __name__ == "__main__":
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "", [
            "input_csv=",
            "output_dir=",
            "output_prefix="
        ])
        opts = dict(opts)

        input_csv = opts["--input_csv"]
        output_dir = opts["--output_dir"]
        output_prefix = opts["--output_prefix"]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path_prefix = os.path.join(output_dir, output_prefix)

    except Exception as e:
        print("Failed to parse arguments:", e)
        sys.exit(2)

    df = pd.read_csv(input_csv, low_memory=False)
    df["Reference"] = pd.to_numeric(df["Reference"], errors="coerce")
    df = df.sort_values("Reference")
    df = df.drop_duplicates(subset="full_seq", keep="first")

    start2end_df = make_combined_start2end_df(df)

    plot_swarm_start2end_general(
        start2end_df, output_path_prefix + "_endpsi_300dpi_clipped_e-2.pdf", "end psi", "End PSI (clipped)"
    )
    plot_swarm_start2end_general(
        start2end_df, output_path_prefix + "_deltapsi_300dpi_clipped_e-2.pdf", "delta psi", "ΔPSI"
    )
    plot_swarm_start2end_general(
        start2end_df, output_path_prefix + "_endlogit_300dpi_clipped_e-2.pdf", "end logit psi", "End Logit PSI"
    )
    plot_swarm_start2end_general(
        start2end_df, output_path_prefix + "_deltalogit_300dpi_clipped_e-2.pdf", "delta logit psi", "ΔLogit PSI"
    )

    plot_swarm_start2end_fixedbins(
        start2end_df, output_path_prefix + "_endpsi_fixedbins.pdf", "end psi", "End PSI (clipped) — Fixed Bins"
    )
    plot_swarm_start2end_fixedbins(
        start2end_df, output_path_prefix + "_deltapsi_fixedbins.pdf", "delta psi", "ΔPSI — Fixed Bins"
    )
    plot_swarm_start2end_fixedbins(
        start2end_df, output_path_prefix + "_endlogit_fixedbins.pdf", "end logit psi", "End Logit PSI — Fixed Bins"
    )
    plot_swarm_start2end_fixedbins(
        start2end_df, output_path_prefix + "_deltalogit_fixedbins.pdf", "delta logit psi", "ΔLogit PSI — Fixed Bins"
    )
