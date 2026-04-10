import os
import sys
import getopt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

def main():
    opts = parse_args()
    df = pd.read_csv(opts["--cell_psi_merged_file"], low_memory=False)

    df = df[df['snp'].apply(lambda x: isinstance(x, str) and x.count(';') == 0)]
    print(f"Retained {len(df)} single-variant rows")

    df = annotate_variants_from_snp(df)
    df = compute_avg_delta_logit(df)

    first_plt, scd_plt = prepare_plot_data(df)

    fx2 = first_plt['var_pos'] - first_plt['exon_start']
    sx2 = scd_plt['var_pos'] - (scd_plt['exon_end'] + 1)
    offset = fx2.max() - sx2.min() + 1
    sx2_shifted = sx2 + offset

    mergedx = np.concatenate((fx2.to_numpy(), sx2_shifted.to_numpy()))
    mergedy = np.concatenate((first_plt['avg_delta_logit'].to_numpy(), scd_plt['avg_delta_logit'].to_numpy()))

    position_counts = Counter(mergedx)
    print("Position\tVariant_Count")
    for pos in sorted(position_counts.keys()):
        print(f"{pos}\t{position_counts[pos]}")

    plot_data(first_plt, fx2, sx2_shifted, mergedx, mergedy, offset, opts["--output_prefix"], opts["--output_dir"])

def parse_args():
    opts, _ = getopt.getopt(sys.argv[1:], "", ["cell_psi_merged_file=", "output_dir=", "output_prefix="])
    opts = dict(opts)
    if not os.path.exists(opts["--output_dir"]):
        os.makedirs(opts["--output_dir"])
    return opts

def annotate_variants_from_snp(df):
    df = df.copy()
    df['var_pos'] = df['snp'].apply(lambda x: int(x.split(':')[0]) if isinstance(x, str) and ':' in x and '>' in x else np.nan)
    return df.dropna(subset=['var_pos'])

def compute_avg_delta_logit(df):
    delta_cols = [col for col in df.columns if col.endswith("_delta_logit_pooled")]
    df['avg_delta_logit'] = df[delta_cols].astype(float).mean(axis=1)
    return df

def prepare_plot_data(df):
    df['intron1_len'] = df['intron1'].str.len()
    df['exon_len'] = df['exon'].str.len()
    df['exon_start'] = df['intron1_len']
    df['exon_end'] = df['intron1_len'] + df['exon_len'] - 1
    df['half_point'] = df['exon_start'] + (df['exon_len'] // 2)
    df['var_pos'] = df['var_pos'].astype(int)

    first_plt = df[df['var_pos'] < df['half_point']].copy()
    scd_plt = df[df['var_pos'] >= df['half_point']].copy()

    overlap_ids = set(first_plt['Reference']) & set(scd_plt['Reference'])
    if overlap_ids:
        print(f"Warning: {len(overlap_ids)} sequences appear in both sides (possible double-counting).")
        print("Examples:", list(overlap_ids)[:5])

    return first_plt.dropna(subset=['avg_delta_logit']), scd_plt.dropna(subset=['avg_delta_logit'])

def plot_data(first_plt, fx2, sx2_shifted, mergedx, mergedy, offset, output_prefix, output_dir):
    def make_plot(x, y, xmin, xmax, title, outname, zoom=False, rel_offset=None):
        print(f"n = {len(y)} points in {outname}")
        fig, ax = plt.subplots(figsize=(2 if zoom else 6, 4))
        all_kernel_vals = []

        for i in range(int(xmin), int(xmax) + 1):
            ys = y[x == i]
            if len(ys) >= 2:
                xs = [i + np.random.uniform(0, 0.01) for _ in range(len(ys))]
                coords = np.vstack([xs, ys])
                kernel = stats.gaussian_kde(coords)(coords)
                kernel = kernel / kernel.max() # Normalize density per position
                all_kernel_vals.extend(kernel)
                idx = np.argsort(kernel)
                sns.scatterplot(x=np.array(xs)[idx], y=np.array(ys)[idx],
                                c=kernel[idx], cmap="plasma", ax=ax,
                                s=8 if zoom else 2, linewidth=0, rasterized=True)
            elif len(ys) == 1:
                sns.scatterplot(x=[i], y=[ys[0]], c=[1], cmap="plasma", ax=ax,
                                s=8 if zoom else 1, linewidth=0, rasterized=True)

        ax.axvline(0, color='black', linestyle='--', lw=0.75)
        ax.axvline(offset, color='black', linestyle='--', lw=0.75)

        midpoint = first_plt['exon_len'].iloc[0] // 2
        midpoint_pos = midpoint
        ax.axvline(midpoint_pos, color='gray', linestyle=':', lw=0.75)

        ytop = ax.get_ylim()[1]
        ax.text(ax.get_xlim()[0] + 3, ytop - 0.5, f"n = {len(y):,}", ha='left', va='top', fontsize=8)
        ax.text(-1, ytop * 0.95, 'SA', ha='center', fontsize=8)
        ax.text(offset + 1, ytop * 0.95, 'SD', ha='center', fontsize=8)
        ax.text(midpoint_pos, ytop * 0.95, f'±{midpoint}', ha='center', fontsize=8)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-10, 7.5)
        ax.set_title(title)
        ax.set_ylabel('Δlogit(PSI)')
        ax.set_xlabel('Position Relative to SD/SA')

        if zoom:
            xticks = list(range(-10, 11, 5))
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(i) for i in xticks])

        if all_kernel_vals:
            cmap = plt.cm.plasma
            norm = plt.Normalize(vmin=min(all_kernel_vals), vmax=max(all_kernel_vals))
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Normalized Density')

        fig.subplots_adjust(left=0.15, right=0.98, top=0.9, bottom=0.15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, outname), dpi=1200, bbox_inches="tight")
        plt.close()

    make_plot(mergedx, mergedy, int(mergedx.min()) - 5, int(mergedx.max()) + 5,
              'Δlogit PSI by Variant Position',
              f"{output_prefix}_avg_delta_logit_pos_plot_plasma.pdf")

    sa_mask = (mergedx >= -10) & (mergedx <= 10)
    make_plot(mergedx[sa_mask], mergedy[sa_mask], -10, 10,
              'Near Splice Acceptor (SA)',
              f"{output_prefix}_zoom_SA.pdf",
              zoom=True)

    sd_mask = (mergedx >= (offset - 10)) & (mergedx <= (offset + 10))
    rel_sd_x = mergedx[sd_mask] - offset
    make_plot(rel_sd_x, mergedy[sd_mask], -10, 10,
              'Near Splice Donor (SD)',
              f"{output_prefix}_zoom_SD.pdf",
              zoom=True)

if __name__ == "__main__":
    main()
