import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import logomaker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import transforms
from pathlib import Path
import seaborn as sns
from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import re
from scipy.stats import pearsonr

def shorten_rbps(rbps, return_notes=False):
    if not rbps or all(x is None or str(x).strip() == "" for x in rbps):
        return ("NA", {}) if return_notes else "NA"

    shorten_ok = {
        "HNRNP", "PTBP", "PCBP", "YB", "SRSF", "CPEB", "KHDRBS", "NOVA",
        "ELAVL", "PABP", "LIN28", "TIA", "IGF2BP", "HNRNPH", "TRA2"
    }

    grouped = defaultdict(list)

    for rbp in rbps:
        rbp = str(rbp).strip()
        if not rbp:
            continue
        matched = False
        for prefix in sorted(shorten_ok, key=len, reverse=True):
            if rbp.startswith(prefix):
                suffix = rbp[len(prefix):]
                grouped[prefix].append(suffix)
                matched = True
                break
        if not matched:
            grouped[rbp].append("")

    condensed_parts = []
    notes = {}

    for prefix, suffixes in grouped.items():
        unique_suffixes = sorted(set(suffixes), key=lambda x: (len(x), x))
        if len(unique_suffixes) == 1 and unique_suffixes[0] == "":
            condensed_parts.append(prefix)
        elif len(unique_suffixes) == 1:
            condensed_parts.append(f"{prefix}{unique_suffixes[0]}")
        else:
            if prefix in shorten_ok:
                condensed_parts.append(f"{prefix}{'/'.join(unique_suffixes)}")
                notes[prefix] = [f"{prefix}{s}" for s in unique_suffixes]
            else:
                condensed_parts.extend([f"{prefix}{s}" for s in unique_suffixes])

    result = ", ".join(condensed_parts)
    return (result, notes) if return_notes else result

def wrap_rbps(text, max_per_line=4):
    if not text or pd.isna(text):
        return "NA"
    tokens = text.split(', ')
    wrapped = "\n".join([", ".join(tokens[i:i+max_per_line]) for i in range(0, len(tokens), max_per_line)])
    return wrapped




def parse_transfac(filepath):
    motifs = {}
    with open(filepath) as f:
        lines = f.readlines()
    current_pwm = []
    motif_name = None
    reading_pwm = False
    for line in lines:
        line = line.strip()
        if line.startswith("ID"):
            motif_name = line.split()[1]
        elif line.startswith("P0"):
            reading_pwm = True
        elif line.startswith("XX") or line == "//":
            if current_pwm and motif_name:
                pwm_df = pd.DataFrame(current_pwm, columns=["A", "C", "G", "T"])
                pwm_df.columns = ['A', 'C', 'G', 'U']
                motifs[motif_name] = pwm_df
                current_pwm = []
                motif_name = None
                reading_pwm = False
        elif reading_pwm:
            tokens = line.split()
            if len(tokens) == 5 and tokens[0].isdigit():
                current_pwm.append([float(x) for x in tokens[1:]])
    return motifs



# === Updated plot_table_bar function

# ---- Data S2/S3 page geometry -------------------------------------------
# The supplementary tables are laid out to fit US Letter portrait pages and
# split across as many pages as the cluster count needs. The original single
# figure was 28 in wide; FONT_SCALE brings the type down proportionally so the
# four columns still hold their text at 8.5 in.
PAGE_W, PAGE_H = 8.5, 11.0
PAGE_MARGIN    = 1.1          # inches reserved for title + header row + colourbar
ROW_IN         = 0.215        # vertical inches per exon row; also the
                             # heatmap cell size, since cells are square
FONT_SCALE     = 0.55
HEAT_RIGHT     = 0.965       # right edge of the heatmap, figure fraction
LABEL_GAP      = 0.175       # space kept between bars and heatmap so the
                             # longest gene label is never clipped
def _fs(x):                   # scale a font size from the original 28in design
    return max(3.5, round(x * FONT_SCALE, 1))

def plot_table_bar(df, pwm_dict, output_file, title, event_effects_by_motif, original_to_display_name, full_effects_df, cluster_to_motifs, scale=None, page_height=None):
    cell_line_cols = [col for col in full_effects_df.columns if col.startswith("Effect Size")]
    cell_line_cols = [c.replace("Effect Size HEK ", "Effect Size HEK293 ") for c in cell_line_cols]
    full_effects_df = full_effects_df.rename(columns=lambda x: x.replace("Effect Size HEK ", "Effect Size HEK293 "))
    df['original_motif'] = pd.Categorical(
        df['original_motif'],
        categories=df.groupby("original_motif")["mean_effect"].mean().sort_values(ascending=False).index,
        ordered=True
    )
    motif_order = df['original_motif'].cat.categories.tolist()
    n = len(motif_order)

    all_effects = [mean for sublist in event_effects_by_motif.values() for (_, mean, _) in sublist]
    # A page-local range would give every page its own colour mapping and its own
    # x limits; `scale` carries the range computed once over all pages.
    if scale is not None:
        true_vmin, true_vmax = scale
    else:
        true_vmin = min(all_effects)
        true_vmax = max(all_effects)
    norm = mcolors.TwoSlopeNorm(vmin=true_vmin, vcenter=0, vmax=true_vmax) if true_vmin != true_vmax else mcolors.Normalize(vmin=true_vmin - 0.5, vmax=true_vmax + 0.5)
    cmap = cm.get_cmap('coolwarm').copy()

    _pairs = []          # (ax_bar, ax_heat, rows, cols), positioned after layout
    bar_spacing = 0.5
    bar_height = 0.68  # leaves a gap between bars; row alignment comes from
                       # the axes positions below, not from this
    row_heights = [len(event_effects_by_motif.get(m, [])) for m in motif_order]
    # Data S2/S3 chunks its pages to fill PAGE_H, so it passes page_height and
    # rows come out at their intended size. A standalone panel like 6D/6E has far
    # fewer rows; on a fixed 11 in page each row is stretched tall and the
    # square-celled heatmap ends up much shorter than the bar block beside it.
    # With no page_height, size the figure to the content instead.
    _fig_h = page_height if page_height is not None else \
             max(3.0, sum(row_heights) * ROW_IN + PAGE_MARGIN)
    fig = plt.figure(figsize=(PAGE_W, _fig_h))
    _top = 1.0 - 0.75 / _fig_h          # room for title + column headers
    _bot = 0.95 / _fig_h                 # x label + horizontal colourbar
    outer_gs = GridSpec(n, 1, height_ratios=row_heights, figure=fig,
                        top=_top, bottom=_bot, left=0.085, right=0.90)

    for i, motif in enumerate(motif_order):
        inner_gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_gs[i], width_ratios=[2.0, 6.0, 5.0, 2], wspace=0.3)
        ax_logo = fig.add_subplot(inner_gs[0])
        ax_rbps = fig.add_subplot(inner_gs[1])
        ax_bar  = fig.add_subplot(inner_gs[2])
        ax_heat = fig.add_subplot(inner_gs[3])

        motif_rows = df[df['original_motif'] == motif]
        display_name = original_to_display_name.get(motif, motif)
        cluster_number = re.sub(r"[^\d]", "", motif)

        if motif in pwm_dict:
            pwm_df = pwm_dict[motif]
            logo_box = inset_axes(ax_logo, width=0.95, height=0.30, loc='center', 
                                  bbox_to_anchor=(0.6, 0.5),
                                  bbox_transform=ax_logo.transAxes,
                                  borderpad=0.2,
                                  axes_class=plt.Axes)
            logo = logomaker.Logo(pwm_df, ax=logo_box, color_scheme='classic')
            logo.ax.set_xticks([])
            logo.ax.set_yticks([])
            logo.style_spines(visible=False)
            ax_logo.axis('off')

        num_motifs_in_cluster = len(cluster_to_motifs.get(motif, []))
        rbp_list_str = motif_rows['rbps_in_cluster'].iloc[0]
        rbp_list = [g.strip() for g in rbp_list_str.split(",") if g.strip()] if isinstance(rbp_list_str, str) else []
        condensed_rbps, _ = shorten_rbps(rbp_list, return_notes=True)

        wrapped = wrap_rbps(condensed_rbps, max_per_line=4)
        rbp_text = f"Number of Motifs in cluster {cluster_number}: {num_motifs_in_cluster}\n{wrapped}"
        ax_rbps.text(0.1, 0.5, rbp_text, ha='left', va='center', fontsize=_fs(14))
        ax_rbps.axis('off')

        event_rows = []
        for exon in motif_rows['gene_exon']:
            row = full_effects_df[(full_effects_df['gene_exon'] == exon) & (full_effects_df['motif'] == motif)]
            values = [float(v) if pd.notna(v) else np.nan for v in row[cell_line_cols].iloc[0].values] if not row.empty else [np.nan]*len(cell_line_cols)
            mean_val = np.nanmean(values)
            sem_val = np.nanstd(values) / np.sqrt(np.count_nonzero(~np.isnan(values)))

            event_rows.append((exon, mean_val, sem_val, values))
        event_rows = sorted(event_rows, key=lambda x: x[1])
        num_rows = len(event_rows)

        heatmap_rows = []
        ax_bar.axvline(0, color='black', linestyle='--', linewidth=1)
        for j, (exon, mean_val, sem_val, values) in enumerate(event_rows):
            color = cmap(norm(mean_val))
            ax_bar.barh(j, mean_val, xerr=sem_val, height=bar_height, color=color, capsize=2, edgecolor='black', linewidth=0.5)
            label_offset = 0.1 + sem_val
            ax_bar.text(mean_val + (label_offset if mean_val >= 0 else -label_offset), j, exon,
                        fontsize=_fs(12), va='center', ha='left' if mean_val >= 0 else 'right')
            heatmap_rows.append(values)

        ax_bar.set_xlim(true_vmin - 0.5, true_vmax + 0.5)
        ax_bar.set_ylim(-0.5, num_rows - 0.5)
        ax_bar.set_yticks([])
        # All blocks share one x scale, so label it once at the foot of the page
        # instead of repeating identical ticks under every cluster.
        if i == n - 1:
            ax_bar.tick_params(axis='x', labelsize=_fs(11))
            ax_bar.set_xlabel('Mean Effect Size ± SEM (All Cell Lines)', fontsize=_fs(13))
        else:
            ax_bar.set_xticklabels([])
            ax_bar.tick_params(axis='x', length=2, labelsize=0)
        # Hide all spines except the bottom (x-axis)
        for spine in ['top', 'right', 'left']:
            ax_bar.spines[spine].set_visible(False)
        ax_bar.spines['bottom'].set_visible(True)
        ax_bar.spines['bottom'].set_linewidth(1.2)


        heat_arr = np.ma.masked_invalid(np.array(heatmap_rows))
        cmap = plt.get_cmap('coolwarm').copy()
        cmap.set_bad(color='white')
        num_rows, num_cols = heat_arr.shape

        im = ax_heat.imshow(
            heat_arr[::-1],
            cmap=cmap,
            norm=norm,
            aspect='auto',
            extent=[0, num_cols, -0.5, num_rows - 0.5],
            zorder=1
        )

        for row in range(num_rows):
            for col in range(num_cols):
                if heat_arr.mask[row, col]:
                    x = col + 0.5
                    y = row
                    ax_heat.text(
                        x, y, 'X',
                        ha='center', va='center',
                        fontsize=_fs(16),
                        color='black',
                        zorder=10
                    )

        ax_heat.set_ylim(-0.5, num_rows - 0.5)
        ax_heat.set_yticks([])

        # Pin the heatmap to the bar block: same top and bottom, so row j of the
        # heatmap lines up with bar j, and the two panels are the same height.
        # Width is then derived so each cell stays square in inches --
        # aspect='equal' would have done that by shrinking the axes instead,
        # which is what made the two columns different heights.
        _pairs.append((ax_bar, ax_heat, num_rows, num_cols))
        if i == n - 1:
            xticks = np.arange(len(cell_line_cols)) + 0.5
            ax_heat.set_xticks(xticks)
            ax_heat.set_xticklabels(
                [c.replace("Effect Size ", "").replace(" Boot 0", "") for c in cell_line_cols],
                rotation=45, ha='right', fontsize=_fs(10)
            )
        else:
            ax_heat.set_xticks([])

        if i == 0:
            pass  # column header already reads "Per-Cell Effects"; a title here collided with it

    # Taller bar with ticks every 0.5 - the default gave only two or three.
    import numpy as _np
    t0 = _np.ceil(true_vmin * 2) / 2
    t1 = _np.floor(true_vmax * 2) / 2
    cbar_ax = fig.add_axes([0.40, 0.12 / _fig_h, 0.24, 0.10 / _fig_h])
    cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                      ticks=_np.arange(t0, t1 + 0.5, 0.5))
    cb.set_label('Δlogit Effect Size', fontsize=_fs(12))
    cb.ax.tick_params(labelsize=_fs(10))

    fig.suptitle(title, fontsize=_fs(24), y=1.0 - 0.14 / _fig_h)
    # Header x-positions were tuned for the original 28 in sheet; at 8.5 in the
    # last two collided. Spread to match the 4 column centres at page width.
    fig.text(0.09, _top + 0.14 / _fig_h, "Motif Logo", fontsize=_fs(14), ha='center')
    fig.text(0.30, _top + 0.14 / _fig_h, "RBP Motifs in Cluster", fontsize=_fs(14), ha='center')
    fig.text(0.63, _top + 0.14 / _fig_h, "Effect Size (Mean ± SEM per Exon)", fontsize=_fs(14), ha='center')
    fig.text(0.875, _top + 0.14 / _fig_h, "Per-Cell Effects", fontsize=_fs(14), ha='center')

    plt.subplots_adjust(left=0.05, right=0.92, top=0.96, bottom=0.05, wspace=0.3, hspace=0.4)

    # Position the heatmaps only now. subplots_adjust recomputes every
    # gridspec-managed axes from its subplotspec, so anything set inside the loop
    # above is silently discarded -- which is why earlier attempts at this had no
    # effect on the output at all.
    #   - heatmap sits against a fixed right margin, same top/bottom as its bars,
    #     so row j lines up with bar j and the two blocks are the same height
    #   - its width is num_cols * row_height, so cells come out square in inches
    #   - the bar axis is pulled back by LABEL_GAP to leave room for the longest
    #     gene label ("NDUFAF5 exon 9") instead of it running under the heatmap
    _fw, _fh = fig.get_size_inches()
    for _axb, _axh, _rows, _cols in _pairs:
        _bb = _axb.get_position()
        _row_h = _bb.height / max(_rows, 1)
        _w = _cols * _row_h * (_fh / _fw)
        _x0 = HEAT_RIGHT - _w
        _axh.set_position([_x0, _bb.y0, _w, _bb.height])
        _axb.set_position([_bb.x0, _bb.y0,
                           max(0.05, _x0 - LABEL_GAP - _bb.x0), _bb.height])
    mpl.rcParams['pdf.fonttype'] = 42   # TrueType
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none' 
    plt.savefig(output_file, dpi=150,
                bbox_inches=None if page_height is not None else 'tight')
    plt.close()

    summary_rows = []
    cell_line_cols = [col for col in full_effects_df.columns if col.startswith("Effect Size") and "Boot 0" in col]
    rename_map = {col: "Mean " + col.replace("Effect Size ", "").replace(" Boot 0", "").replace("HEK", "HEK293") for col in cell_line_cols}
    
    for motif in motif_order:
        motif_rows = df[df['original_motif'] == motif]
        for exon in motif_rows['gene_exon']:
            match = full_effects_df[(full_effects_df['gene_exon'] == exon) & (full_effects_df['motif'] == motif)]
            if match.empty:
                continue
            row = match.iloc[0]
            rbp_str = motif_rows['rbps_in_cluster'].iloc[0]
            cell_vals = {rename_map[col]: float(row[col]) if pd.notna(row[col]) else np.nan for col in cell_line_cols}
            mean_val = np.nanmean(list(cell_vals.values()))
            summary_rows.append({
                'motif': motif,
                'gene_exon': exon,
                'rbps_in_cluster': rbp_str,
                **cell_vals,
                'mean_effect': mean_val
            })
    
    summary_df = pd.DataFrame(summary_rows)
    csv_path = output_file.replace('.pdf', '_summary_full.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Exported full summary to {csv_path}")






def plot_effect_size_vs_average(merged_df, full_effects_df, output_path_prefix, file_suffix='effects_exon', cluster_to_motifs=None, motif_rbps=None):
    event_keys = set(zip(merged_df['motif'], merged_df['event_id']))
    df = full_effects_df[full_effects_df.apply(lambda r: (r['motif'], r['event_id']) in event_keys, axis=1)].copy()

    cell_lines = ['HEK', 'HeLa', 'K562', 'HMC3', 'MCF7']
    effect_cols = [f'Effect Size {cl} Boot 0' for cl in cell_lines]
    df['Average All Cell Lines'] = df[effect_cols].mean(axis=1)

    # Filter to motif/event_id pairs where all 5 cell lines have values
    counts = df.groupby(['motif', 'event_id'])[effect_cols].apply(lambda x: x.notna().all().all())
    valid_keys = set(counts[counts].index)
    df = df[df.apply(lambda r: (r['motif'], r['event_id']) in valid_keys, axis=1)].copy()
    
    # Keep only motifs with effects in at least 3 exon families
    motif_counts = df.groupby('motif')['event_id'].nunique()
    valid_motifs = motif_counts[motif_counts >= 3].index
    df = df[df['motif'].isin(valid_motifs)].copy()


    df_long = df.melt(id_vars=['Average All Cell Lines', 'motif', 'event_id'], value_vars=effect_cols,
                      var_name='Cell Line', value_name='Effect Size')
    df_long['Cell Line'] = df_long['Cell Line'].str.extract(r'Effect Size (.+) Boot 0')
    df_long['label'] = df_long['motif'].str.replace(r'cluster_0*(\d+)', r'cluster \1', regex=True)
    df_long['Distance from Line'] = abs(df_long['Effect Size'] - df_long['Average All Cell Lines'])

    custom_rgb_colors = {
        'HEK': '#E984B6',
        'HeLa': '#7FBE7E',
        'MCF7': '#807CB9',
        'HMC3': '#EF4025',
        'K562': '#F9AE33'
    }

    # === Per-Exon Scatter Plot ===
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axline((0, 0), slope=1, color='grey', linestyle='--', linewidth=1)
    sns.scatterplot(
        data=df_long,
        x='Average All Cell Lines',
        y='Effect Size',
        hue='Cell Line',
        palette=custom_rgb_colors,
        alpha=0.7,
        s=40,
        edgecolor="none",
        ax=ax
    )
    ax.set_xlabel('Average Effect Size (All Cell Lines)', fontsize=_fs(14))
    ax.set_ylabel('Individual Effect Size', fontsize=_fs(14))
    ax.set_title('Individual vs. Average Effect', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Cell Line', title_fontsize=_fs(12), fontsize=11)

    outliers = df_long[df_long['Distance from Line'] > 0.4]
    for _, row in outliers.iterrows():
        x, y = row['Average All Cell Lines'], row['Effect Size']
        dx = 0.05
        ha = 'left' if x <= y else 'right'
        x_text = x + dx if ha == 'left' else x - dx
        ax.text(x_text, y, row['label'], fontsize=8, ha=ha, va='center', color='black', zorder=10)

    fig.tight_layout()
    fig.savefig(f"{output_path_prefix}_{file_suffix}_individual_vs_average_plot.pdf")
    plt.close()
    df_long.to_csv(f"{output_path_prefix}_{file_suffix}_data.csv", index=False)

    labeled_outliers = []
    for _, row in outliers.iterrows():
        motif_cluster = row['motif']
        sub_motifs = cluster_to_motifs.get(motif_cluster, []) if cluster_to_motifs else []
        rbp_set = set()
        for sub_motif in sub_motifs:
            rbps = motif_rbps.get(sub_motif, "")
            if rbps:
                rbp_set.update([r.strip() for r in rbps.split(",")])
        labeled_outliers.append({
            'motif_cluster': motif_cluster,
            'rbps': ', '.join(sorted(rbp_set)) if rbp_set else "NA",
            'event_id': row['event_id'],
            'cell_line': row['Cell Line'],
            'average_effect': row['Average All Cell Lines'],
            'individual_effect': row['Effect Size'],
            'distance_from_line': row['Distance from Line']
        })
    pd.DataFrame(labeled_outliers).to_csv(f"{output_path_prefix}_{file_suffix}_labeled_outliers.csv", index=False)

    # === Cluster-Level Scatter Plot ===
    avg_df = df_long.groupby(['motif', 'Cell Line']).agg({
        'Effect Size': 'mean',
        'Average All Cell Lines': 'mean'
    }).reset_index()
    avg_df['label'] = avg_df['motif'].str.replace(r'cluster_0*(\d+)', r'cluster \1', regex=True)
    avg_df['Distance from Line'] = abs(avg_df['Effect Size'] - avg_df['Average All Cell Lines'])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axline((0, 0), slope=1, color='grey', linestyle='--', linewidth=1)
    sns.scatterplot(
        data=avg_df,
        x='Average All Cell Lines',
        y='Effect Size',
        hue='Cell Line',
        palette=custom_rgb_colors,
        alpha=0.7,
        s=60,
        edgecolor="none",
        ax=ax
    )
    ax.set_xlabel('Average Effect Size (All Cell Lines)', fontsize=_fs(14))
    ax.set_ylabel('Cluster Mean Effect Size (Per Cell Line)', fontsize=_fs(14))
    ax.set_title('Cluster-Level Avg vs. All-Cell Avg', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Cell Line', title_fontsize=_fs(12), fontsize=11)

    outliers_avg = avg_df[avg_df['Distance from Line'] > 0.25]
    for _, row in outliers_avg.iterrows():
        x, y = row['Average All Cell Lines'], row['Effect Size']
        dx = 0.05
        ha = 'left' if x <= y else 'right'
        x_text = x + dx if ha == 'left' else x - dx
        ax.text(x_text, y, row['label'], fontsize=8, ha=ha, va='center', color='black', zorder=10)
    r, p = pearsonr(avg_df['Average All Cell Lines'], avg_df['Effect Size'])

    # Annotate on plot
    ax.text(0.05, 0.95, f"$r$ = {r:.2f}", transform=ax.transAxes,
        ha='left', va='top', fontsize=_fs(12))
    fig.tight_layout()
    fig.savefig(f"{output_path_prefix}_{file_suffix}_cluster_level_mean_plot.pdf")
    plt.close()

    avg_df.to_csv(f"{output_path_prefix}_{file_suffix}_cluster_level_data.csv", index=False)
    outliers_avg.to_csv(f"{output_path_prefix}_{file_suffix}_cluster_level_outliers.csv", index=False)

    # === Optional: Export motif-cell summary ===
    grouped_rows = []
    df_grouped = df_long.groupby(['motif', 'event_id', 'Cell Line'])
    for (motif_cluster, event_id, cell_line), group in df_grouped:
        avg_effect = group['Average All Cell Lines'].iloc[0]
        ind_effect = group['Effect Size'].iloc[0]
        sub_motifs = cluster_to_motifs.get(motif_cluster, []) if cluster_to_motifs else []
        rbp_set = set()
        for sub_motif in sub_motifs:
            rbps = motif_rbps.get(sub_motif, "")
            if rbps:
                rbp_set.update([r.strip() for r in rbps.split(",")])
        grouped_rows.append({
            'motif_cluster': motif_cluster,
            'rbps': ', '.join(sorted(rbp_set)) if rbp_set else "NA",
            'event_id': event_id,
            'cell_line': cell_line,
            'average_effect': avg_effect,
            'individual_effect': ind_effect
        })
    expanded_df = pd.DataFrame(grouped_rows)
    expanded_df.to_csv(f"{output_path_prefix}_{file_suffix}_motifs_in_cluster.csv", index=False)
    print(f"Exported deduplicated RBP-mapped motifs to {output_path_prefix}_{file_suffix}_motifs_in_cluster.csv")



def plot_table_bar_paged(df, pwm_dict, output_file, title, event_effects_by_motif,
                         original_to_display_name, full_effects_df, cluster_to_motifs):
    """Render plot_table_bar across as many US Letter pages as the clusters need,
    then merge them into one multi-page PDF."""
    import tempfile, os as _os
    from PyPDF2 import PdfMerger

    order = (df.groupby("original_motif")["mean_effect"].mean()
               .sort_values(ascending=False).index.tolist())
    budget = int((PAGE_H - PAGE_MARGIN) / ROW_IN)

    pages, cur, used = [], [], 0
    for m in order:
        r = max(1, len(event_effects_by_motif.get(m, [])))
        if cur and used + r > budget:
            pages.append(cur); cur, used = [], 0
        cur.append(m); used += r
    if cur:
        pages.append(cur)

    # One colour scale and one x range for the whole document.
    _all = [m for v in event_effects_by_motif.values() for (_, m, _) in v]
    scale = (min(_all), max(_all)) if _all else None

    tmp = tempfile.mkdtemp()
    parts = []
    for i, page_motifs in enumerate(pages, 1):
        sub = df[df["original_motif"].isin(page_motifs)].copy()
        sub["original_motif"] = sub["original_motif"].astype(str)
        ee = {m: v for m, v in event_effects_by_motif.items() if m in page_motifs}
        f = _os.path.join(tmp, "p%02d.pdf" % i)
        plot_table_bar(sub, pwm_dict, f,
                       "%s  (page %d of %d)" % (title, i, len(pages)),
                       ee, original_to_display_name, full_effects_df, cluster_to_motifs,
                       scale=scale, page_height=PAGE_H)
        parts.append(f)

    mg = PdfMerger()
    for f in parts:
        mg.append(f)
    mg.write(output_file); mg.close()
    print("  %s -> %d pages, %d clusters" % (_os.path.basename(output_file), len(pages), len(order)))

def plot_top_bottom_dotplot_with_logos(merged_df, pwm_dict, output_file, original_to_display_name, full_effects_df, top_n=10):
    event_keys = set(zip(merged_df['motif'], merged_df['event_id']))
    df = full_effects_df[full_effects_df.apply(lambda r: (r['motif'], r['event_id']) in event_keys, axis=1)].copy()

    cell_lines = ['HEK', 'HeLa', 'K562', 'HMC3', 'MCF7']
    effect_cols = [f'Effect Size {cl} Boot 0' for cl in cell_lines]
    df['mean'] = df[effect_cols].mean(axis=1)
    df['sem'] = df[effect_cols].std(axis=1, ddof=0) / np.sqrt(df[effect_cols].count(axis=1))

    df['gene_exon'] = merged_df.set_index(['motif', 'event_id'])['gene_exon'].reindex(df.set_index(['motif', 'event_id']).index).values
    df_sorted = df.sort_values('mean')
    df_top_bottom = pd.concat([df_sorted.head(top_n), df_sorted.tail(top_n)]).sort_values('mean')
    df_top_bottom['y_label'] = df_top_bottom.apply(lambda row: f"{original_to_display_name.get(row['motif'], row['motif'])} ({row['gene_exon']})", axis=1)
    # Export CSV
    export_file = output_file.replace('.pdf', '_data.csv')
    df_top_bottom.to_csv(export_file, index=False)
    print(f"Exported data to {export_file}")
    
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(df_top_bottom) + 1))
    y_pos = np.arange(len(df_top_bottom))
    ax.errorbar(df_top_bottom['mean'], y_pos, xerr=df_top_bottom['sem'], fmt='o', color='black', capsize=3)
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top_bottom['y_label'], fontsize=10)
    ax.set_xlabel('Mean Effect Size (Δlogit) ± SEM')
    ax.set_title('Top & Bottom 10 Motif–Exon Effects')
    plt.subplots_adjust(left=0.3)
    
    
    for i, (_, row) in enumerate(df_top_bottom.iterrows()):
        motif = row['motif']
        if motif in pwm_dict:
            y_frac = (i - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
            logo_ax = fig.add_axes([0.01, y_frac - 0.02, 0.03, 0.04])
            pwm = pwm_dict[motif]
            logomaker.Logo(pwm, ax=logo_ax, color_scheme='classic').style_spines(visible=False)
            logo_ax.set_xticks([]); logo_ax.set_yticks([]); logo_ax.axis('off')

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()





def main():
    base_dir = '/ESL/ESL_MPRA/Figure_6/outputs/effect_size_MAY_full'
    supertable_path = '/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/st_corrected.csv'
    # MAY reprocessing changed event_id from construct coords (span 160) to exon
    # coords (span ~74). The older st_final_* tables still carry the old convention
    # and join at 0/128 against the MAY effect sizes, which silently produced
    # all-NaN rows: no bars and an all-X heatmap. st_corrected.csv joins 128/128.
    cluster_tab = '/ESL/ESL_MPRA/Figure_6/inputs/motif_clusters/clusters.tab'
    transfac_file = '/ESL/ESL_MPRA/Figure_6/inputs/motif_clusters/Root_motifs.tf'
    rbp_mapping_file = '/ESL/ESL_MPRA/Figure_6/inputs/motif_clusters/rncmpt_to_rbpname_full_mapping.csv'
    plot_dir = '/ESL/ESL_MPRA/Figure_6/outputs/concordance_MAY_full'
    output_dir = '/ESL/ESL_MPRA/Figure_6/outputs/figure6_elements_MAY_full'
    os.makedirs(output_dir, exist_ok=True)

    supertable_df = pd.read_csv(supertable_path)
    motif_map_df = pd.read_csv(rbp_mapping_file)
    motif_map_df['Motif_ID'] = motif_map_df['Motif_ID'].str.strip()
    motif_map_df['RBP_Name'] = motif_map_df['RBP_Name'].str.strip()
    motif_renamer = dict(zip(motif_map_df['Motif_ID'], motif_map_df['RBP_Name']))
    print("=== Sample motif renamings ===")
    print(list(motif_renamer.items())[:10])

    pwm_dict = parse_transfac(transfac_file)

    cluster_to_rbps = {}
    cluster_to_motifs = {}
    
    for line in open(cluster_tab, 'r'):
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            cluster = parts[0].strip()
            motif_list_full = parts[1].split(",")
            motif_list_short = []
            for motif_id in motif_list_full:
                match = re.search(r'(motif\d+)', motif_id)
                if match:
                    motif_list_short.append(match.group(1))
            cluster_to_motifs[cluster] = motif_list_short

            rbp_field = parts[-1]
            rbps = []
            for motif_entry in rbp_field.split(','):
                try:
                    name_part = motif_entry.split(':', 1)[1].strip()
                    gene_match = re.match(r'[A-Za-z0-9]+', name_part)
                    if gene_match:
                        rbp_name = gene_match.group(0)
                        rbps.append(motif_renamer.get(rbp_name, rbp_name).upper())
                except Exception:
                    continue
            cluster_to_rbps[cluster] = sorted(set(rbps))

    motif_to_cluster = {}
    for cluster, motifs in cluster_to_motifs.items():
        for motif_id in motifs:
            motif_short = re.search(r'(motif\d+)', motif_id)
            if motif_short:
                motif_key = motif_short.group(1)
                motif_to_cluster[motif_key] = cluster

    cell_lines = ["HEK", "HeLa", "K562", "HMC3", "MCF7"]
    fimo_base = '/ESL/ESL_MPRA/Figure_6/outputs/effect_size_MAY_full'

    for region in ['exon', 'intron1', 'intron2']:
        print(f'Processing region: {region}')
        dfs = []
        for cell in cell_lines:
            f = f'{plot_dir}/high_concordance_{cell}_{region}.csv'
            if not os.path.exists(f):
                print(f"Missing file: {f}")
                continue
            df_cell = pd.read_csv(f)
            df_cell = df_cell[['motif', 'event_id', f'Effect Size {cell} Boot 0']]
            dfs.append(df_cell)

        if len(dfs) == 0:
            print(f"No files available for region {region}")
            continue

        from functools import reduce
        merged_df = reduce(lambda left, right: pd.merge(left, right, on=['motif', 'event_id'], how='outer'), dfs)

        cell_line_cols = [col for col in merged_df.columns if col.startswith("Effect Size")]
        merged_df['n_cells'] = merged_df[cell_line_cols].notna().sum(axis=1)
        merged_df = merged_df[merged_df['n_cells'] >= 2].dropna(subset=['motif', 'event_id'])
        if merged_df.empty:
            print(f"No valid entries with ≥2 cell lines for {region}")
            continue

        fimo_file = f"{fimo_base}/out_MAY_full_df_{region}.csv"
        if not os.path.exists(fimo_file):
            print(f"Skipping region {region} — FIMO file not found: {fimo_file}")
            continue
        fimo_df = pd.read_csv(fimo_file, low_memory=False)

        # Extract all motif IDs (motifXXX) from 'motifs_in_cluster'
        fimo_df['motif_id_list'] = fimo_df['motifs_in_cluster'].fillna('').str.findall(r'motif\d+')
        
        # Filter by event_id overlap with merged_df
        valid_event_ids = set(merged_df['event_id'].dropna())
        fimo_filtered = fimo_df[fimo_df['event_id'].isin(valid_event_ids)].copy()
        
        # Collect all unique motifs from filtered region
        fimo_motifs_present = set(m for sublist in fimo_filtered['motif_id_list'] for m in sublist)
        
        print(f"All FIMO motifs present (first 10): {list(fimo_motifs_present)[:10]}")
        print(f"FIMO motif count for {region}: {len(fimo_motifs_present)}")


        merged_df['gene_exon'] = merged_df['event_id'].map(dict(zip(supertable_df['event_id'], supertable_df['gene_exon'])))
        merged_df['original_motif'] = merged_df['motif'].str.strip()
        merged_df['motif_id_short'] = merged_df['original_motif'].str.extract(r'cluster_0*(\d+)')[0]
        merged_df['motif_id_short'] = 'motif' + merged_df['motif_id_short']

        # === Build cluster_to_rbps_filtered based only on FIMO-supported motifXXX IDs ===
        cluster_to_rbps_filtered = {}
        
        motif_to_rbps_all = defaultdict(set)
        with open(cluster_tab, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                rbp_field = parts[-1]
                entries = rbp_field.split(',')
                for entry in entries:
                    if "|" in entry and ":" in entry:
                        motif_id = entry.split("|")[0].strip()  # e.g., motif293
                        try:
                            gene_part = entry.split(":", 1)[1].strip()
                            gene_match = re.match(r'([A-Za-z0-9]+)', gene_part)
                            if gene_match:
                                rbp = gene_match.group(1)
                                rbp_sem = motif_renamer.get(rbp, rbp).upper()
                                motif_to_rbps_all[motif_id].add(rbp_sem)
                        except Exception:
                            continue
        used_clusters = merged_df['original_motif'].unique()  # cluster_054, etc.

        cluster_to_rbps_filtered = {}
        for cluster_id, motifs in cluster_to_motifs.items():
            fimo_motifs_in_cluster = [m for m in motifs if m in fimo_motifs_present]
            rbps = set()
            for motif in fimo_motifs_in_cluster:
                rbps.update(motif_to_rbps_all.get(motif, []))
            cluster_to_rbps_filtered[cluster_id] = sorted(rbps) 


        # === Add RBP display field ===
        merged_df['rbps_in_cluster'] = merged_df['original_motif'].map(
        lambda x: ', '.join(cluster_to_rbps_filtered.get(x, [])) if cluster_to_rbps_filtered.get(x) else "NA")


        print("=== Sample RBP assignments ===")
        print(merged_df[['original_motif', 'rbps_in_cluster']].drop_duplicates().head(20))
        original_to_display_name = {m: motif_renamer.get(m, m) for m in merged_df['original_motif'].unique()}

        motif_to_events = {}
        for motif, subdf in merged_df.groupby('original_motif'):
            event_list = []
            for _, row in subdf.iterrows():
                exon = row['gene_exon']
                values = row[cell_line_cols].astype(float).values
                mean_val = np.nanmean(values)
                sem_val = np.nanstd(values) / np.sqrt(np.count_nonzero(~np.isnan(values)))

                event_list.append((exon, mean_val, sem_val))
            motif_to_events[motif] = event_list

        merged_df['mean_effect'] = merged_df[cell_line_cols].mean(axis=1)

        full_effects_path = f'{base_dir}/out_MAY_full_boots_0_{region}.csv'
        full_effects_df = pd.read_csv(full_effects_path)
        full_effects_df['gene_exon'] = full_effects_df['event_id'].map(dict(zip(supertable_df['event_id'], supertable_df['gene_exon'])))

        # --- Figure 6D / 6E: the same renderer restricted to the clusters called
        # out in the manuscript panels, so their styling matches Data S2/S3.
        # Identified from the V4 figure by logo consensus, effect sign and the
        # RBP names in the legend:
        #   6D exon    050 YBX1 | 054 SRSF1 | 023 FUS/ZNF346 | 061 HNRNPH1/H3
        #   6E intron1 026 ZCRB1 | 143 PTBP1 | 075 PABPN1     | 118 RC3H1
        PANEL_CLUSTERS = {
            'exon':    ['cluster_050', 'cluster_054', 'cluster_023', 'cluster_061'],
            'intron1': ['cluster_026', 'cluster_143', 'cluster_075', 'cluster_118'],
        }
        PANEL_NAME = {'exon': '6D', 'intron1': '6E'}
        if region in PANEL_CLUSTERS:
            keep = PANEL_CLUSTERS[region]
            panel_df = merged_df[merged_df['original_motif'].isin(keep)].copy()
            panel_ee = {m: v for m, v in motif_to_events.items() if m in keep}
            missing = [c for c in keep if c not in set(panel_df['original_motif'])]
            if missing:
                print("  [WARN] %s: clusters absent from the May data: %s" % (PANEL_NAME[region], missing))
            if len(panel_df):
                plot_table_bar(
                    panel_df,
                    pwm_dict=pwm_dict,
                    output_file=os.path.join(output_dir, 'fig%s_%s_panel.pdf' % (PANEL_NAME[region], region)),
                    title='Effect Sizes (All Cell Lines) - %s' % ('Exon' if region == 'exon' else "5' Intron"),
                    event_effects_by_motif=panel_ee,
                    original_to_display_name=original_to_display_name,
                    full_effects_df=full_effects_df,
                    cluster_to_motifs=cluster_to_motifs,
                )
                print("  fig%s_%s_panel.pdf -> %d clusters, %d rows" % (
                    PANEL_NAME[region], region, len(keep) - len(missing), len(panel_df)))

        plot_table_bar_paged(
            merged_df,
            pwm_dict=pwm_dict,
            output_file=os.path.join(output_dir, f'final_motif_{region}_barplot_all_cells.pdf'),
            title=f'Motif Effects Across {region.capitalize()} Contexts (Mean ± SEM, All Cell Lines)',
            event_effects_by_motif=motif_to_events,
            original_to_display_name=original_to_display_name,
            full_effects_df=full_effects_df,
            cluster_to_motifs=cluster_to_motifs
        )

        plot_top_bottom_dotplot_with_logos(
            merged_df,
            pwm_dict,
            os.path.join(output_dir, f'final_motif_{region}_top_bottom_all_cells.pdf'),
            original_to_display_name,
            full_effects_df,
            top_n=10
        )
        
        motif_rbps = {}
        for cluster_id, motifs in cluster_to_motifs.items():
            for motif in motifs:
                rbps = motif_to_rbps_all.get(motif, [])
                if rbps:
                    motif_rbps[motif] = ', '.join(sorted(rbps))

        plot_effect_size_vs_average(
            merged_df,
            full_effects_df,
            output_path_prefix=str(Path(output_dir) / f'filtered_scatter_{region}'),
            file_suffix=f'{region}',
            cluster_to_motifs=cluster_to_motifs,
            motif_rbps=motif_rbps
        )


        
        

if __name__ == "__main__":
    main()
