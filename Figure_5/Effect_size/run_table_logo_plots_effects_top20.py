import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import re
import logomaker
from matplotlib.gridspec import GridSpec

def load_data(effects_csv, super_table_csv, cluster_table):
    rename_mapping = {
        'Mean HEK Average': 'HEK',
        'Mean HeLa Average': 'HeLa',
        'Mean K562 Average': 'K562',
        'Mean HMC3 Average': 'HMC3',
        'Mean MCF7 Average': 'MCF7'
    }
    effect_size_columns = list(rename_mapping.values())

    effects_df = pd.read_csv(effects_csv)
    effects_df = effects_df.rename(columns=rename_mapping)

    supertable_df = pd.read_csv(super_table_csv)
    eventid_to_geneexon = dict(zip(supertable_df['event_id'], supertable_df['gene_exon']))
    effects_df['gene_exon'] = effects_df['event_id'].map(eventid_to_geneexon)

    cluster_to_rbps = {}
    with open(cluster_table, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                cluster = parts[0].strip()
                rbp_field = parts[-1]
                rbps = []
                motif_entries = rbp_field.split(',')
                for motif_entry in motif_entries:
                    try:
                        name_part = motif_entry.split(':', 1)[1]
                        gene_match = re.match(r'[A-Za-z0-9]+', name_part)
                        if gene_match:
                            rbps.append(gene_match.group(0).upper())
                    except Exception:
                        continue
                cluster_to_rbps[cluster] = sorted(set(rbps))
    effects_df['rbps_in_cluster'] = effects_df['motif'].map(
        lambda m: ', '.join(cluster_to_rbps.get(m, []))
    )

    effects_df['mean_effect'] = effects_df[effect_size_columns].mean(axis=1)
    effects_df['std_effect'] = effects_df[effect_size_columns].std(axis=1)

    return effects_df, effect_size_columns

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

def wrap_rbps(text, max_per_line=5):
    if not text or pd.isna(text):
        return "NA"
    tokens = text.split(', ')
    wrapped = "\n".join([", ".join(tokens[i:i+max_per_line]) for i in range(0, len(tokens), max_per_line)])
    return wrapped

def plot_table_bar(df, value_column, error_column, pwm_dict, output_file, title):
    n = len(df)
    fig = plt.figure(figsize=(24, n * 1.3))
    gs = GridSpec(n, 5, width_ratios=[2.4, 1.8, 1.6, 6.0, 5.0], figure=fig)

    true_vmin = df[value_column].min()
    true_vmax = df[value_column].max()
    norm = mcolors.TwoSlopeNorm(vmin=true_vmin, vcenter=0, vmax=true_vmax)
    cmap = plt.get_cmap('coolwarm')

    padded_vmin = true_vmin - 0.6
    padded_vmax = true_vmax + 0.6

    df = df.sort_values(by=value_column, ascending=False)

    for i, (_, row) in enumerate(df.iterrows()):
        motif = row['motif']
        effect = row[value_column]
        error = row[error_column] if error_column else None

        # Column 1: Motif Logo
        ax_logo = fig.add_subplot(gs[i, 0])
        if motif in pwm_dict:
            pwm_df = pwm_dict[motif]
            logomaker.Logo(pwm_df, ax=ax_logo, color_scheme='classic')
            ax_logo.set_xticks([])
            ax_logo.set_yticks([])
            ax_logo.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        else:
            ax_logo.text(0.5, 0.5, "No Logo", ha='center', va='center', color='red', fontsize=11)
            ax_logo.axis('off')

        # Column 2: Gene Exon
        ax_exon = fig.add_subplot(gs[i, 1])
        ax_exon.text(0.5, 0.5, row['gene_exon'] if pd.notna(row['gene_exon']) else "NA",
                     ha='center', va='center', fontsize=16, fontweight='bold')
        ax_exon.axis('off')

        # Column 3: Cluster ID
        ax_cluster = fig.add_subplot(gs[i, 2])
        ax_cluster.text(0.5, 0.5, motif, ha='center', va='center', fontsize=16, fontweight='bold')
        ax_cluster.axis('off')

        # Column 4: RBP motifs
        ax_rbps = fig.add_subplot(gs[i, 3])
        rbp_text = wrap_rbps(row['rbps_in_cluster'], max_per_line=5)
        ax_rbps.text(0, 0.5, rbp_text, ha='left', va='center', fontsize=13)
        ax_rbps.axis('off')

        # Column 5: Bar plot
        ax_bar = fig.add_subplot(gs[i, 4])
        color = cmap(norm(effect))
        ax_bar.barh([motif], [effect], color=color, xerr=[[error]] if error else None,
                    capsize=5, edgecolor='black')
        ax_bar.axvline(0, color='black', linestyle='--')
        ax_bar.set_xlim(padded_vmin, padded_vmax)
        ax_bar.set_yticks([])
        if i != n - 1:
            ax_bar.set_xticks([])
        else:
            ax_bar.set_xlabel('Effect Size', fontsize=17)

        for spine in ax_bar.spines.values():
            spine.set_visible(False)

    # Title + Header
    fig.suptitle(title, fontsize=30, fontweight='bold', y=1.03)
    fig.text(0.07, 1.015, "Motif Logo", fontsize=18, fontweight='bold')
    fig.text(0.27, 1.015, "Gene/Exon", fontsize=18, fontweight='bold')
    fig.text(0.41, 1.015, "Cluster", fontsize=18, fontweight='bold')
    fig.text(0.59, 1.015, "RBP Motifs in Cluster", fontsize=18, fontweight='bold')
    fig.text(0.89, 1.015, "Effect Size", fontsize=18, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file, dpi=100)
    plt.close()

def main():
    effects_csv = '/ESL/Figures_SK/effect_sizes_debug/out_04_28_2025_delta_logit_on_clusters_on_pooled_no_rev_strand/out_04_28_2025_delta_logit_on_clusters_on_pooled_no_rev_strand_boots_0_exon.csv'
    super_table_csv = '/ESL/Figures_SK/General_preprocessing/fix_supertable/final_super_table_04_21_2025_with_event_and_gene_exon.csv'
    cluster_tab = '/ESL/Figures_SK/Cluster_motifs/rsat_out_04_28_2025/motifs_04_28_2025_tables/clusters.tab'
    transfac_file = '/ESL/Figures_SK/Cluster_motifs/rsat_out_04_28_2025/motifs_04_28_2025_motifs/root_motifs/Root_motifs.tf'
    output_dir = '/ESL/Figures_SK/effect_sizes_debug/out_04_28_2025_delta_logit_on_clusters_on_pooled_no_rev_strand/plots/bar_plot'
    os.makedirs(output_dir, exist_ok=True)

    effects_df, cl_cols = load_data(effects_csv, super_table_csv, cluster_tab)
    pwm_dict = parse_transfac(transfac_file)

    combined_df = effects_df[['motif', 'mean_effect', 'std_effect', 'gene_exon', 'rbps_in_cluster']].copy()
    top20 = combined_df.nlargest(20, 'mean_effect')
    bottom20 = combined_df.nsmallest(20, 'mean_effect')
    combined = pd.concat([top20, bottom20])

    plot_table_bar(
        combined, 'mean_effect', 'std_effect', pwm_dict,
        f"{output_dir}/combined_top20_bottom20_table_barplot.pdf",
        "Top and Bottom 20 Motifs (Combined Across Cell Lines)"
    )

    for cl in cl_cols:
        df = effects_df[['motif', cl, 'gene_exon', 'rbps_in_cluster']].rename(columns={cl: 'effect'})
        top20 = df.nlargest(20, 'effect')
        bottom20 = df.nsmallest(20, 'effect')
        combined = pd.concat([top20, bottom20])
        plot_table_bar(
            combined, 'effect', None, pwm_dict,
            f"{output_dir}/top20_bottom20_table_barplot_{cl}.pdf",
            f"Top and Bottom 20 Motifs ({cl})"
        )

if __name__ == "__main__":
    main()
