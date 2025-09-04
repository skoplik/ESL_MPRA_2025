import os
import sys
import getopt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import MatplotlibDeprecationWarning
from collections import defaultdict
from scipy.stats import kruskal
import numpy as np
import math
import statistics
import ast
import warnings
from tqdm import tqdm
import re
import random
from scipy.stats import zscore
import scipy.stats as stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from scipy.stats import kruskal, mannwhitneyu
import warnings
warnings.filterwarnings("ignore")




# Helper Functions
pd.set_option('display.max_colwidth', None)  # Show full column content
def parse_arguments():
    """Parse command-line arguments."""
    try:
        opts, _ = getopt.getopt(
            sys.argv[1:], 
            "", 
            ["fimo_out_file=","numboots=", "output_dir=", "output_prefix=","indv_cell_lines_file=", "cluster_file="]
        )
        opts = dict(opts)

        # Check if all required keys are present
        required_keys = ['--fimo_out_file','--numboots','--indv_cell_lines_file','--cluster_file', '--output_dir', '--output_prefix']
        for key in required_keys:
            if key not in opts:
                raise KeyError(f"Missing required argument: {key}")
        
        return opts
    except getopt.GetoptError as e:
        print(f"Failed to parse arguments: {str(e)}")
        sys.exit(2)
    except KeyError as e:
        print(f"Error: {str(e)}")
        sys.exit(2)


def create_output_dir(output_dir):
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")
import re

def extract_var_positions(seq_type_str):
    if pd.isna(seq_type_str):
        return []
    if seq_type_str=='none':
        return []
    return [int(m.group(1)) for m in re.finditer(r'(\d+):\s*[ACGT]>[ACGT]', seq_type_str)]
    
def load_data(fimo_out_file, indv_cell_lines_file, cluster_file):
    """Load necessary data files into pandas DataFrames."""
    print("Loading data files...")
    # Build motif → cluster map
    cluster_df = pd.read_csv(cluster_file, sep='\t')
    motif_to_cluster = {}
    for _, row in cluster_df.iterrows():
        cluster_id = row["cluster"]
        for full_id in row["id"].split(","):
            match = re.search(r"(motif\d+)", full_id)
            if match:
                motif = match.group(1)
                motif_to_cluster[motif] = cluster_id
                
    # Load files
    fimo_df = pd.read_csv(
    fimo_out_file,
    sep='\t',
    header=None,
    names=[
        'motif_id', 'alt_id', 'Reference', 'start', 'stop',
        'strand', 'score', 'p-value', 'q-value', 'matched_sequence'
    ],
    usecols=[0, 2, 3, 4, 5 , 7, 9],  # drop col 1,4
    dtype=str,
    index_col=False)
    
    print(fimo_df)
  
    # Extract 'motif###' (e.g., from 'motif123|Attract:XYZ')
    fimo_df['motif_numeric'] = fimo_df['motif_id'].str.extract(r'motif(\d+)', expand=False).astype(int)
    
    # Sort by motif number
    fimo_df = fimo_df.sort_values('motif_numeric').drop(columns='motif_numeric')
  
    print(fimo_df)
    
    fimo_df["motif_id"] = fimo_df["motif_id"].str.extract(r"(motif\d+)", expand=False)
    # Map motif_id to cluster
    fimo_df["cluster"] = fimo_df["motif_id"].map(motif_to_cluster)
    fimo_df = fimo_df.dropna(subset=["cluster"])
    
    # Group to annotate which motifs matched per (Reference, cluster)
    motif_annotations = (
        fimo_df.groupby(["Reference", "cluster"])["motif_id"]
        .unique()
        .apply(lambda x: ",".join(sorted(x)))
        .reset_index()
        .rename(columns={"motif_id": "motifs_in_cluster"})
    )
    
    # Merge back with original FIMO hits
    fimo_df = pd.merge(
        fimo_df,
        motif_annotations,
        on=["Reference", "cluster"],
        how="left"
    )
    
    # Deduplicate to one row per (Reference, cluster)
    fimo_df = fimo_df.drop_duplicates(subset=["Reference", "cluster"])
    
    # Replace motif_id with cluster and retain annotation
    fimo_df["motif_id"] = fimo_df["cluster"]
    fimo_df.drop(columns=["cluster"], inplace=True)   
    
    indv_cell_lines_df = pd.read_csv(indv_cell_lines_file, sep=',', low_memory=False)  # PSIs from each cell line
    indv_cell_lines_df['intron1_start'] = 0
    indv_cell_lines_df['intron1_end'] = indv_cell_lines_df['intron1'].apply(len)-1
    indv_cell_lines_df['exon_start'] = indv_cell_lines_df['intron1'].apply(len) 
    indv_cell_lines_df['exon_end'] = indv_cell_lines_df['intron1'].apply(len) + indv_cell_lines_df['exon'].apply(len)-1
    indv_cell_lines_df['intron2_start'] = indv_cell_lines_df['intron1'].apply(len) + indv_cell_lines_df['exon'].apply(len) 
    indv_cell_lines_df['var_pos'] = indv_cell_lines_df['snp'].apply(extract_var_positions)
        
    return fimo_df, indv_cell_lines_df



def merge_data_fimo(fimo_df, psi_supertable_merged):
    """Merge FIMO results with PSI data."""
    print("Merging FIMO results with PSI data...")
    fimo_df['Reference'] = fimo_df['Reference'].astype(int)
    merged_df = pd.merge(fimo_df, psi_supertable_merged, on='Reference', how='inner')  # 'inner' keeps only matching rows
    print("Data merged successfully.")
    # Clean up the 'motif' column due to extra \n in formatting and replacing delimeters for easier parsing
    merged_df['motif_id'] = merged_df['motif_id'].str.replace(r'MOTIF |\n', '', regex=True)
    merged_df['start']=  merged_df['start'].astype(int) - 1 #fimo does motif indexing as 1 index, shift to 0
    merged_df['stop']=  merged_df['stop'].astype(int) - 1 #fimo does motif indexing as 1 index, shift to 0
    merged_df.rename(columns={'start': 'motif_start', 'stop':'motif_stop'}, inplace=True)
    
    return merged_df

    

def filter_splice_sites(merged_df):
    """Filter out splice site regions from the DataFrame efficiently."""
    print(f"Filtering splice site regions from {len(merged_df)} records...")
    
       # Using 0-based indexing for positions
    block_5ss = merged_df.apply(
        lambda row: set(range(row.exon_start - 1 - 4, row.exon_start - 1 + 4)), axis=1
    )
    block_3ss = merged_df.apply(
        lambda row: set(range(row.exon_end - 1 - 3, row.exon_end - 1 + 5)), axis=1
    )


    # Merge both sets using set union
    all_block_ss = block_5ss.combine(block_3ss, lambda x, y: x | y)
    # Convert var_pos to sets and check for overlap efficiently
    mask = merged_df.apply(lambda row: bool(set(row.var_pos) & all_block_ss[row.name]), axis=1)

    # Filter out rows where variant positions overlap with splice site regions
    filtered_df = merged_df[~mask].copy()

    print(f"Filtered splice site regions. Remaining records: {len(filtered_df)}")
    return filtered_df

    
def split_results_by_loc(df_merged):
    """Split FIMO/PSI results into intron1, exon, and intron2 regions."""
    print("Splitting results by location...")
    #this filtering prevents the fimo hits from overlapping the splice junctions (unable to determine the region eg. intronic vs. exonic), 
    #but also this prevents picking up splice site disruption effects.
    df_intron1 = df_merged[df_merged['motif_stop'] <= (df_merged['intron1_end'])] #motif must end at or before the end of intron 1 (5'-3')
    df_intron2 = df_merged[df_merged['motif_start']>= (df_merged['intron2_start'])] #motif must start at or after the start of intron 2 (5'-3')
    df_exon = df_merged[(df_merged['motif_start'] >= (df_merged['exon_start'])) & (df_merged['motif_stop'] <= (df_merged['exon_end']))] # motif must start at or after the start of the exon and/or stop at or before the end of the exon
    print("Results split into intron1, exon, and intron2.")
    
    # Combine all filtered DataFrames (df_intron1, df_intron2, df_exon)
    df_combined = pd.concat([df_intron1, df_intron2, df_exon])

    # Check if every row in df is in one of the three DataFrames
    #the ones missing (printed here) are the ones that overlap regions (ie over intron and exon etc.)
    missing_rows = df_merged.loc[~df_merged.index.isin(df_combined.index)]
    
    #sanity checks to make sure all fimo results are accounted for
    if missing_rows.empty:
        print("All rows in the original DataFrame are present in one of the three DataFrames.")
    else:
        print(f"{len(missing_rows)} row(s) from the original DataFrame are not in any of the three DataFrames.")
    return df_intron1, df_exon, df_intron2



    
    
def save_split_data(df_intron1, df_exon, df_intron2, output_path_prefix):
    """Save the split DataFrames to CSV files."""
    print("Saving split DataFrames to CSV files...")
    df_intron1.to_csv(f'{output_path_prefix}_df_intron1.csv', index=False)
    df_exon.to_csv(f'{output_path_prefix}_df_exon.csv', index=False)
    df_intron2.to_csv(f'{output_path_prefix}_df_intron2.csv', index=False)
    print("Split DataFrames saved successfully.")
    
    
def open_split_data(output_path_prefix):
    """Save the split DataFrames to CSV files."""
    print("opening split DataFrames CSV files...")
    df_intron1=pd.read_csv(f'{output_path_prefix}_df_intron1.csv')
    df_exon=pd.read_csv(f'{output_path_prefix}_df_exon.csv')
    df_intron2=pd.read_csv(f'{output_path_prefix}_df_intron2.csv')
    return df_intron1,df_exon,df_intron2
    


    
def full_bootstrap_effect_size_with_plots(df_with_fimo, df_without_fimo, mincov, n_bootstraps=2, plot=False, output_dir_plots=None):
    cell_lines = ['HEK', 'HeLa', 'K562', 'HMC3', 'MCF7']
    var_logits = [f'{cl}_pooled_logit' for cl in cell_lines]
    var_included = [f'{cl}_pooled_included' for cl in cell_lines]
    var_total = [f'{cl}_pooled_total' for cl in cell_lines]
    total_incl_cols = var_included + var_total

    df_with_fimo_filtered = df_with_fimo[df_with_fimo['Reference'].isin(df_without_fimo['Reference'])]

    def filter_event_ids_by_coverage(motif_data, non_motif_data, cell_lines, mincov):
        valid_event_ids_per_cl = []
        all_event_ids = set(motif_data['event_id']).union(set(non_motif_data['event_id']))
        per_cl_passed = {}

        for cl in cell_lines:
            col = f'{cl}_pooled_logit'
            if col not in motif_data.columns or col not in non_motif_data.columns:
                print(f"[WARN] Column {col} not found in data. Skipping {cl}.")
                continue

            motif_counts = motif_data.groupby('event_id')[col].apply(lambda x: x.notna().sum())
            non_motif_counts = non_motif_data.groupby('event_id')[col].apply(lambda x: x.notna().sum())

            passed = motif_counts[(motif_counts >= mincov) & (non_motif_counts >= mincov)].index
            per_cl_passed[cl] = set(passed)
            valid_event_ids_per_cl.append(set(passed))

        if not valid_event_ids_per_cl:
            return set(), per_cl_passed  # no cell lines processed

        valid_event_ids = set.intersection(*valid_event_ids_per_cl)

        # partially_failed_event_ids = set.union(*valid_event_ids_per_cl) - valid_event_ids
        # if partially_failed_event_ids:
        #     print(f'Event IDs failing final intersection (did not pass in ALL cell lines):')
        #     for eid in partially_failed_event_ids:
        #         failing_cls = [cl for cl, passed_set in per_cl_passed.items() if eid not in passed_set]
        #         print(f'  - {eid}: failed in {", ".join(failing_cls)}')

        return valid_event_ids, per_cl_passed

    def effect_size_for_bootstrap(df_with, df_without, boot_id):
        motif_list = df_with['motif_id'].unique()
        results = []

        for motif in motif_list:
            print(motif)
            all_seqs_with_motif = df_with[df_with['motif_id'] == motif]['Reference'].unique()

            motif_data = df_with[(df_with['motif_id'] == motif) & (df_with['snp'] != 'none')]
            motif_data_wts = df_with[(df_with['motif_id'] == motif) & (df_with['snp'] == 'none')]
            non_motif_data = df_without[(~df_without['Reference'].isin(all_seqs_with_motif)) & (df_without['snp'] != 'none')]
            non_motif_data_wts = df_without[(~df_without['Reference'].isin(all_seqs_with_motif)) & (df_without['snp'] == 'none')]

            valid_event_ids, per_cl_passed = filter_event_ids_by_coverage(motif_data, non_motif_data, cell_lines, mincov)

            if not valid_event_ids:
                continue

            group_with = motif_data.groupby('event_id')[var_logits].mean()
            group_without = non_motif_data.groupby('event_id')[var_logits].mean()

            motif_contexts = []
            for event in group_with.index:
                has_wt_with_motif = not motif_data_wts[motif_data_wts['event_id'] == event].empty
                motif_contexts.append("knock_out" if has_wt_with_motif else "knock_in")

            # Filter effect size calculation per cell line based on per_cl_passed
            filtered_diff = {}
            for cl in cell_lines:
                col = f'{cl}_pooled_logit'
                if col not in group_with.columns or col not in group_without.columns:
                    continue
                passing_events = group_with.index.intersection(per_cl_passed.get(cl, set()))
                col_diff = []
                for eid in group_with.index:
                    if eid in passing_events:
                        col_diff.append(group_with.loc[eid, col] - group_without.loc[eid, col])
                    else:
                        col_diff.append(float('nan'))  # Skip effect size calc for failing combo
                filtered_diff[col] = col_diff

            df_diff = pd.DataFrame({
                'motif': motif,
                'event_id': group_with.index,
                'motif_context': motif_contexts,
                **{f'Effect Size {cl} Boot {boot_id}': filtered_diff[f'{cl}_pooled_logit'] for cl in cell_lines if f'{cl}_pooled_logit' in filtered_diff}
            })

            results.append(df_diff)

            if plot and boot_id == 0:
                for eid in group_with.index:
                    plot_effect_strips(
                        eid, motif, motif_data, non_motif_data, df_diff,
                        motif_data_wts, non_motif_data_wts, mincov, output_dir_plots)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    if n_bootstraps == 0:
        bootstrap_df = effect_size_for_bootstrap(df_with_fimo_filtered, df_without_fimo.copy(), 0)
    else:
        boot_results = []
        for boot in range(n_bootstraps):
            sampled = df_without_fimo.copy()
            for cl in cell_lines:
                col = f'Mean {cl}'
                sampled[col] = sampled[col].sample(frac=1, replace=True).reset_index(drop=True)
            seq_sample = sampled.dropna(subset=[f'Mean {cl}' for cl in cell_lines])['Reference'].unique()
            fimo_sample = df_with_fimo_filtered[df_with_fimo_filtered['Reference'].isin(seq_sample)]
            result = effect_size_for_bootstrap(fimo_sample, sampled, boot)
            boot_results.append(result)

        bootstrap_df = boot_results[0]
        for next_df in boot_results[1:]:
            bootstrap_df = pd.merge(bootstrap_df, next_df, on=['motif', 'event_id', 'motif_context'], how='inner')

    # Identify rows with at least one non-NaN effect size across all cell lines
    effect_cols = [col for col in bootstrap_df.columns if col.startswith('Effect Size')]
    bootstrap_df = bootstrap_df[bootstrap_df[effect_cols].notna().any(axis=1)].reset_index(drop=True)
    
    # Proceed with summary calculations
    summary = {}
    for cl in cell_lines:
        col_key = cl
        boot_cols = [col for col in bootstrap_df.columns if f'Effect Size {col_key} Boot' in col]
        if not boot_cols:
            continue  # Skip if no columns for this CL
    
        data = bootstrap_df[boot_cols]
        summary[f'Mean {cl} Average'] = data.mean(axis=1)
        summary[f'Mean {cl} 95% CI Lower'] = data.quantile(0.025, axis=1)
        summary[f'Mean {cl} 95% CI Upper'] = data.quantile(0.975, axis=1)
    
    summary_df = pd.DataFrame(summary)
    return pd.concat([bootstrap_df, summary_df], axis=1)



def plot_effect_strips(event_id, motif, motif_data, non_motif_data, effect, motif_data_wts, non_motif_data_wts, mincov, output_dir):
    cell_lines = ['HEK', 'HeLa', 'K562', 'HMC3', 'MCF7']
    columns_to_average = [f'{cl}_pooled_logit' for cl in cell_lines]
    colors = {'Motif Present': 'green', 'Motif Absent': 'red'}

    fig, axs = plt.subplots(1, 5, figsize=(18, 6), sharey=True)
    axs = axs.flatten()

    gene_exon = None
    if 'gene_exon' in motif_data.columns and event_id in motif_data['event_id'].values:
        gene_exon = motif_data.loc[motif_data['event_id'] == event_id, 'gene_exon'].dropna().unique()
    elif 'gene_exon' in non_motif_data.columns and event_id in non_motif_data['event_id'].values:
        gene_exon = non_motif_data.loc[non_motif_data['event_id'] == event_id, 'gene_exon'].dropna().unique()
    gene_exon = gene_exon[0] if isinstance(gene_exon, (list, np.ndarray, pd.Series)) and len(gene_exon) > 0 else "unknown"

    plotted_any = False

    for i, col in enumerate(columns_to_average):
        cell_line = col.split("_")[0]
        ax = axs[i]

        data_with_cl = motif_data[motif_data['event_id'] == event_id][[col]].dropna().rename(columns={col: 'logit variants'})
        data_with_cl['Motif Presence'] = 'Motif Present'
        data_without_cl = non_motif_data[non_motif_data['event_id'] == event_id][[col]].dropna().rename(columns={col: 'logit variants'})
        data_without_cl['Motif Presence'] = 'Motif Absent'
        combined = pd.concat([data_with_cl, data_without_cl], ignore_index=True)

        n_with = len(data_with_cl)
        n_without = len(data_without_cl)

        mean_with = data_with_cl['logit variants'].mean() if n_with > 0 else np.nan
        mean_without = data_without_cl['logit variants'].mean() if n_without > 0 else np.nan
        delta = mean_with - mean_without if pd.notna(mean_with) and pd.notna(mean_without) else None

        pval = None
        if n_with > 0 and n_without > 0:
            try:
                stat, pval = mannwhitneyu(data_with_cl['logit variants'], data_without_cl['logit variants'])
            except:
                pass

        subtitle_parts = []
        if pd.notna(pval): subtitle_parts.append(f"p={pval:.2g}")
        if pd.notna(delta): subtitle_parts.append(f"Δlogit={delta:.2f}")
        if n_with < mincov or n_without < mincov: subtitle_parts.append(f"N<{mincov}")
        subtitle = ", ".join(subtitle_parts)
        ax.set_title(f"{cell_line}\n{subtitle}", fontsize=9)

        if not combined.empty:
            sns.stripplot(
            data=combined,
            x='Motif Presence',
            y='logit variants',
            ax=ax,
            palette=colors,
            size=4,
            jitter=True,
            linewidth=0.3,
            edgecolor='black',
            alpha=1,
            zorder=1)
        
            # sns.boxplot(
            # data=combined,
            # x='Motif Presence',
            # y='logit variants',
            # ax=ax,
            # showcaps=True,
            # boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1, 'zorder': 2},
            # whiskerprops={'linewidth': 1, 'color': 'black', 'zorder': 2},
            # medianprops={'color': 'black', 'linewidth': 1, 'zorder': 3},
            # showfliers=False,
            # meanline=True)
            
        if n_with >= mincov and n_without >= mincov:
            plotted_any = True

        if pd.notna(mean_with):
            ax.axhline(mean_with, color='green', linestyle='dashed', linewidth=1)
        if pd.notna(mean_without):
            ax.axhline(mean_without, color='red', linestyle='dashed', linewidth=1)

        ax.axhline(0, color='black', linestyle='dashed', linewidth=1)

        ax.set_ylim(-16, 12)
        ax.set_xlabel("")
        ax.set_ylabel("logit variants" if i % 5 == 0 else "")
        ax.text(0, -14.2, f"N={n_with}", ha='center', va='top', fontsize=8, transform=ax.transData)
        ax.text(1, -14.2, f"N={n_without}", ha='center', va='top', fontsize=8, transform=ax.transData)

        ref_val = np.nan
        x_pos = None
        color = None
        if motif_data_wts['event_id'].isin([event_id]).any():
            ref_val = motif_data_wts[motif_data_wts['event_id'] == event_id][f'{cell_line}_pooled_psi_raw'].iloc[0]
            x_pos = 0
            color = 'green'
        if non_motif_data_wts['event_id'].isin([event_id]).any():
            ref_val = non_motif_data_wts[non_motif_data_wts['event_id'] == event_id][f'{cell_line}_pooled_psi_raw'].iloc[0]
            x_pos = 1
            color = 'red'
        if pd.notna(ref_val) and x_pos is not None:
            ax.text(x_pos, -15.2, f"Ref PSI={ref_val:.2f}", color=color, ha='center', va='top', fontsize=8, transform=ax.transData)

    if plotted_any:
        plt.suptitle(f"{motif} | {event_id} | {gene_exon}", fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0.03, 0.1, 0.97, 0.9])
        os.makedirs(output_dir, exist_ok=True)
        motif_clean = motif.split('/')[0].split(';')[0]
        plt.savefig(os.path.join(output_dir, f"strip_{motif_clean}_{event_id}_allcells.pdf"))
        plt.close()

        
        
def plot_effect_size(df_from_boots, output_path_prefix):
    # Define the cell lines
    cell_lines = ["HEK", "HeLa", "K562", "HMC3", "MCF7"]
    
    # Extract relevant columns
    columns_to_keep = ["motif", "event_id"] + [
        f"Mean {cell} Average" for cell in cell_lines
    ] + [
        f"Mean {cell} 95% CI Lower" for cell in cell_lines
    ] + [
        f"Mean {cell} 95% CI Upper" for cell in cell_lines
    ]
    
    df_subset = df_from_boots[columns_to_keep]
    
    # Reshape into long format
    plot_data = []
    for cell in cell_lines:
        for _, row in df_subset.iterrows():
            plot_data.append({
                "Cell Line": cell,
                "Motif": row["motif"],
                "Event ID": row["event_id"],
                "Effect Size Mean": row[f"Mean {cell} Average"],
                "CI Lower": row[f"Mean {cell} 95% CI Lower"],
                "CI Upper": row[f"Mean {cell} 95% CI Upper"],
            })
    
    df_long = pd.DataFrame(plot_data)
    
    # Compute mean effect size per motif for sorting
    motif_means = df_long.groupby(["Cell Line", "Motif"])["Effect Size Mean"].mean().reset_index()
    
    # Generate colors for each unique event_id
    num_colors = df_long["Event ID"].nunique()
    hues = np.linspace(0, 1, num_colors, endpoint=False)
    colors = [plt.cm.hsv(hue) for hue in hues]
    random.shuffle(colors)
    event_id_to_color = dict(zip(df_long["Event ID"].unique(), colors))
    df_long["color"] = df_long["Event ID"].map(event_id_to_color)

    # Create a separate plot for each cell line
    for cell in cell_lines:
        df_cell = df_long[df_long["Cell Line"] == cell]
    
        # Sort motifs by their mean effect size across all event_ids
        sorted_motifs = motif_means[motif_means["Cell Line"] == cell].sort_values("Effect Size Mean")["Motif"]
        df_cell["Motif"] = pd.Categorical(df_cell["Motif"], categories=sorted_motifs, ordered=True)
        df_cell = df_cell.sort_values("Motif")
    
        # Assign numerical index without jitter
        unique_motifs = df_cell["Motif"].unique()
        motif_to_index = {motif: index for index, motif in enumerate(unique_motifs)}
    
        # Create figure
        plt.figure(figsize=(20,30))
    
        for _, row in df_cell.iterrows():
            y_value = motif_to_index[row["Motif"]]  # No jittering
            lower_error = row["Effect Size Mean"] - row["CI Lower"]
            upper_error = row["CI Upper"] - row["Effect Size Mean"]
    
            plt.errorbar(row["Effect Size Mean"], y_value, xerr=[[lower_error], [upper_error]], 
                         fmt="o", color=row["color"], markersize=8, alpha=0.7, capsize=3)
    
        plt.title(f"Effect Size by Motif in {cell}")
        plt.xlabel("Effect Size Mean")
        # Apply split to each motif name individually to plot short version
        plt.yticks(range(len(unique_motifs)), [motif.split('/')[0].split(';')[0] for motif in unique_motifs], fontsize=9)
        plt.grid(axis="x", linestyle="--", alpha=0.7)
    
        # Save each plot separately
        plt.tight_layout()
        output_filename = f"{output_path_prefix}_effect_size_{cell}.pdf"
        plt.savefig(output_filename)
        plt.close()    


def plot_motif_context_correlations_per_cell_line(df_out, context_col='motif_context', output_dir='context_corrs'):
    """
    Plot and save pairwise Spearman correlation heatmaps of motif effect sizes across event_ids
    for each cell line, split by motif context (all, knock-in, knock-out).

    Parameters:
        df_out (pd.DataFrame): Output from full_bootstrap_effect_size_with_plots (must include 'motif_context')
        context_col (str): Column with motif context labels
        output_dir (str): Directory to save the correlation plots
    """
    os.makedirs(output_dir, exist_ok=True)
    cell_lines = ['HEK', 'HeLa', 'K562', 'HMC3', 'MCF7']

    for cl in cell_lines:
        col = f"Mean {cl} Average"
        
        df_all = df_out.dropna(subset=[col])
        df_knock_in = df_all[df_all[context_col] == 'knock_in']
        df_knock_out = df_all[df_all[context_col] == 'knock_out']

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"{cl} | Spearman correlations of Δlogit across event_ids", fontsize=14)

        for i, (df_subset, label) in enumerate(zip(
            [df_all, df_knock_in, df_knock_out],
            ["All", "Knock-in", "Knock-out"]
        )):
            if df_subset.empty:
                ax[i].set_title(f"{label} (no data)")
                ax[i].axis('off')
                continue

            matrix = df_subset.pivot(index='event_id', columns='motif', values=col)
            corr = matrix.corr(method='spearman')

            sns.heatmap(corr, ax=ax[i], cmap='vlag', center=0, cbar=False,
                        xticklabels=False, yticklabels=False)
            ax[i].set_title(label)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = os.path.join(output_dir, f"correlation_heatmap_{cl}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        

def plot_context_dependence_facetgrid(csv_path, region, output_dir,mean_thresh=0.0, std_thresh=1.0, filter_low_es_thresh=0):
    create_output_dir(os.path.join(output_dir,f"facet_plots_{region}"))
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Filter motifs with ≥2 events
    event_counts = df.groupby("motif")["event_id"].nunique()
    valid_motifs = event_counts[event_counts >= 2].index
    filtered_df = df[df["motif"].isin(valid_motifs)]

    # List of cell line columns
    cell_lines = {
        "HEK": "Mean HEK Average",
        "HeLa": "Mean HeLa Average",
        "K562": "Mean K562 Average",
        "HMC3": "Mean HMC3 Average",
        "MCF7": "Mean MCF7 Average"
    }
    print(len(filtered_df))
    if filter_low_es_thresh > 0:
        for col in cell_lines.values():
            filtered_df[col] = filtered_df[col].where(filtered_df[col].abs() > filter_low_es_thresh, np.nan)


   
    print(len(filtered_df))
    # Build a combined dataframe
    rows = []
    for cell, col in cell_lines.items():
        g = filtered_df.groupby("motif")[col].agg(['mean', 'std'])
        g.columns = ['mean', 'std']
        g['motif'] = g.index

        def flip(vals):
            return (vals > 0).any() and (vals < 0).any()

        g['sign_flip'] = filtered_df.groupby("motif")[col].agg(flip)
        g['Cell Line'] = cell
        rows.append(g)

    combined_df = pd.concat(rows)
    
     # Save motifs where |mean| > mean_thresh and std < std_thresh
    gated_df = combined_df[(combined_df['mean'].abs() > mean_thresh) & (combined_df['std'] < std_thresh)]
    if not gated_df.empty:
        # Get only motifs that passed gating
        gated_motifs = gated_df['motif'].unique()
        event_details = filtered_df[filtered_df['motif'].isin(gated_motifs)].copy()
    
        # Get mean effects per motif/cell line from gated_df
        mean_effect_map = gated_df.set_index(["motif", "Cell Line"])[["mean"]].to_dict()["mean"]
    
        # Collect rows for detailed event export
        detail_rows = []
        for _, row in event_details.iterrows():
            motif = row["motif"]
            event_id = row["event_id"]
    
            row_data = {
                "motif": motif,
                "event_id": event_id
            }
    
            # Add per-cell-line individual and average effect
            for cell, col in cell_lines.items():
                individual_effect = row[col]
                mean_effect = mean_effect_map.get((motif, cell), None)
                row_data[f"{cell}_individual_effect"] = individual_effect
                row_data[f"{cell}_mean_effect"] = mean_effect
    
            detail_rows.append(row_data)
    
        # Build DataFrame and export
        detailed_df = pd.DataFrame(detail_rows)
        detail_path = os.path.join(output_dir, f"facet_plots_{region}", f"gated_event_details_mean_thresh_{mean_thresh}_std_thresh_{std_thresh}_filter_thresh_{filter_low_es_thresh}.csv")
        detailed_df.to_csv(detail_path, index=False)
        print(f"Exported detailed gated motif-event data to: {detail_path}")
    else:
        print("No motifs passed the gating thresholds.")
        
    # Plot using FacetGrid
    g = sns.FacetGrid(
        combined_df,
        col="Cell Line",
        col_order=["HEK", "HeLa", "K562", "HMC3", "MCF7"],
        height=4,
        aspect=1,
        sharey=True
    )
    g.map_dataframe(
        sns.scatterplot,
        x="mean",
        y="std",
        hue="sign_flip",
        palette={True: "crimson", False: "teal"},
        edgecolor="k",
        alpha=0.6
    )

    g.set_axis_labels("Mean Motif Effect Size Across Exon Families", "Standard deviation Across Exon Families")
    g.set_titles("{col_name}")
    g.add_legend(title="Sign flip")

    for ax in g.axes.flatten():
        ax.axhline(std_thresh, color='gray', linestyle='--', lw=1)
        ax.axvline(mean_thresh, color='black', linestyle=':', lw=1)
        ax.axvline(-1*mean_thresh, color='black', linestyle=':', lw=1)

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Context Dependence of RBP Motifs Across Cell Lines (≥2 Events) - {region}")
    plt.tight_layout()

    output_path = os.path.join(output_dir,f"facet_plots_{region}",f"mean_thresh_{mean_thresh}_std_thresh_{std_thresh}_filter_thresh_{filter_low_es_thresh}.pdf")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved facetgrid plot to: {output_path}")



def plot_combined_region_effects(region_dfs, region_labels, output_dir, cell_line='HEK', min_motifs=2):
    """
    Create a scatter plot of mean effect size per event_id vs each individual motif-level effect,
    followed by a strip plot of average motif effect sizes by event_id, colored by region.

    Parameters:
        region_dfs (List[pd.DataFrame]): List of DataFrames, each corresponding to a region.
        region_labels (List[str]): List of region names, in the same order as region_dfs.
        output_dir (str): Directory to save the plots.
        cell_line (str): Cell line to use for effect size column (default: 'HEK').
        min_motifs (int): Minimum number of motifs per event_id to include.

    Output:
        Saves a scatterplot and a stripplot PDF to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    col = f"Mean {cell_line} Average"

    # Collect relevant data with region labels
    all_data = []
    for df, region in zip(region_dfs, region_labels):
        if col not in df.columns:
            print(f"[WARN] Column '{col}' not found in region '{region}', skipping.")
            continue

        region_df = df[["event_id", "motif", col]].copy()
        region_df["region"] = region
        all_data.append(region_df)

    if not all_data:
        print("[ERROR] No valid dataframes with the required column. Nothing to plot.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Filter for event_ids with enough motifs total
    event_counts = combined_df.groupby("event_id")["motif"].nunique()
    valid_events = event_counts[event_counts >= min_motifs].index

    # Compute mean effect size per event_id and sort
    event_means = combined_df.groupby("event_id")[col].mean()
    sorted_events = event_means.loc[valid_events].sort_values().index.tolist()

    # Final filtered and sorted data
    plot_df = combined_df[combined_df["event_id"].isin(sorted_events)].copy()
    plot_df["event_id"] = pd.Categorical(plot_df["event_id"], categories=sorted_events, ordered=True)

    # Compute mean effect sizes for scatter plot
    mean_effects = plot_df.groupby("event_id")[col].mean().reset_index()
    scatter_df = plot_df.merge(mean_effects, on="event_id", suffixes=("", "_mean"))

    # Scatter plot
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=scatter_df, x=f"{col}_mean", y=col, hue="region", alpha=0.7)
    plt.plot([scatter_df[f"{col}_mean"].min(), scatter_df[f"{col}_mean"].max()],
             [scatter_df[f"{col}_mean"].min(), scatter_df[f"{col}_mean"].max()],
             linestyle='--', color='gray', label='y = x')
    plt.xlabel("Mean Effect Size per Event")
    plt.ylabel("Motif Effect Size")
    plt.title("Mean vs Motif Effect Sizes")
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f"scatter_means_vs_motifs_{cell_line}.pdf")
    plt.savefig(scatter_path)
    plt.close()
    print(f"[INFO] Saved scatter plot to {scatter_path}")

    
# Main Function
def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
    warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
    pd.set_option('display.max_columns', None)
    # Parse command-line arguments
    opts = parse_arguments()
    # Extract arguments
    fimo_out_file = opts.get("--fimo_out_file")
    indv_cell_lines_file = opts.get("--indv_cell_lines_file")
    cluster_file = opts.get("--cluster_file")
    output_dir = opts.get("--output_dir")
    output_prefix = opts.get("--output_prefix")
    numboots = int(opts.get("--numboots", 1))  # Default to 1 if not provided

    # Validate essential arguments
    required_args = [fimo_out_file, indv_cell_lines_file, cluster_file, output_dir, output_prefix]
    if not all(required_args):
        print("Missing one or more required arguments.")
        sys.exit(2)
    
    # Create output directory
    create_output_dir(output_dir)
    output_path_prefix = f"{output_dir}/{output_prefix}"
    print(output_path_prefix)

    
    #Load data
    '''
    fimo_df,psi_supertable_merged_df = load_data(fimo_out_file,indv_cell_lines_file,cluster_file)
    
    print(fimo_df)
    print(psi_supertable_merged_df)
    psi_supertable_merged_df = filter_splice_sites(psi_supertable_merged_df)
    psi_supertable_merged_df.to_csv(f'{output_path_prefix}_psi_supetable_merged_NO_FIMO.csv', sep=',', index=False) #save the file with merged fimo and PSI results after filtering splice sites
    psi_supetable_fimo_merged_df = merge_data_fimo(fimo_df, psi_supertable_merged_df)
    print(psi_supetable_fimo_merged_df)
    psi_supetable_fimo_merged_df.to_csv(f'{output_path_prefix}_psi_supetable_fimo_merged.csv', sep=',', index=False) #save the file with merged fimo and PSI results after filtering splice sites
    #split fimo data on where the motif binds (eg. intron or exon)
    df_intron1, df_exon, df_intron2 = split_results_by_loc(psi_supetable_fimo_merged_df)
    save_split_data(df_intron1, df_exon, df_intron2, output_path_prefix)
    '''
    
    df_intron1, df_exon, df_intron2 = open_split_data(output_path_prefix)
    psi_supertable_merged_df = pd.read_csv(f'{output_path_prefix}_psi_supetable_merged_NO_FIMO.csv') #no fimo data, just all sequences
    

    mincov=10

    
    boots_exon_df=full_bootstrap_effect_size_with_plots(
    df_with_fimo=df_exon,
    df_without_fimo=psi_supertable_merged_df,
    mincov=10,
    n_bootstraps=numboots,
    plot=False,
    output_dir_plots=f"{output_dir}/plots/stripplots_exon")


    boots_exon_df.to_csv(f'{output_path_prefix}_boots_{numboots}_exon.csv', sep=',', index=False)
    boots_exon_df = pd.read_csv(f'{output_path_prefix}_boots_{numboots}_exon.csv', sep=',')
    plot_effect_size(boots_exon_df, f'{output_dir}_boots_{numboots}_exon')
    
 

    boots_intron1_df=full_bootstrap_effect_size_with_plots(
    df_with_fimo=df_intron1,
    df_without_fimo=psi_supertable_merged_df,
    mincov=10,
    n_bootstraps=numboots,
    plot=False,
    output_dir_plots=f"{output_dir}_plots/stripplots_intron1")

    boots_intron1_df.to_csv(f'{output_path_prefix}_boots_{numboots}_intron1.csv', sep=',', index=False)
    plot_effect_size(boots_intron1_df, f'{output_path_prefix}_boots_{numboots}_intron1')
    
    
    boots_intron2_df=full_bootstrap_effect_size_with_plots(
    df_with_fimo=df_intron2,
    df_without_fimo=psi_supertable_merged_df,
    mincov=10,
    n_bootstraps=numboots,
    plot= False,
    output_dir_plots=f"{output_path_prefix}_plots/stripplots_intron2")

    boots_intron2_df.to_csv(f'{output_path_prefix}_boots_{numboots}_intron2.csv', sep=',', index=False)
    plot_effect_size(boots_intron1_df, f'{output_path_prefix}_boots_{numboots}_intron2')
    
    
    
    #exonic
    output_dir_new=f'{output_dir}/plots'
    csv_path=f'{output_path_prefix}_boots_{numboots}_exon.csv'
    region='exon'
    plot_context_dependence_facetgrid(csv_path, region, output_dir_new, mean_thresh=1, std_thresh=2.0, filter_low_es_thresh=0.25)
    
  
    
    #intron1
    
    csv_path=f'{output_path_prefix}_boots_{numboots}_intron1.csv'
    region='intron1'
    plot_context_dependence_facetgrid(csv_path, region, output_dir_new, mean_thresh=1, std_thresh=2.0,filter_low_es_thresh=0.25)
    

     #intron2
    
    csv_path=f'{output_path_prefix}_boots_{numboots}_intron2.csv'
    region='intron2'

    plot_context_dependence_facetgrid(csv_path, region, output_dir_new, mean_thresh=1, std_thresh=2.0, filter_low_es_thresh=0.25)
    
  
    # Load your effect size CSV for a region (e.g., exon)
    exon_df = pd.read_csv(f"{output_path_prefix}_boots_{numboots}_exon.csv")
    intron1_df = pd.read_csv(f"{output_path_prefix}_boots_{numboots}_intron1.csv")
    intron2_df = pd.read_csv(f"{output_path_prefix}_boots_{numboots}_intron2.csv")
    # Set where to save the plot
    swarm_output_dir = os.path.join(output_dir,"plots","swarm_plots")
    region_labels = ['exon', 'intron1', 'intron2']
    region_dfs = [exon_df, intron1_df, intron2_df]

    # # Call the plotting function
    # plot_combined_region_effects(
    # region_dfs=region_dfs,
    # region_labels=region_labels,
    # output_dir=swarm_output_dir,
    # cell_line='HEK',
    # min_motifs=2)





if __name__ == "__main__":
    main()    
    