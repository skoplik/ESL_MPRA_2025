import numpy as np
import pandas as pd
import os
import sys, getopt
from scipy.special import logit
from functools import reduce

def offset_psis(rep_df, clip=1e-3):
    print(" Running offset_psis(): Clipping PSI and computing logit(PSI)...")
    rep_df['PSI_clipped'] = rep_df['PSI'].clip(lower=clip, upper=1 - clip)
    rep_df['logit_PSI'] = logit(rep_df['PSI_clipped'])
    return rep_df

def build_fullseqs_df(supertable):
    full_seqs = pd.DataFrame({
        'Reference': np.arange(1, len(supertable)+1),
        'event_id': supertable['event_id'],
        'gene_exon': supertable['gene_exon'],
        'snp': supertable['snp'],
        'source': supertable['source'],
        'seq_type': supertable['seq_type'],
        'intron1': supertable['intron1'],
        'exon': supertable['exon'],
        'intron2': supertable['intron2'],
        'full_seq': supertable['full_seq'],
    })
    return full_seqs

def compute_pooled_stats(df_list, full_seqs, label, clip=1e-3):
    rep_refs = pd.concat([df[['Reference']] for df in df_list])
    ref_counts = rep_refs['Reference'].value_counts()
    valid_refs = ref_counts[ref_counts >= 2].index.tolist()

    all_df = pd.concat(df_list)
    all_df = all_df[all_df['Reference'].isin(valid_refs)]

    grouped = all_df.groupby('Reference')
    pooled = grouped.apply(lambda group: pd.Series({
        f'{label}_pooled_included': (group['Coverage'] * group['PSI']).sum(),
        f'{label}_pooled_excluded': (group['Coverage'] * (1 - group['PSI'])).sum()
    })).reset_index()

    pooled[f'{label}_total_pooled'] = pooled[f'{label}_pooled_included'] + pooled[f'{label}_pooled_excluded']
    pooled[f'{label}_pooled_psi_raw'] = pooled[f'{label}_pooled_included'] / pooled[f'{label}_total_pooled']
    pooled[f'{label}_pooled_psi_clipped'] = pooled[f'{label}_pooled_psi_raw'].clip(lower=clip, upper=1 - clip)
    pooled[f'{label}_pooled_logit'] = logit(pooled[f'{label}_pooled_psi_clipped'])

    pooled = pd.merge(pooled, full_seqs, on='Reference', how='left')

    wt_pool = pooled[pooled['snp'] == 'none'][['event_id', f'{label}_pooled_psi_raw', f'{label}_pooled_logit']]
    wt_pool = wt_pool.rename(columns={
        f'{label}_pooled_psi_raw': f'{label}_wt_pooled_psi_raw',
        f'{label}_pooled_logit': f'{label}_wt_pooled_logit'
    })
    pooled = pooled.merge(wt_pool, on='event_id', how='left')
    pooled[f'{label}_dpsi_pooled'] = pooled[f'{label}_pooled_psi_raw'] - pooled[f'{label}_wt_pooled_psi_raw']
    pooled[f'{label}_delta_logit_pooled'] = pooled[f'{label}_pooled_logit'] - pooled[f'{label}_wt_pooled_logit']
    pooled.loc[pooled['snp'] == 'none', [f'{label}_dpsi_pooled', f'{label}_delta_logit_pooled']] = np.nan

    return pooled

def attach_replicate_info(pooled_df, replicate_dfs, label):
    for i, df in enumerate(replicate_dfs):
        tag = f"{label}_rep{i+1}"
        if df.empty:
            continue
        df = df[['Reference', 'Coverage', 'PSI', 'PSI_clipped', 'logit_PSI']].copy()
        df[f'{tag}_included'] = df['Coverage'] * df['PSI']
        df[f'{tag}_excluded'] = df['Coverage'] * (1 - df['PSI'])
        df[f'{tag}_psi_raw'] = df['PSI']
        df[f'{tag}_psi_clipped'] = df['PSI_clipped']
        df[f'{tag}_logit'] = df['logit_PSI']
        df = df.drop(columns=['Coverage', 'PSI', 'PSI_clipped', 'logit_PSI'])
        pooled_df = pd.merge(pooled_df, df, on='Reference', how='left')
    return pooled_df

def write_outputs_with_and_without_wt(pooled_df, output_prefix, label):
    wt_refs = pooled_df[pooled_df['snp'] == 'none']
    var_refs = pooled_df[pooled_df['snp'] != 'none']
    wt_event_ids = set(wt_refs['event_id'].dropna().unique())
    var_event_ids = set(var_refs['event_id'].dropna().unique())
    ok_event_ids = sorted(wt_event_ids & var_event_ids)

    filtered_df = pooled_df[pooled_df['event_id'].isin(ok_event_ids)].copy()
    filtered_df = filtered_df.drop_duplicates(subset=['Reference', 'event_id'])
    filtered_df.to_csv(f"{output_prefix}_{label}_WITH_WT.csv", index=False)

    var_out = filtered_df[filtered_df['snp'] != 'none'].copy()
    delta_cols = [col for col in var_out.columns if 'dpsi' in col or 'delta_logit' in col]
    var_out = var_out.drop(columns=delta_cols)
    var_out.to_csv(f"{output_prefix}_{label}_VARIANTS_ONLY.csv", index=False)

    # Corrected: write pooled_df as-is, keeping all delta columns (NaNs allowed)
    pooled_df.to_csv(f"{output_prefix}_{label}_WTS_VARS_NO_DELTAS.csv", index=False)


def main():
    opts, args = getopt.getopt(sys.argv[1:], "", [
        "HEK293_Rep1_PSI=", "HEK293_Rep2_PSI=", "HEK293_WT_Rep1_PSI=", "HEK293_WT_Rep2_PSI=",
        "HeLa_Rep1_PSI=", "HeLa_Rep2_PSI=", "K562_Rep1_PSI=", "K562_Rep2_PSI=",
        "MCF7_Rep1_PSI=", "MCF7_Rep2_PSI=", "HMC3_Rep1_PSI=", "HMC3_Rep2_PSI=",
        "supertable_file=", "output_dir=", "output_prefix=", "clip="])
    opts = dict(opts)
    clip_val = float(opts.get("--clip", 1e-3))
    output_prefix = os.path.join(opts["--output_dir"], opts["--output_prefix"])
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    supertable = pd.read_csv(opts["--supertable_file"], sep=',')
    supertable.iloc[:, 0] = supertable.iloc[:, 0].astype(int) + 1
    full_seqs = build_fullseqs_df(supertable)

    def read(path, label):
        df = pd.read_csv(opts[path], sep='\\t')
        df = pd.merge(df, full_seqs[['Reference']], on='Reference', how='inner')
        df = offset_psis(df, clip=clip_val)
        df['cell_line'] = label
        return df

    data = {
        'HeLa': [read("--HeLa_Rep1_PSI", "HeLa"), read("--HeLa_Rep2_PSI", "HeLa")],
        'K562': [read("--K562_Rep1_PSI", "K562"), read("--K562_Rep2_PSI", "K562")],
        'MCF7': [read("--MCF7_Rep1_PSI", "MCF7"), read("--MCF7_Rep2_PSI", "MCF7")],
        'HMC3': [read("--HMC3_Rep1_PSI", "HMC3"), read("--HMC3_Rep2_PSI", "HMC3")],
        'HEK': [read("--HEK293_Rep1_PSI", "HEK"), read("--HEK293_Rep2_PSI", "HEK"),
                read("--HEK293_WT_Rep1_PSI", "HEK"), read("--HEK293_WT_Rep2_PSI", "HEK")]
    }
    '''
    all_pooled = []
    for label, reps in data.items():
        pooled_df = compute_pooled_stats(reps, full_seqs, label, clip=clip_val)
        pooled_df = attach_replicate_info(pooled_df, reps, label)
        write_outputs_with_and_without_wt(pooled_df, output_prefix, label)
        df = pd.read_csv(f"{output_prefix}_{label}_WITH_WT.csv")
        all_pooled.append(df)

    if all_pooled:
        merged_df = reduce(lambda left, right: pd.merge(
            left, right, on=['Reference', 'event_id', 'gene_exon','intron1','exon','intron2','full_seq','snp','source','seq_type'], how='outer'
        ), all_pooled)

        seq_cols = ['Reference', 'event_id', 'gene_exon', 'snp', 'source', 'seq_type', 'intron1', 'exon', 'intron2', 'full_seq']
        other_cols = [col for col in merged_df.columns if col not in seq_cols]
        cell_lines = ['HeLa', 'K562', 'MCF7', 'HMC3', 'HEK']
        col_order = seq_cols.copy()
        for cl in cell_lines:
            cl_pooled = [col for col in other_cols if col.startswith(f'{cl}_pooled_')]
            cl_wt = [col for col in other_cols if col.startswith(f'{cl}_wt_')]
            cl_deltas = [col for col in other_cols if f'{cl}_dpsi' in col or f'{cl}_delta_logit' in col]
            cl_reps = [col for col in other_cols if col.startswith(f'{cl}_rep')]
            col_order.extend(cl_pooled + cl_wt + cl_deltas + cl_reps)

        merged_df = merged_df[col_order]
        merged_df.to_csv(f"{output_prefix}_ALL_WITH_WT.csv", index=False)
        variants_only = merged_df[merged_df['snp'] != 'none'].copy()
        variants_only.to_csv(f"{output_prefix}_ALL_VARIANTS_ONLY.csv", index=False)'''
        
    # === Merge all WTS_VARS_NO_DELTAS.csv files ===
    all_no_deltas = []
    cell_lines = ['HeLa', 'K562', 'MCF7', 'HMC3', 'HEK']
    for cl in cell_lines:
        path = f"{output_prefix}_{cl}_WTS_VARS_NO_DELTAS.csv"
        if os.path.exists(path):
            print(f"Loading {path}")
            df = pd.read_csv(path)
            all_no_deltas.append(df)
        else:
            print(f"WARNING: Missing {path}")

    if all_no_deltas:
        merged_no_deltas = reduce(lambda left, right: pd.merge(
            left, right,
            on=['Reference', 'event_id', 'gene_exon', 'intron1', 'exon', 'intron2',
                'full_seq', 'snp', 'source', 'seq_type'],
            how='outer'
        ), all_no_deltas)

        seq_cols = ['Reference', 'event_id', 'gene_exon', 'snp', 'source', 'seq_type',
                    'intron1', 'exon', 'intron2', 'full_seq']
        other_cols = [col for col in merged_no_deltas.columns if col not in seq_cols]
        col_order = seq_cols + other_cols
        merged_no_deltas = merged_no_deltas[col_order]

        merged_no_deltas.to_csv(f"{output_prefix}_ALL_WTS_VARS_NO_DELTAS.csv", index=False)

if __name__ == "__main__":
    main()
