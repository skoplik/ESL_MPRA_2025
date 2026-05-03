import numpy as np
import pandas as pd
import os
import sys, getopt
import traceback
from scipy.special import logit
from functools import reduce

def offset_psis(rep_df, clip=1e-3):
    print(" Running offset_psis(): Clipping PSI and computing logit(PSI)...")
    rep_df['PSI_clipped'] = rep_df['PSI'].clip(lower=clip, upper=1 - clip)
    rep_df['logit_PSI'] = logit(rep_df['PSI_clipped'])
    return rep_df

def build_fullseqs_df(supertable):
    # Carries event_id_161 (original SE: id), event_id (chr:exon_start-exon_end:strand),
    # transcript_class (MANE/alt/duplicate), and the two alt-transcript columns.
    # All rows kept — including alt rows of ambiguous events. WT/variant pairing
    # in compute_pooled_stats groups by the new event_id, which puts each
    # transcript-annotation family into its own group.
    full_seqs = pd.DataFrame({
        'Reference': np.arange(1, len(supertable)+1),
        'event_id': supertable['event_id'],
        'event_id_161': supertable['event_id_161'] if 'event_id_161' in supertable.columns else supertable['event_id'],
        'gene_exon': supertable['gene_exon'],
        'snp': supertable['snp'],
        'source': supertable['source'],
        'seq_type': supertable['seq_type'],
        'intron1': supertable['intron1'],
        'exon': supertable['exon'],
        'intron2': supertable['intron2'],
        'full_seq': supertable['full_seq'],
        'transcript_id': supertable['transcript_id'],
        'exon_start_hg38': supertable['exon_start_hg38'],
        'exon_end_hg38': supertable['exon_end_hg38'],
        'variant_hg38': supertable['variant_hg38'],
        'transcript_class': supertable['transcript_class'] if 'transcript_class' in supertable.columns else 'MANE',
        'alt_transcripts_in_supertable': supertable['alt_transcripts_in_supertable'] if 'alt_transcripts_in_supertable' in supertable.columns else '',
        'alt_transcripts_gencode_only': supertable['alt_transcripts_gencode_only'] if 'alt_transcripts_gencode_only' in supertable.columns else '',
    })
    return full_seqs

def compute_pooled_stats(df_list, full_seqs, label, clip=1e-3):
    rep_refs = pd.concat([df[['Reference']] for df in df_list])
    ref_counts = rep_refs['Reference'].value_counts()
    valid_refs = set(ref_counts[ref_counts >= 2].index)

    all_df = pd.concat(df_list)
    all_df = all_df[all_df['Reference'].isin(valid_refs)]

    # Vectorize: pre-compute inc/exc per row, then a single groupby.sum()
    inc_col = f'{label}_pooled_included'
    exc_col = f'{label}_pooled_excluded'
    all_df[inc_col] = all_df['Coverage'] * all_df['PSI']
    all_df[exc_col] = all_df['Coverage'] * (1 - all_df['PSI'])
    pooled = (all_df.groupby('Reference', sort=False)[[inc_col, exc_col]]
                    .sum().reset_index())

    pooled[f'{label}_total_pooled'] = pooled[f'{label}_pooled_included'] + pooled[f'{label}_pooled_excluded']
    pooled[f'{label}_pooled_psi_raw'] = pooled[f'{label}_pooled_included'] / pooled[f'{label}_total_pooled']
    pooled[f'{label}_pooled_psi_clipped'] = pooled[f'{label}_pooled_psi_raw'].clip(lower=clip, upper=1 - clip)
    pooled[f'{label}_pooled_logit'] = logit(pooled[f'{label}_pooled_psi_clipped'])

    pooled = pd.merge(pooled, full_seqs, on='Reference', how='left')

    # Multiple WT rows can share an event_id (e.g. true duplicates with same
    # full_seq AND same SJ). Average across them so the merge below produces
    # one match per row, not a cartesian-style row blow-up.
    wt_pool = (pooled[pooled['snp'] == 'none']
               [['event_id', f'{label}_pooled_psi_raw', f'{label}_pooled_logit']]
               .groupby('event_id', as_index=False).mean())
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

def compute_replicate_sds(pooled_df, label, var_rep_indices=None, wt_rep_indices=None, clip=1e-3):
    """
    Compute SD across replicates for PSI, logit(PSI), dPSI, and delta logit PSI.

    For standard cell lines (HeLa, K562, MCF7, HMC3): all reps share the same
    rows; WT rows are identified by snp=='none'. Leave var_rep_indices and
    wt_rep_indices as None.

    For HEK: variant reps (1,2) and WT reps (3,4) are separate rep columns.
    Pass var_rep_indices=[1,2], wt_rep_indices=[3,4]. The WT PSI for dPSI is
    computed as the mean across wt_rep_indices columns on snp=='none' rows,
    then mapped per event_id onto variant rows.
    """
    all_rep_psi_cols = [col for col in pooled_df.columns
                        if col.startswith(f'{label}_rep') and col.endswith('_psi_raw')]
    all_rep_indices = [int(col.replace(f'{label}_rep', '').replace('_psi_raw', ''))
                       for col in all_rep_psi_cols]

    # Which reps to use for PSI/logit SD (variant reps only for HEK)
    sd_rep_indices = var_rep_indices if var_rep_indices is not None else all_rep_indices
    sd_psi_cols = [f'{label}_rep{i}_psi_raw' for i in sd_rep_indices
                   if f'{label}_rep{i}_psi_raw' in pooled_df.columns]
    sd_logit_cols = [f'{label}_rep{i}_logit' for i in sd_rep_indices
                     if f'{label}_rep{i}_logit' in pooled_df.columns]

    if len(sd_psi_cols) >= 2:
        pooled_df[f'{label}_sd_psi'] = pooled_df[sd_psi_cols].std(axis=1, ddof=1)
    else:
        pooled_df[f'{label}_sd_psi'] = np.nan

    if len(sd_logit_cols) >= 2:
        pooled_df[f'{label}_sd_logit'] = pooled_df[sd_logit_cols].std(axis=1, ddof=1)
    else:
        pooled_df[f'{label}_sd_logit'] = np.nan

    rep_dpsi_cols = []
    rep_dlogit_cols = []

    if var_rep_indices is not None and wt_rep_indices is not None:
        # HEK: WT values come from wt_rep_indices columns on snp=='none' rows,
        # averaged across WT reps per event_id, then mapped onto all rows.
        wt_psi_cols = [f'{label}_rep{i}_psi_raw' for i in wt_rep_indices
                       if f'{label}_rep{i}_psi_raw' in pooled_df.columns]
        wt_logit_cols = [f'{label}_rep{i}_logit' for i in wt_rep_indices
                         if f'{label}_rep{i}_logit' in pooled_df.columns]

        wt_rows = pooled_df[pooled_df['snp'] == 'none'][['event_id'] + wt_psi_cols + wt_logit_cols].copy()

        wt_psi_map = None
        wt_logit_map = None
        if wt_psi_cols:
            wt_rows['wt_mean_psi'] = wt_rows[wt_psi_cols].mean(axis=1)
            wt_psi_map = wt_rows.groupby('event_id')['wt_mean_psi'].mean()
        if wt_logit_cols:
            wt_rows['wt_mean_logit'] = wt_rows[wt_logit_cols].mean(axis=1)
            wt_logit_map = wt_rows.groupby('event_id')['wt_mean_logit'].mean()

        for i in var_rep_indices:
            psi_col = f'{label}_rep{i}_psi_raw'
            logit_col = f'{label}_rep{i}_logit'
            dpsi_col = f'{label}_rep{i}_dpsi'
            dlogit_col = f'{label}_rep{i}_delta_logit'
            if psi_col not in pooled_df.columns:
                continue
            if wt_psi_map is not None:
                pooled_df[dpsi_col] = pooled_df[psi_col] - pooled_df['event_id'].map(wt_psi_map)
            if wt_logit_map is not None:
                pooled_df[dlogit_col] = pooled_df[logit_col] - pooled_df['event_id'].map(wt_logit_map)
            pooled_df.loc[pooled_df['snp'] == 'none', [dpsi_col, dlogit_col]] = np.nan
            rep_dpsi_cols.append(dpsi_col)
            rep_dlogit_cols.append(dlogit_col)

    else:
        # Standard: WT rows identified by snp=='none', same rep columns for all rows.
        # Multiple WT rows can share an event_id (e.g. true duplicates with same
        # full_seq AND same SJ within an exon family). Average across them so
        # set_index doesn't fail on duplicate index values.
        wt_rows = pooled_df[pooled_df['snp'] == 'none'][['event_id'] + sd_psi_cols + sd_logit_cols].copy()
        wt_rows = wt_rows.groupby('event_id', as_index=True).mean()

        wt_psi_lookup = {}
        wt_logit_lookup = {}
        for i, psi_col in zip(sd_rep_indices, sd_psi_cols):
            wt_psi_lookup[i] = wt_rows[psi_col]
        for i, logit_col in zip(sd_rep_indices, sd_logit_cols):
            wt_logit_lookup[i] = wt_rows[logit_col]

        for i in sd_rep_indices:
            psi_col = f'{label}_rep{i}_psi_raw'
            logit_col = f'{label}_rep{i}_logit'
            dpsi_col = f'{label}_rep{i}_dpsi'
            dlogit_col = f'{label}_rep{i}_delta_logit'
            if psi_col not in pooled_df.columns:
                continue
            wt_psi_for_rep = pooled_df['event_id'].map(wt_psi_lookup.get(i, pd.Series(dtype=float)))
            wt_logit_for_rep = pooled_df['event_id'].map(wt_logit_lookup.get(i, pd.Series(dtype=float)))
            pooled_df[dpsi_col] = pooled_df[psi_col] - wt_psi_for_rep
            pooled_df[dlogit_col] = pooled_df[logit_col] - wt_logit_for_rep
            pooled_df.loc[pooled_df['snp'] == 'none', [dpsi_col, dlogit_col]] = np.nan
            rep_dpsi_cols.append(dpsi_col)
            rep_dlogit_cols.append(dlogit_col)

    if len(rep_dpsi_cols) >= 2:
        pooled_df[f'{label}_sd_dpsi'] = pooled_df[rep_dpsi_cols].std(axis=1, ddof=1)
    else:
        pooled_df[f'{label}_sd_dpsi'] = np.nan

    if len(rep_dlogit_cols) >= 2:
        pooled_df[f'{label}_sd_delta_logit'] = pooled_df[rep_dlogit_cols].std(axis=1, ddof=1)
    else:
        pooled_df[f'{label}_sd_delta_logit'] = np.nan

    pooled_df = pooled_df.drop(columns=rep_dpsi_cols + rep_dlogit_cols)

    return pooled_df

def fill_single_rep_columns(merged_df, rep_dfs_by_label, clip=1e-3):
    """For refs present in merged_df but with NaN pooled PSI for a cell line,
    fill individual rep columns from the raw per-rep dataframes."""
    cell_lines = list(rep_dfs_by_label.keys())
    hek_var_reps = {1, 2}  # only variant reps for HEK

    for label, reps in rep_dfs_by_label.items():
        pool_col = f'{label}_pooled_psi_raw'
        if pool_col not in merged_df.columns:
            continue
        missing_mask = merged_df[pool_col].isna()
        if not missing_mask.any():
            continue

        rep_indices = range(1, len(reps) + 1)
        if label == 'HEK':
            rep_indices = hek_var_reps

        for i, df in zip(range(1, len(reps) + 1), reps):
            if label == 'HEK' and i not in hek_var_reps:
                continue
            tag = f'{label}_rep{i}'
            inc_col  = f'{tag}_included'
            exc_col  = f'{tag}_excluded'
            psi_col  = f'{tag}_psi_raw'
            clip_col = f'{tag}_psi_clipped'
            log_col  = f'{tag}_logit'
            if psi_col not in merged_df.columns:
                continue

            rep_sub = df[['Reference', 'Coverage', 'PSI', 'PSI_clipped', 'logit_PSI']].copy()
            rep_sub = rep_sub.rename(columns={
                'PSI': psi_col, 'PSI_clipped': clip_col, 'logit_PSI': log_col
            })
            rep_sub[inc_col] = rep_sub['Coverage'] * rep_sub[psi_col]
            rep_sub[exc_col] = rep_sub['Coverage'] * (1 - rep_sub[psi_col])
            rep_sub = rep_sub.drop(columns=['Coverage'])
            rep_sub = rep_sub.set_index('Reference')

            fill_refs = merged_df.loc[missing_mask, 'Reference']
            for col in [inc_col, exc_col, psi_col, clip_col, log_col]:
                if col in merged_df.columns and col in rep_sub.columns:
                    merged_df.loc[missing_mask, col] = fill_refs.map(rep_sub[col]).values

    return merged_df


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

    pooled_df.to_csv(f"{output_prefix}_{label}_WTS_VARS_NO_DELTAS.csv", index=False)


def main():
    opts, args = getopt.getopt(sys.argv[1:], "", [
        "HEK293_Rep1_PSI=", "HEK293_Rep2_PSI=", "HEK293_WT_Rep1_PSI=", "HEK293_WT_Rep2_PSI=",
        "HeLa_Rep1_PSI=", "HeLa_Rep2_PSI=", "K562_Rep1_PSI=", "K562_Rep2_PSI=",
        "MCF7_Rep1_PSI=", "MCF7_Rep2_PSI=", "HMC3_Rep1_PSI=", "HMC3_Rep2_PSI=",
        "supertable_file=", "output_dir=", "output_prefix=", "clip="])
    opts = dict(opts)
    clip_val = float(opts.get("--clip", 1e-3))
    output_dir = opts["--output_dir"]
    output_prefix = os.path.join(output_dir, opts["--output_prefix"])
    os.makedirs(output_dir, exist_ok=True)
    supertable = pd.read_csv(opts["--supertable_file"], sep=',')
    supertable.iloc[:, 0] = supertable.iloc[:, 0].astype(int) + 1
    full_seqs = build_fullseqs_df(supertable)

    def read(path, label):
        df = pd.read_csv(opts[path], sep='\t', engine='python')
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

    # HEK: reps 1+2 are variant libraries, reps 3+4 are WT libraries
    hek_sd_kwargs = dict(var_rep_indices=[1, 2], wt_rep_indices=[3, 4])

    all_no_deltas = []
    cell_lines = ['HeLa', 'K562', 'MCF7', 'HMC3', 'HEK']
    rep_dfs_by_label = {}  # saved for single-rep fill-in pass

    for label, reps in data.items():
        rep_dfs_by_label[label] = reps
        try:
            print(f"\n=== Processing {label} ===")
            pooled_df = compute_pooled_stats(reps, full_seqs, label, clip=clip_val)
            print(f"  compute_pooled_stats done: {pooled_df.shape}")
            pooled_df = attach_replicate_info(pooled_df, reps, label)
            print(f"  attach_replicate_info done: {pooled_df.shape}")
            sd_kwargs = hek_sd_kwargs if label == 'HEK' else {}
            pooled_df = compute_replicate_sds(pooled_df, label, clip=clip_val, **sd_kwargs)
            print(f"  compute_replicate_sds done: {pooled_df.shape}")
            write_outputs_with_and_without_wt(pooled_df, output_prefix, label)
            print(f"  Files written for {label}")
        except Exception as e:
            print(f"ERROR processing {label}: {e}")
            traceback.print_exc()

    for cl in cell_lines:
        path = f"{output_prefix}_{cl}_WTS_VARS_NO_DELTAS.csv"
        if os.path.exists(path):
            print(f"Loading {path}")
            df = pd.read_csv(path)
            all_no_deltas.append(df)
        else:
            print(f"WARNING: Missing {path}")

    if all_no_deltas:
        # Outer-merge on Reference only (each Reference is unique within a
        # cell-line file). Merging on full metadata columns explodes to a
        # cartesian-like product because NaN keys don't match across files —
        # this fix avoids a 14+ GiB allocation. Take metadata from the union
        # of all files via a concat → drop_duplicates pass.
        meta_cols = ['Reference', 'event_id', 'event_id_161', 'gene_exon', 'snp',
                     'source', 'seq_type', 'intron1', 'exon', 'intron2',
                     'full_seq', 'transcript_id', 'exon_start_hg38',
                     'exon_end_hg38', 'variant_hg38']
        extra_meta = [c for c in ('transcript_class',
                                  'alt_transcripts_in_supertable',
                                  'alt_transcripts_gencode_only',
                                  'n_alt_transcripts_in_supertable',
                                  'n_alt_transcripts_gencode_only')
                      if all(c in df.columns for df in all_no_deltas)]
        all_meta_cols = meta_cols + extra_meta

        meta_df = pd.concat(
            [df[[c for c in all_meta_cols if c in df.columns]] for df in all_no_deltas],
            ignore_index=True
        ).drop_duplicates(subset='Reference', keep='first')

        # Strip metadata cols from each cell-line df, keep PSI cols + Reference
        psi_dfs = [df.drop(columns=[c for c in all_meta_cols if c != 'Reference' and c in df.columns])
                   for df in all_no_deltas]
        merged_no_deltas = reduce(lambda l, r: pd.merge(l, r, on='Reference', how='outer'),
                                  psi_dfs)
        merged_no_deltas = pd.merge(meta_df, merged_no_deltas, on='Reference', how='outer')

        col_order = all_meta_cols + [c for c in merged_no_deltas.columns if c not in all_meta_cols]
        merged_no_deltas = merged_no_deltas[col_order]

        # Fill individual rep columns for refs that passed >=2 reps in at least
        # one cell line but have only single-rep coverage in another cell line.
        print("\nFilling single-rep columns for refs with NaN pooled stats...")
        merged_no_deltas = fill_single_rep_columns(merged_no_deltas, rep_dfs_by_label, clip=clip_val)
        print("  Done.")

        # Drop barcode-cluster duplicate rows: same full_seq + same SJ + same
        # transcript_id as a lower-Reference row → identical PSI computation.
        # Verified: duplicate rows have identical pooled PSI / WT PSI / dPSI /
        # delta_logit values across all cell lines (same canonical pkl key,
        # same junction). Keep the canonical (lowest Reference) per
        # (full_seq, intron1_len, exon_len) combo only.
        if 'transcript_class' in merged_no_deltas.columns:
            n_dup = int((merged_no_deltas['transcript_class'] == 'duplicate').sum())
            merged_no_deltas = merged_no_deltas[merged_no_deltas['transcript_class'] != 'duplicate'].copy()
            print(f"\nDropped {n_dup:,} barcode-cluster duplicate rows (transcript_class=='duplicate'); "
                  f"identical PSI to canonical row.")

        merged_no_deltas.to_csv(f"{output_prefix}_ALL_WTS_VARS_NO_DELTAS.csv", index=False)
        merged_no_deltas.to_csv(f"{output_prefix}_ALL_WTS_VARS_NO_DELTAS.csv.gz", index=False, compression='gzip')
        print(f"\nWrote merged file: {output_prefix}_ALL_WTS_VARS_NO_DELTAS.csv(.gz)")

        # Also write merged ALL_WITH_WT.csv: keep only event_ids that contain
        # at least one WT row (snp == 'none') AND at least one variant row
        # (snp != 'none').
        wt_eids   = set(merged_no_deltas.loc[merged_no_deltas['snp'] == 'none', 'event_id'].dropna().unique())
        var_eids  = set(merged_no_deltas.loc[merged_no_deltas['snp'] != 'none', 'event_id'].dropna().unique())
        both_eids = wt_eids & var_eids
        all_with_wt = merged_no_deltas[merged_no_deltas['event_id'].isin(both_eids)].copy()
        all_with_wt.to_csv(f"{output_prefix}_ALL_WITH_WT.csv", index=False)
        all_with_wt.to_csv(f"{output_prefix}_ALL_WITH_WT.csv.gz", index=False, compression='gzip')
        print(f"Wrote merged file: {output_prefix}_ALL_WITH_WT.csv(.gz) "
              f"({len(all_with_wt):,} rows; "
              f"WT={int((all_with_wt['snp']=='none').sum()):,} "
              f"Var={int((all_with_wt['snp']!='none').sum()):,})")

if __name__ == "__main__":
    main()