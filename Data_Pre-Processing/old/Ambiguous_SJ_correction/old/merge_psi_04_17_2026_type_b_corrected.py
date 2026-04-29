# -*- coding: utf-8 -*-
"""
merge_psi_04_17_2026_type_b_corrected.py

Identical to merge_psi_07_18_2025_clipping_no_psuedo.py, except that
before pooling, PSI values for Type B (var/ref boundary mismatch) References
are replaced with PSI recomputed at the reference junction.

Type B: a variant's intron1/exon boundary differs from its reference's boundary
within the same event_id. This is a transcript annotation artifact — the variant
construct was assigned to a different Gencode transcript than its reference,
giving the two rows different annotated exon coordinates. The boundary difference
is not caused by the variant sequence. The per-replicate PSI txt files have PSI
at the variant's (incorrectly annotated) junction; this script overrides those
values with PSI at the reference junction so that both members of each ref/variant
pair are measured at the same junction and delta logit is a valid comparison.

If a Type B Reference has insufficient coverage at the reference junction
(<MINCOV reads) in a given replicate, that replicate row is dropped (treated
as missing), exactly as if it had not been sequenced at that junction.

Extra arguments (added on top of the base merge PSI args):
  --boundary_mismatches  path to boundary_mismatches.csv (from script 01)
  --pkl_HeLa_rep1        path to HeLa rep1 splicing count pickle
  --pkl_HeLa_rep2
  --pkl_K562_rep1
  --pkl_K562_rep2
  --pkl_MCF7_rep1
  --pkl_MCF7_rep2
  --pkl_HMC3_rep1
  --pkl_HMC3_rep2
  --pkl_HEK_rep1
  --pkl_HEK_rep2
  --pkl_HEK_wt_rep1
  --pkl_HEK_wt_rep2
"""

import pickle
import numpy as np
import pandas as pd
import os
import sys
import getopt
import traceback
from scipy.special import logit
from functools import reduce

MINCOV    = 10
SHARED_5P = 286
E_JXN     = (26, 871)


# ── Junction correction helpers ───────────────────────────────────────────
def ref_junctions(intron1_len, exon_len):
    i1 = (26, SHARED_5P + intron1_len)
    i2 = (SHARED_5P + intron1_len + exon_len + 1, 871)
    return i1, i2

def psi_at_ref_jxn(pkl_dict, fasta_ref, i1, i2):
    """Return (psi, coverage) at reference junction; (nan, 0) if below MINCOV."""
    if fasta_ref not in pkl_dict:
        return np.nan, 0
    jd  = pkl_dict[fasta_ref]
    inc = min(jd.get(i1, 0), jd.get(i2, 0))
    exc = jd.get(E_JXN, 0)
    tot = inc + exc
    if tot < MINCOV:
        return np.nan, 0
    return inc / tot, tot

def build_type_b_map(bm_csv):
    """
    Returns dict: Reference (int) -> {ref_intron1_len, ref_exon_len}
    Only for variant rows (snp != 'none' implied by presence in boundary_mismatches).
    """
    bm = pd.read_csv(bm_csv)
    result = {}
    for _, row in bm.iterrows():
        result[int(row["Reference"])] = {
            "ref_intron1_len": int(row["ref_intron1_len"]),
            "ref_exon_len":    int(row["ref_exon_len"]),
        }
    return result

def apply_type_b_corrections(df, pkl_dict, type_b_map, clip):
    """
    For rows in df whose Reference is in type_b_map, recompute PSI/Coverage
    using the reference junction from the pickle. Rows below MINCOV are dropped.
    Returns the modified df.
    """
    df = df.copy()
    type_b_refs = [r for r in df["Reference"].unique() if r in type_b_map]
    if not type_b_refs:
        return df

    corrected_rows = []
    drop_refs = []
    for ref in type_b_refs:
        info = type_b_map[ref]
        i1, i2 = ref_junctions(info["ref_intron1_len"], info["ref_exon_len"])
        psi, cov = psi_at_ref_jxn(pkl_dict, ref, i1, i2)
        if np.isnan(psi):
            drop_refs.append(ref)
        else:
            corrected_rows.append({"Reference": ref, "PSI_new": psi, "Coverage_new": cov})

    # Drop refs with no coverage at reference junction
    if drop_refs:
        df = df[~df["Reference"].isin(drop_refs)].copy()

    # Update PSI and Coverage for remaining Type B refs
    if corrected_rows:
        corr_df = pd.DataFrame(corrected_rows).set_index("Reference")
        mask = df["Reference"].isin(corr_df.index)
        df.loc[mask, "PSI"]      = df.loc[mask, "Reference"].map(corr_df["PSI_new"])
        df.loc[mask, "Coverage"] = df.loc[mask, "Reference"].map(corr_df["Coverage_new"])
        # Recompute clipped PSI and logit
        df.loc[mask, "PSI_clipped"] = df.loc[mask, "PSI"].clip(lower=clip, upper=1 - clip)
        df.loc[mask, "logit_PSI"]   = logit(df.loc[mask, "PSI_clipped"])

    return df


# ── Merge PSI functions (unchanged from 07_18_2025 version) ───────────────
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

def compute_replicate_sds(pooled_df, label, var_rep_indices=None, wt_rep_indices=None, clip=1e-3):
    all_rep_psi_cols = [col for col in pooled_df.columns
                        if col.startswith(f'{label}_rep') and col.endswith('_psi_raw')]
    all_rep_indices = [int(col.replace(f'{label}_rep', '').replace('_psi_raw', ''))
                       for col in all_rep_psi_cols]

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
        wt_rows = pooled_df[pooled_df['snp'] == 'none'][['event_id'] + sd_psi_cols + sd_logit_cols].copy()

        wt_psi_lookup = {}
        wt_logit_lookup = {}
        for i, psi_col in zip(sd_rep_indices, sd_psi_cols):
            wt_psi_lookup[i] = wt_rows.set_index('event_id')[psi_col]
        for i, logit_col in zip(sd_rep_indices, sd_logit_cols):
            wt_logit_lookup[i] = wt_rows.set_index('event_id')[logit_col]

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

def write_outputs(pooled_df, output_prefix, label):
    wt_refs  = pooled_df[pooled_df['snp'] == 'none']
    var_refs = pooled_df[pooled_df['snp'] != 'none']
    wt_eids  = set(wt_refs['event_id'].dropna().unique())
    var_eids = set(var_refs['event_id'].dropna().unique())
    ok_eids  = sorted(wt_eids & var_eids)

    filtered_df = pooled_df[pooled_df['event_id'].isin(ok_eids)].copy()
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
        "supertable_file=", "output_dir=", "output_prefix=", "clip=",
        "boundary_mismatches=",
        "pkl_HeLa_rep1=", "pkl_HeLa_rep2=",
        "pkl_K562_rep1=", "pkl_K562_rep2=",
        "pkl_MCF7_rep1=", "pkl_MCF7_rep2=",
        "pkl_HMC3_rep1=", "pkl_HMC3_rep2=",
        "pkl_HEK_rep1=",  "pkl_HEK_rep2=",
        "pkl_HEK_wt_rep1=", "pkl_HEK_wt_rep2=",
    ])
    opts = dict(opts)
    clip_val = float(opts.get("--clip", 1e-3))
    output_dir = opts["--output_dir"]
    output_prefix = os.path.join(output_dir, opts["--output_prefix"])
    os.makedirs(output_dir, exist_ok=True)

    supertable = pd.read_csv(opts["--supertable_file"], sep=',')
    supertable.iloc[:, 0] = supertable.iloc[:, 0].astype(int) + 1
    full_seqs = build_fullseqs_df(supertable)

    # ── Load Type B boundary mismatch map ─────────────────────────────────
    type_b_map = {}
    if "--boundary_mismatches" in opts:
        print("Loading boundary mismatches...")
        type_b_map = build_type_b_map(opts["--boundary_mismatches"])
        print(f"  Type B References to correct: {len(type_b_map):,}")

    # ── Load splicing count pickles ───────────────────────────────────────
    pkl_keys = [
        "pkl_HeLa_rep1", "pkl_HeLa_rep2",
        "pkl_K562_rep1", "pkl_K562_rep2",
        "pkl_MCF7_rep1", "pkl_MCF7_rep2",
        "pkl_HMC3_rep1", "pkl_HMC3_rep2",
        "pkl_HEK_rep1",  "pkl_HEK_rep2",
        "pkl_HEK_wt_rep1", "pkl_HEK_wt_rep2",
    ]
    pkls = {}
    if type_b_map:
        print("Loading splicing count pickles for Type B correction...")
        for key in pkl_keys:
            arg = f"--{key}"
            if arg in opts:
                print(f"  {key}...", end=" ", flush=True)
                with open(opts[arg], "rb") as f:
                    pkls[key] = pickle.load(f)
                print(f"ok ({len(pkls[key]):,} refs)")

    # ── Rep-to-pickle mapping (mirrors REP_PKLS in correction scripts) ────
    # Each list entry = (txt_arg_name, pkl_key)
    rep_pkl_map = {
        "--HeLa_Rep1_PSI":     "pkl_HeLa_rep1",
        "--HeLa_Rep2_PSI":     "pkl_HeLa_rep2",
        "--K562_Rep1_PSI":     "pkl_K562_rep1",
        "--K562_Rep2_PSI":     "pkl_K562_rep2",
        "--MCF7_Rep1_PSI":     "pkl_MCF7_rep1",
        "--MCF7_Rep2_PSI":     "pkl_MCF7_rep2",
        "--HMC3_Rep1_PSI":     "pkl_HMC3_rep1",
        "--HMC3_Rep2_PSI":     "pkl_HMC3_rep2",
        "--HEK293_Rep1_PSI":   "pkl_HEK_rep1",
        "--HEK293_Rep2_PSI":   "pkl_HEK_rep2",
        "--HEK293_WT_Rep1_PSI":"pkl_HEK_wt_rep1",
        "--HEK293_WT_Rep2_PSI":"pkl_HEK_wt_rep2",
    }

    def read(path_arg, label):
        df = pd.read_csv(opts[path_arg], sep='\t', engine='python')
        df = pd.merge(df, full_seqs[['Reference']], on='Reference', how='inner')
        df = offset_psis(df, clip=clip_val)
        df['cell_line'] = label
        # Apply Type B junction correction if applicable
        if type_b_map and path_arg in rep_pkl_map:
            pkl_key = rep_pkl_map[path_arg]
            if pkl_key in pkls:
                n_before = len(df)
                df = apply_type_b_corrections(df, pkls[pkl_key], type_b_map, clip_val)
                n_after  = len(df)
                n_corrected = sum(1 for r in type_b_map if r in df["Reference"].values)
                print(f"    Type B correction for {path_arg}: "
                      f"{n_corrected} refs corrected, {n_before - n_after} dropped (below MINCOV at ref junction)")
        return df

    data = {
        'HeLa': [read("--HeLa_Rep1_PSI", "HeLa"), read("--HeLa_Rep2_PSI", "HeLa")],
        'K562': [read("--K562_Rep1_PSI", "K562"), read("--K562_Rep2_PSI", "K562")],
        'MCF7': [read("--MCF7_Rep1_PSI", "MCF7"), read("--MCF7_Rep2_PSI", "MCF7")],
        'HMC3': [read("--HMC3_Rep1_PSI", "HMC3"), read("--HMC3_Rep2_PSI", "HMC3")],
        'HEK': [read("--HEK293_Rep1_PSI", "HEK"), read("--HEK293_Rep2_PSI", "HEK"),
                read("--HEK293_WT_Rep1_PSI", "HEK"), read("--HEK293_WT_Rep2_PSI", "HEK")]
    }

    hek_sd_kwargs = dict(var_rep_indices=[1, 2], wt_rep_indices=[3, 4])
    all_no_deltas = []
    cell_lines = ['HeLa', 'K562', 'MCF7', 'HMC3', 'HEK']

    for label, reps in data.items():
        try:
            print(f"\n=== Processing {label} ===")
            pooled_df = compute_pooled_stats(reps, full_seqs, label, clip=clip_val)
            print(f"  compute_pooled_stats done: {pooled_df.shape}")
            pooled_df = attach_replicate_info(pooled_df, reps, label)
            print(f"  attach_replicate_info done: {pooled_df.shape}")
            sd_kwargs = hek_sd_kwargs if label == 'HEK' else {}
            pooled_df = compute_replicate_sds(pooled_df, label, clip=clip_val, **sd_kwargs)
            print(f"  compute_replicate_sds done: {pooled_df.shape}")
            write_outputs(pooled_df, output_prefix, label)
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
        print(f"\nWrote: {output_prefix}_ALL_WTS_VARS_NO_DELTAS.csv")


if __name__ == "__main__":
    main()
