"""
04_merge_si_into_data.py — Merge SI alt-junction PSIs into the main data files.

Produces two new files where each SI row appears alongside the main-table rows,
allowing downstream analysis to see all measurements (main + rescued alts) in
one place. The original main files and SI file are left unchanged.

Outputs:
  1e-2_ALL_WITH_WT_with_SI.csv             — ALL_WITH_WT + SI rows
  1e-2_ALL_WTS_VARS_NO_DELTAS_with_SI.csv  — NO_DELTAS + SI rows

A `source_table` column distinguishes original rows ('main') from SI rows ('si').
SI rows that didn't pass the SI coverage-pair filter are not included (they
weren't in the SI file to begin with).

SI rows are normalised into the main-file column layout:
  - Reference stays the same (SI keys by the MANE-row Reference).
  - intron1/exon/intron2 reconstructed by slicing full_seq at SI's
    (alt_intron1_len, alt_exon_len).
  - transcript_id <- alt_transcript_id; junction lengths <- alt values.
  - Per-cell *_alt count/PSI columns renamed to match main schema (drop _alt).
  - SD columns and HEK_rep3/HEK_rep4 columns set to NaN (SI only carries 2 reps
    per cell, no SDs).
"""

import os
import pandas as pd
import numpy as np

BASE_OUT = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output"
ST       = os.path.join(BASE_OUT, "st_corrected.csv")
SI       = os.path.join(BASE_OUT, "ambiguous_sjs", "SI_alt_transcript_psi.csv")
NO_DEL   = os.path.join(BASE_OUT, "1e-2_ALL_WTS_VARS_NO_DELTAS.csv")
WITH_WT  = os.path.join(BASE_OUT, "1e-2_ALL_WITH_WT.csv")

OUT_ND   = os.path.join(BASE_OUT, "1e-2_ALL_WTS_VARS_NO_DELTAS_with_SI.csv")
OUT_WT   = os.path.join(BASE_OUT, "1e-2_ALL_WITH_WT_with_SI.csv")

CELLS = ["HeLa", "K562", "MCF7", "HMC3", "HEK"]


def si_to_main_layout(si_df: pd.DataFrame, main_columns: list, st_lookup: dict) -> pd.DataFrame:
    """Project SI rows into the main-file column layout.

    Parameters
    ----------
    si_df : SI table (post-Stage-3 output)
    main_columns : list of column names from the main file
    st_lookup : dict mapping data-Reference (Reference + 1 from supertable)
                -> dict of supertable fields (full_seq, variant_hg38,
                alt_transcripts_in_supertable, alt_transcripts_gencode_only)

    Returns a DataFrame with `len(si_df)` rows and exactly `main_columns +
    ['source_table']` columns (in that order), with NaN for fields SI doesn't
    carry (SDs, HEK rep3/rep4).
    """
    out = {col: pd.Series([pd.NA] * len(si_df), dtype="object") for col in main_columns}

    # Look up supertable fields per SI row using `original_reference` (the
    # MANE-row Ref in the main file). SI's `Reference` is synthetic
    # (244,001+) and is NOT in the supertable, so the lookup must use
    # original_reference instead.
    refs = si_df["original_reference"].astype(int).tolist()
    fseqs           = [st_lookup.get(r, {}).get("full_seq", "")              for r in refs]
    variant_hg38s   = [st_lookup.get(r, {}).get("variant_hg38", pd.NA)       for r in refs]
    alt_in_st       = [st_lookup.get(r, {}).get("alt_transcripts_in_supertable", "") for r in refs]
    alt_gc_only     = [st_lookup.get(r, {}).get("alt_transcripts_gencode_only", "")  for r in refs]

    i1_lens = si_df["alt_intron1_len"].astype(int).tolist()
    ex_lens = si_df["alt_exon_len"].astype(int).tolist()

    intron1s = [fs[:i1]               for fs, i1 in zip(fseqs, i1_lens)]
    exons    = [fs[i1:i1+ex]          for fs, i1, ex in zip(fseqs, i1_lens, ex_lens)]
    intron2s = [fs[i1+ex:]             for fs, i1, ex in zip(fseqs, i1_lens, ex_lens)]

    # Direct passthroughs
    out["Reference"]                       = si_df["Reference"].values
    out["event_id"]                        = si_df["event_id"].values
    out["event_id_161"]                    = si_df["event_id_161"].values
    out["gene_exon"]                       = si_df["gene_exon"].values
    out["snp"]                             = si_df["snp"].values
    out["source"]                          = si_df["source"].values
    out["seq_type"]                        = si_df["seq_type"].values
    out["intron1"]                         = intron1s
    out["exon"]                            = exons
    out["intron2"]                         = intron2s
    out["full_seq"]                        = fseqs
    out["transcript_id"]                   = si_df["alt_transcript_id"].values
    out["exon_start_hg38"]                 = si_df["alt_exon_start_hg38"].values
    out["exon_end_hg38"]                   = si_df["alt_exon_end_hg38"].values
    out["variant_hg38"]                    = variant_hg38s
    out["transcript_class"]                = "alt"
    out["alt_transcripts_in_supertable"]   = alt_in_st
    out["alt_transcripts_gencode_only"]    = alt_gc_only

    # Per-cell columns: SI uses _alt suffix; main schema drops it.
    for cell in CELLS:
        # Pooled
        for src, dst in [
            (f"{cell}_pooled_included_alt",    f"{cell}_pooled_included"),
            (f"{cell}_pooled_excluded_alt",    f"{cell}_pooled_excluded"),
            (f"{cell}_total_pooled_alt",       f"{cell}_total_pooled"),
            (f"{cell}_pooled_psi_raw_alt",     f"{cell}_pooled_psi_raw"),
            (f"{cell}_pooled_psi_clipped_alt", f"{cell}_pooled_psi_clipped"),
            (f"{cell}_pooled_logit_alt",       f"{cell}_pooled_logit"),
            (f"{cell}_wt_pooled_psi_raw_alt",  f"{cell}_wt_pooled_psi_raw"),
            (f"{cell}_wt_pooled_logit_alt",    f"{cell}_wt_pooled_logit"),
            (f"{cell}_dpsi_pooled_alt",        f"{cell}_dpsi_pooled"),
            (f"{cell}_delta_logit_pooled_alt", f"{cell}_delta_logit_pooled"),
        ]:
            if src in si_df.columns and dst in out:
                out[dst] = si_df[src].values
        # Per rep — SI has rep1/rep2 only (HEK rep3/rep4 stay NaN)
        for ri in (1, 2):
            for src, dst in [
                (f"{cell}_rep{ri}_included_alt",    f"{cell}_rep{ri}_included"),
                (f"{cell}_rep{ri}_excluded_alt",    f"{cell}_rep{ri}_excluded"),
                (f"{cell}_rep{ri}_psi_raw_alt",     f"{cell}_rep{ri}_psi_raw"),
                (f"{cell}_rep{ri}_psi_clipped_alt", f"{cell}_rep{ri}_psi_clipped"),
                (f"{cell}_rep{ri}_logit_alt",       f"{cell}_rep{ri}_logit"),
            ]:
                if src in si_df.columns and dst in out:
                    out[dst] = si_df[src].values

    df = pd.DataFrame({col: out[col] for col in main_columns})
    df["source_table"] = "si"
    return df


def merge(main_path: str, si_df: pd.DataFrame, st_lookup: dict, out_path: str):
    print(f"\n=== {os.path.basename(main_path)} → {os.path.basename(out_path)} ===")
    main = pd.read_csv(main_path, low_memory=False)
    print(f"  main rows: {len(main):,}")
    main_cols = list(main.columns)

    # Drop low_coverage_rescue=True rows from the SI subset before merging —
    # those don't meet the standard pair-coverage threshold and are kept only
    # in the standalone SI file for diagnostic use.
    si_in_scope = si_df[si_df.get("low_coverage_rescue", False) != True].copy()
    n_dropped = len(si_df) - len(si_in_scope)
    print(f"  SI rows after dropping low_coverage_rescue: {len(si_in_scope):,}  (dropped {n_dropped:,})")

    # For ALL_WITH_WT: only include SI rows whose event_id_161 already has a row
    # in this main file (preserve the WT-pairing constraint).
    if "WITH_WT" in os.path.basename(main_path):
        eids_with_wt = set(main["event_id_161"].dropna().unique())
        si_subset = si_in_scope[si_in_scope["event_id_161"].isin(eids_with_wt)].copy()
        print(f"  SI rows (filtered to events in WITH_WT): {len(si_subset):,}")
    else:
        si_subset = si_in_scope
        print(f"  SI rows (no further filter): {len(si_subset):,}")

    si_as_main = si_to_main_layout(si_subset, main_cols, st_lookup)
    main = main.assign(source_table="main")
    merged = pd.concat(
        [main, si_as_main[main_cols + ["source_table"]]],
        ignore_index=True,
    )
    print(f"  merged rows: {len(merged):,}  ({(merged['source_table']=='main').sum():,} main + {(merged['source_table']=='si').sum():,} si)")
    merged.to_csv(out_path, index=False)
    print(f"  saved → {out_path}")


def main():
    print("Loading inputs...")
    si_df = pd.read_csv(SI, low_memory=False)
    print(f"  SI: {len(si_df):,} rows")

    # Supertable lookup: data Reference (= supertable Ref + 1) -> needed fields
    print("Building supertable lookup...")
    st = pd.read_csv(ST,
                     usecols=["Reference", "full_seq", "variant_hg38",
                              "alt_transcripts_in_supertable",
                              "alt_transcripts_gencode_only"],
                     low_memory=False)
    st["data_ref"] = st["Reference"].astype(int) + 1
    st_lookup = (st.set_index("data_ref")
                   [["full_seq", "variant_hg38",
                     "alt_transcripts_in_supertable",
                     "alt_transcripts_gencode_only"]]
                   .to_dict(orient="index"))
    print(f"  supertable lookup entries: {len(st_lookup):,}")

    merge(NO_DEL,  si_df, st_lookup, OUT_ND)
    merge(WITH_WT, si_df, st_lookup, OUT_WT)
    print("\nDone.")


if __name__ == "__main__":
    main()
