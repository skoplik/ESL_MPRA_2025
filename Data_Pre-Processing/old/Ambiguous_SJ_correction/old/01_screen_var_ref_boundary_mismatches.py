"""
Screen for transcript annotation inconsistencies where a variant row's
intron1/exon boundary differs from its reference's boundary within the same
event_id.

This is an annotation artifact: the variant construct was assigned to a
different Gencode transcript than its reference construct, and those two
transcripts have different exon coordinates for the same event. The boundary
difference is not caused by the variant sequence itself. Because PSI is counted
at different junctions for the variant and reference, delta logit is not a
valid comparison for these rows.

For each mismatch, also looks up the raw junction counts from splicing
count pickles to show the actual read evidence at both the variant's
junction and the reference's junction.

Outputs:
  boundary_mismatches.csv  — all mismatch rows with junction evidence
"""

import pickle
import pandas as pd
import numpy as np
from scipy.special import logit

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH   = "/ESL/Figures_SK/General_preprocessing/output_03_16_2026/03_16_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"
OUTPUT_DIR  = "/ESL/Figures_SK/ambiguous_sjs"
OUTPUT_CSV  = f"{OUTPUT_DIR}/boundary_mismatches.csv"

PKL_BASE = "/ESL/Analysis/STAR_alignment/separate_concat_2023_09_19_d1c_ms75_from_s3_2023_11_27/recount_SJs"
PKL_2024  = PKL_BASE + "/2024_07_26"
WT_BASE   = "/ESL/Analysis/WT_Library/separate/recount_SJs"

REPS = {
    "HeLa_rep1":   PKL_BASE + "/HeLa_Rep1_separate_splicing_profiles/HeLa_Rep1_separate_all_splicing_counts.p",
    "HeLa_rep2":   PKL_BASE + "/HeLa_Rep3_20231031_separate_splicing_profiles/HeLa_Rep3_20231031_separate_all_splicing_counts.p",
    "K562_rep1":   PKL_BASE + "/K562_Rep1_separate_splicing_profiles/K562_Rep1_separate_all_splicing_counts.p",
    "K562_rep2":   PKL_BASE + "/K562_Rep3_20231031_separate_splicing_profiles/K562_Rep3_20231031_separate_all_splicing_counts.p",
    "MCF7_rep1":   PKL_2024  + "/MCF7_Rep1_separate_splicing_profiles/MCF7_Rep1_all_splicing_counts.p",
    "MCF7_rep2":   PKL_2024  + "/MCF7_Rep2_separate_splicing_profiles/MCF7_Rep2_all_splicing_counts.p",
    "HMC3_rep1":   PKL_2024  + "/HMC3_Rep1_separate_splicing_profiles/HMC3_Rep1_all_splicing_counts.p",
    "HMC3_rep2":   PKL_2024  + "/HMC3_Rep2_separate_splicing_profiles/HMC3_Rep2_all_splicing_counts.p",
    "HEK_rep1":    PKL_BASE + "/HEK293_Rep2_separate_splicing_profiles/HEK293_Rep2_separate_all_splicing_counts.p",
    "HEK_rep2":    PKL_BASE + "/HEK293_Rep3_separate_splicing_profiles/HEK293_Rep3_separate_all_splicing_counts.p",
    "HEK_wt_rep1": WT_BASE  + "/HEK_WT_Rep1_20231101_separate_splicing_profiles/HEK_WT_Rep1_20231101_separate_all_splicing_counts.p",
    "HEK_wt_rep2": WT_BASE  + "/HEK_WT_Rep2_20231101_separate_splicing_profiles/HEK_WT_Rep2_20231101_separate_all_splicing_counts.p",
}

SHARED_5P_LEN = 286
MINCOV        = 10
CLIP          = 0.01
E_JXN         = (26, 871)


def i1_jxn(intron1_len):
    return (26, SHARED_5P_LEN + intron1_len)

def i2_jxn(intron1_len, exon_len):
    return (SHARED_5P_LEN + intron1_len + exon_len + 1, 871)

def psi_at(counts, i1, i2, e=E_JXN):
    n_i1 = counts.get(i1, 0)
    n_i2 = counts.get(i2, 0)
    n_e  = counts.get(e,  0)
    inc  = min(n_i1, n_i2)
    tot  = inc + n_e
    return (inc / tot, tot, n_i1, n_i2, n_e) if tot >= MINCOV else (np.nan, tot, n_i1, n_i2, n_e)


def main():
    print("Loading COMPASS data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["intron1_len"] = df["intron1"].str.len()
    df["exon_len"]    = df["exon"].str.len()
    df["intron2_len"] = df["intron2"].str.len()

    refs = (df[df["snp"] == "none"]
            [["event_id", "Reference", "intron1_len", "exon_len", "intron2_len"]]
            .copy()
            .rename(columns={
                "Reference":   "ref_Reference",
                "intron1_len": "ref_intron1_len",
                "exon_len":    "ref_exon_len",
                "intron2_len": "ref_intron2_len",
            }))

    vars_df = df[df["snp"] != "none"].copy()
    merged  = vars_df.merge(refs, on="event_id", how="left")

    mismatches = merged[
        (merged["intron1_len"] != merged["ref_intron1_len"]) |
        (merged["exon_len"]    != merged["ref_exon_len"])
    ].copy()

    print(f"Total variants: {len(vars_df):,}")
    print(f"Variants with boundary mismatch vs reference: {len(mismatches):,}")
    print(f"Unique event_ids affected: {mismatches['event_id'].nunique():,}")
    print(f"intron1 diff distribution:")
    mismatches["intron1_diff"] = mismatches["intron1_len"] - mismatches["ref_intron1_len"]
    print(mismatches["intron1_diff"].value_counts().sort_index().to_string())
    print()

    print("Loading splicing count pickles...")
    pkl_data = {}
    for name, path in REPS.items():
        print(f"  {name}...", end=" ", flush=True)
        try:
            with open(path, "rb") as f:
                pkl_data[name] = pickle.load(f)
            print(f"ok ({len(pkl_data[name]):,} refs)")
        except Exception as ex:
            print(f"FAILED: {ex}")
            pkl_data[name] = {}

    print(f"\nExamining junction counts for {len(mismatches):,} mismatch rows...")
    records = []

    for _, row in mismatches.iterrows():
        if pd.isna(row["ref_Reference"]):
            continue
        var_ref   = int(row["Reference"])
        ref_ref   = int(row["ref_Reference"])

        var_i1 = i1_jxn(row["intron1_len"])
        var_i2 = i2_jxn(row["intron1_len"], row["exon_len"])
        ref_i1 = i1_jxn(row["ref_intron1_len"])
        ref_i2 = i2_jxn(row["ref_intron1_len"], row["ref_exon_len"])

        rec = {
            "Reference":        row["Reference"],
            "ref_Reference":    row["ref_Reference"],
            "event_id":         row["event_id"],
            "gene_exon":        row["gene_exon"],
            "snp":              row["snp"],
            "source":           row["source"],
            "seq_type":         row["seq_type"],
            "var_intron1_len":  row["intron1_len"],
            "ref_intron1_len":  row["ref_intron1_len"],
            "intron1_diff":     row["intron1_len"] - row["ref_intron1_len"],
            "var_exon_len":     row["exon_len"],
            "ref_exon_len":     row["ref_exon_len"],
            "var_i1_jxn":       str(var_i1),
            "ref_i1_jxn":       str(ref_i1),
            "var_i2_jxn":       str(var_i2),
            "ref_i2_jxn":       str(ref_i2),
        }

        for rep_name, counts in pkl_data.items():
            var_c = counts.get(var_ref, {})
            ref_c = counts.get(ref_ref, {})

            psi_var_at_var, cov_var_at_var, vi1_at_vj, vi2_at_vj, ve_at_vj = psi_at(var_c, var_i1, var_i2)
            psi_var_at_ref, cov_var_at_ref, vi1_at_rj, vi2_at_rj, _        = psi_at(var_c, ref_i1, ref_i2)
            psi_ref_at_ref, cov_ref_at_ref, ri1_at_rj, ri2_at_rj, re_at_rj = psi_at(ref_c, ref_i1, ref_i2)
            psi_ref_at_var, cov_ref_at_var, ri1_at_vj, ri2_at_vj, _         = psi_at(ref_c, var_i1, var_i2)

            rec[f"var_psi_at_var_jxn_{rep_name}"]      = psi_var_at_var
            rec[f"var_psi_at_ref_jxn_{rep_name}"]      = psi_var_at_ref
            rec[f"ref_psi_at_ref_jxn_{rep_name}"]      = psi_ref_at_ref
            rec[f"ref_psi_at_var_jxn_{rep_name}"]      = psi_ref_at_var
            rec[f"var_i1_reads_at_var_jxn_{rep_name}"] = vi1_at_vj
            rec[f"var_i1_reads_at_ref_jxn_{rep_name}"] = vi1_at_rj
            rec[f"ref_i1_reads_at_ref_jxn_{rep_name}"] = ri1_at_rj
            rec[f"ref_i1_reads_at_var_jxn_{rep_name}"] = ri1_at_vj

        records.append(rec)

    out = pd.DataFrame(records)

    rep_names = list(REPS.keys())
    ref_reads_at_var_cols = [f"ref_i1_reads_at_var_jxn_{r}" for r in rep_names]
    var_reads_at_ref_cols = [f"var_i1_reads_at_ref_jxn_{r}" for r in rep_names]

    out["ref_any_reads_at_var_jxn"] = out[ref_reads_at_var_cols].fillna(0).gt(0).any(axis=1)
    out["var_any_reads_at_ref_jxn"] = out[var_reads_at_ref_cols].fillna(0).gt(0).any(axis=1)
    out["both_jxns_have_reads"]     = out["ref_any_reads_at_var_jxn"] & out["var_any_reads_at_ref_jxn"]

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  ({len(out):,} rows)")

    print("\n=== Summary ===")
    print(f"Mismatch rows: {len(out):,}")
    print(f"  Reference has reads at variant's annotated junction:  {out['ref_any_reads_at_var_jxn'].sum():,}")
    print(f"  Variant has reads at reference's annotated junction:  {out['var_any_reads_at_ref_jxn'].sum():,}")
    print(f"  Both junctions have reads (delta logit invalid):      {out['both_jxns_have_reads'].sum():,}")

    print("\nAffected event_ids:")
    print(out[["event_id", "gene_exon", "intron1_diff", "both_jxns_have_reads"]]
          .drop_duplicates("event_id").to_string(index=False))


if __name__ == "__main__":
    main()
