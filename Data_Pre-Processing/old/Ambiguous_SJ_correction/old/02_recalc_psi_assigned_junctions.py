"""
Recalculate PSI for ambiguous splice junction sequences using each row's
assigned transcript junction positions (from intron1/exon lengths), rather
than the first-occurring transcript in the supertable.

For events with multiple valid splice junctions (different transcripts, same
full_seq), the current pipeline computes PSI using the first-occurring
transcript's junction positions. This script looks up raw junction counts from
the STAR splicing count pickles and recomputes PSI at the assigned positions.

Junction position formula (0-indexed construct coordinates):
  i1 = (26, 286 + len(intron1))
  i2 = (286 + len(intron1) + len(exon) + 1, 871)
  e  = (26, 871)   [exon-skipping, same for all]

PSI = min(i1_count, i2_count) / (min(i1_count, i2_count) + e_count)

Output: CSV comparing current PSI vs recalculated PSI for each ambiguous row,
per replicate.
"""

import pickle
import pandas as pd
import numpy as np
from scipy.special import logit

# ── Paths ──────────────────────────────────────────────────────────────────
SUPERTABLE  = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv"
CURRENT_PSI = "/ESL/Figures_SK/General_preprocessing/output_03_16_2026/03_16_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"
OUTPUT_DIR  = "/ESL/Figures_SK/ambiguous_sjs"

PKL_BASE = "/ESL/Analysis/STAR_alignment/separate_concat_2023_09_19_d1c_ms75_from_s3_2023_11_27/recount_SJs"
PKL_2024  = PKL_BASE + "/2024_07_26"
WT_BASE   = "/ESL/Analysis/WT_Library/separate/recount_SJs"

REPS = {
    "HeLa_rep1":  PKL_BASE + "/HeLa_Rep1_separate_splicing_profiles/HeLa_Rep1_separate_all_splicing_counts.p",
    "HeLa_rep2":  PKL_BASE + "/HeLa_Rep3_20231031_separate_splicing_profiles/HeLa_Rep3_20231031_separate_all_splicing_counts.p",
    "K562_rep1":  PKL_BASE + "/K562_Rep1_separate_splicing_profiles/K562_Rep1_separate_all_splicing_counts.p",
    "K562_rep2":  PKL_BASE + "/K562_Rep3_20231031_separate_splicing_profiles/K562_Rep3_20231031_separate_all_splicing_counts.p",
    "MCF7_rep1":  PKL_2024  + "/MCF7_Rep1_separate_splicing_profiles/MCF7_Rep1_all_splicing_counts.p",
    "MCF7_rep2":  PKL_2024  + "/MCF7_Rep2_separate_splicing_profiles/MCF7_Rep2_all_splicing_counts.p",
    "HMC3_rep1":  PKL_2024  + "/HMC3_Rep1_separate_splicing_profiles/HMC3_Rep1_all_splicing_counts.p",
    "HMC3_rep2":  PKL_2024  + "/HMC3_Rep2_separate_splicing_profiles/HMC3_Rep2_all_splicing_counts.p",
    "HEK_rep1":   PKL_BASE + "/HEK293_Rep2_separate_splicing_profiles/HEK293_Rep2_separate_all_splicing_counts.p",
    "HEK_rep2":   PKL_BASE + "/HEK293_Rep3_separate_splicing_profiles/HEK293_Rep3_separate_all_splicing_counts.p",
    "HEK_wt_rep1": WT_BASE + "/HEK_WT_Rep1_20231101_separate_splicing_profiles/HEK_WT_Rep1_20231101_separate_all_splicing_counts.p",
    "HEK_wt_rep2": WT_BASE + "/HEK_WT_Rep2_20231101_separate_splicing_profiles/HEK_WT_Rep2_20231101_separate_all_splicing_counts.p",
}

SHARED_5P_LEN = 286
MINCOV = 10

def compute_psi(jxn_counts, i1_jxn, i2_jxn, e_jxn=(26, 871)):
    i1 = jxn_counts.get(i1_jxn, 0)
    i2 = jxn_counts.get(i2_jxn, 0)
    e  = jxn_counts.get(e_jxn,  0)
    included = min(i1, i2)
    total = included + e
    if total < MINCOV:
        return np.nan, total
    return included / total, total

def expected_junctions(intron1_len, exon_len):
    i1 = (26, SHARED_5P_LEN + intron1_len)
    i2 = (SHARED_5P_LEN + intron1_len + exon_len + 1, 871)
    e  = (26, 871)
    return i1, i2, e

def main():
    print("Loading supertable...")
    st = pd.read_csv(SUPERTABLE, low_memory=False)

    print("Loading current PSI results...")
    curr = pd.read_csv(CURRENT_PSI, low_memory=False)

    coords_per_event = (st.groupby("event_id")[["exon_start_hg38", "exon_end_hg38"]]
                        .apply(lambda g: g.drop_duplicates().shape[0]))
    ambig_eids = set(coords_per_event[coords_per_event > 1].index)
    print(f"Ambiguous event_ids: {len(ambig_eids)}")

    # Pickle keys are 1-indexed; supertable Reference is 0-indexed → add 1
    seen_seq = {}
    for _, r in st.iterrows():
        if r["full_seq"] not in seen_seq:
            seen_seq[r["full_seq"]] = int(r["Reference"]) + 1

    ambig_rows = st[st["event_id"].isin(ambig_eids)].copy()
    ambig_rows["fasta_ref"] = ambig_rows["full_seq"].map(seen_seq)
    ambig_rows["intron1_len"] = ambig_rows["intron1"].str.len()
    ambig_rows["exon_len"]    = ambig_rows["exon"].str.len()

    ambig_rows["is_primary"] = ambig_rows["fasta_ref"] == ambig_rows["Reference"] + 1
    n_primary = ambig_rows["is_primary"].sum()
    n_alt     = (~ambig_rows["is_primary"]).sum()
    print(f"  Primary (current PSI correct):   {n_primary}")
    print(f"  Alt transcript (PSI may differ): {n_alt}")

    print("\nLoading pickles...")
    pkl_data = {}
    for name, path in REPS.items():
        print(f"  {name}...", end=" ")
        try:
            with open(path, "rb") as f:
                pkl_data[name] = pickle.load(f)
            print(f"ok ({len(pkl_data[name])} refs)")
        except Exception as ex:
            print(f"FAILED: {ex}")
            pkl_data[name] = {}

    print("\nRecalculating PSI...")
    records = []

    for _, row in ambig_rows.iterrows():
        fasta_ref = row["fasta_ref"]
        i1_jxn, i2_jxn, e_jxn = expected_junctions(row["intron1_len"], row["exon_len"])

        rec = {
            "Reference":    row["Reference"],
            "event_id":     row["event_id"],
            "transcript_id": row["transcript_id"],
            "snp":          row["snp"],
            "seq_type":     row["seq_type"],
            "is_primary":   row["is_primary"],
            "fasta_ref":    fasta_ref,
            "intron1_len":  row["intron1_len"],
            "exon_len":     row["exon_len"],
            "i1_jxn":       str(i1_jxn),
            "i2_jxn":       str(i2_jxn),
        }

        for rep_name, counts in pkl_data.items():
            if fasta_ref in counts:
                psi, cov = compute_psi(counts[fasta_ref], i1_jxn, i2_jxn, e_jxn)
            else:
                psi, cov = np.nan, 0
            rec[f"recalc_psi_{rep_name}"]      = psi
            rec[f"recalc_coverage_{rep_name}"] = cov

        records.append(rec)

    recalc = pd.DataFrame(records)

    curr_psi_cols = [c for c in curr.columns if c.endswith("_psi_raw") and "wt_" not in c and "sd_" not in c]
    curr_sub = curr[["Reference"] + curr_psi_cols].copy()
    curr_sub["Reference_0idx"] = curr_sub["Reference"] - 1
    curr_sub = curr_sub.drop(columns=["Reference"]).rename(columns={"Reference_0idx": "Reference"})
    curr_sub.columns = ["Reference"] + [f"current_{c}" for c in curr_psi_cols]

    recalc = recalc.merge(curr_sub, on="Reference", how="left")

    rep_to_curr = {
        "HeLa_rep1":   "HeLa_rep1_psi_raw",
        "HeLa_rep2":   "HeLa_rep2_psi_raw",
        "K562_rep1":   "K562_rep1_psi_raw",
        "K562_rep2":   "K562_rep2_psi_raw",
        "MCF7_rep1":   "MCF7_rep1_psi_raw",
        "MCF7_rep2":   "MCF7_rep2_psi_raw",
        "HMC3_rep1":   "HMC3_rep1_psi_raw",
        "HMC3_rep2":   "HMC3_rep2_psi_raw",
        "HEK_rep1":    "HEK_rep1_psi_raw",
        "HEK_rep2":    "HEK_rep2_psi_raw",
        "HEK_wt_rep1": "HEK_rep3_psi_raw",
        "HEK_wt_rep2": "HEK_rep4_psi_raw",
    }

    for rep_name, curr_col in rep_to_curr.items():
        recalc_col   = f"recalc_psi_{rep_name}"
        current_col  = f"current_{curr_col}"
        if recalc_col in recalc.columns and current_col in recalc.columns:
            recalc[f"delta_psi_{rep_name}"] = recalc[recalc_col] - recalc[current_col]

    outpath = f"{OUTPUT_DIR}/recalc_psi_ambiguous_junctions.csv"
    recalc.to_csv(outpath, index=False)
    print(f"\nSaved: {outpath}  ({len(recalc):,} rows)")

    print("\n=== Summary: PSI difference (recalc - current) for ALT transcript rows ===")
    alt_rows = recalc[~recalc["is_primary"]]
    delta_cols = [c for c in recalc.columns if c.startswith("delta_psi_")]
    for col in delta_cols:
        vals = alt_rows[col].dropna()
        if len(vals) > 0:
            print(f"  {col}: n={len(vals)}, mean={vals.mean():.4f}, "
                  f"median={vals.median():.4f}, max_abs={vals.abs().max():.4f}")

if __name__ == "__main__":
    main()
