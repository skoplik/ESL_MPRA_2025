"""
Build the union of two classes of sequences with alternative junction annotations:

  Type A — Ambiguous-event alt transcripts:
    Same full_seq appears under multiple transcript annotations in the same
    event_id. PSI was computed using the first-occurring transcript's junction,
    so all sequences have PSI calculated at the same junction,
    but there are alternative junctions for these transcripts annotated in Gencode.

  Type B — Var/ref boundary mismatches:
    A variant's intron1/exon boundary differs from its reference's boundary
    within the same event_id. PSI is counted at a different junction than
    the reference, making delta logit invalid.

Outputs:
  alt_junction_exclusions.csv
      Union table of all affected References with type/reason metadata.

  04_17_2026_1e-2_ALL_WTS_VARS_NO_DELTAS_junction_filtered.csv
      Main COMPASS file with all Type B variant rows removed.
      Type A rows stay in the main file (ref and var use same junction,
      so delta is internally consistent).
"""

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH    = "/ESL/Figures_SK/General_preprocessing/output_03_16_2026/03_16_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"
AMB_PATH     = "/ESL/Figures_SK/ambiguous_sjs/recalc_psi_ambiguous_junctions.csv"
BM_PATH      = "/ESL/Figures_SK/ambiguous_sjs/boundary_mismatches.csv"
OUT_DIR      = "/ESL/Figures_SK/ambiguous_sjs"
OUT_MAIN_DIR = "/ESL/Figures_SK/General_preprocessing/output_04_17_2026"

OUT_EXCL     = f"{OUT_DIR}/alt_junction_exclusions.csv"
OUT_FILTERED = f"{OUT_MAIN_DIR}/04_17_2026_1e-2_ALL_WTS_VARS_NO_DELTAS_junction_filtered.csv"

print("Loading data...")
df   = pd.read_csv(DATA_PATH, low_memory=False)
amb  = pd.read_csv(AMB_PATH)
bm   = pd.read_csv(BM_PATH)

meta = df[["Reference", "gene_exon"]].drop_duplicates("Reference")

# ── Type A: non-primary ambiguous-event rows ─────────────────────────────
type_a = amb[~amb["is_primary"]][["Reference", "event_id", "snp", "seq_type",
                                   "intron1_len", "i1_jxn", "i2_jxn"]].copy()
type_a = type_a.merge(meta, on="Reference", how="left")
type_a["exclusion_type"]      = "A"
type_a["reason"]              = "alt transcript: same full_seq mapped to multiple transcripts; PSI at wrong junction"
type_a["var_intron1_len"]     = type_a["intron1_len"]
type_a["ref_intron1_len"]     = np.nan
type_a["intron1_diff"]        = np.nan
type_a["var_i1_jxn"]          = type_a["i1_jxn"]
type_a["ref_i1_jxn"]          = np.nan
type_a["both_jxns_have_reads"]= np.nan
type_a = type_a.drop(columns=["intron1_len", "i1_jxn", "i2_jxn"])

# ── Type B: var/ref boundary mismatches ──────────────────────────────────
type_b = bm[["Reference", "event_id", "gene_exon", "snp", "seq_type",
             "var_intron1_len", "ref_intron1_len", "intron1_diff",
             "var_i1_jxn", "ref_i1_jxn", "both_jxns_have_reads"]].copy()
type_b["exclusion_type"] = "B"
type_b["reason"]         = "transcript annotation inconsistency: variant assigned to different transcript than reference; intron1 length differs, PSI counted at different junction"

shared_cols = ["Reference", "event_id", "gene_exon", "snp", "seq_type",
               "exclusion_type", "reason",
               "var_intron1_len", "ref_intron1_len", "intron1_diff",
               "var_i1_jxn", "ref_i1_jxn", "both_jxns_have_reads"]

union = pd.concat([type_a[shared_cols], type_b[shared_cols]], ignore_index=True)

both_mask = union.duplicated("Reference", keep=False)
union.loc[both_mask, "exclusion_type"] = "A+B"
union = union.drop_duplicates("Reference", keep="first")
union = union.sort_values(["event_id", "snp"]).reset_index(drop=True)

print(f"\nType A (ambiguous alt transcripts):  {(union['exclusion_type'].isin(['A','A+B'])).sum():,} references")
print(f"Type B (var/ref boundary mismatch):  {(union['exclusion_type'].isin(['B','A+B'])).sum():,} references")
print(f"Both:                                {(union['exclusion_type']=='A+B').sum():,} references")
print(f"Total unique references to exclude:  {len(union):,}")
print(f"Unique event_ids:                    {union['event_id'].nunique():,}")

union.to_csv(OUT_EXCL, index=False)
print(f"\nSaved alt junction exclusion table: {OUT_EXCL}")

# ── Filter main data: remove Type B variants only ─────────────────────────
type_b_refs = set(union.loc[union["exclusion_type"].isin(["B", "A+B"]), "Reference"])

before = len(df)
drop_mask = (df["Reference"].isin(type_b_refs)) & (df["snp"] != "none")
df_filtered = df[~drop_mask].copy()
after = len(df_filtered)

print(f"\nMain data: {before:,} rows → {after:,} rows ({before-after:,} removed)")
print(f"  References (snp='none') kept: {(df_filtered['snp']=='none').sum():,}")
print(f"  Variants kept:               {(df_filtered['snp']!='none').sum():,}")

df_filtered.to_csv(OUT_FILTERED, index=False)
print(f"Saved filtered data: {OUT_FILTERED}")
