"""
For ambiguous events, update transcript_id, exon_start_hg38, exon_end_hg38
in the main table for rows whose intron1 length doesn't match the chosen
junction's intron1 length.

Two cases:
  Case 1: the same SNP also exists as a row with the chosen intron1_len →
          copy intron1/exon/intron2/full_seq AND metadata from that row.
  Case 2: no matching chosen-intron1_len row exists for this SNP (the variant
          was designed only for the non-chosen transcript) →
          update only transcript_id, exon_start_hg38, exon_end_hg38 from
          the supertable's chosen_junction_* columns.
          Sequences are left as-is (they reflect the actual synthesized construct).

Does NOT modify the supertable — supertable changes are handled separately.

Inputs:
  output_04_24_2026/04_24_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv  (from script 07)
  output_04_24_2026/st_04_24_2026.csv                            (supertable with chosen_junction_* cols)

Outputs (new files, source not modified):
  output_04_24_2026/04_24_2026_1e-2_ALL_WTS_VARS_NO_DELTAS_seqfix.csv
  output_04_24_2026/04_24_2026_1e-2_ALL_WITH_WT_seqfix.csv
"""

import pandas as pd
import numpy as np
import os

MAIN_CSV   = "/ESL/Figures_SK/General_preprocessing/output_04_24_2026/04_24_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"
ST_IN      = "/ESL/Figures_SK/General_preprocessing/output_04_24_2026/st_04_24_2026.csv"
OUT_DIR    = "/ESL/Figures_SK/General_preprocessing/output_04_24_2026"
OUT_PREFIX = "04_24_2026_1e-2"

SEQ_COLS  = ["intron1", "exon", "intron2", "full_seq",
             "transcript_id", "exon_start_hg38", "exon_end_hg38"]
META_COLS = ["transcript_id", "exon_start_hg38", "exon_end_hg38"]

print("Loading main CSV...")
df = pd.read_csv(MAIN_CSV, low_memory=False)
df["_intron1_len"] = df["intron1"].str.len()

print("Loading supertable (with chosen_junction_* cols)...")
st = pd.read_csv(ST_IN, low_memory=False)

# Build event -> chosen intron1_len and chosen metadata from supertable
chosen_i1   = {}
chosen_meta = {}

ref_rows = st[st["snp"] == "none"].copy()
ref_rows["_intron1_len"] = ref_rows["intron1"].str.len()

for eid in ref_rows["event_id"].unique():
    sub = ref_rows[ref_rows["event_id"] == eid]
    if "chosen_junction_transcript_id" not in sub.columns:
        continue
    chosen_tx = str(sub["chosen_junction_transcript_id"].iloc[0])
    if chosen_tx == "nan":
        continue
    tx_base = chosen_tx.split(".")[0]
    match = sub[sub["transcript_id"].str.startswith(tx_base)]
    if len(match):
        row0 = match.iloc[0]
        chosen_i1[eid]   = int(row0["_intron1_len"])
        chosen_meta[eid] = {
            "transcript_id":   row0["transcript_id"],
            "exon_start_hg38": row0["exon_start_hg38"],
            "exon_end_hg38":   row0["exon_end_hg38"],
        }
    else:
        # Fall back to chosen_junction_* columns directly
        r0 = sub.iloc[0]
        chosen_meta[eid] = {
            "transcript_id":   r0["chosen_junction_transcript_id"],
            "exon_start_hg38": r0.get("chosen_junction_exon_start_hg38", np.nan),
            "exon_end_hg38":   r0.get("chosen_junction_exon_end_hg38",   np.nan),
        }

print(f"Events with chosen junction: {len(chosen_i1)}")

# Identify mismatched rows: intron1_len != chosen_i1
df["_chosen_i1"] = df["event_id"].map(chosen_i1)
mismatch_mask = df["_chosen_i1"].notna() & (df["_intron1_len"] != df["_chosen_i1"])
print(f"Mismatched rows: {mismatch_mask.sum():,}")

# Build seq_map from main table non-mismatched rows in ambiguous events
ambig_eids = set(chosen_i1.keys())
seq_source = df[df["event_id"].isin(ambig_eids) & ~mismatch_mask].copy()
seq_map = {}
for _, r in seq_source.iterrows():
    key = (r["event_id"], str(r["snp"]), int(r["_intron1_len"]))
    if key not in seq_map:
        seq_map[key] = {col: r[col] for col in SEQ_COLS}
print(f"Seq source rows: {len(seq_source):,}  unique keys: {len(seq_map):,}")

# Update mismatched rows
print("\nUpdating mismatched rows...")
df_out = df.copy()
updated_full = 0
updated_meta = 0
no_update    = 0

for idx, row in df[mismatch_mask].iterrows():
    eid = row["event_id"]
    key = (eid, str(row["snp"]), int(row["_chosen_i1"]))

    if key in seq_map:
        # Case 1: matching row found — copy full sequences + metadata
        src = seq_map[key]
        for col in SEQ_COLS:
            df_out.at[idx, col] = src[col]
        updated_full += 1
    elif eid in chosen_meta:
        # Case 2: no matching row — update metadata only (sequences stay as-is)
        for col in META_COLS:
            df_out.at[idx, col] = chosen_meta[eid][col]
        updated_meta += 1
    else:
        no_update += 1

print(f"  Case 1 (full sequence update): {updated_full:,}")
print(f"  Case 2 (metadata only):        {updated_meta:,}")
if no_update:
    print(f"  WARNING: {no_update} rows had no update source")

# Drop temp columns
df_out = df_out.drop(columns=[c for c in df_out.columns if c.startswith("_")])

# Save
no_deltas_path = os.path.join(OUT_DIR, f"{OUT_PREFIX}_ALL_WTS_VARS_NO_DELTAS_seqfix.csv")
df_out.to_csv(no_deltas_path, index=False)
print(f"\nSaved: {no_deltas_path}")

wt_eids  = set(df_out.loc[df_out["snp"] == "none", "event_id"])
var_eids = set(df_out.loc[df_out["snp"] != "none", "event_id"])
with_wt  = df_out[df_out["event_id"].isin(wt_eids & var_eids)].copy()
with_wt_path = os.path.join(OUT_DIR, f"{OUT_PREFIX}_ALL_WITH_WT_seqfix.csv")
with_wt.to_csv(with_wt_path, index=False)
print(f"Saved: {with_wt_path}")
print(f"  Rows: {len(with_wt):,}  WT: {(with_wt['snp']=='none').sum():,}  Var: {(with_wt['snp']!='none').sum():,}")
print("\nDone.")
