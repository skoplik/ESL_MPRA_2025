"""
Add transcript_id, exon_start_hg38, exon_end_hg38, variant_hg38 columns
to the merged ALL_WTS_VARS_NO_DELTAS.csv from the supertable.
Saves in-place.
"""
import pandas as pd

SUPERTABLE = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv"
WTS_FILE   = "/ESL/Figures_SK/General_preprocessing/output_04_17_2026/04_17_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"

print(f"Loading supertable: {SUPERTABLE}")
sup = pd.read_csv(SUPERTABLE)
sup["Reference"] = sup["Reference"].astype(int) + 1   # 0-indexed → 1-indexed

print(f"Loading WTS/VARS file: {WTS_FILE}")
wts = pd.read_csv(WTS_FILE)
print(f"  Shape: {wts.shape}")

annot_cols = ["Reference", "transcript_id", "exon_start_hg38", "exon_end_hg38", "variant_hg38"]
sup_sub = sup[annot_cols]

merged = pd.merge(wts, sup_sub, on="Reference", how="left", validate="one_to_one")

prefix = [
    "Reference", "event_id", "gene_exon", "snp", "source", "seq_type",
    "intron1", "exon", "intron2", "full_seq",
    "transcript_id", "exon_start_hg38", "exon_end_hg38", "variant_hg38"
]
rest = [c for c in merged.columns if c not in prefix]
merged = merged[prefix + rest]

print(f"  Merged shape: {merged.shape}")
for col in ["transcript_id", "exon_start_hg38", "exon_end_hg38", "variant_hg38"]:
    print(f"  Non-null {col}: {merged[col].notna().sum()}/{len(merged)}")

merged.to_csv(WTS_FILE, index=False)
print(f"Saved (in-place): {WTS_FILE}")
