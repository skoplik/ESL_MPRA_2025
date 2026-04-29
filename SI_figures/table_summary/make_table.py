import pandas as pd
import numpy as np

# === Paths ===
input_file = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/ALL_WITH_WT.csv"
supertable_file = "/ESL/ESL_MPRA/Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25.csv.gz"
output_file = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/table_gene_exon_coords_psi_dlogits.tsv"

# === Load ===
df = pd.read_csv(input_file)
st = pd.read_csv(supertable_file)

# Ensure merge key types match
df["Reference"] = df["Reference"].astype(str)
st["Reference"] = st["Reference"].astype(str)

# Identify columns
psi_raw_cols = [c for c in df.columns if c.endswith("_psi_raw")]
dlogit_pooled_cols = [c for c in df.columns if c.endswith("delta_logit_pooled")]

if not psi_raw_cols:
    raise ValueError("No *_psi_raw columns found!")
if not dlogit_pooled_cols:
    raise ValueError("No *_delta_logit_pooled columns found!")

# Split WT and variants
wt_df = df[df["snp"] == "none"].copy()
var_df = df[df["snp"] != "none"].copy()

# === WT reference PSI per event_id ===
wt_df["row_ref_psi"] = wt_df[psi_raw_cols].mean(axis=1)
wt_refpsi = (
    wt_df.groupby("event_id")["row_ref_psi"]
    .mean()
    .reset_index(name="ref_PSI")
)

# === Per-variant average Δlogit across pooled cols ===
var_df["variant_avg_dlogit"] = var_df[dlogit_pooled_cols].mean(axis=1)

# === Collapse variants per event_id ===
var_summary = (
    var_df.groupby("event_id")
    .agg(
        delta_logit_min=("variant_avg_dlogit", "min"),
        delta_logit_max=("variant_avg_dlogit", "max"),
        num_variants=("variant_avg_dlogit", "count")
    )
    .reset_index()
)

# === Combine WT and variant summaries ===
summary_df = wt_refpsi.merge(var_summary, on="event_id", how="left")

# === Merge with supertable to add gene_exon and hg38 coords ===
merged = summary_df.merge(
    st[["event_id", "gene_exon", "exon_start_hg38", "exon_end_hg38"]],
    on="event_id",
    how="left"
).drop_duplicates("event_id")

# === Format coords ===
merged["coords_hg38"] = merged["exon_start_hg38"].astype(str) + "-" + merged["exon_end_hg38"].astype(str)

# === Keep only events with >=300 variants ===
filtered = merged[merged["num_variants"] >= 300]

# === Final table (include event_id first) ===
out_df = filtered[["gene_exon", "event_id", "coords_hg38", "ref_PSI", "delta_logit_min", "delta_logit_max", "num_variants"]]

# === Save ===
out_df.to_csv(output_file, sep="\t", index=False)

print(f"Saved table with ≥300 variants: {output_file}")
