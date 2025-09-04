# We use GSM6031960 because it corresponds to RNA-seq expression data 
# from the HMC3 human microglial cell line. This sample
# provides transcript-per-million (TPM) values that serve as a 
# representative expression profile for microglia. TPMs from GSM6031960 
# are used to quantify gene expression levels in HMC3

import pandas as pd

# === File paths ===
annotation_file = "/ESL/Figures_SK/effect_size_final/Expression_analysis/Human.GRCh38.p13.annot.tsv"
expression_file = "/ESL/Figures_SK/effect_size_final/Expression_analysis/GSE200354_norm_counts_TPM_GRCh38.p13_NCBI.tsv"
output_file = "/ESL/Figures_SK/effect_size_final/Expression_analysis/processed_nTPMs/HMC3_merged_with_ensg.tsv"


# === Load data ===
ann = pd.read_csv(annotation_file, sep="\t", dtype=str)
expr = pd.read_csv(expression_file, sep="\t", dtype=str)

# === Convert types ===
ann["GeneID"] = pd.to_numeric(ann["GeneID"], errors="coerce")
expr["GeneID"] = pd.to_numeric(expr["GeneID"], errors="coerce")
expr["GSM6031960"] = pd.to_numeric(expr["GSM6031960"], errors="coerce")

# === Filter to rows with valid EnsemblGeneID ===
ann = ann[ann["EnsemblGeneID"].str.startswith("ENSG", na=False)]

# === Merge on GeneID ===
merged = pd.merge(ann[["GeneID", "EnsemblGeneID"]], expr[["GeneID", "GSM6031960"]], on="GeneID", how="inner")

# === Compute nTPM ===
total_tpm = merged["GSM6031960"].sum()
merged["nTPM"] = (merged["GSM6031960"] / total_tpm) * 1e6

# === Output ENSG and nTPM only ===
merged[["EnsemblGeneID", "nTPM"]].to_csv(output_file, sep="\t", index=False)

