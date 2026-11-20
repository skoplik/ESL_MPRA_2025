import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import os

# === Input files ===
hpa_file = "/ESL/ESL_MPRA/SI_figures/RBP_expression_data/hpa_rna_celline.tsv"
hmc3_file = "/ESL/ESL_MPRA/SI_figures/RBP_expression_data/processed_nTPMs/HMC3_merged_with_ensg.tsv"
output_merged = "/ESL/ESL_MPRA/SI_figures/RBP_expression_data/processed_nTPMs/nTPM_merged_HMC3_HEK_HeLa_MCF7_K562.tsv"
output_dir = "/ESL/ESL_MPRA/SI_figures/RBP_expression_data/processed_nTPMs"

os.makedirs(output_dir, exist_ok=True)

# === Load HPA file ===
hpa = pd.read_csv(hpa_file, sep="\t", dtype=str)
hpa["nTPM"] = pd.to_numeric(hpa["nTPM"], errors="coerce")

# === Cell lines of interest ===
cell_lines = {
    "HEK293": "HEK293",
    "HeLa": "HeLa",
    "MCF7": "MCF-7",
    "K562": "K-562"
}

cell_line_files = {}


# === Extract one row per ENSG per cell line and store separately ===
cell_line_dfs = {}
for label, hpa_name in cell_lines.items():
    df = hpa[hpa["Cell line"] == hpa_name].copy()
    df = df[["Gene", "Gene name", "nTPM"]].rename(columns={
        "Gene": "EnsemblGeneID",
        "nTPM": f"{label}_nTPM"
    })
    df = df.drop_duplicates("EnsemblGeneID")
    df.to_csv(os.path.join(output_dir, f"{label}_nTPM.tsv"), sep="\t", index=False)
    cell_line_dfs[label] = df

# === Load HMC3 file ===
hmc3 = pd.read_csv(hmc3_file, sep="\t", dtype=str)
hmc3["nTPM"] = pd.to_numeric(hmc3["nTPM"], errors="coerce")
hmc3 = hmc3.rename(columns={"nTPM": "HMC3_nTPM"})

# === Merge everything stepwise ===
merged = hmc3[["EnsemblGeneID", "HMC3_nTPM"]].copy()

for label, df in cell_line_dfs.items():
    df = df.drop(columns=["Gene name"])  # avoid duplication
    merged = pd.merge(merged, df, on="EnsemblGeneID", how="outer")

# === Get master gene name mapping from HPA ===
gene_name_map = hpa[["Gene", "Gene name"]].drop_duplicates().rename(columns={"Gene": "EnsemblGeneID"})
merged = pd.merge(gene_name_map, merged, on="EnsemblGeneID", how="right")

hpa_cols = ["HEK293_nTPM", "HeLa_nTPM", "MCF7_nTPM", "K562_nTPM"]
merged = merged.dropna(subset=hpa_cols)

# === Save final merged output ===
merged.to_csv(output_merged, sep="\t", index=False)
