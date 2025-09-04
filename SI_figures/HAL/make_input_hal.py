import pandas as pd
import os
import zipfile
from io import StringIO

# === Input and output paths ===
input_path = "/ESL/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WITH_WT.csv"
output_hal_zip_path = "/ESL/Figures_SK/HAL/hal_input_variants_only_avgwtpsi_exon6nt.tsv.zip"
output_plotting_path = "/ESL/Figures_SK/HAL/hal_plotting_input_avg_psi_dpsi.csv"

# === Load input file ===
df = pd.read_csv(input_path)

# === List of cell lines for WT PSI and delta PSI (raw ΔPSI) ===
wt_psi_columns = [
    "HEK_wt_pooled_psi_raw",
    "HeLa_wt_pooled_psi_raw",
    "K562_wt_pooled_psi_raw",
    "MCF7_wt_pooled_psi_raw",
    "HMC3_wt_pooled_psi_raw"
]
dpsi_columns = [
    "HEK_dpsi_pooled",
    "HeLa_dpsi_pooled",
    "K562_dpsi_pooled",
    "MCF7_dpsi_pooled",
    "HMC3_dpsi_pooled"
]

# === Filter variant and WT rows ===
df_var = df[df["snp"] != "none"].copy()
df_wt = df[df["snp"] == "none"].copy()

# === Compute average WT PSI (0–100), clip to 0.01–99.99 ===
df_wt["WILDTYPE_PSI"] = (df_wt[wt_psi_columns].mean(axis=1) * 100).clip(lower=0.01, upper=99.99)

# === Compute average dPSI for variants ===
df_var["dPSI"] = df_var[dpsi_columns].mean(axis=1)

# === Merge WT into variants ===
df_merged = pd.merge(
    df_var,
    df_wt[["event_id", "exon", "intron2", "WILDTYPE_PSI"]],
    on="event_id",
    how="inner"
)

# === Construct WT and variant sequences: exon + first 6 nt of intron2 (lowercase) ===
df_merged["WILDTYPE_SEQ"] = df_merged["exon_y"] + df_merged["intron2_y"].str[:6].str.lower()
df_merged["VARIANT_SEQ"] = df_merged["exon_x"] + df_merged["intron2_x"].str[:6].str.lower()

# === Ensure same length and not identical ===
same_len = df_merged["WILDTYPE_SEQ"].str.len() == df_merged["VARIANT_SEQ"].str.len()
diff_seq = df_merged["WILDTYPE_SEQ"] != df_merged["VARIANT_SEQ"]
df_filtered = df_merged[same_len & diff_seq].copy()

# === Count nucleotide differences ===
def count_differences(seq1, seq2):
    return sum(a != b for a, b in zip(seq1, seq2))

df_filtered["NT_DIFFS"] = [
    count_differences(wt, var)
    for wt, var in zip(df_filtered["WILDTYPE_SEQ"], df_filtered["VARIANT_SEQ"])
]

# === Count number of variants implied by snp field ===
# e.g. "24G>A;0A>C" → 2, "0:T>A" → 1
df_filtered["SNP_COUNT"] = df_filtered["snp"].str.split(";").str.len()

# === Keep only rows where SNP count matches observed NT differences ===
df_valid = df_filtered[df_filtered["NT_DIFFS"] == df_filtered["SNP_COUNT"]].copy()

# === Prefix VARIANT_NAME with 'Seq_' to avoid numeric-only IDs ===
df_valid["VARIANT_NAME"] = "Seq_" + df_valid["Reference"].astype(str)

# === Create final HAL input DataFrame ===
hal_df = pd.DataFrame({
    "VARIANT_NAME": df_valid["VARIANT_NAME"],
    "WILDTYPE_SEQ": df_valid["WILDTYPE_SEQ"],
    "VARIANT_SEQ": df_valid["VARIANT_SEQ"],
    "WILDTYPE_PSI": df_valid["WILDTYPE_PSI"]
})

# === Save HAL input to zip archive ===
with zipfile.ZipFile(output_hal_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    buffer = StringIO()
    hal_df.to_csv(buffer, sep="\t", index=False, header=False)
    zipf.writestr("hal_input_variants_only_avgwtpsi_exon6nt.tsv", buffer.getvalue())

# === Save plotting file with dPSI (CSV with header) ===
plot_df = pd.DataFrame({
    "VARIANT_NAME": df_valid["VARIANT_NAME"],
    "WILDTYPE_PSI": df_valid["WILDTYPE_PSI"],
    "dPSI": df_valid["dPSI"]
})

print(len(plot_df))
plot_df.to_csv(output_plotting_path, index=False)

