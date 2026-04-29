import argparse
import os
import zipfile
from io import StringIO
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", required=True, help="ALL_WITH_WT.csv")
parser.add_argument("--output_zip", required=True)
parser.add_argument("--output_plot_csv", required=True)
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output_zip), exist_ok=True)

wt_psi_columns = [
    "HEK_wt_pooled_psi_raw", "HeLa_wt_pooled_psi_raw", "K562_wt_pooled_psi_raw",
    "MCF7_wt_pooled_psi_raw", "HMC3_wt_pooled_psi_raw"
]
dpsi_columns = [
    "HEK_dpsi_pooled", "HeLa_dpsi_pooled", "K562_dpsi_pooled",
    "MCF7_dpsi_pooled", "HMC3_dpsi_pooled"
]

print("Loading data...")
df = pd.read_csv(args.input_csv, low_memory=False)

df_var = df[df["snp"] != "none"].copy()
df_wt  = df[df["snp"] == "none"].copy()

df_wt["WILDTYPE_PSI"] = (df_wt[wt_psi_columns].mean(axis=1) * 100).clip(lower=0.01, upper=99.99)
df_var["dPSI"] = df_var[dpsi_columns].mean(axis=1)

df_merged = pd.merge(
    df_var,
    df_wt[["event_id", "exon", "intron2", "WILDTYPE_PSI"]],
    on="event_id",
    how="inner"
)

df_merged["WILDTYPE_SEQ"] = df_merged["exon_y"] + df_merged["intron2_y"].str[:6].str.lower()
df_merged["VARIANT_SEQ"]  = df_merged["exon_x"] + df_merged["intron2_x"].str[:6].str.lower()

same_len  = df_merged["WILDTYPE_SEQ"].str.len() == df_merged["VARIANT_SEQ"].str.len()
diff_seq  = df_merged["WILDTYPE_SEQ"] != df_merged["VARIANT_SEQ"]
df_filtered = df_merged[same_len & diff_seq].copy()

def count_differences(s1, s2):
    return sum(a != b for a, b in zip(s1, s2))

df_filtered["NT_DIFFS"]  = [count_differences(w, v) for w, v in zip(df_filtered["WILDTYPE_SEQ"], df_filtered["VARIANT_SEQ"])]
df_filtered["SNP_COUNT"] = df_filtered["snp"].str.split(";").str.len()
df_valid = df_filtered[df_filtered["NT_DIFFS"] == df_filtered["SNP_COUNT"]].copy()
df_valid["VARIANT_NAME"] = "Seq_" + df_valid["Reference"].astype(str)

hal_df = pd.DataFrame({
    "VARIANT_NAME": df_valid["VARIANT_NAME"],
    "WILDTYPE_SEQ": df_valid["WILDTYPE_SEQ"],
    "VARIANT_SEQ":  df_valid["VARIANT_SEQ"],
    "WILDTYPE_PSI": df_valid["WILDTYPE_PSI"]
})

with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    buf = StringIO()
    hal_df.to_csv(buf, sep="\t", index=False, header=False)
    zipf.writestr("hal_input_variants_only_avgwtpsi_exon6nt.tsv", buf.getvalue())

pd.DataFrame({
    "VARIANT_NAME": df_valid["VARIANT_NAME"],
    "WILDTYPE_PSI": df_valid["WILDTYPE_PSI"],
    "dPSI":         df_valid["dPSI"]
}).to_csv(args.output_plot_csv, index=False)

print(f"HAL input zip:   {args.output_zip}  ({len(hal_df):,} variants)")
print(f"Plotting CSV:    {args.output_plot_csv}")
print("\nDone. Submit the zip to HAL manually.")
