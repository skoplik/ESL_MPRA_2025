import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
import json

def logit_clip(p, eps=0.01):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
    
    
# === Constants ===
CITRINE_EXON1 = ("ATGGTGTCCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTCAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAACTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGATGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAAGTGAAGTTCGAGGGCGACACCCTCGTGAACCGCATCGAGCTAAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAAGTGAACTTCAAGATCCGCCACAACATCGAG")
CITRINE_EXON2 = ("GACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAG")
SMN2_INTRON6 = ("GTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA")
SMN2_INTRON7 = ("AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAAAAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCCAAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTGTGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTAGGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTCTGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAATTCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAG")

LEN_CITRINE1 = len(CITRINE_EXON1)
LEN_SMN2_5 = len(SMN2_INTRON6)
LEN_SMN2_3 = len(SMN2_INTRON7)
LEN_CITRINE2 = len(CITRINE_EXON2)

# === Paths ===
main_data_path = "/ESL/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WITH_WT.csv"
wt_path = "/ESL/Figures_SK/Spliceai/redo_07_23_25/spliceai_raw_wt_preds.tsv"
snv_path = "/ESL/Figures_SK/Spliceai/redo_07_23_25/spliceai_raw_preds.tsv"
output_dir = "/ESL/Figures_SK/Spliceai/redo_07_23_25/final_07_27_25"
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# === Load
print("Loading main data...")
main_df = pd.read_csv(main_data_path)
print(f"Main data loaded: {len(main_df):,} rows")

print("Loading WT spliceai predictions...")
wt_df = pd.read_csv(wt_path, sep="\t", dtype=str)
print(f"WT predictions loaded: {len(wt_df):,} rows")

print("Loading SNV spliceai predictions...")
snv_df = pd.read_csv(snv_path, sep="\t", dtype=str)
print(f"SNV predictions loaded: {len(snv_df):,} rows")

main_df["Reference"] = main_df["Reference"].astype(str)

# === Coordinates
print("Computing exon coordinates...")
main_df["intron1_len"] = main_df["intron1"].str.len()
main_df["exon_len"] = main_df["exon"].str.len()
main_df["exon_start"] = LEN_CITRINE1 + LEN_SMN2_5 + main_df["intron1_len"]
main_df["exon_end"] = main_df["exon_start"] + main_df["exon_len"] - 1

# === Parse SpliceAI scores
print("Parsing WT scores...")
wt_scores = {}
for idx, (ref, score) in enumerate(zip(wt_df["Reference"], wt_df["spliceai_scores"])):
    if idx > 0 and idx % 100 == 0:
        print(f"Parsed {idx:,} WT entries...")
    if isinstance(score, str) and score.startswith("["):
        try:
            wt_scores[ref] = json.loads(score)
        except Exception as e:
            print(f"[WT] Failed to parse {ref}: {e}")

print(f"WT parsing complete. Parsed {len(wt_scores):,} valid entries.")

print("Parsing SNV scores...")
snv_scores = {}
for idx, (ref, score) in enumerate(zip(snv_df["Reference"], snv_df["spliceai_scores"])):
    if idx > 0 and idx % 10000 == 0:
        print(f"Parsed {idx:,} SNV entries...")
    if isinstance(score, str) and score.startswith("["):
        try:
            snv_scores[ref] = json.loads(score)
        except Exception as e:
            print(f"[SNV] Failed to parse {ref}: {e}")

print(f"SNV parsing complete. Parsed {len(snv_scores):,} valid entries.")

# === Subset main dataframe
print("Subsetting to matched WTs and SNVs...")
wt_main = main_df[(main_df["snp"] == "none") & main_df["Reference"].isin(wt_scores)].copy()
snv_main = main_df[(main_df["snp"] != "none") & main_df["Reference"].isin(snv_scores)].copy()

wt_main["spliceai"] = wt_main["Reference"].map(wt_scores)
snv_main["spliceai"] = snv_main["Reference"].map(snv_scores)

# === Compute SpliceAI predictions
def extract_predictions(df, name):
    print(f"Extracting predictions for {name}...")
    prod_result = np.full(len(df), np.nan)
    logit_sum_result = np.full(len(df), np.nan)
    starts = df["exon_start"].astype(int).to_numpy()
    ends = df["exon_end"].astype(int).to_numpy()
    spliceai = df["spliceai"].to_numpy()
    references = df["Reference"].to_numpy()
    index_lut = dict(zip(df.index.to_numpy(), range(len(df))))
    for count, (i, s, e, v, ref) in enumerate(zip(df.index, starts, ends, spliceai, references)):
        if count > 0 and count % 10000 == 0:
            print(f"[{name}] Processed {count:,} entries...")
        if not isinstance(v, list): continue
        try:
            if not (0 <= s < len(v)) or not (0 <= e < len(v)): continue
            sa = v[s][1]
            sd = v[e][2]
            prod_result[index_lut[i]] = sa * sd
            logit_sum_result[index_lut[i]] = logit_clip(sa) + logit_clip(sd)
        except Exception as e:
            print(f"[{name}] Failed at ref {ref} s={s} e={e}: {e}")
            continue
    print(f"[{name}] Done.")
    return prod_result, logit_sum_result

wt_main["spliceai_product"], wt_main["spliceai_logit"] = extract_predictions(wt_main, "WT")
snv_main["spliceai_product"], snv_main["spliceai_logit"] = extract_predictions(snv_main, "SNV")

# === Compute experimental averages
print("Computing average PSI and logit across cell lines...")
psi_cols = [
    "HeLa_wt_pooled_psi_raw", "K562_wt_pooled_psi_raw", "MCF7_wt_pooled_psi_raw",
    "HMC3_wt_pooled_psi_raw", "HEK_wt_pooled_psi_raw"
]
logit_cols = [
    "HeLa_pooled_logit", "K562_pooled_logit", "MCF7_pooled_logit",
    "HMC3_pooled_logit", "HEK_pooled_logit"
]
main_df[psi_cols + logit_cols] = main_df[psi_cols + logit_cols].apply(pd.to_numeric, errors="coerce")
main_df["avg_psi"] = main_df[psi_cols].mean(axis=1)
main_df["avg_logit"] = main_df[logit_cols].mean(axis=1)

# === Assign experimental avg_psi and avg_logit to SNVs
print("Assigning experimental avg_psi and avg_logit to SNV rows by Reference...")
ref_to_psi = main_df.set_index("Reference")["avg_psi"].to_dict()
ref_to_logit = main_df.set_index("Reference")["avg_logit"].to_dict()

snv_main["avg_psi"] = snv_main["Reference"].map(ref_to_psi)
snv_main["avg_logit"] = snv_main["Reference"].map(ref_to_logit)

print("Joining WT SpliceAI and experimental values by event_id...")
wt_spliceai = wt_main.set_index("event_id")[["spliceai_product", "spliceai_logit"]]
wt_exp = main_df[main_df["snp"] == "none"].set_index("event_id")[["avg_psi", "avg_logit"]]



snv_main["wt_spliceai_product"] = snv_main["event_id"].map(wt_spliceai["spliceai_product"])
snv_main["wt_spliceai_logit"] = snv_main["event_id"].map(wt_spliceai["spliceai_logit"])
snv_main["wt_exp_psi"] = snv_main["event_id"].map(wt_exp["avg_psi"])
snv_main["wt_exp_logit"] = snv_main["event_id"].map(wt_exp["avg_logit"])

snv_main["delta_spliceai_product"] = snv_main["spliceai_product"] - snv_main["wt_spliceai_product"]
snv_main["delta_spliceai_logit"] = snv_main["spliceai_logit"] - snv_main["wt_spliceai_logit"]
snv_main["delta_exp_psi"] = snv_main["avg_psi"] - snv_main["wt_exp_psi"]
snv_main["delta_exp_logit"] = snv_main["avg_logit"] - snv_main["wt_exp_logit"]
# === Correlation plot function (with n)
def plot_corr(df, x, y, label):
    print(f"Plotting correlation: {label} ({x} vs {y})...")
    subset = df[[x, y]].dropna()
    print(f"Points in plot: {len(subset):,}")
    if len(subset) < 2:
        print("Skipping plot: not enough data")
        return
    r, _ = pearsonr(subset[x], subset[y])
    n = len(subset)
    plt.figure(figsize=(4, 4))
    sns.scatterplot(data=subset, x=x, y=y, s=8, alpha=0.2)
    plt.title(f"{label}\nr = {r:.2f}, n = {n:,}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    path = os.path.join(plot_dir, f"{label}_{x}_vs_{y}.pdf")
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")
    return r, n

# === Filter WT PSI edge cases
print("Filtering out variants with WT PSI == 0 or 1 in any cell line...")
wt_psi_cols = [
    "HeLa_wt_pooled_psi_raw", "K562_wt_pooled_psi_raw", "MCF7_wt_pooled_psi_raw",
    "HMC3_wt_pooled_psi_raw", "HEK_wt_pooled_psi_raw"
]
main_df["wt_psi_edge"] = main_df[wt_psi_cols].apply(lambda row: any(row == 0) or any(row == 1), axis=1)
non_edge_refs = set(main_df[~main_df["wt_psi_edge"]]["Reference"])
snv_filtered = snv_main[snv_main["Reference"].isin(non_edge_refs)].copy()

# === Plot configurations
plot_tasks = [
    ("spliceai_product", "avg_psi", "SpliceAI_vs_Exp_PSI"),
    ("spliceai_logit", "avg_logit", "SpliceAI_vs_Exp_Logit"),
    ("delta_spliceai_logit", "delta_exp_logit", "SpliceAI_vs_Exp_Delta_Logit"),
]

# === Run plots and record r values
corr_summary = {}

# Unfiltered
print("Running plots on ALL variants...")
for x, y, label in plot_tasks:
    r, _ = plot_corr(snv_main, x, y, label)
    corr_summary[label] = round(r, 4)

# Filtered
print("Running plots on FILTERED variants (no WT PSI == 0 or 1)...")
for x, y, label in plot_tasks:
    label_filtered = label + "_no_edge_wtpsi"
    r, _ = plot_corr(snv_filtered, x, y, label_filtered)
    corr_summary[label_filtered] = round(r, 4)

# === Save correlation summary
summary_path = os.path.join(output_dir, "spliceai_vs_exp_correlation_summary.json")
with open(summary_path, "w") as f:
    json.dump(corr_summary, f, indent=2)
print(f"Saved correlation summary: {summary_path}")
