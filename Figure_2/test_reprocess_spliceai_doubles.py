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
doubles_path = "/ESL/Figures_SK/Spliceai/redo_07_23_25/spliceai_raw_preds_doubles.tsv"
output_dir = "/ESL/Figures_SK/Spliceai/redo_07_23_25/final_07_27_25_with_doubles"
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# === Load
print("Loading main data...")
main_df = pd.read_csv(main_data_path)
main_df["Reference"] = main_df["Reference"].astype(str)
print(f"Main data loaded: {len(main_df):,} rows")

print("Loading WT spliceai predictions...")
wt_df = pd.read_csv(wt_path, sep="\t", dtype=str)
print(f"WT predictions loaded: {len(wt_df):,} rows")

print("Loading SNV spliceai predictions...")
snv_df = pd.read_csv(snv_path, sep="\t", dtype=str)
print(f"SNV predictions loaded: {len(snv_df):,} rows")

print("Loading double mutant spliceai predictions...")
double_df = pd.read_csv(doubles_path, sep="\t", dtype=str)
print(f"Double predictions loaded: {len(double_df):,} rows")

# === Coordinates
print("Computing exon coordinates...")
main_df["intron1_len"] = main_df["intron1"].str.len()
main_df["exon_len"] = main_df["exon"].str.len()
main_df["exon_start"] = LEN_CITRINE1 + LEN_SMN2_5 + main_df["intron1_len"]
main_df["exon_end"] = main_df["exon_start"] + main_df["exon_len"] - 1

# === Parse SpliceAI scores
def parse_scores(df):
    parsed = {}
    for idx, (ref, score) in enumerate(zip(df["Reference"], df["spliceai_scores"])):
        if isinstance(score, str) and score.startswith("["):
            try:
                parsed[ref] = json.loads(score)
            except Exception as e:
                print(f"Failed to parse {ref}: {e}")
    return parsed

wt_scores = parse_scores(wt_df)
snv_scores = parse_scores(snv_df)
double_scores = parse_scores(double_df)

# === Merge variant predictions
variant_df = main_df[(main_df["snp"] != "none")].copy()
variant_df = variant_df[variant_df["Reference"].isin(snv_scores) | variant_df["Reference"].isin(double_scores)].copy()
variant_df["spliceai"] = variant_df["Reference"].map({**snv_scores, **double_scores})

# === Compute predictions
def extract_predictions(df):
    prod_result = np.full(len(df), np.nan)
    logit_sum_result = np.full(len(df), np.nan)
    starts = df["exon_start"].astype(int).to_numpy()
    ends = df["exon_end"].astype(int).to_numpy()
    spliceai = df["spliceai"].to_numpy()
    index_lut = dict(zip(df.index.to_numpy(), range(len(df))))
    for count, (i, s, e, v) in enumerate(zip(df.index, starts, ends, spliceai)):
        if not isinstance(v, list): continue
        try:
            if not (0 <= s < len(v)) or not (0 <= e < len(v)): continue
            sa = v[s][1]
            sd = v[e][2]
            prod_result[index_lut[i]] = sa * sd
            logit_sum_result[index_lut[i]] = logit_clip(sa*sd)
        except Exception as e:
            print(f"Failed at index {i}, s={s}, e={e}: {e}")
            continue
    return prod_result, logit_sum_result

variant_df["spliceai_product"], variant_df["spliceai_logit"] = extract_predictions(variant_df)

# === Compute experimental values
psi_cols = ["HeLa_wt_pooled_psi_raw", "K562_wt_pooled_psi_raw", "MCF7_wt_pooled_psi_raw", "HMC3_wt_pooled_psi_raw", "HEK_wt_pooled_psi_raw"]
logit_cols = ["HeLa_pooled_logit", "K562_pooled_logit", "MCF7_pooled_logit", "HMC3_pooled_logit", "HEK_pooled_logit"]
main_df[psi_cols + logit_cols] = main_df[psi_cols + logit_cols].apply(pd.to_numeric, errors="coerce")
main_df["avg_psi"] = main_df[psi_cols].mean(axis=1)
main_df["avg_logit"] = main_df[logit_cols].mean(axis=1)

ref_to_psi = main_df.set_index("Reference")["avg_psi"].to_dict()
ref_to_logit = main_df.set_index("Reference")["avg_logit"].to_dict()

variant_df["avg_psi"] = variant_df["Reference"].map(ref_to_psi)
variant_df["avg_logit"] = variant_df["Reference"].map(ref_to_logit)

# === Get WT values
wt_main = main_df[(main_df["snp"] == "none") & main_df["Reference"].isin(wt_scores)].copy()
wt_main["spliceai"] = wt_main["Reference"].map(wt_scores)
wt_main["spliceai_product"], wt_main["spliceai_logit"] = extract_predictions(wt_main)
wt_main["avg_psi"] = wt_main[psi_cols].mean(axis=1)
wt_main["avg_logit"] = wt_main[logit_cols].mean(axis=1)

wt_spliceai = wt_main.set_index("event_id")[["spliceai_product", "spliceai_logit"]]
wt_exp = wt_main.set_index("event_id")[["avg_psi", "avg_logit"]]

variant_df["wt_spliceai_product"] = variant_df["event_id"].map(wt_spliceai["spliceai_product"])
variant_df["wt_spliceai_logit"] = variant_df["event_id"].map(wt_spliceai["spliceai_logit"])
variant_df["wt_exp_psi"] = variant_df["event_id"].map(wt_exp["avg_psi"])
variant_df["wt_exp_logit"] = variant_df["event_id"].map(wt_exp["avg_logit"])

variant_df["delta_spliceai_product"] = variant_df["spliceai_product"] - variant_df["wt_spliceai_product"]
variant_df["delta_spliceai_logit"] = variant_df["spliceai_logit"] - variant_df["wt_spliceai_logit"]
variant_df["delta_exp_psi"] = variant_df["avg_psi"] - variant_df["wt_exp_psi"]
variant_df["delta_exp_logit"] = variant_df["avg_logit"] - variant_df["wt_exp_logit"]

# === Filter WT PSI edge
main_df["wt_psi_edge"] = main_df[psi_cols].apply(lambda row: any(row == 0) or any(row == 1), axis=1)
non_edge_refs = set(main_df[~main_df["wt_psi_edge"]]["Reference"])
variant_filtered = variant_df[variant_df["Reference"].isin(non_edge_refs)].copy()

# === Correlation plotting
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
    ax = sns.scatterplot(
        data=subset,
        x=x,
        y=y,
        s=8,
        alpha=0.15,
        color='#732B8E',
        edgecolor='none'
    )

    if ax.collections:
        ax.collections[0].set_rasterized(True)

    ticks = np.arange(-10, 10 + 1e-9, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylim(ticks[0], ticks[-1])

    ax.set_aspect('equal', adjustable='box')

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lo = max(min(x0, x1), min(y0, y1))
    hi = min(max(x0, x1), max(y0, y1))
    ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', linewidth=1)

    ax.set_title(f"{label}\nr = {r:.2f}, n = {n:,}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()

    path = os.path.join(plot_dir, f"{label}_{x}_vs_{y}.pdf")
    plt.savefig(path, dpi=1200)
    plt.close()
    print(f"Saved plot: {path}")
    return r, n

plot_tasks = [
    ("spliceai_product", "avg_psi", "SpliceAI_vs_Exp_PSI"),
    ("spliceai_logit", "avg_logit", "SpliceAI_vs_Exp_Logit"),
    ("delta_spliceai_logit", "delta_exp_logit", "SpliceAI_vs_Exp_Delta_Logit"),
]

corr_summary = {}

print("Running plots on ALL variants (SNVs + Doubles)...")
for x, y, label in plot_tasks:
    r, _ = plot_corr(variant_df, x, y, label)
    corr_summary[label] = round(r, 4)

print("Running plots on FILTERED variants (no WT PSI == 0 or 1)...")
for x, y, label in plot_tasks:
    label_filtered = label + "_no_edge_wtpsi"
    r, _ = plot_corr(variant_filtered, x, y, label_filtered)
    corr_summary[label_filtered] = round(r, 4)

summary_path = os.path.join(output_dir, "spliceai_vs_exp_correlation_summary.json")
with open(summary_path, "w") as f:
    json.dump(corr_summary, f, indent=2)
print(f"Saved correlation summary: {summary_path}")
