import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm
import os
import json

def logit_clip(p, eps=0.01):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

# === Constants ===
CITRINE_EXON1 = ("ATGGTGTCCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTCAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAACTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGATGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAAGTGAAGTTCGAGGGCGACACCCTCGTGAACCGCATCGAGCTAAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAAGTGAACTTCAAGATCCGCCACAACATCGAG")
SMN2_INTRON6 = ("GTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA")

LEN_CITRINE1 = len(CITRINE_EXON1)
LEN_SMN2_5 = len(SMN2_INTRON6)

# === Paths ===
main_data_path = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz"
wt_path = "/ESL/ESL_MPRA/Figure_3/SpliceAI/output/spliceai_raw_wt_preds.tsv"
snv_path = "/ESL/ESL_MPRA/Figure_3/SpliceAI/output/spliceai_raw_preds.tsv"
doubles_path = "/ESL/ESL_MPRA/Figure_3/SpliceAI/output/spliceai_raw_preds_doubles.tsv"
output_dir = "/ESL/ESL_MPRA/Figure_3/SpliceAI/output"
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# === Load ===
print("Loading main data...")
main_df = pd.read_csv(main_data_path, low_memory=False)
main_df["Reference"] = main_df["Reference"].astype(str)
print(f"Main data loaded: {len(main_df):,} rows")

print("Loading SpliceAI predictions...")
wt_df = pd.read_csv(wt_path, sep="\t", dtype=str)
snv_df = pd.read_csv(snv_path, sep="\t", dtype=str)
double_df = pd.read_csv(doubles_path, sep="\t", dtype=str)
print(f"WT: {len(wt_df):,}  SNV: {len(snv_df):,}  Doubles: {len(double_df):,}")

# === Exon coordinates ===
main_df["intron1_len"] = main_df["intron1"].str.len()
main_df["exon_len"] = main_df["exon"].str.len()
main_df["exon_start"] = LEN_CITRINE1 + LEN_SMN2_5 + main_df["intron1_len"]
main_df["exon_end"] = main_df["exon_start"] + main_df["exon_len"] - 1

# === Parse SpliceAI scores ===
def parse_scores(df):
    parsed = {}
    for ref, score in tqdm(zip(df["Reference"], df["spliceai_scores"]), total=len(df), desc="  parsing"):
        if isinstance(score, str) and score.startswith("["):
            try:
                parsed[ref] = json.loads(score)
            except Exception as e:
                print(f"Failed to parse {ref}: {e}")
    return parsed

print("Parsing WT scores...")
wt_scores = parse_scores(wt_df)
print(f"  WT parsed: {len(wt_scores):,}")
print("Parsing SNV scores...")
snv_scores = parse_scores(snv_df)
print(f"  SNV parsed: {len(snv_scores):,}")
print("Parsing doubles scores...")
double_scores = parse_scores(double_df)
print(f"  Doubles parsed: {len(double_scores):,}")

# === Extract splice site probabilities ===
def extract_predictions(df):
    prod_result = np.full(len(df), np.nan)
    logit_result = np.full(len(df), np.nan)
    starts = df["exon_start"].astype(int).to_numpy()
    ends = df["exon_end"].astype(int).to_numpy()
    spliceai = df["spliceai"].to_numpy()
    index_lut = dict(zip(df.index.to_numpy(), range(len(df))))
    for i, s, e, v in tqdm(zip(df.index, starts, ends, spliceai), total=len(df), desc="  extracting"):
        if not isinstance(v, list):
            continue
        try:
            if not (0 <= s < len(v)) or not (0 <= e < len(v)):
                continue
            sa = v[s][1]
            sd = v[e][2]
            prod_result[index_lut[i]] = sa * sd
            logit_result[index_lut[i]] = logit_clip(sa * sd)
        except Exception as exc:
            print(f"Failed at index {i}, s={s}, e={e}: {exc}")
    return prod_result, logit_result

# === WT predictions ===
print("Extracting WT splice site predictions...")
wt_main = main_df[(main_df["snp"] == "none") & main_df["Reference"].isin(wt_scores)].copy()
wt_main["spliceai"] = wt_main["Reference"].map(wt_scores)
wt_main["spliceai_product"], wt_main["spliceai_logit"] = extract_predictions(wt_main)
wt_spliceai = wt_main.set_index("event_id")[["spliceai_product", "spliceai_logit"]]
print(f"  WT predictions done: {wt_main['spliceai_logit'].notna().sum():,} valid")

# === Variant predictions ===
print("Extracting variant splice site predictions...")
variant_df = main_df[main_df["snp"] != "none"].copy()
variant_df = variant_df[
    variant_df["Reference"].isin(snv_scores) | variant_df["Reference"].isin(double_scores)
].copy()
print(f"  Variants to process: {len(variant_df):,}")
variant_df["spliceai"] = variant_df["Reference"].map({**snv_scores, **double_scores})
variant_df["spliceai_product"], variant_df["spliceai_logit"] = extract_predictions(variant_df)
print(f"  Variant predictions done: {variant_df['spliceai_logit'].notna().sum():,} valid")

# === Delta SpliceAI ===
print("Computing delta SpliceAI logit...")
variant_df["wt_spliceai_product"] = variant_df["event_id"].map(wt_spliceai["spliceai_product"])
variant_df["wt_spliceai_logit"] = variant_df["event_id"].map(wt_spliceai["spliceai_logit"])
variant_df["delta_spliceai_product"] = variant_df["spliceai_product"] - variant_df["wt_spliceai_product"]
variant_df["delta_spliceai_logit"] = variant_df["spliceai_logit"] - variant_df["wt_spliceai_logit"]

# === Average experimental delta logit across cell lines ===
delta_logit_cols = [
    "HeLa_delta_logit_pooled", "K562_delta_logit_pooled", "MCF7_delta_logit_pooled",
    "HMC3_delta_logit_pooled", "HEK_delta_logit_pooled"
]
for col in delta_logit_cols:
    variant_df[col] = pd.to_numeric(variant_df[col], errors="coerce")
variant_df["avg_delta_logit_pooled"] = variant_df[delta_logit_cols].mean(axis=1)

# === Save merged predictions CSV ===
out_cols = (
    ["Reference", "event_id", "gene_exon", "snp", "source", "seq_type",
     "spliceai_product", "spliceai_logit",
     "wt_spliceai_product", "wt_spliceai_logit",
     "delta_spliceai_product", "delta_spliceai_logit"]
    + delta_logit_cols
    + ["avg_delta_logit_pooled"]
)
out_cols = [c for c in out_cols if c in variant_df.columns]
out_path = os.path.join(output_dir, "spliceai_merged_predictions_04_09_2026.csv")
variant_df[out_cols].to_csv(out_path, index=False)
print(f"Saved merged predictions: {out_path} ({len(variant_df):,} rows)")

col = variant_df["delta_spliceai_logit"].dropna()
print(f"delta_spliceai_logit range: min={col.min():.3f}, max={col.max():.3f}  (n={len(col):,})")

# === WT PSI edge filter ===
wt_psi_cols = [
    "HeLa_wt_pooled_psi_raw", "K562_wt_pooled_psi_raw", "MCF7_wt_pooled_psi_raw",
    "HMC3_wt_pooled_psi_raw", "HEK_wt_pooled_psi_raw"
]
main_df[wt_psi_cols] = main_df[wt_psi_cols].apply(pd.to_numeric, errors="coerce")
main_df["wt_psi_edge"] = main_df[wt_psi_cols].apply(lambda row: any(row == 0) or any(row == 1), axis=1)
non_edge_refs = set(main_df[~main_df["wt_psi_edge"]]["Reference"])
variant_filtered = variant_df[variant_df["Reference"].isin(non_edge_refs)].copy()

# === Correlation plots ===
def plot_corr(df, x, y, label):
    subset = df[[x, y]].dropna()
    if len(subset) < 2:
        print(f"Skipping {label}: not enough data")
        return None, None
    r, _ = pearsonr(subset[x], subset[y])
    n = len(subset)
    print(f"{label}: r={r:.3f}, n={n:,}")

    fig, ax = plt.subplots(figsize=(4, 4))
    sc = ax.scatter(subset[x], subset[y], s=8, alpha=0.15, color="#732B8E", linewidths=0, rasterized=True)

    ticks = np.arange(-10, 10 + 1e-9, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylim(ticks[0], ticks[-1])
    ax.set_aspect("equal", adjustable="box")

    lo, hi = max(ticks[0], ticks[0]), min(ticks[-1], ticks[-1])
    ax.plot([ticks[0], ticks[-1]], [ticks[0], ticks[-1]], color="black", linestyle="--", linewidth=1)

    ax.set_title(f"{label}\nr = {r:.2f}, n = {n:,}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()

    path = os.path.join(plot_dir, f"{label}.pdf")
    plt.savefig(path, dpi=1200)
    plt.close()
    return r, n

plot_tasks = [
    ("delta_spliceai_logit", "avg_delta_logit_pooled", "SpliceAI_vs_Exp_delta_logit_all"),
    ("delta_spliceai_logit", "avg_delta_logit_pooled", "SpliceAI_vs_Exp_delta_logit_no_edge"),
]

corr_summary = {}

r, n = plot_corr(variant_df, "delta_spliceai_logit", "avg_delta_logit_pooled", "SpliceAI_vs_Exp_delta_logit_all")
if r is not None:
    corr_summary["all"] = {"r": round(r, 4), "n": n}

r, n = plot_corr(variant_filtered, "delta_spliceai_logit", "avg_delta_logit_pooled", "SpliceAI_vs_Exp_delta_logit_no_edge_wtpsi")
if r is not None:
    corr_summary["no_edge_wtpsi"] = {"r": round(r, 4), "n": n}

summary_path = os.path.join(output_dir, "spliceai_vs_exp_correlation_summary.json")
with open(summary_path, "w") as f:
    json.dump(corr_summary, f, indent=2)
print(f"Saved correlation summary: {summary_path}")
