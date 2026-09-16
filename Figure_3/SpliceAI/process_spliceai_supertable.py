import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm

CITRINE_EXON1 = "ATGGTGTCCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTCAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAACTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGATGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAAGTGAAGTTCGAGGGCGACACCCTCGTGAACCGCATCGAGCTAAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAAGTGAACTTCAAGATCCGCCACAACATCGAG"
SMN2_INTRON6 = "GTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA"
LEN_CITRINE1 = len(CITRINE_EXON1)
LEN_SMN2_5 = len(SMN2_INTRON6)

parser = argparse.ArgumentParser()
parser.add_argument("--raw_preds", required=True, help="spliceai_raw_preds_supertable.tsv")
parser.add_argument("--supertable", required=True, help="st_corrected.csv")
parser.add_argument("--data", required=True, help="1e-2_ALL_WITH_WT.csv (COMPASS data)")
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
plot_dir = os.path.join(args.output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

def logit_clip(p, eps=0.01):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

# === Load inputs ===
print("Loading supertable...")
st = pd.read_csv(args.supertable, low_memory=False)
st["snp"] = st["snp"].astype(str).str.strip()
st["intron1_len"] = st["intron1"].str.len()
st["exon_len"] = st["exon"].str.len()
st["exon_start"] = LEN_CITRINE1 + LEN_SMN2_5 + st["intron1_len"]
st["exon_end"] = st["exon_start"] + st["exon_len"] - 1
print(f"Supertable: {len(st):,} rows")

print("Loading raw SpliceAI predictions...")
preds_df = pd.read_csv(args.raw_preds, sep="\t", dtype=str)
print(f"Raw preds: {len(preds_df):,} rows")

print("Loading COMPASS data file...")
data_df = pd.read_csv(args.data, low_memory=False)
data_refs = set(data_df["Reference"].astype(str))
print(f"COMPASS data: {len(data_df):,} rows; {len(data_refs):,} unique References")

# === Parse spliceai_scores ===
print("Parsing SpliceAI scores...")
scores_by_ref = {}
for ref, score in tqdm(zip(preds_df["Reference"], preds_df["spliceai_scores"]), total=len(preds_df)):
    if isinstance(score, str) and score.startswith("["):
        try:
            scores_by_ref[str(ref)] = json.loads(score)
        except Exception as e:
            print(f"Failed to parse ref={ref}: {e}")
print(f"Parsed {len(scores_by_ref):,} predictions")

# === Merge preds onto supertable ===
st["Reference"] = st["Reference"].astype(str)
st = st[st["Reference"].isin(scores_by_ref)].copy()
st["spliceai"] = st["Reference"].map(scores_by_ref)

# === Extract splice site probabilities ===
print("Extracting splice site probabilities...")
spliceai_product = np.full(len(st), np.nan)
spliceai_logit = np.full(len(st), np.nan)
starts = st["exon_start"].astype(int).to_numpy()
ends = st["exon_end"].astype(int).to_numpy()
score_vals = st["spliceai"].to_numpy()

for i, (s, e, v) in enumerate(tqdm(zip(starts, ends, score_vals), total=len(st))):
    if not isinstance(v, list):
        continue
    try:
        if not (0 <= s < len(v)) or not (0 <= e < len(v)):
            continue
        sa = v[s][1]
        sd = v[e][2]
        prod = sa * sd
        spliceai_product[i] = prod
        spliceai_logit[i] = logit_clip(prod)
    except Exception as exc:
        print(f"Failed at row {i}, s={s}, e={e}: {exc}")

st["spliceai_product"] = spliceai_product
st["spliceai_logit"] = spliceai_logit
print(f"Valid predictions: {st['spliceai_logit'].notna().sum():,}")

# === Compute delta: variant logit - WT logit ===
print("Computing delta SpliceAI logit...")
wt_mask = st["snp"].str.lower() == "none"
wt_df = st[wt_mask].drop_duplicates(subset="event_id")
wt_logit_map = wt_df.set_index("event_id")["spliceai_logit"]
wt_product_map = wt_df.set_index("event_id")["spliceai_product"]

st["wt_spliceai_logit"] = st["event_id"].map(wt_logit_map)
st["wt_spliceai_product"] = st["event_id"].map(wt_product_map)
st["delta_spliceai_logit"] = st["spliceai_logit"] - st["wt_spliceai_logit"]
st["delta_spliceai_product"] = st["spliceai_product"] - st["wt_spliceai_product"]

# === Output 1: full supertable predictions ===
out_cols_st = [
    "Reference", "event_id", "gene_exon", "snp", "source", "seq_type",
    "spliceai_product", "spliceai_logit",
    "wt_spliceai_product", "wt_spliceai_logit",
    "delta_spliceai_product", "delta_spliceai_logit",
]
out_cols_st = [c for c in out_cols_st if c in st.columns]
out_path_st = os.path.join(args.output_dir, "spliceai_supertable_predictions.csv")
st[out_cols_st].to_csv(out_path_st, index=False)
print(f"Saved full supertable predictions: {out_path_st} ({len(st):,} rows)")

# === Output 2: data-subset predictions (COMPASS data) ===
print("Building data-subset predictions...")
data_subset = st[st["Reference"].isin(data_refs)].copy()
print(f"Data subset: {len(data_subset):,} rows")

delta_logit_cols = [
    "HeLa_delta_logit_pooled", "K562_delta_logit_pooled", "MCF7_delta_logit_pooled",
    "HMC3_delta_logit_pooled", "HEK_delta_logit_pooled"
]
data_df["Reference"] = data_df["Reference"].astype(str)
exp_cols = ["Reference"] + [c for c in delta_logit_cols if c in data_df.columns]
data_subset = data_subset.merge(data_df[exp_cols], on="Reference", how="left")

for col in delta_logit_cols:
    if col in data_subset.columns:
        data_subset[col] = pd.to_numeric(data_subset[col], errors="coerce")
present_delta_cols = [c for c in delta_logit_cols if c in data_subset.columns]
if present_delta_cols:
    data_subset["avg_delta_logit_pooled"] = data_subset[present_delta_cols].mean(axis=1)

out_cols_data = out_cols_st + [c for c in present_delta_cols + ["avg_delta_logit_pooled"] if c in data_subset.columns]
out_path_data = os.path.join(args.output_dir, "spliceai_data_predictions.csv")
data_subset[out_cols_data].to_csv(out_path_data, index=False)
print(f"Saved data predictions: {out_path_data} ({len(data_subset):,} rows)")

# === Correlation plots (variants only, data subset) ===
variant_subset = data_subset[data_subset["snp"].str.lower() != "none"].copy()

def plot_corr(df, x, y, label):
    subset = df[[x, y]].dropna()
    if len(subset) < 2:
        print(f"Skipping {label}: not enough data")
        return None, None
    r, _ = pearsonr(subset[x], subset[y])
    n = len(subset)
    print(f"{label}: r={r:.3f}, n={n:,}")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(subset[x], subset[y], s=8, alpha=0.15, color="#732B8E", linewidths=0, rasterized=True)
    ticks = np.arange(-10, 10 + 1e-9, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylim(ticks[0], ticks[-1])
    ax.set_aspect("equal", adjustable="box")
    ax.plot([ticks[0], ticks[-1]], [ticks[0], ticks[-1]], color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{label}\nr = {r:.2f}, n = {n:,}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{label}.pdf"), dpi=1200)
    plt.close()
    return r, n

corr_summary = {}

if "avg_delta_logit_pooled" in variant_subset.columns:
    r, n = plot_corr(variant_subset, "delta_spliceai_logit", "avg_delta_logit_pooled", "SpliceAI_vs_Exp_delta_logit_all")
    if r is not None:
        corr_summary["all"] = {"r": round(r, 4), "n": n}

    wt_psi_cols = [c for c in [
        "HeLa_wt_pooled_psi_raw", "K562_wt_pooled_psi_raw", "MCF7_wt_pooled_psi_raw",
        "HMC3_wt_pooled_psi_raw", "HEK_wt_pooled_psi_raw"
    ] if c in data_df.columns]
    if wt_psi_cols:
        data_df[wt_psi_cols] = data_df[wt_psi_cols].apply(pd.to_numeric, errors="coerce")
        data_df["wt_psi_edge"] = data_df[wt_psi_cols].apply(lambda row: any(row == 0) or any(row == 1), axis=1)
        non_edge_refs = set(data_df[~data_df["wt_psi_edge"]]["Reference"].astype(str))
        variant_filtered = variant_subset[variant_subset["Reference"].isin(non_edge_refs)].copy()
        r, n = plot_corr(variant_filtered, "delta_spliceai_logit", "avg_delta_logit_pooled", "SpliceAI_vs_Exp_delta_logit_no_edge_wtpsi")
        if r is not None:
            corr_summary["no_edge_wtpsi"] = {"r": round(r, 4), "n": n}

summary_path = os.path.join(args.output_dir, "spliceai_vs_exp_correlation_summary.json")
with open(summary_path, "w") as f:
    json.dump(corr_summary, f, indent=2)
print(f"Saved correlation summary: {summary_path}")
print("Done.")
