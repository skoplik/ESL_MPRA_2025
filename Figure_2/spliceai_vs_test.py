#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# === Paths ===
spliceai_file = "/ESL/Figures_SK/Spliceai/all/spliceai_psi_logit.tsv"
mmsplice_test_file = "/ESL/ESL_MPRA/Figure_2/model_output_csv/retrained_mmsplice_model_predictions_test_dataset7_model10.csv"
supertable_file = "/ESL/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WITH_WT.csv"

outdir = "/ESL/Figures_SK/Spliceai/redo_07_23_25/spliceai_vs_exp_refs"
os.makedirs(outdir, exist_ok=True)

# === Load files ===
spliceai_df = pd.read_csv(spliceai_file, sep="\t")
mmsplice_df = pd.read_csv(mmsplice_test_file)
supertable_df = pd.read_csv(supertable_file)

print(f"Loaded SpliceAI: {len(spliceai_df):,} rows")
print(f"Loaded MMSplice: {len(mmsplice_df):,} rows")
print(f"Loaded Supertable: {len(supertable_df):,} rows")

# === Ensure IDs ===
spliceai_df["Reference"] = spliceai_df["Reference"].astype(int) + 1  # shift 0→1 idx
spliceai_df["Reference"] = spliceai_df["Reference"].astype(str)

mmsplice_df["Reference_ID"] = mmsplice_df["Reference_ID"].astype(str)
mmsplice_df["Variant_ID"] = mmsplice_df["Variant_ID"].astype(str)

supertable_df["Reference"] = supertable_df["Reference"].astype(str)
supertable_df["event_id"] = supertable_df["event_id"].astype(str)

# === Δlogit for SpliceAI ===
def compute_dlogit(group):
    if "none" not in group["snp"].values:
        return pd.DataFrame()
    wt_logit = group.loc[group["snp"] == "none", "spliceai_logit"].iloc[0]
    group = group.copy()
    group["delta_spliceai_logit"] = group["spliceai_logit"] - wt_logit
    return group

spliceai_df = spliceai_df.groupby("event_id", group_keys=False).apply(compute_dlogit)

# === Average experimental Δlogit pooled (across cell lines) ===
logit_cols = [c for c in supertable_df.columns if c.endswith("delta_logit_pooled")]
supertable_df[logit_cols] = supertable_df[logit_cols].apply(pd.to_numeric, errors="coerce")
supertable_df["avg_delta_logit_pooled"] = supertable_df[logit_cols].mean(axis=1)

# === Filter out WT PSI edge cases (any wt pooled psi == 0 or 1) ===
psi_cols = [
    "HeLa_wt_pooled_psi_raw",
    "K562_wt_pooled_psi_raw",
    "MCF7_wt_pooled_psi_raw",
    "HMC3_wt_pooled_psi_raw",
    "HEK_wt_pooled_psi_raw"
]
supertable_df["wt_psi_edge"] = supertable_df[psi_cols].apply(
    lambda row: any(row == 0) or any(row == 1), axis=1
)
non_edge_refs = set(supertable_df.loc[~supertable_df["wt_psi_edge"], "Reference"])
supertable_filtered = supertable_df[supertable_df["Reference"].isin(non_edge_refs)].copy()
print(f"Supertable filtered (non-edge WTs): {len(supertable_filtered):,}")

# === Add back MMSplice refs dropped by edge filter ===
all_refs_before = set(supertable_df["Reference"])
refs_after = set(supertable_filtered["Reference"])
dropped_refs = all_refs_before - refs_after

mm_refs = set(mmsplice_df["Reference_ID"])
mm_dropped = mm_refs & dropped_refs
print(f"MMSplice Reference_IDs dropped by edge filter: {len(mm_dropped):,}")
if mm_dropped:
    add_back = supertable_df[supertable_df["Reference"].isin(mm_dropped)]
    supertable_filtered = pd.concat([supertable_filtered, add_back], ignore_index=True)
    print(f"Supertable after adding back MMSplice refs: {len(supertable_filtered):,}")

# === Restrict Supertable to MMSplice event_ids ===
ref_to_event = supertable_filtered.set_index("Reference")["event_id"].to_dict()
mm_event_ids = mmsplice_df["Reference_ID"].map(ref_to_event).dropna().unique().tolist()
supertable_subset = supertable_filtered[supertable_filtered["event_id"].isin(mm_event_ids)].copy()
print(f"Sequences in filtered Supertable for MMSplice event_ids: {len(supertable_subset):,}")

# === Merge SpliceAI with filtered Supertable subset ===
merged_subset = spliceai_df.merge(
    supertable_subset[["Reference", "event_id", "snp", "intron1", "exon", "intron2", "avg_delta_logit_pooled"]],
    on=["Reference", "event_id", "snp"],
    how="inner"
)
print(f"Merged SpliceAI + filtered Supertable (MMSplice event_ids): {len(merged_subset):,}")

# === Distance-to-splice-site filter (≤100 nt, supports multiple SNPs) ===
def snp_within_100(row):
    snp = row["snp"]
    if snp == "none" or pd.isna(snp):
        return False
    try:
        intron1_len = len(str(row["intron1"])) if pd.notnull(row["intron1"]) else 0
        exon_len = len(str(row["exon"])) if pd.notnull(row["exon"]) else 0
        exon_start = intron1_len
        exon_end = intron1_len + exon_len - 1

        for e in str(snp).split(";"):
            if ":" not in e:
                continue
            pos_str, change = e.split(":")
            pos = int(pos_str)
            dist = min(abs(pos - exon_start), abs(pos - exon_end))
            if dist <= 100:
                return True
        return False
    except Exception:
        return False

merged_dist100 = merged_subset[merged_subset.apply(snp_within_100, axis=1)].copy()
print(f"After distance filter (≤100 nt, MMSplice subset): {len(merged_dist100):,} variants")

# === Correlation plot ===
def plot_corr(df, label, outname):
    subset = df[["delta_spliceai_logit", "avg_delta_logit_pooled"]].dropna()
    print(f"Points in {label}: {len(subset):,}")
    if len(subset) > 1:
        r, _ = pearsonr(subset["delta_spliceai_logit"], subset["avg_delta_logit_pooled"])
        n = len(subset)
        print(f"{label} Δlogit correlation: r = {r:.2f}, n = {n}")

        plt.figure(figsize=(4, 4))
        ax = sns.scatterplot(
            data=subset,
            x="delta_spliceai_logit",
            y="avg_delta_logit_pooled",
            s=8,
            alpha=0.15,
            color="#732B8E",
            edgecolor="none"
        )
        if ax.collections:
            ax.collections[0].set_rasterized(True)

        lo = min(subset.min().min(), -5)
        hi = max(subset.max().max(), 5)
        ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", linewidth=1)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        ax.set_title(f"{label}\nr = {r:.2f}, n = {n:,}")
        ax.set_xlabel("SpliceAI Δlogit(PSI)")
        ax.set_ylabel("Experimental Δlogit(PSI)")

        outpath = os.path.join(outdir, outname)
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.close()
        print(f"Saved plot: {outpath}")

# === Plot MMSplice subset ===
plot_corr(merged_dist100, "SpliceAI vs Experimental (MMSplice subset, ≤100 nt)", "spliceai_vs_exp_dlogit_mmsplice_dist100.pdf")

# === Now do full filtered supertable (non-edge WTs, ≤100 nt) ===
supertable_all_filtered = supertable_filtered.copy()
merged_all = spliceai_df.merge(
    supertable_all_filtered[["Reference", "event_id", "snp", "intron1", "exon", "intron2", "avg_delta_logit_pooled"]],
    on=["Reference", "event_id", "snp"],
    how="inner"
)
merged_all_dist100 = merged_all[merged_all.apply(snp_within_100, axis=1)].copy()
print(f"Merged SpliceAI + filtered Supertable (all, ≤100 nt): {len(merged_all_dist100):,}")

plot_corr(merged_all_dist100, "SpliceAI vs Experimental (all variants, ≤100 nt)", "spliceai_vs_exp_dlogit_all_dist100.pdf")
