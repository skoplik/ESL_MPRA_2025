import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.special import expit
import os

# === Paths ===
mmsplice_pred_path = "/ESL/Figures_SK/mmsplice2/output_all_preds_with_doubles/mmsplice_predictions_with_ref_idx.csv"
supertable_path = "/ESL/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WITH_WT.csv"
output_dir = "/ESL/Figures_SK/mmsplice2/output_all_preds_with_doubles/toss_0_1_refs/plots"
output_csv = os.path.join(output_dir, "mmsplice_vs_supertable_vcfid_filtered.csv")
os.makedirs(output_dir, exist_ok=True)

# === Constants ===
CELL_LINES = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]
WT_PSI_FIELDS = [f"{cl}_wt_pooled_psi_raw" for cl in CELL_LINES]
WT_LOGIT_FIELDS = [f"{cl}_wt_pooled_logit" for cl in CELL_LINES]
VARIANT_LOGIT_FIELDS = [f"{cl}_pooled_logit" for cl in CELL_LINES]
VARIANT_PSI_FIELDS = [f"{cl}_pooled_psi_raw" for cl in CELL_LINES]

# === Load data ===
mm = pd.read_csv(mmsplice_pred_path)
print(f"Loaded MMSplice predictions: {len(mm)} rows")

sup = pd.read_csv(supertable_path)
print(f"Loaded Supertable: {len(sup)} rows")

mm["Reference"] = mm["Reference"].astype(str)
sup["Reference"] = sup["Reference"].astype(str)

# === Cast relevant columns
for col in WT_PSI_FIELDS + WT_LOGIT_FIELDS + VARIANT_PSI_FIELDS + VARIANT_LOGIT_FIELDS:
    sup[col] = pd.to_numeric(sup[col], errors="coerce")

# === Compute Δlogit per cell line
for cl in CELL_LINES:
    sup[f"{cl}_delta_logit_pooled"] = sup[f"{cl}_pooled_logit"] - sup[f"{cl}_wt_pooled_logit"]

# === Compute average values
sup["wt_avg_psi"] = sup[WT_PSI_FIELDS].mean(axis=1)
sup["wt_avg_logit"] = sup[WT_LOGIT_FIELDS].mean(axis=1)
sup["mean_delta_logit_pooled"] = sup[[f"{cl}_delta_logit_pooled" for cl in CELL_LINES]].mean(axis=1)

# === Merge
merged = mm.merge(sup, on="Reference", how="inner")
print(f"Merged entries: {len(merged)}")


# === Check which Supertable rows were not matched in the merge
sup_refs = set(sup["Reference"])
merged_refs = set(merged["Reference"])
missing_from_merged = sup_refs - merged_refs

if missing_from_merged:
    missing_sup = sup[sup["Reference"].isin(missing_from_merged)].copy()
    print(f"Supertable entries missing from merged result: {len(missing_sup)}")
    print("Breakdown by 'snp' for missing entries:")
    print(missing_sup["snp"].value_counts())

    # Only consider variant rows (exclude WTs)
    missing_vars = missing_sup[missing_sup["snp"] != "none"].copy()

    # Lookup WT per event_id
    wt_lookup = sup[sup["snp"] == "none"][["event_id", "Reference"]].drop_duplicates()
    missing_vars = missing_vars.merge(wt_lookup, on="event_id", how="left", suffixes=("", "_wt"))
    missing_vars["has_wt_in_sup"] = missing_vars["Reference_wt"].notnull()

    # Classify SNVs vs DNVs
    is_double = missing_vars["snp"].str.count(";").fillna(0).gt(0)
    is_snv = ~is_double

    snv_missing_wt = missing_vars[is_snv & (~missing_vars["has_wt_in_sup"])]
    dnv_missing_wt = missing_vars[is_double & (~missing_vars["has_wt_in_sup"])]

    snv_with_wt = missing_vars[is_snv & (missing_vars["has_wt_in_sup"])]
    dnv_with_wt = missing_vars[is_double & (missing_vars["has_wt_in_sup"])]

    print(f"Missing SNVs with no WT in Supertable: {len(snv_missing_wt)}")
    print(f"Missing double mutants with no WT in Supertable: {len(dnv_missing_wt)}")
    print(f"Missing SNVs that HAVE WT in Supertable: {len(snv_with_wt)}")
    print(f"Missing double mutants that HAVE WT in Supertable: {len(dnv_with_wt)}")

else:
    print("All Supertable entries are present in merged result.")
    
# === Filter out entries with WT PSI == 0 or 1 in any cell line
merged = merged[~((merged[WT_PSI_FIELDS] == 0.0).any(axis=1) | (merged[WT_PSI_FIELDS] == 1.0).any(axis=1))]
print(f"Filtered merged entries: {len(merged)}")

# === Compute predicted PSI/logit and experimental ΔPSI
merged["delta_logit_psi"] = pd.to_numeric(merged["delta_logit_psi"], errors="coerce")
merged["pred_variant_logit"] = merged["wt_avg_logit"] + merged["delta_logit_psi"]
merged["predicted_variant_psi"] = expit(merged["pred_variant_logit"])
merged["avg_variant_psi"] = merged[VARIANT_PSI_FIELDS].mean(axis=1)
merged["experimental_delta_psi"] = merged["avg_variant_psi"] - merged["wt_avg_psi"]
merged["predicted_delta_psi"] = merged["predicted_variant_psi"] - merged["wt_avg_psi"]

# === Rename for clarity
merged["MMsplice Ref ID"] = merged["Reference"]
merged["MMSplice VCF Var"] = merged["snp"]

# === Output CSV columns
meta_cols = [
    "Reference", "event_id", "gene_exon", "snp", "source", "seq_type",
    "intron1", "exon", "intron2", "full_seq"
]

export_cols = (
    ["MMsplice Ref ID", "MMSplice VCF Var", "region", "delta_logit_psi",
     "mean_delta_logit_pooled", "wt_avg_psi", "wt_avg_logit",
     "predicted_variant_psi", "avg_variant_psi",
     "predicted_delta_psi", "experimental_delta_psi"] +
    [f"{cl}_delta_logit_pooled" for cl in CELL_LINES] +
    meta_cols
)

merged[export_cols].drop_duplicates().to_csv(output_csv, index=False)
print(f"Saved merged CSV to: {output_csv}")

# === Define subsets
region_subsets = {
    "all_regions": merged,
    "intronic_only": merged[merged["region"].str.contains("intron", case=False, na=False)],
    "exonic_only": merged[merged["region"].str.contains("exon", case=False, na=False)],
    "donor_only": merged[merged["region"].isin(["donor", "donor_dinu"])],
    "acceptor_only": merged[merged["region"].isin(["acceptor", "acceptor_dinu"])],
    "donor_acceptor_only": merged[merged["region"].isin(["donor", "donor_dinu", "acceptor", "acceptor_dinu"])],
    "dinu_only": merged[merged["region"].isin(["donor_dinu", "acceptor_dinu"])]
}

# === Plotting
def plot_corr(df, tag):
    df = df.dropna(subset=["delta_logit_psi", "mean_delta_logit_pooled"])
    n = len(df)
    if n == 0:
        print(f" No data for plot: {tag}")
        return

    print(f"\nCorrelation stats for: {tag} (n = {n})")
    for cl in CELL_LINES:
        col = f"{cl}_delta_logit_pooled"
        valid = df[["delta_logit_psi", col]].dropna()
        if len(valid) >= 2:
            pr, pp = pearsonr(valid["delta_logit_psi"], valid[col])
            sr, sp = spearmanr(valid["delta_logit_psi"], valid[col])
            print(f"{cl:5s}: Pearson r = {pr:.3f}, p = {pp:.2e} | Spearman r = {sr:.3f}, p = {sp:.2e}")

    if n >= 2:
        pear_mean, _ = pearsonr(df["delta_logit_psi"], df["mean_delta_logit_pooled"])
        spear_mean, _ = spearmanr(df["delta_logit_psi"], df["mean_delta_logit_pooled"])
        print(f"{'mean':5s}: Pearson r = {pear_mean:.3f} | Spearman r = {spear_mean:.3f}")

        plt.figure(figsize=(5, 5))
        plt.scatter(df["delta_logit_psi"], df["mean_delta_logit_pooled"], alpha=0.1, s=8, c='black')
        # plt.axhline(0, ls="--", color="black")
        # plt.axvline(0, ls="--", color="black")
        min_val = min(df["delta_logit_psi"].min(), df["mean_delta_logit_pooled"].min())
        max_val = max(df["delta_logit_psi"].max(), df["mean_delta_logit_pooled"].max())
        plt.plot([min_val, max_val], [min_val, max_val], ls="--", color="green", linewidth=2)
        plt.xlabel("MMSplice Δlogit(ψ)")
        plt.ylabel("Supertable avg Δlogit")
        plt.title(f"{tag} – Δlogit\nn={n} | r={pear_mean:.2f}, ρ={spear_mean:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"mmsplice_vs_supertable_{tag}.pdf"))
        plt.close()

        psi_df = df.dropna(subset=["predicted_variant_psi", "avg_variant_psi"])
        if len(psi_df) >= 2:
            pr_psi, _ = pearsonr(psi_df["predicted_variant_psi"], psi_df["avg_variant_psi"])
            sr_psi, _ = spearmanr(psi_df["predicted_variant_psi"], psi_df["avg_variant_psi"])
            print(f"PSI corr: Pearson r = {pr_psi:.3f}, Spearman r = {sr_psi:.3f}")

            plt.figure(figsize=(5, 5))
            plt.scatter(psi_df["predicted_variant_psi"], psi_df["avg_variant_psi"], alpha=0.2, s=8)
            plt.plot([0, 1], [0, 1], ls="--", color="gray")
            plt.xlabel("Predicted PSI (MMSplice)")
            plt.ylabel("Observed PSI")
            plt.title(f"{tag} – PSI\nr={pr_psi:.2f}, ρ={sr_psi:.2f}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"psi_corr_{tag}.pdf"))
            plt.close()

        dpsi_df = df.dropna(subset=["predicted_delta_psi", "experimental_delta_psi"])
        if len(dpsi_df) >= 2:
            pr_dpsi, _ = pearsonr(dpsi_df["predicted_delta_psi"], dpsi_df["experimental_delta_psi"])
            sr_dpsi, _ = spearmanr(dpsi_df["predicted_delta_psi"], dpsi_df["experimental_delta_psi"])
            print(f"ΔPSI corr: Pearson r = {pr_dpsi:.3f}, Spearman r = {sr_dpsi:.3f}")

            plt.figure(figsize=(5, 5))
            plt.scatter(dpsi_df["predicted_delta_psi"], dpsi_df["experimental_delta_psi"], alpha=0.2, s=8)
            plt.axhline(0, ls="--", color="gray")
            plt.axvline(0, ls="--", color="gray")
            plt.xlabel("Predicted ΔPSI (MMSplice)")
            plt.ylabel("Experimental ΔPSI")
            plt.title(f"{tag} – ΔPSI\nr={pr_dpsi:.2f}, ρ={sr_dpsi:.2f}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"dpsi_corr_{tag}.pdf"))
            plt.close()

# === Generate plots
for tag, df in region_subsets.items():
    plot_corr(df, tag)
