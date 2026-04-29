import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.special import logit
import os

# === Input paths ===
hal_pred_path = "/ESL/ESL_MPRA/Figure_3/model_processing/outputs/hal/hal_predictions.tsv"
supertable_path = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/ALL_WITH_WT.csv"
output_base = "/ESL/ESL_MPRA/Figure_3/model_processing/outputs/hal/plots/"

# === Load HAL predictions ===
pred_df = pd.read_csv(hal_pred_path, sep="\t", header=0)
pred_df.rename(columns={
    "WT_SEQ": "WILDTYPE_SEQ",
    "MUT_SEQ": "VARIANT_SEQ",
    "WT_PSI": "WILDTYPE_PSI",
    "MUT_PSI": "VARIANT_PSI",
    "DELTA_PSI": "DELTA_PSI"
}, inplace=True)

pred_df["WILDTYPE_PSI"] = pd.to_numeric(pred_df["WILDTYPE_PSI"], errors="coerce") / 100
pred_df["VARIANT_PSI"] = pd.to_numeric(pred_df["VARIANT_PSI"], errors="coerce") / 100
pred_df["DELTA_PSI"] = pd.to_numeric(pred_df["DELTA_PSI"], errors="coerce") / 100
pred_df["Reference"] = pred_df["VARIANT_NAME"].str.replace("Seq_", "").astype(int)

# === Load Supertable ===
sup = pd.read_csv(supertable_path)

# === Compute average experimental dPSI and delta logit ===
dpsi_cols = [c for c in sup.columns if c.endswith("_dpsi_pooled")]
dlogit_cols = [c for c in sup.columns if c.endswith("_delta_logit_pooled")]
wt_cols = [c for c in sup.columns if c.endswith("_wt_pooled_psi_raw")]

sup["exp_dpsi_avg"] = sup[dpsi_cols].mean(axis=1, skipna=True)
sup["exp_dlogit_avg"] = sup[dlogit_cols].mean(axis=1, skipna=True)

# === Create unfiltered merged dataset ===
merged_unfiltered = pd.merge(pred_df, sup[["Reference", "exp_dpsi_avg", "exp_dlogit_avg"]], on="Reference", how="inner")

# === Filter: remove rows where any wt PSI is 0 or 1 ===
wt_mask = ~((sup[wt_cols] == 0.0).any(axis=1) | (sup[wt_cols] == 1.0).any(axis=1))
valid_refs = set(sup[wt_mask]["Reference"])
merged_filtered = merged_unfiltered[merged_unfiltered["Reference"].isin(valid_refs)]

# === Compute HAL delta logit ===
def clip_psi(x, eps=1e-2):
    return x.clip(lower=eps, upper=1 - eps)

def compute_predicted_logit(df):
    df = df.copy()
    df["hal_logit_wt"] = logit(clip_psi(df["WILDTYPE_PSI"]))
    df["hal_logit_var"] = logit(clip_psi(df["VARIANT_PSI"]))
    df["hal_delta_logit"] = df["hal_logit_var"] - df["hal_logit_wt"]
    return df

merged_unfiltered = compute_predicted_logit(merged_unfiltered)
merged_filtered = compute_predicted_logit(merged_filtered)

# === Plotting function ===
def make_plot(df, xcol, ycol, xlabel, ylabel, label, out_path):
    x = df[xcol]
    y = df[ycol]
    r, pval = pearsonr(x, y)
    print(f"{label}: pearson r = {r:.4f}, p = {pval:.4e}, n = {len(x)}")

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.1, s=8, rasterized=True, color='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{label}\nn = {len(x)}\npearson r = {r:.3f}")
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    plt.plot([min_val, max_val], [min_val, max_val], ls="--", color="green", linewidth=2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# === Plot all versions ===
make_plot(
    merged_unfiltered,
    "DELTA_PSI",
    "exp_dpsi_avg",
    "HAL Predicted ΔPSI",
    "Measured ΔPSI (mean across cell lines)",
    "HAL ΔPSI vs Experimental ΔPSI (unfiltered)",
    f"{output_base}_dpsi_unfiltered.pdf"
)

make_plot(
    merged_filtered,
    "DELTA_PSI",
    "exp_dpsi_avg",
    "Predicted ΔPSI (HAL)",
    "Measured ΔPSI (mean across cell lines)",
    "HAL ΔPSI vs Experimental ΔPSI (filtered)",
    f"{output_base}_dpsi_filtered.pdf"
)

make_plot(
    merged_unfiltered,
    "hal_delta_logit",
    "exp_dlogit_avg",
    "HAL Predicted Δlogit PSI",
    "Measured Mean Δlogit PSI",
    "HAL Δlogit vs Experimental Δlogit (unfiltered)",
    f"{output_base}_dlogit_unfiltered.pdf"
)

make_plot(
    merged_filtered,
    "hal_delta_logit",
    "exp_dlogit_avg",
    "HAL Predicted Δlogit PSI",
    "Measured Mean Δlogit PSI",
    "HAL Δlogit vs Experimental Δlogit (filtered)",
    f"{output_base}_dlogit_filtered.pdf"
)
