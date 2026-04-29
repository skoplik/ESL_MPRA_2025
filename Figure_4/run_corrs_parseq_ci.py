import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import ast


# === File paths ===
parse_seq_path = "/ESL/ESL_MPRA/Figure_4/41467_2024_52474_MOESM7_ESM.csv"
supertable_path = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz"
variant_info_path = "/ESL/ESL_MPRA/Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25.csv.gz"
clinvar_path = "/ESL/Figures/Variant_analyses/heatmap_plots/supertable_ClinVar_matched_in_supertable.txt"
output_dir = "/ESL/ESL_MPRA/Figure_4/outputs/fig_parse_seq_ci"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "parse_seq_corr_output.tsv")

# === ClinVar color map ===
CLINVAR_COLORS = {
    "Pathogenic": "red",
    "Likely_pathogenic": "orange",
    "Uncertain Significance (VUS)": "slateblue",
    "Likely_benign": "#9ACD32",
    "Benign": "#258d4c",
    "Conflicting (CL)": "orchid",
}
CLINVAR_ORDER = list(CLINVAR_COLORS.keys())

# === Functions ===
def parse_clinvar_significance(clinvar_item):
    sigs = set()
    if isinstance(clinvar_item, dict) and "ClinVar_info" in clinvar_item:
        clinvar_item = clinvar_item["ClinVar_info"]
    if isinstance(clinvar_item, list):
        for entry in clinvar_item:
            if isinstance(entry, tuple) and len(entry) == 4:
                meta = entry[3]
                if "CLNSIG=" in meta:
                    for part in meta.split(";"):
                        if part.startswith("CLNSIG="):
                            raw_sigs = part.split("=")[1].replace(" ", "_").split("|")
                            sigs.update(raw_sigs)
    elif isinstance(clinvar_item, str):
        for part in clinvar_item.split(";"):
            if part.startswith("CLNSIG="):
                raw_sigs = part.split("=")[1].replace(" ", "_").split("|")
                sigs.update(raw_sigs)
    normalized = set()
    for s in sigs:
        s_lower = s.strip().lower()
        if s_lower in {"conflicting_classifications_of_pathogenicity", "conflicting"}:
            normalized.add("Conflicting (CL)")
        elif s_lower == "uncertain_significance":
            normalized.add("Uncertain Significance (VUS)")
        elif s_lower == "not_provided":
            normalized.add("Not_provided")
        else:
            normalized.add(s.replace(" ", "_"))
    if not normalized:
        return "Not_provided"
    elif len(normalized) > 1:
        return "Conflicting (CL)"
    else:
        return list(normalized)[0]

def format_hgvs_to_variant_hg38(hgvs_str):
    chrom, pos, ref, alt = hgvs_str.split("-")
    return f"chr{chrom}:{pos}:{ref}>{alt}"

def report_corr(x, y, label):
    valid = merged[[x, y]].dropna()
    r, p = stats.pearsonr(valid[x], valid[y])
    print(f"{label}: r = {r:.3f}, p = {p:.2e}, N = {len(valid)}")
    return r, p, len(valid)

def regression_ci_band(x_vals, y_vals, x_line, ci=0.95):
    """Return (y_fit, lower, upper) for x_line given paired x_vals, y_vals."""
    slope, intercept, r, p, se = stats.linregress(x_vals, y_vals)
    y_fit = slope * x_line + intercept

    n = len(x_vals)
    x_mean = np.mean(x_vals)
    residuals = y_vals - (slope * x_vals + intercept)
    s_err = np.sqrt((residuals ** 2).sum() / (n - 2))
    t_val = stats.t.ppf((1 + ci) / 2, df=n - 2)
    x_dev = x_line - x_mean
    ci_band = t_val * s_err * np.sqrt(1.0 / n + x_dev ** 2 / ((x_vals - x_mean) ** 2).sum())

    return y_fit, y_fit - ci_band, y_fit + ci_band, slope, intercept, r, p

def plot_scatter(x, y, filename):
    df = merged[[x, y, "ClinVar Classification", "gene_exon"]].dropna()
    if df.empty:
        print(f"Skipping {filename} — no data.")
        return

    gene_exon = df["gene_exon"].iloc[0]
    r, p = stats.pearsonr(df[x], df[y])
    n = len(df)

    # 95% CI band
    x_vals = df[x].values
    y_vals = df[y].values
    x_line = np.linspace(x_vals.min(), x_vals.max(), 300)
    y_fit, lower, upper, slope, intercept, _, _ = regression_ci_band(x_vals, y_vals, x_line)

    plt.figure(figsize=(4.5, 4.5))
    ax = plt.gca()

    # Regression line + CI band (drawn first, behind dots)
    ax.fill_between(x_line, lower, upper, color="grey", alpha=0.2, zorder=0, label="95% CI")
    ax.plot(x_line, y_fit, color="black", linewidth=1, zorder=1, label="Regression")

    for label in CLINVAR_ORDER:
        sub = df[df["ClinVar Classification"] == label]
        if not sub.empty:
            fill_color = mcolors.to_rgba(CLINVAR_COLORS[label], alpha=0.6)
            edge_color = mcolors.to_rgba(CLINVAR_COLORS[label], alpha=1.0)
            ax.scatter(
                sub[x],
                sub[y],
                s=40,
                facecolors=fill_color,
                edgecolors=edge_color,
                linewidth=0.7,
                label=label,
                zorder=2
            )

    plt.xlabel("HEK293 Mean PSI" if "psi_mean" in x else "HEK293 Mean ΔPSI (COMPASS)")
    plt.ylabel("ParSE-seq HEK293 Mean PSI" if "psi_mean" in y else "ParSE-seq HEK293 Mean ΔPSI")
    plt.title(f"{gene_exon}\nr = {r:.2f}, p = {p:.1e}, N = {n}")

    handles, labels_list = ax.get_legend_handles_labels()
    # Keep CI and regression entries, then ClinVar
    by_label = dict(zip(labels_list, handles))
    ordered = []
    for entry in ["Regression", "95% CI"] + CLINVAR_ORDER:
        if entry in by_label:
            ordered.append((entry, by_label[entry]))
    ax.legend([h for _, h in ordered], [l for l, _ in ordered], fontsize=7, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved: {os.path.join(output_dir, filename)}")

# === Load data ===
parse_df = pd.read_csv(parse_seq_path)
psi_df = pd.read_csv(supertable_path, low_memory=False)
clinvar_df = pd.read_csv(clinvar_path, sep="\t")

# === Parse ClinVar and annotate classification
clinvar_df["ClinVar_item"] = clinvar_df["ClinVar_item"].apply(ast.literal_eval)
clinvar_df["ClinVar Classification"] = clinvar_df["ClinVar_item"].apply(parse_clinvar_significance)

# === Merge ClinVar Classification (variant_hg38 already present in psi_df)
psi_df = pd.merge(psi_df, clinvar_df[["mut_ref", "ClinVar Classification"]], left_on="Reference", right_on="mut_ref", how="left")
psi_df["ClinVar Classification"] = psi_df["ClinVar Classification"].fillna("Not_provided")

# === Format ParSE variant_hg38
parse_df["variant_hg38"] = parse_df["HGVS Variant"].apply(format_hgvs_to_variant_hg38)

# === Normalize PSI values from percent to proportion
rep_cols = ["HEK PSI Rep 1", "HEK PSI Rep 2", "HEK PSI Rep 3"]
for col in rep_cols:
    parse_df[col] = parse_df[col] / 100

parse_df["parse_psi_mean"] = parse_df[rep_cols].mean(axis=1)
parse_df["parse_delta_psi"] = parse_df["delta_psi"] / 100
parse_df["parse_delta_psi_norm"] = parse_df["delta_psi_norm"] / 100

# === Merge with ParSE-seq
merged = pd.merge(psi_df, parse_df, on="variant_hg38", how="inner", suffixes=("", "_parse"))

# === Compute mean PSI and delta PSI from replicates
hek_reps = [c for c in ["HEK_rep1_psi_raw", "HEK_rep2_psi_raw", "HEK_rep3_psi_raw", "HEK_rep4_psi_raw"] if c in merged.columns]
merged["our_psi_mean"] = merged[hek_reps].mean(axis=1)

wt_df = psi_df[psi_df["snp"] == "none"][["event_id", "HEK_pooled_psi_clipped"]]
wt_df = wt_df.rename(columns={"HEK_pooled_psi_clipped": "our_wt_psi"})
merged = pd.merge(merged, wt_df, on="event_id", how="left")
merged["our_delta_psi"] = merged["our_psi_mean"] - merged["our_wt_psi"]

# === Filter to ClinVar categories with colors
merged = merged[merged["ClinVar Classification"].isin(CLINVAR_COLORS.keys())]

# === Correlation reports
report_corr("our_psi_mean", "parse_psi_mean", "PSI mean correlation")
report_corr("our_delta_psi", "parse_delta_psi", "ΔPSI correlation")
report_corr("our_delta_psi", "parse_delta_psi_norm", "ΔPSI normalized correlation")

# === Plots with 95% CI band
plot_scatter("our_psi_mean", "parse_psi_mean", "parse_vs_our_psi_mean_ci.pdf")
plot_scatter("our_delta_psi", "parse_delta_psi", "parse_vs_our_delta_psi_ci.pdf")

# === Save output
merged.to_csv(output_path, sep="\t", index=False)
print(f"Saved merged output to {output_path}")
