import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from matplotlib.patches import Patch
import matplotlib.colors as mcolors


# === File paths ===
parse_seq_path = "/ESL/ESL_MPRA/Figure_3/41467_2024_52474_MOESM7_ESM.csv"
supertable_path = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz"
variant_info_path = "/ESL/ESL_MPRA/Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25.csv.gz"
clinvar_path = "/ESL/Figures/Variant_analyses/heatmap_plots/supertable_ClinVar_matched_in_supertable.txt"
output_dir = "/ESL/ESL_MPRA/Figure_3/outputs/fig_parse_seq"
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
    r, p = pearsonr(valid[x], valid[y])
    print(f"{label}: r = {r:.3f}, p = {p:.2e}, N = {len(valid)}")
    return r, p, len(valid)

def plot_scatter(x, y, filename):
    df = merged[[x, y, "ClinVar Classification", "gene_exon"]].dropna()
    if df.empty:
        print(f"Skipping {filename} — no data.")
        return

    gene_exon = df["gene_exon"].iloc[0]
    r, p = pearsonr(df[x], df[y])
    n = len(df)

    plt.figure(figsize=(4.5, 4.5))
    ax = plt.gca()

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
                label=label
            )

    plt.xlabel("HEK293 Mean PSI" if "psi_mean" in x else "HEK293 Mean ΔPSI (Our MPPRA)")
    plt.ylabel("ParSE-seq HEK293 Mean PSI" if "psi_mean" in y else "ParSE-seq HEK293 Mean ΔPSI")
    plt.title(f"{gene_exon}\nr = {r:.2f}, p = {p:.1e}, N = {n}")
    handles, labels = ax.get_legend_handles_labels()
    handles_labels = {l: h for h, l in zip(handles, labels) if l in CLINVAR_ORDER}
    ordered_handles = [handles_labels[l] for l in CLINVAR_ORDER if l in handles_labels]
    plt.legend(handles=ordered_handles, fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# === Load data ===
parse_df = pd.read_csv(parse_seq_path)
psi_df = pd.read_csv(supertable_path)
variant_info = pd.read_csv(variant_info_path)
clinvar_df = pd.read_csv(clinvar_path, sep="\t")

# === Make variant_info 1-indexed
variant_info["Reference"] = variant_info["Reference"] + 1

# === Parse ClinVar and annotate classification
clinvar_df["ClinVar_item"] = clinvar_df["ClinVar_item"].apply(ast.literal_eval)
clinvar_df["ClinVar Classification"] = clinvar_df["ClinVar_item"].apply(parse_clinvar_significance)

# === Merge variant_hg38 and ClinVar Classification
psi_df = pd.merge(psi_df, variant_info[["Reference", "variant_hg38"]], on="Reference", how="left")
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

# === Compute mean PSI and delta PSI from your replicates
hek_reps = ["HEK_rep1_psi_raw", "HEK_rep2_psi_raw", "HEK_rep3_psi_raw", "HEK_rep4_psi_raw"]
hek_reps = [c for c in hek_reps if c in merged.columns]
merged["our_psi_mean"] = merged[hek_reps].mean(axis=1)

wt_df = psi_df[psi_df["snp"] == "none"][["event_id", "HEK_pooled_psi_clipped"]]
wt_df = wt_df.rename(columns={"HEK_pooled_psi_clipped": "our_wt_psi"})
merged = pd.merge(merged, wt_df, on="event_id", how="left")
merged["our_delta_psi"] = merged["our_psi_mean"] - merged["our_wt_psi"]

# === Filter ClinVar values
merged = merged[merged["ClinVar Classification"].isin(CLINVAR_COLORS.keys())]

# === Correlation reports
report_corr("our_psi_mean", "parse_psi_mean", "PSI mean correlation")
report_corr("our_delta_psi", "parse_delta_psi", "ΔPSI correlation")
report_corr("our_delta_psi", "parse_delta_psi_norm", "ΔPSI normalized correlation")

# === Plots
plot_scatter("our_psi_mean", "parse_psi_mean", "parse_vs_our_psi_mean.pdf")
plot_scatter("our_delta_psi", "parse_delta_psi", "parse_vs_our_delta_psi.pdf")

# === Save output
merged.to_csv(output_path, sep="\t", index=False)
print(f"Saved merged output to {output_path}")
