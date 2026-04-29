import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Paths ===
tf_list_path = "/ESL/ESL_MPRA/SI_figures/TFs_overlap/TF_names_v_1.01.txt"
variant_counts_path = "/ESL/ESL_MPRA/SI_figures/highly_saturated_exons/outputs/all_gene_exon_variant_counts.tsv"
allseq_file = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz"
clinvar_path = "/ESL/ESL_MPRA/Figure_4/outputs/clinvar/swarm_delta_logit.csv"

output_overlap_path = "/ESL/ESL_MPRA/SI_figures/TFs_overlap/outputs/tf_overlap_variant_counts.tsv"
output_plot = "/ESL/ESL_MPRA/SI_figures/TFs_overlap/outputs/tf_overlap_clinvar_stripplot.pdf"

# === ClinVar colors ===
CLINVAR_COLORS = {
    "Pathogenic": "red",
    "Likely_pathogenic": "orange",
    "Conflicting": "orchid",
    "Uncertain_significance": "slateblue",
    "Benign": "#258d4c",
    "Likely_benign": "#9ACD32"
}

# === Load TF list ===
tf_genes = set(pd.read_csv(tf_list_path, header=None)[0].str.strip().unique())

# === Load variant counts and all sequences ===
variant_counts = pd.read_csv(variant_counts_path, sep="\t")
allseq = pd.read_csv(allseq_file)
clinvar_df = pd.read_csv(clinvar_path)

# Extract gene name from "gene_exon"
variant_counts["gene"] = variant_counts["gene_exon"].str.split(" ").str[0]

# === Overlap TFs ===
overlap = variant_counts[variant_counts["gene"].isin(tf_genes)].copy()
overlap = overlap.sort_values("variant_count", ascending=False)
overlap.to_csv(output_overlap_path, sep="\t", index=False)
print(f"Found {len(overlap)} overlapping TFs. Saved to {output_overlap_path}")

# Keep only >100 variants
overlap = overlap[overlap["variant_count"] > 100]

# === Average Δlogit(PSI) ===
delta_cols = [c for c in allseq.columns if c.endswith("_delta_logit_pooled")]
allseq[delta_cols] = allseq[delta_cols].apply(pd.to_numeric, errors="coerce")
allseq["avg_delta_logit"] = allseq[delta_cols].mean(axis=1)

tf_gene_exons = overlap["gene_exon"].unique().tolist()
variants = allseq[(allseq["gene_exon"].isin(tf_gene_exons)) & (allseq["snp"] != "none")].copy()

# Map exons to y positions + jitter
y_positions = {exon: i for i, exon in enumerate(tf_gene_exons)}
variants["y_pos"] = variants["gene_exon"].map(y_positions)
rng = np.random.default_rng(seed=42)
variants["y_jitter"] = variants["y_pos"] + rng.uniform(-0.2, 0.2, size=len(variants))

# === Merge ClinVar by Reference + gene_exon ===
variants["Reference"] = variants["Reference"].astype(str)
clinvar_df["Reference"] = clinvar_df["Reference"].astype(str)

annot_variants = variants.merge(
    clinvar_df[["Reference", "gene_exon", "CLNSIG_category"]],
    on=["Reference", "gene_exon"],
    how="inner"
)

print(f"Total ClinVar overlap variants (TFs, >100 vars): {len(annot_variants)}")

# === Plot ===
fig, ax = plt.subplots(figsize=(12, 8))

# Grey = all variants
ax.scatter(
    variants["avg_delta_logit"],
    variants["y_jitter"],
    color="lightgrey",
    alpha=0.6,
    s=20,
    edgecolor="none",
    label="All variants"
)

# Colored = ClinVar overlaps
plot_order = ["Pathogenic", "Likely_pathogenic", "Conflicting", "Uncertain_significance", "Benign", "Likely_benign"]
for category in plot_order:
    subset = annot_variants[annot_variants["CLNSIG_category"] == category]
    if not subset.empty:
        ax.scatter(
            subset["avg_delta_logit"],
            subset["y_jitter"],
            color=CLINVAR_COLORS[category],
            s=20,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label=category
        )

# === Y-axis labels with inline n ===
yticklabels = []
for exon in tf_gene_exons:
    n = overlap.loc[overlap["gene_exon"] == exon, "variant_count"].iloc[0]
    yticklabels.append(f"{exon}, n={n}")

ax.set_yticks(range(len(tf_gene_exons)))
ax.set_yticklabels(yticklabels)

ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Average Δlogit(PSI) Across Cell Lines")
ax.set_ylabel("TF Gene/Exon")
ax.set_title("TF Overlap \nΔlogit(PSI) with ClinVar Annotation")

# Legend dedup
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.savefig(output_plot, bbox_inches="tight")
plt.close()

print(f"Saved TF overlap + ClinVar stripplot to {output_plot}")
