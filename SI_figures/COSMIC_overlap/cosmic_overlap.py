import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Paths ===
input_path = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz"
variant_count_path = "/ESL/ESL_MPRA/SI_figures/highly_saturated_exons/outputs/all_gene_exon_variant_counts.tsv"
cosmic_path = "/ESL/ESL_MPRA/SI_figures/COSMIC_overlap/Cosmic_CancerGeneCensus_v102_GRCh37.tsv"
clinvar_path = "/ESL/ESL_MPRA/Figure_3/outputs/clinvar/swarm_delta_logit.csv"
output_plot = "/ESL/ESL_MPRA/SI_figures/COSMIC_overlap/outputs/cosmic_overlap_clinvar_stripplot.pdf"

# === ClinVar colors ===
CLINVAR_COLORS = {
    "Pathogenic": "red",
    "Likely_pathogenic": "orange",
    "Conflicting": "orchid",
    "Uncertain_significance": "slateblue",
    "Benign": "#258d4c",
    "Likely_benign": "#9ACD32"
}

# === Load ===
df = pd.read_csv(input_path)
variant_counts = pd.read_csv(variant_count_path, sep="\t")
cosmic_df = pd.read_csv(cosmic_path, sep="\t")
clinvar_df = pd.read_csv(clinvar_path)

# === COSMIC Tier 1 genes with >100 variants ===
cosmic_df = cosmic_df[cosmic_df["TIER"] == 1]
merged = variant_counts.merge(
    cosmic_df[["GENE_SYMBOL", "ROLE_IN_CANCER", "TIER"]],
    left_on=variant_counts["gene_exon"].str.split().str[0],
    right_on="GENE_SYMBOL",
    how="inner"
)
merged = merged[merged["variant_count"] > 100]

# === Delta logit cols ===
delta_cols = [c for c in df.columns if c.endswith("_delta_logit_pooled")]
df[delta_cols] = df[delta_cols].apply(pd.to_numeric, errors="coerce")
df["avg_delta_logit"] = df[delta_cols].mean(axis=1)

# === Unique event_id + exon pairs, sorted by event_id ===
cosmic_event_exons = df[
    (df["gene_exon"].isin(merged["gene_exon"])) & (df["snp"] != "none")
][["event_id", "gene_exon"]].drop_duplicates()

cosmic_event_exons = cosmic_event_exons.sort_values("event_id").reset_index(drop=True)

# Attach counts and role info (exon-level)
cosmic_event_exons = cosmic_event_exons.merge(
    merged[["gene_exon", "variant_count", "ROLE_IN_CANCER", "TIER"]].drop_duplicates(),
    on="gene_exon",
    how="left"
)

# Map to variants
variants = df[(df["event_id"].isin(cosmic_event_exons["event_id"])) & (df["snp"] != "none")].copy()

# Map event_id to y position
y_positions = {ev: i for i, ev in enumerate(cosmic_event_exons["event_id"])}
variants["y_pos"] = variants["event_id"].map(y_positions)

# Jitter
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

print(f"Total ClinVar overlap variants (COSMIC Tier 1, >100 vars): {len(annot_variants)}")

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

# === Y-axis labels with inline n, role ===
yticklabels = []
for _, row in cosmic_event_exons.iterrows():
    yticklabels.append(f"{row['gene_exon']}, n={row['variant_count']}, {row['ROLE_IN_CANCER']}")

ax.set_yticks(range(len(cosmic_event_exons)))
ax.set_yticklabels(yticklabels)

ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Average Δlogit(PSI) Across Cell Lines")
ax.set_ylabel("COSMIC Tier 1 Gene/Exon")
ax.set_title("COSMIC Tier 1 Overlap \nΔlogit(PSI) with ClinVar Annotation")

# Legend dedup
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.savefig(output_plot, bbox_inches="tight")
plt.close()

print(f"Saved COSMIC overlap + ClinVar stripplot to {output_plot}")
