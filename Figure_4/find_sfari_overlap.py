import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# === Make PDF text editable in Illustrator ===
mpl.rcParams['pdf.fonttype'] = 42   # Embed TrueType fonts, not paths
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text editable if you also save SVGs

# === Paths ===
sfari_file = "/ESL/ESL_MPRA/Figure_4/SFARI-Gene_genes_07-08-2025release_08-20-2025export.csv"
hgnc_file = "/ESL/Figures_SK/Disease_relevant_genes/hgnc_complete_set.txt"
supertable_file = "/ESL/ESL_MPRA/Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25.csv.gz"
allseq_file = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz"
clinvar_path = "/ESL/ESL_MPRA/Figure_4/outputs/clinvar/swarm_delta_logit.csv"

output_plot = "/ESL/ESL_MPRA/SI_figures/ACMG_overlap/outputs/sfari_overlap_clinvar_stripplot.pdf"

# === ClinVar colors ===
CLINVAR_COLORS = {
    "Pathogenic": "red",
    "Likely_pathogenic": "orange",
    "Uncertain_significance": "slateblue",
    "Likely_benign": "#9ACD32",
    "Benign": "#258d4c",
    "Conflicting": "orchid"
}

# === Load data ===
sfari = pd.read_csv(sfari_file)
hgnc = pd.read_csv(hgnc_file, sep="\t", dtype=str, low_memory=False)
st = pd.read_csv(supertable_file, dtype=str, low_memory=False)
allseq = pd.read_csv(allseq_file, dtype=str, low_memory=False)
clinvar_df = pd.read_csv(clinvar_path)

# === Normalize SFARI ===
sfari = sfari.rename(columns={"ensembl-id": "ensembl_gene_id", "gene-symbol": "symbol"})

# === HGNC alias map ===
def expand_aliases(x):
    if pd.isna(x):
        return []
    return [a.strip() for a in str(x).split("|")]

alias_map = {}
for _, row in hgnc.iterrows():
    all_names = set([row["symbol"]] + expand_aliases(row["alias_symbol"]) + expand_aliases(row["prev_symbol"]))
    for name in all_names:
        if pd.notna(row["ensembl_gene_id"]):
            alias_map[name] = row["ensembl_gene_id"]

sfari["ensembl_gene_id_mapped"] = sfari["symbol"].map(alias_map)
sfari["ensembl_gene_id_final"] = sfari["ensembl_gene_id"].combine_first(sfari["ensembl_gene_id_mapped"])

# === Supertable mapping ===
st_genes = st[["Reference", "event_id", "gene_exon", "gene_name", "hgnc_id"]].copy()
st_genes = st_genes.merge(hgnc[["hgnc_id", "ensembl_gene_id"]], how="left", on="hgnc_id")

# === Overlap with SFARI ===
overlap = st_genes.merge(
    sfari[["symbol", "ensembl_gene_id_final", "gene-score", "syndromic", "genetic-category"]],
    how="inner",
    left_on="ensembl_gene_id",
    right_on="ensembl_gene_id_final"
).drop_duplicates()

# Add variant counts
event_counts = allseq["event_id"].value_counts().rename("variant_count").reset_index().rename(columns={"index": "event_id"})
overlap = overlap.merge(event_counts, on="event_id", how="left")

# Keep only >30 variants
overlap = overlap[overlap["variant_count"] > 30]
overlap["variant_count"] = overlap["variant_count"].astype(int)

# === Average Δlogit(PSI) ===
delta_cols = [c for c in allseq.columns if c.endswith("_delta_logit_pooled")]
allseq[delta_cols] = allseq[delta_cols].apply(pd.to_numeric, errors="coerce")
allseq["avg_delta_logit"] = allseq[delta_cols].mean(axis=1)

# === Unique event_id + exon pairs, sorted alphabetically by gene_exon ===
sfari_event_exons = overlap[["event_id", "gene_exon", "gene_name", "variant_count"]].drop_duplicates()
sfari_event_exons = sfari_event_exons.sort_values("gene_exon").reset_index(drop=True)

# Label: "GENE exonX, n=###"
sfari_event_exons["label"] = sfari_event_exons.apply(
    lambda r: f"{r['gene_exon']}, n={r['variant_count']}", axis=1
)

# Map to variants
variants = allseq[(allseq["event_id"].isin(sfari_event_exons["event_id"])) & (allseq["snp"] != "none")].copy()

# Map exons to positions
x_positions = {ev: i for i, ev in enumerate(sfari_event_exons["event_id"])}
variants["x_pos"] = variants["event_id"].map(x_positions)
rng = np.random.default_rng(seed=42)
variants["x_jitter"] = variants["x_pos"] + rng.uniform(-0.3, 0.3, size=len(variants))

# === Merge ClinVar ===
variants["Reference"] = variants["Reference"].astype(str)
clinvar_df["Reference"] = clinvar_df["Reference"].astype(str)

annot_variants = variants.merge(
    clinvar_df[["Reference", "event_id", "CLNSIG_category"]],
    on=["Reference", "event_id"],
    how="inner"
)

print(f"Total ClinVar overlap variants (SFARI, >30 vars): {len(annot_variants)}")

# === Plot ===
fig, ax = plt.subplots(figsize=(11, 6))

# All variants (grey, rasterized)
ax.scatter(
    variants["x_jitter"],
    variants["avg_delta_logit"],
    color="lightgrey",
    alpha=0.6,
    s=30,
    edgecolor="none",
    label="All variants",
    rasterized=True
)

# ClinVar categories (fill + stroke trick, rasterized dots only)
plot_order = ["Likely_benign", "Benign", "Conflicting", "Uncertain_significance", "Likely_pathogenic", "Pathogenic"]
for category in plot_order:
    subset = annot_variants[annot_variants["CLNSIG_category"] == category]
    if not subset.empty:
        # semi-transparent fill
        ax.scatter(
            subset["x_jitter"],
            subset["avg_delta_logit"],
            facecolors=CLINVAR_COLORS[category],
            edgecolors="none",
            s=35,
            alpha=0.6,
            rasterized=True
        )
        # solid edge
        ax.scatter(
            subset["x_jitter"],
            subset["avg_delta_logit"],
            facecolors="none",
            edgecolors=CLINVAR_COLORS[category],
            s=35,
            linewidth=0.6,
            alpha=1.0,
            label=category,
            rasterized=True
        )

# === X-axis labels ===
ax.set_xticks(range(len(sfari_event_exons)))
ax.set_xticklabels(sfari_event_exons["label"], rotation=45, ha="right", fontsize=8)

ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_ylabel("Average Δlogit(PSI) Across Cell Lines")
ax.set_xlabel("SFARI Gene / Exon (alphabetical)")
ax.set_title("SFARI Overlap Δlogit(PSI) with ClinVar Annotation")

# Legend dedup
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.01, 1), loc="upper left")

plt.tight_layout()
plt.savefig(output_plot, bbox_inches="tight", dpi=600)  # Rasterized dots at high dpi, editable text
plt.close()

print(f"Saved SFARI overlap + ClinVar stripplot to {output_plot}")
