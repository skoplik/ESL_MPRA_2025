import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === Paths ===
acmg_file = "/ESL/ESL_MPRA/SI_figures/ACMG_overlap/1-s2.0-S1098360023008791-mmc1.csv"
variant_counts_path = "/ESL/ESL_MPRA/SI_figures/highly_saturated_exons/outputs/all_gene_exon_variant_counts.tsv"
allseq_file = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz"
clinvar_path = "/ESL/ESL_MPRA/Figure_4/outputs/clinvar/swarm_delta_logit.csv"

output_overlap_path = "/ESL/ESL_MPRA/SI_figures/ACMG_overlap/outputs/acmg_overlap_variant_counts.tsv"
output_aliases_path = "/ESL/ESL_MPRA/SI_figures/ACMG_overlap/acmg_aliases_expanded.txt"
output_plot = "/ESL/ESL_MPRA/SI_figures/ACMG_overlap/outputs/acmg_overlap_clinvar_stripplot.pdf"

# === ClinVar colors ===
CLINVAR_COLORS = {
    "Pathogenic": "red",
    "Likely_pathogenic": "orange",
    "Conflicting": "orchid",
    "Uncertain_significance": "slateblue",
    "Benign": "#258d4c",
    "Likely_benign": "#9ACD32"
}

# === Load ACMG gene list ===
acmg = pd.read_csv(acmg_file)

def expand_aliases(x):
    items = []
    a_clean = str(x).upper().replace(" ", "")
    a_nohyphen = a_clean.replace("-", "")
    items.append(a_nohyphen)
    items.append(a_clean)
    if "-" not in a_clean and any(ch.isdigit() for ch in a_clean):
        hyphenated = "".join(
            [c + "-" if c.isalpha() and i + 1 < len(a_clean) and a_clean[i+1].isdigit() else c
             for i, c in enumerate(a_clean)]
        )
        items.append(hyphenated)
    return set(items)

acmg_genes = set()
for g in acmg["Gene"].dropna():
    acmg_genes.update(expand_aliases(g))

print(f"Total ACMG gene symbols/aliases collected: {len(acmg_genes)}")
with open(output_aliases_path, "w") as f:
    for g in sorted(acmg_genes):
        f.write(g + "\n")
print(f"Saved expanded ACMG aliases to {output_aliases_path}")

# === Load variant counts and all sequences ===
variant_counts = pd.read_csv(variant_counts_path, sep="\t")
allseq = pd.read_csv(allseq_file)
clinvar_df = pd.read_csv(clinvar_path)

# Extract gene name from "gene_exon"
variant_counts["gene"] = variant_counts["gene_exon"].str.split(" ").str[0].str.upper()

# === Overlap ACMG genes ===
overlap = variant_counts[variant_counts["gene"].isin(acmg_genes)].copy()
overlap = overlap.sort_values("variant_count", ascending=False)
overlap.to_csv(output_overlap_path, sep="\t", index=False)
print(f"Found {len(overlap)} overlapping ACMG exons. Saved to {output_overlap_path}")

# Keep only >100 variants
overlap = overlap[overlap["variant_count"] > 100]

# === Average Δlogit(PSI) ===
delta_cols = [c for c in allseq.columns if c.endswith("_delta_logit_pooled")]
allseq[delta_cols] = allseq[delta_cols].apply(pd.to_numeric, errors="coerce")
allseq["avg_delta_logit"] = allseq[delta_cols].mean(axis=1)

# === Unique event_id + exon pairs, sorted by event_id ===
acmg_event_exons = allseq[
    (allseq["gene_exon"].isin(overlap["gene_exon"])) & (allseq["snp"] != "none")
][["event_id", "gene_exon"]].drop_duplicates()

acmg_event_exons = acmg_event_exons.sort_values("event_id").reset_index(drop=True)

# Attach counts from overlap (per exon, not per event_id)
acmg_event_exons = acmg_event_exons.merge(
    overlap[["gene_exon", "variant_count"]].drop_duplicates(),
    on="gene_exon",
    how="left"
)

# Map to variants
variants = allseq[(allseq["event_id"].isin(acmg_event_exons["event_id"])) & (allseq["snp"] != "none")].copy()

# Map event_id to y position
y_positions = {ev: i for i, ev in enumerate(acmg_event_exons["event_id"])}
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

print(f"Total ClinVar overlap variants (ACMG): {len(annot_variants)}")

# === Plot ===
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(
    variants["avg_delta_logit"],
    variants["y_jitter"],
    color="lightgrey", alpha=0.6, s=20, edgecolor="none", label="All variants"
)
plot_order = ["Pathogenic", "Likely_pathogenic", "Conflicting", "Uncertain_significance", "Benign", "Likely_benign"]
for category in plot_order:
    subset = annot_variants[annot_variants["CLNSIG_category"] == category]
    if not subset.empty:
        ax.scatter(
            subset["avg_delta_logit"], subset["y_jitter"],
            color=CLINVAR_COLORS[category],
            s=20, alpha=0.8, edgecolor="black", linewidth=0.5,
            label=category
        )

# === Y-axis labels with inline n (from overlap, exon-level counts) ===
yticklabels = []
for _, row in acmg_event_exons.iterrows():
    yticklabels.append(f"{row['gene_exon']}, n={row['variant_count']}")

ax.set_yticks(range(len(acmg_event_exons)))
ax.set_yticklabels(yticklabels)
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Average Δlogit(PSI) Across Cell Lines")
ax.set_ylabel("ACMG Gene/Exon")
ax.set_title("ACMG Gene Overlap\nΔlogit(PSI) with ClinVar Annotation")

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(output_plot, bbox_inches="tight")
plt.close()

print(f"Saved ACMG overlap + ClinVar stripplot to {output_plot}")
