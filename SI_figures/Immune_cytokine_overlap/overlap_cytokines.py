import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Paths ===
input_path = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz"
supertable_meta = "/ESL/ESL_MPRA/Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25.csv.gz"
cytokine_file = "/ESL/ESL_MPRA/SI_figures/Immune_cytokine_overlap/CytokineRegistry.November_2015.csv"
clinvar_path = "/ESL/ESL_MPRA/Figure_4/outputs/clinvar/swarm_delta_logit.csv"

cyto_alias_file = "/ESL/ESL_MPRA/SI_figures/Immune_cytokine_overlap/cytokine_aliases_expanded.txt"
receptor_alias_file = "/ESL/ESL_MPRA/SI_figures/Immune_cytokine_overlap/receptor_aliases_expanded.txt"

output_plot = "/ESL/ESL_MPRA/SI_figures/Immune_cytokine_overlap/outputs/cytokine_receptor_overlap_clinvar_stripplot.pdf"

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
super_meta = pd.read_csv(supertable_meta, dtype=str)
cytokine_df = pd.read_csv(cytokine_file, dtype=str)
clinvar_df = pd.read_csv(clinvar_path)

# Normalize column names
super_meta.columns = super_meta.columns.str.strip().str.lower()
cytokine_df.columns = cytokine_df.columns.str.strip().str.lower()

super_meta["hgnc_id"] = super_meta["hgnc_id"].fillna("").str.strip()
super_meta["gene_name"] = super_meta["gene_name"].fillna("").str.strip()

# Load aliases
def load_aliases(path):
    with open(path) as f:
        return set(line.strip().upper() for line in f if line.strip())

cytokine_aliases = load_aliases(cyto_alias_file)
receptor_aliases = load_aliases(receptor_alias_file)

# Detect usable keys
hgnc_col = None
if "hgnc id" in cytokine_df.columns:
    hgnc_col = "hgnc id"
elif "hgnc_id" in cytokine_df.columns:
    hgnc_col = "hgnc_id"

symbol_col = None
if "entrezgene symbol (human)" in cytokine_df.columns:
    symbol_col = "entrezgene symbol (human)"
elif "symbol" in cytokine_df.columns:
    symbol_col = "symbol"

# === HGNC matches ===
merged_by_hgnc = pd.DataFrame()
if hgnc_col:
    merged_by_hgnc = super_meta.merge(
        cytokine_df,
        left_on="hgnc_id",
        right_on=hgnc_col,
        how="inner"
    )
merged_by_symbol = pd.DataFrame()
if symbol_col:
    missing_hgnc = cytokine_df if hgnc_col is None else cytokine_df[cytokine_df[hgnc_col] == ""]
    merged_by_symbol = super_meta.merge(
        missing_hgnc,
        left_on="gene_name",
        right_on=symbol_col,
        how="inner"
    )

hgnc_overlap = pd.concat([merged_by_hgnc, merged_by_symbol], ignore_index=True)

# === Alias matches ===
super_meta["gene_name_upper"] = super_meta["gene_name"].str.upper()
cytokine_alias_overlap = super_meta[super_meta["gene_name_upper"].isin(cytokine_aliases)].copy()
receptor_alias_overlap = super_meta[super_meta["gene_name_upper"].isin(receptor_aliases)].copy()

# Combine everything
combined_overlap = pd.concat([hgnc_overlap, cytokine_alias_overlap, receptor_alias_overlap], ignore_index=True)

# === Delta logit cols ===
delta_cols = [c for c in df.columns if c.endswith("_delta_logit_pooled")]
df[delta_cols] = df[delta_cols].apply(pd.to_numeric, errors="coerce")
df["avg_delta_logit"] = df[delta_cols].mean(axis=1)

# === Event_id/exons ===
event_exons = df[
    (df["gene_exon"].isin(combined_overlap["gene_exon"])) & (df["snp"] != "none")
][["event_id", "gene_exon"]].drop_duplicates()

event_exons = event_exons.sort_values("event_id").reset_index(drop=True)

# Count variants
variant_counts = (
    df[df["gene_exon"].isin(event_exons["gene_exon"]) & (df["snp"] != "none")]
    .groupby("gene_exon")
    .size()
    .reset_index(name="variant_count")
)

event_exons = event_exons.merge(variant_counts, on="gene_exon", how="left")

# Map to variants
variants = df[(df["event_id"].isin(event_exons["event_id"])) & (df["snp"] != "none")].copy()
y_positions = {ev: i for i, ev in enumerate(event_exons["event_id"])}
variants["y_pos"] = variants["event_id"].map(y_positions)

rng = np.random.default_rng(seed=42)
variants["y_jitter"] = variants["y_pos"] + rng.uniform(-0.2, 0.2, size=len(variants))

# ClinVar merge
variants["Reference"] = variants["Reference"].astype(str)
clinvar_df["Reference"] = clinvar_df["Reference"].astype(str)
annot_variants = variants.merge(
    clinvar_df[["Reference", "gene_exon", "CLNSIG_category"]],
    on=["Reference", "gene_exon"],
    how="inner"
)

print(f"Total ClinVar overlap variants (cytokines + receptors): {len(annot_variants)}")

# === Plot ===
fig, ax = plt.subplots(figsize=(12, 10))

# Grey = all variants
ax.scatter(
    variants["avg_delta_logit"], variants["y_jitter"],
    color="lightgrey", alpha=0.6, s=20, edgecolor="none", label="All variants"
)

# Colored = ClinVar overlaps
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

# Y-axis labels (just gene_exon + n)
yticklabels = []
for _, row in event_exons.iterrows():
    yticklabels.append(f"{row['gene_exon']}, n={row['variant_count']}")

ax.set_yticks(range(len(event_exons)))
ax.set_yticklabels(yticklabels)

ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Average Δlogit(PSI) Across Cell Lines")
ax.set_ylabel("Gene/Exon")
ax.set_title("Cytokine + Receptor Overlap\nΔlogit(PSI) with ClinVar Annotation")

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.savefig(output_plot, bbox_inches="tight")
plt.close()

print(f"Saved combined cytokine + receptor ClinVar stripplot to {output_plot}")
