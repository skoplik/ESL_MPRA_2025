import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import matplotlib as mpl

# === Paths ===
acmg_file = "/ESL/Figures_SK/Disease_relevant_genes/ACMG/1-s2.0-S1098360023008791-mmc1.csv"
allseq_file = "/ESL/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WITH_WT.csv"
clinvar_path = "/ESL/Figures_SK/redo_pop_variants/clinvar_updated_7_15_25/swarm_delta_logit.csv"

output_aliases_path = "/ESL/Figures_SK/Disease_relevant_genes/ACMG/acmg_aliases_expanded.txt"
output_plot_snv = "/ESL/Figures_SK/Disease_relevant_genes/ACMG/all_exons_overlap_clinvar_stripplot_snvonly.pdf"
output_plot_all = "/ESL/Figures_SK/Disease_relevant_genes/ACMG/all_exons_overlap_clinvar_stripplot_allvars.pdf"

# === Fonts: keep text editable in Illustrator ===
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'


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

# === Load all sequences and ClinVar ===
allseq_master = pd.read_csv(allseq_file)
clinvar_df = pd.read_csv(clinvar_path)

# === Average Δlogit(PSI) ===
delta_cols = [c for c in allseq_master.columns if c.endswith("_delta_logit_pooled")]
allseq_master[delta_cols] = allseq_master[delta_cols].apply(pd.to_numeric, errors="coerce")
allseq_master["avg_delta_logit"] = allseq_master[delta_cols].mean(axis=1)

# === Filter helper for SNVs ===
def is_single_snv(s):
    if pd.isna(s) or s == "none":
        return False
    if ";" in s:  # doubles or multiples
        return False
    return bool(re.match(r"^\d+[: ]?[ACGT]>\s*[ACGT]$", s.strip(), flags=re.IGNORECASE))

# === Function to plot (vertical, snug, flipped order) ===
def make_plot(allseq, min_variants, outpath, title_suffix):
    counts = allseq.groupby("gene_exon").size().reset_index(name="variant_count")
    counts["gene"] = counts["gene_exon"].str.split(" ").str[0].str.upper()
    counts["in_ACMG"] = counts["gene"].isin(acmg_genes)

    counts = counts[counts["variant_count"] >= min_variants].sort_values("gene_exon")
    print(f"Exons with ≥{min_variants} variants ({title_suffix}): {len(counts)}")

    allseq = allseq[allseq["gene_exon"].isin(counts["gene_exon"])]

    event_exons = allseq[["event_id", "gene_exon"]].drop_duplicates()
    event_exons = event_exons.merge(
        counts[["gene_exon", "variant_count"]],
        on="gene_exon",
        how="left"
    ).sort_values("gene_exon").reset_index(drop=True)

    # Flip order so A is on top, Z at bottom
    event_exons = event_exons.iloc[::-1].reset_index(drop=True)

    y_positions = {ev: i for i, ev in enumerate(event_exons["event_id"])}
    allseq["y_pos"] = allseq["event_id"].map(y_positions)
    rng = np.random.default_rng(seed=42)
    allseq["y_jitter"] = allseq["y_pos"] + rng.uniform(-0.2, 0.2, size=len(allseq))

    # Merge ClinVar
    allseq["Reference"] = allseq["Reference"].astype(str)
    clinvar_df["Reference"] = clinvar_df["Reference"].astype(str)
    annot_variants = allseq.merge(
        clinvar_df[["Reference", "event_id", "CLNSIG_category"]],
        on=["Reference", "event_id"],
        how="inner"
    )

    print(f"Total ClinVar overlap variants ({title_suffix}): {len(annot_variants)}")

    # Plot (portrait 8.5 x 11)
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.scatter(
        allseq["avg_delta_logit"],
        allseq["y_jitter"],
        color="lightgrey", alpha=0.6, s=12, edgecolor="none", label="All variants"
    )

    plot_order = ["Likely_benign", "Benign", "Conflicting", "Uncertain_significance", "Likely_pathogenic", "Pathogenic"]
    for category in plot_order:
        subset = annot_variants[annot_variants["CLNSIG_category"] == category]
        if not subset.empty:
            ax.scatter(
                subset["avg_delta_logit"], subset["y_jitter"],
                color=CLINVAR_COLORS[category],
                s=12, alpha=0.8, edgecolor="black", linewidth=0.5,
                label=category
            )

    yticklabels = [f"{row['gene_exon']}, n={row['variant_count']}" for _, row in event_exons.iterrows()]
    ax.set_yticks(range(len(event_exons)))
    ax.set_yticklabels(yticklabels, fontsize=6)

    # Snug y-limits
    ax.set_ylim(-1, len(event_exons) +0.1)

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Average Δlogit(PSI) Across Cell Lines")
    ax.set_ylabel(f"Gene/Exon (≥{min_variants} variants, {title_suffix})")
    ax.set_title(f"ACMG + Non-ACMG Exons (≥{min_variants} variants, {title_suffix})\nΔlogit(PSI) with ClinVar Annotation")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=600)
    plt.close()

    print(f"Saved plot to {outpath}")

# === Single SNVs only, ≥150 ===
allseq_snv = allseq_master[allseq_master["snp"].apply(is_single_snv)]
print(f"Variants after SNV-only filter: {len(allseq_snv)}")
make_plot(allseq_snv, min_variants=150, outpath=output_plot_snv, title_suffix="single SNVs only")

# === All variants, ≥300 ===
allseq_all = allseq_master.copy()
print(f"Variants (all types): {len(allseq_all)}")
make_plot(allseq_all, min_variants=300, outpath=output_plot_all, title_suffix="all variants")
