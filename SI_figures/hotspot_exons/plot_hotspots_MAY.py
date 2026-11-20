#!/usr/bin/env python3
"""SI figure: dlogit(PSI) distributions for variants in mutational hotspot exons.

May 2026 rebuild of Figures_SK/Disease_relevant_genes/hot_spot_exons/plot_hotspots.py.
Changes from the original:
  * May data (1e-2_ALL_WTS_VARS_NO_DELTAS) instead of the July 2025 table.
  * Everything is per locus (event_id), not per gene_exon label. The original
    counted, filtered and labelled by gene_exon, so two loci sharing a label
    (CACNA1C "exon 31") got one pooled n and identical y labels.
  * Labels go through SI_figures/exon_label_fixes.py (MANE numbering) and the
    script fails if two plotted loci still share a label.
  * Hotspot overlap is recomputed from the May exon coordinates.
  * ClinVar comes from the Figure 4 table, joined on Reference with no shift
    (verified: 2166/2166 SNP matches at +0, 7/2040 at +1).
  * Filter is >= MIN_VARIANTS (legend: "at least 100"); the original used > 100.

Prints the numbers quoted in the legend.
"""
import ast, importlib.util, os, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

REPO = "/ESL/ESL_MPRA"
sys.path.insert(0, f"{REPO}/SI_figures")
from exon_label_fixes import label as fix_label

MAIN = f"{REPO}/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz"
HOTSPOTS = f"{REPO}/SI_figures/hotspot_exons/inputs/mapped_hg38_hotspot_annotated.tsv"
CLINVAR = f"{REPO}/Figure_4/inputs/supertable_ClinVar_matched_in_supertable.txt"
PANEL_SCRIPT = f"{REPO}/Figure_4/ssm_single_muts_logits_06_04_2025_with_clinvar_patched.py"
OUT_DIR = f"{REPO}/SI_figures/hotspot_exons/outputs"
CELLS = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]
MIN_VARIANTS = 100

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

spec = importlib.util.spec_from_file_location("ssm_mod", PANEL_SCRIPT)
ssm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssm)
CLINVAR_COLORS = ssm.CLINVAR_COLORS
PLOT_ORDER = ["Pathogenic", "Likely_pathogenic", "Conflicting",
              "Uncertain_significance", "Benign", "Likely_benign"]

# --- load ---
dl_cols = [c + "_delta_logit_pooled" for c in CELLS]
df = pd.read_csv(MAIN, low_memory=False,
                 usecols=["Reference", "event_id", "gene_exon", "seq_type",
                          "exon_start_hg38", "exon_end_hg38"] + dl_cols)
df[dl_cols] = df[dl_cols].apply(pd.to_numeric, errors="coerce")
df["avg_delta_logit"] = df[dl_cols].mean(axis=1, skipna=True)
variants = df[df["seq_type"].ne("none") & df["avg_delta_logit"].notna()].copy()

# --- hotspot overlap, per locus, from May exon coordinates ---
hs = pd.read_csv(HOTSPOTS, sep="\t").drop_duplicates(["chrom", "start", "end", "gene_name", "exon_number"])
loci = (variants.groupby("event_id")
        .agg(gene_exon=("gene_exon", "first"),
             start=("exon_start_hg38", "first"), end=("exon_end_hg38", "first"),
             n=("Reference", "size"))
        .reset_index())
loci["chrom"] = loci["event_id"].str.split(":").str[0]
hit = []
for _, r in loci.iterrows():
    h = hs[(hs.chrom == r.chrom) & (hs.start < r.end) & (hs.end > r.start)]
    hit.append(not h.empty)
hot = loci[hit].copy()
hot["gene"] = hot["gene_exon"].astype(str).str.split().str[0]
hot["label"] = [fix_label(e, g) for e, g in zip(hot.event_id, hot.gene_exon)]

print(f"hotspot exons represented in the library: {len(hot)} loci from {hot.gene.nunique()} genes")
plot_loci = hot[hot.n >= MIN_VARIANTS].sort_values("event_id").reset_index(drop=True)
print(f"with >= {MIN_VARIANTS} measured variants: {len(plot_loci)} loci from {plot_loci.gene.nunique()} genes")
dup = plot_loci[plot_loci.label.duplicated(keep=False)]
if len(dup):
    raise SystemExit("two plotted loci share a label; add them to exon_label_fixes.py:\n"
                     + dup.to_string(index=False))

# --- variants + ClinVar ---
v = variants[variants.event_id.isin(plot_loci.event_id)].copy()
y_of = {e: i for i, e in enumerate(plot_loci.event_id)}
v["y"] = v.event_id.map(y_of) + np.random.default_rng(42).uniform(-0.2, 0.2, len(v))

cv = pd.read_csv(CLINVAR, sep="\t")
cv["ClinVar_item"] = cv["ClinVar_item"].apply(ast.literal_eval)
cv["CLNSIG_category"] = cv["ClinVar_item"].apply(ssm.parse_clinvar_significance)
cv = cv.drop_duplicates("mut_ref")
annot = v.merge(cv[["mut_ref", "CLNSIG_category"]], left_on="Reference", right_on="mut_ref", how="inner")
annot = annot[annot.CLNSIG_category.isin(PLOT_ORDER)]
print(f"ClinVar-annotated SNVs plotted: {len(annot)}")
print(annot.CLNSIG_category.value_counts().to_string())

# --- plot ---
fig, ax = plt.subplots(figsize=(12, 0.45 * len(plot_loci) + 2))
ax.scatter(v.avg_delta_logit, v.y, color="lightgrey", alpha=0.6, s=20,
           edgecolor="none", label="All variants")
for cat in PLOT_ORDER:
    s = annot[annot.CLNSIG_category == cat]
    if len(s):
        ax.scatter(s.avg_delta_logit, s.y, color=CLINVAR_COLORS[cat], s=20, alpha=0.8,
                   edgecolor="black", linewidth=0.5, label=cat.replace("_", " "))
ax.set_yticks(range(len(plot_loci)))
ax.set_yticklabels([f"{r.label}, n={r.n}" for r in plot_loci.itertuples()])
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Average Δlogit(PSI) across cell lines")
ax.set_ylabel("Hotspot exons")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
for ext in ("pdf", "png"):
    out = f"{OUT_DIR}/hotspot_overlap_clinvar_stripplot_MAY.{ext}"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print("wrote", out)
plot_loci.to_csv(f"{OUT_DIR}/hotspot_loci_MAY.csv", index=False)
hot.to_csv(f"{OUT_DIR}/hotspot_loci_all_MAY.csv", index=False)
