#!/usr/bin/env python3
"""SI figure: dlogit(PSI) distributions for highly saturated exons (>= 300 variants).

May 2026 rebuild of acmg_overlap_no_overlap.py (the ">=300, all variants" plot).
Changes from the original:
  * Counted per locus (event_id), not per gene_exon label, and only measured
    non-reference variants are counted - the same rule as Supplementary Table 1
    (SI_figures/build_supplementary_table_1.py), so the figure and the table list
    the same exons with the same n.
  * Labels go through SI_figures/exon_label_fixes.py (MANE numbering); the script
    fails if two plotted loci share a label.
  * ClinVar from the Figure 4 table, joined on Reference with no shift
    (verified 2166/2166 SNP matches at +0).
Prints the exon count quoted in the legend.
"""
import ast, importlib.util, os, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

REPO = "/ESL/ESL_MPRA"
sys.path.insert(0, f"{REPO}/SI_figures")
from exon_label_fixes import label as fix_label, primary_loci

MAIN = f"{REPO}/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz"
CLINVAR = f"{REPO}/Figure_4/inputs/supertable_ClinVar_matched_in_supertable.txt"
PANEL_SCRIPT = f"{REPO}/Figure_4/ssm_single_muts_logits_06_04_2025_with_clinvar_patched.py"
OUT_DIR = f"{REPO}/SI_figures/highly_saturated_exons/outputs"
CELLS = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]
MIN_VARIANTS = 300

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

spec = importlib.util.spec_from_file_location("ssm_mod", PANEL_SCRIPT)
ssm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssm)
CLINVAR_COLORS = ssm.CLINVAR_COLORS
PLOT_ORDER = ["Likely_benign", "Benign", "Conflicting", "Uncertain_significance",
              "Likely_pathogenic", "Pathogenic"]

psi_cols = [c + "_pooled_psi_raw" for c in CELLS]
dl_cols = [c + "_delta_logit_pooled" for c in CELLS]
df = pd.read_csv(MAIN, low_memory=False,
                 usecols=["Reference", "event_id", "event_id_161", "gene_exon", "seq_type"] + psi_cols + dl_cols)
df[dl_cols] = df[dl_cols].apply(pd.to_numeric, errors="coerce")
df["avg_delta_logit"] = df[dl_cols].mean(axis=1, skipna=True)
# same variant definition as Supplementary Table 1
variants = df[df["seq_type"].ne("none") & df[psi_cols].notna().any(axis=1)].copy()
variants = variants[variants["event_id"].isin(primary_loci(variants))]   # one locus per construct

loci = (variants.groupby("event_id")
        .agg(gene_exon=("gene_exon", "first"), n=("Reference", "size")).reset_index())
loci = loci[loci.n >= MIN_VARIANTS].copy()
loci["label"] = [fix_label(e, g) for e, g in zip(loci.event_id, loci.gene_exon)]
loci["gene"] = loci["label"].astype(str).str.split().str[0]
dup = loci[loci.label.duplicated(keep=False)]
if len(dup):
    raise SystemExit("two plotted loci share a label; add them to exon_label_fixes.py:\n"
                     + dup.to_string(index=False))
# alphabetical, A at the top
loci = loci.sort_values("label", ascending=False).reset_index(drop=True)
print(f"exons with >= {MIN_VARIANTS} variants: {len(loci)} loci from {loci.gene.nunique()} genes")

v = variants[variants.event_id.isin(loci.event_id)].copy()
v = v[v.avg_delta_logit.notna()]
y_of = {e: i for i, e in enumerate(loci.event_id)}
v["y"] = v.event_id.map(y_of) + np.random.default_rng(42).uniform(-0.2, 0.2, len(v))

cv = pd.read_csv(CLINVAR, sep="\t")
cv["ClinVar_item"] = cv["ClinVar_item"].apply(ast.literal_eval)
cv["CLNSIG_category"] = cv["ClinVar_item"].apply(ssm.parse_clinvar_significance)
cv = cv.drop_duplicates("mut_ref")
annot = v.merge(cv[["mut_ref", "CLNSIG_category"]], left_on="Reference", right_on="mut_ref")
annot = annot[annot.CLNSIG_category.isin(PLOT_ORDER)]
print(f"ClinVar-annotated variants plotted: {len(annot)}")

fig, ax = plt.subplots(figsize=(8.5, 11))
ax.scatter(v.avg_delta_logit, v.y, color="lightgrey", alpha=0.6, s=12,
           edgecolor="none", label="All variants")
for cat in PLOT_ORDER:
    s = annot[annot.CLNSIG_category == cat]
    if len(s):
        ax.scatter(s.avg_delta_logit, s.y, color=CLINVAR_COLORS[cat], s=12, alpha=0.8,
                   edgecolor="black", linewidth=0.5, label=cat.replace("_", " "))
ax.set_yticks(range(len(loci)))
ax.set_yticklabels([f"{r.label}, n={r.n}" for r in loci.itertuples()], fontsize=6)
ax.set_ylim(-1, len(loci) + 0.1)
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Average Δlogit(PSI) across cell lines")
ax.set_ylabel(f"Exons with ≥{MIN_VARIANTS} variants")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
for ext in ("pdf", "png"):
    out = f"{OUT_DIR}/saturated_exons_clinvar_stripplot_MAY.{ext}"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print("wrote", out)
loci.to_csv(f"{OUT_DIR}/saturated_exons_loci_MAY.csv", index=False)
