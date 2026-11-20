#!/usr/bin/env python3
"""
Regenerate the SCN5A exon 24 NSSM map (Figure 4E) from the MAY 2026 supertable.

Reuses the plotting code in ssm_single_muts_logits_06_04_2025_with_clinvar_patched.py
unchanged; this only supplies the May input and filters to one exon, instead of
looping over every event_id (which emits thousands of PDFs).

Note the repo runner still points at the JULY 2025 table and at the old
/ESL/Figures_SK copy of the script, so it does NOT reproduce current data.
"""
import os, ast, argparse, importlib.util
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend import Legend

# The plotting script predates matplotlib 3.9, which renamed Legend.legendHandles
# to .legend_handles. Restore the old name so the script runs unmodified.
if tuple(int(x) for x in matplotlib.__version__.split(".")[:2]) >= (3, 9):
    Legend.legendHandles = property(lambda self: self.legend_handles)

SCRIPT     = "/ESL/ESL_MPRA/Figure_4/ssm_single_muts_logits_06_04_2025_with_clinvar_patched.py"
SUPERTABLE = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz"
CLINVAR    = "/ESL/ESL_MPRA/Figure_4/inputs/supertable_ClinVar_matched_in_supertable.txt"

ap = argparse.ArgumentParser()
ap.add_argument("--gene-exon", default="scn5a exon 24")
ap.add_argument("--supertable", default=SUPERTABLE)
ap.add_argument("--clinvar", default=CLINVAR)
ap.add_argument("--script", default=SCRIPT)
ap.add_argument("--outdir", default="/ESL/ESL_MPRA/Figure_4/outputs/scn5a_ssm_MAY")
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("ssm_mod", a.script)
ssm  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssm)          # top level is imports + defs only; main() is guarded

print(f"[LOAD] supertable {a.supertable}")
df = pd.read_csv(a.supertable, low_memory=False)
print(f"       {len(df):,} rows")
print(f"[LOAD] clinvar {a.clinvar}")
dfc = pd.read_csv(a.clinvar, sep="\t")
dfc["ClinVar_item"] = dfc["ClinVar_item"].apply(ast.literal_eval)

merged = df.merge(dfc, left_on="Reference", right_on="mut_ref", how="left")
sub = merged[merged["gene_exon"].astype(str).str.lower().str.strip() == a.gene_exon.lower().strip()]
print(f"[FILTER] {a.gene_exon}: {len(sub):,} rows, event_ids = {sorted(set(sub.event_id.astype(str)))}")
if sub.empty:
    hits = [g for g in merged["gene_exon"].dropna().unique() if a.gene_exon.split()[0].lower() in str(g).lower()]
    raise SystemExit(f"no rows. candidates containing that gene: {hits[:10]}")

try:
    cmap = matplotlib.colormaps["coolwarm"].copy()
except (AttributeError, KeyError):
    cmap = plt.cm.get_cmap("coolwarm").copy()
cmap.set_bad("white")
cmap_red = LinearSegmentedColormap.from_list("white_red", ["white", "red"])

os.makedirs(a.outdir, exist_ok=True)
for eid, group in sub.groupby("event_id"):
    print(f"[RUN] {eid}  ({len(group):,} rows)")
    ssm.make_heatmap(group, a.outdir, cmap, cmap_red)

print("\n[DONE] files written:")
for root, _, files in os.walk(a.outdir):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f"   {os.path.getsize(p):>8,}  {p}")
