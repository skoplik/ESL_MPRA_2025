#!/usr/bin/env python3
"""
Build a single merged prediction file combining all splicing model predictions
with COMPASS experimental delta logit(PSI) values.

Output: one row per variant (Reference), NaN where a model has no prediction.

Models included:
  - SpliceAI          : delta_spliceai_logit
  - AlphaGenome       : alphagenome_delta_logit
  - Pangolin          : pangolin_delta_logit
  - HAL               : hal_delta_logit  (computed from WT_PSI / MUT_PSI)
  - Baseline MMSplice : baseline_mmsplice_delta_logit
  - Retrained MMSplice: retrained_mmsplice_delta_logit  +  mmsplice_is_test (True/False/NaN)

Experimental columns (from COMPASS main data):
  - HeLa_delta_logit_pooled, K562_delta_logit_pooled, MCF7_delta_logit_pooled,
    HMC3_delta_logit_pooled, HEK_delta_logit_pooled
  - avg_delta_logit_pooled  (mean of the five above)

Filtering (applied before saving):
  - Variants only (snp != 'none')
  - WT PSI not exactly 0 or 1 in any cell line
  - At least one valid delta logit across cell lines
"""

import os
import numpy as np
import pandas as pd
from scipy.special import logit as scipy_logit

CELL_LINES   = ['HeLa', 'K562', 'MCF7', 'HMC3', 'HEK']
WT_PSI_COLS  = [f'{c}_wt_pooled_psi_raw'  for c in CELL_LINES]
DELTA_COLS   = [f'{c}_delta_logit_pooled' for c in CELL_LINES]

# ── Paths ──────────────────────────────────────────────────────────────────────
MAIN_DATA   = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz"

MODEL_DIR   = "/ESL/ESL_MPRA/Figure_3/MMSplice/model_output_csv"
RETRAINED_FULL = f"{MODEL_DIR}/retrained_mmsplice_model_predictions_dataset7_model10_04_09_26_updated.csv"
RETRAINED_TEST = f"{MODEL_DIR}/retrained_mmsplice_model_predictions_test_dataset7_model10_04_09_26_updated.csv"
BASELINE_FULL  = f"{MODEL_DIR}/retrained_mmsplice_baseline_aggregate_dataset7_model10_04_09_26_updated.csv"

SPLICEAI_FILE   = "/ESL/ESL_MPRA/Figure_3/SpliceAI/output/spliceai_merged_predictions.csv"
ALPHAGENOME_FILE = "/ESL/ESL_MPRA/Figure_3/AlphaGenome/alphagenome_16k_all_variants.tsv"
PANGOLIN_FILE   = "/ESL/ESL_MPRA/SI_figures/Pangolin/pangolin_variant_deltas_all_exons.tsv"
HAL_PRED_FILE   = "/ESL/ESL_MPRA/Figure_4/model_processing/outputs/hal/hal_predictions.tsv"

OUTDIR           = "/ESL/ESL_MPRA/Figure_3/model_comparison"
OUTFILE          = os.path.join(OUTDIR, "mega_pred_file.csv")
OUTFILE_FILTERED = os.path.join(OUTDIR, "mega_pred_file_filtered.csv")

PSI_CLIP = 1e-2

# ── Load COMPASS base data ─────────────────────────────────────────────────────
print("Loading COMPASS main data...")
KEEP_COLS = (["Reference", "event_id", "gene_exon", "snp", "source", "seq_type"]
             + WT_PSI_COLS + DELTA_COLS)

main = pd.read_csv(MAIN_DATA, low_memory=False, usecols=KEEP_COLS)
main["avg_delta_logit_pooled"] = main[DELTA_COLS].mean(axis=1, skipna=True)
print(f"  {len(main):,} total rows")

mega = main.copy()

# ── SpliceAI ───────────────────────────────────────────────────────────────────
print("Loading SpliceAI...")
spliceai = pd.read_csv(SPLICEAI_FILE, usecols=["Reference", "delta_spliceai_logit"])
spliceai["Reference"] = spliceai["Reference"].astype(int)
spliceai = spliceai.rename(columns={"delta_spliceai_logit": "spliceai_delta_logit"})
before = len(mega)
mega = mega.merge(spliceai.drop_duplicates("Reference"), on="Reference", how="left")
print(f"  Matched: {mega['spliceai_delta_logit'].notna().sum():,} / {before:,}")

# ── AlphaGenome ────────────────────────────────────────────────────────────────
print("Loading AlphaGenome...")
ag = pd.read_csv(ALPHAGENOME_FILE, sep="\t", usecols=["Reference", "alphagenome_delta_logit"])
ag["Reference"] = ag["Reference"].astype(int)
ag = ag.drop_duplicates("Reference")
mega = mega.merge(ag, on="Reference", how="left")
print(f"  Matched: {mega['alphagenome_delta_logit'].notna().sum():,} / {before:,}")

# ── Pangolin ───────────────────────────────────────────────────────────────────
print("Loading Pangolin...")
pan = pd.read_csv(PANGOLIN_FILE, sep="\t", usecols=["Reference", "pangolin_delta_logit"])
pan["Reference"] = pan["Reference"].astype(int)
pan = pan.drop_duplicates("Reference")
mega = mega.merge(pan, on="Reference", how="left")
print(f"  Matched: {mega['pangolin_delta_logit'].notna().sum():,} / {before:,}")

# ── HAL ────────────────────────────────────────────────────────────────────────
print("Loading HAL...")
hal = pd.read_csv(HAL_PRED_FILE, sep="\t",
                  usecols=["VARIANT_NAME", "WT_PSI", "MUT_PSI"])
hal["Reference"] = hal["VARIANT_NAME"].str.replace("Seq_", "", regex=False).astype(int)
# PSI is on 0-100 scale → convert to fraction, clip, compute delta logit
hal["wt_frac"]  = (hal["WT_PSI"]  / 100).clip(PSI_CLIP, 1 - PSI_CLIP)
hal["mut_frac"] = (hal["MUT_PSI"] / 100).clip(PSI_CLIP, 1 - PSI_CLIP)
hal["hal_delta_logit"] = scipy_logit(hal["mut_frac"]) - scipy_logit(hal["wt_frac"])
hal = hal[["Reference", "hal_delta_logit"]].drop_duplicates("Reference")
mega = mega.merge(hal, on="Reference", how="left")
print(f"  Matched: {mega['hal_delta_logit'].notna().sum():,} / {before:,}")

# ── Baseline MMSplice ──────────────────────────────────────────────────────────
print("Loading Baseline MMSplice (full)...")
base = pd.read_csv(BASELINE_FULL, usecols=["Variant_ID", "Retrained_MMSplice_Predicted_Delta_Logit"])
base = base.rename(columns={
    "Variant_ID": "Reference",
    "Retrained_MMSplice_Predicted_Delta_Logit": "baseline_mmsplice_delta_logit"
})
base["Reference"] = base["Reference"].astype(int)
base = base.drop_duplicates("Reference")
mega = mega.merge(base, on="Reference", how="left")
print(f"  Matched: {mega['baseline_mmsplice_delta_logit'].notna().sum():,} / {before:,}")

# ── Retrained MMSplice ─────────────────────────────────────────────────────────
print("Loading Retrained MMSplice (full + test)...")
ret_full = pd.read_csv(RETRAINED_FULL, usecols=["Variant_ID", "Retrained_MMSplice_Predicted_Delta_Logit"])
ret_full = ret_full.rename(columns={
    "Variant_ID": "Reference",
    "Retrained_MMSplice_Predicted_Delta_Logit": "retrained_mmsplice_delta_logit"
})
ret_full["Reference"] = ret_full["Reference"].astype(int)
ret_full = ret_full.drop_duplicates("Reference")

ret_test = pd.read_csv(RETRAINED_TEST, usecols=["Variant_ID"])
test_refs = set(ret_test["Variant_ID"].astype(int))
ret_full["mmsplice_is_test"] = ret_full["Reference"].isin(test_refs)

mega = mega.merge(ret_full, on="Reference", how="left")
print(f"  Matched retrained: {mega['retrained_mmsplice_delta_logit'].notna().sum():,} / {before:,}")
print(f"  Of those, test set: {(mega['mmsplice_is_test'] == True).sum():,}")

# ── Final column order ─────────────────────────────────────────────────────────
MODEL_COLS = ["spliceai_delta_logit", "alphagenome_delta_logit", "pangolin_delta_logit",
              "hal_delta_logit", "baseline_mmsplice_delta_logit",
              "retrained_mmsplice_delta_logit", "mmsplice_is_test"]
col_order = (
    ["Reference", "event_id", "gene_exon", "snp", "source", "seq_type"]
    + WT_PSI_COLS + DELTA_COLS
    + ["avg_delta_logit_pooled"]
    + MODEL_COLS
)
mega = mega[col_order]

mega.to_csv(OUTFILE, index=False)
print(f"\nSaved: {OUTFILE}  ({len(mega):,} rows, {len(mega.columns)} columns)")

# ── Filtered version ───────────────────────────────────────────────────────────
print("\nApplying filter...")
# 1. Variants only
filt = mega[mega["snp"] != "none"].copy()
print(f"  After snp != 'none':                  {len(filt):,}")
# 2. Remove WT PSI edge cases (exactly 0 or 1 in any cell line)
edge_mask = filt[WT_PSI_COLS].apply(lambda row: (row == 0).any() or (row == 1).any(), axis=1)
filt = filt[~edge_mask]
print(f"  After removing WT PSI edge cases:     {len(filt):,}")
# 3. Must have at least one valid experimental delta logit
filt = filt[filt[DELTA_COLS].notna().any(axis=1)]
print(f"  After requiring ≥1 valid delta logit: {len(filt):,}")

filt.to_csv(OUTFILE_FILTERED, index=False)
print(f"\nSaved: {OUTFILE_FILTERED}  ({len(filt):,} rows, {len(filt.columns)} columns)")

print("\nCoverage in filtered set:")
for col in MODEL_COLS[:-1]:  # skip mmsplice_is_test
    n = filt[col].notna().sum()
    print(f"  {col:40s}: {n:,}")
