"""
Build SI table for Type B (transcript annotation inconsistency) rows.

Recomputes PSI at the reference junction for all affected rows (both the
annotation-mismatched variants and their references), and saves as
SI_boundary_mismatch.csv.

Type B rows are cases where a variant construct was assigned to a different
Gencode transcript than its reference construct, giving the two rows different
annotated intron1/exon boundaries for the same event_id. This is an annotation
artifact — the boundary difference is not caused by the variant sequence.
Because the variant and reference were originally measured at different
junctions, delta logit was not a valid comparison. The SI table documents PSI
for all affected rows measured at the reference junction (so variant and
reference use the same junction), providing traceability.

The main rebuilt data file (via merge_psi_04_17_2026_type_b_corrected.py)
already applies this correction at source.

Output: /ESL/Figures_SK/ambiguous_sjs/SI_boundary_mismatch.csv
"""

import pickle
import pandas as pd
import numpy as np
from scipy.special import logit as scipy_logit

# ── Paths ──────────────────────────────────────────────────────────────────
MAIN_CSV = "/ESL/Figures_SK/General_preprocessing/output_04_17_2026/04_17_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"
BM_CSV   = "/ESL/Figures_SK/ambiguous_sjs/boundary_mismatches.csv"
SI_OUT   = "/ESL/Figures_SK/ambiguous_sjs/SI_boundary_mismatch.csv"

PKL_BASE = "/ESL/Analysis/STAR_alignment/separate_concat_2023_09_19_d1c_ms75_from_s3_2023_11_27/recount_SJs"
PKL_2024  = PKL_BASE + "/2024_07_26"
WT_BASE   = "/ESL/Analysis/WT_Library/separate/recount_SJs"

REP_PKLS = {
    "HeLa_rep1":   PKL_BASE + "/HeLa_Rep1_separate_splicing_profiles/HeLa_Rep1_separate_all_splicing_counts.p",
    "HeLa_rep2":   PKL_BASE + "/HeLa_Rep3_20231031_separate_splicing_profiles/HeLa_Rep3_20231031_separate_all_splicing_counts.p",
    "K562_rep1":   PKL_BASE + "/K562_Rep1_separate_splicing_profiles/K562_Rep1_separate_all_splicing_counts.p",
    "K562_rep2":   PKL_BASE + "/K562_Rep3_20231031_separate_splicing_profiles/K562_Rep3_20231031_separate_all_splicing_counts.p",
    "MCF7_rep1":   PKL_2024  + "/MCF7_Rep1_separate_splicing_profiles/MCF7_Rep1_all_splicing_counts.p",
    "MCF7_rep2":   PKL_2024  + "/MCF7_Rep2_separate_splicing_profiles/MCF7_Rep2_all_splicing_counts.p",
    "HMC3_rep1":   PKL_2024  + "/HMC3_Rep1_separate_splicing_profiles/HMC3_Rep1_all_splicing_counts.p",
    "HMC3_rep2":   PKL_2024  + "/HMC3_Rep2_separate_splicing_profiles/HMC3_Rep2_all_splicing_counts.p",
    "HEK_rep1":    PKL_BASE + "/HEK293_Rep2_separate_splicing_profiles/HEK293_Rep2_separate_all_splicing_counts.p",
    "HEK_rep2":    PKL_BASE + "/HEK293_Rep3_separate_splicing_profiles/HEK293_Rep3_separate_all_splicing_counts.p",
    "HEK_wt_rep1": WT_BASE  + "/HEK_WT_Rep1_20231101_separate_splicing_profiles/HEK_WT_Rep1_20231101_separate_all_splicing_counts.p",
    "HEK_wt_rep2": WT_BASE  + "/HEK_WT_Rep2_20231101_separate_splicing_profiles/HEK_WT_Rep2_20231101_separate_all_splicing_counts.p",
}

CELLS     = ["HeLa", "K562", "MCF7", "HMC3", "HEK"]
CELL_REPS = {
    "HeLa": ["HeLa_rep1", "HeLa_rep2"],
    "K562": ["K562_rep1", "K562_rep2"],
    "MCF7": ["MCF7_rep1", "MCF7_rep2"],
    "HMC3": ["HMC3_rep1", "HMC3_rep2"],
    "HEK":  ["HEK_rep1",  "HEK_rep2"],
}
HEK_WT_REPS = ["HEK_wt_rep1", "HEK_wt_rep2"]

SHARED_5P = 286
MINCOV    = 10
CLIP      = 1e-2
E_JXN     = (26, 871)


def junctions(i1_len, ex_len):
    return (26, SHARED_5P + i1_len), (SHARED_5P + i1_len + ex_len + 1, 871)

def clip_logit(psi):
    return float(scipy_logit(np.clip(psi, CLIP, 1 - CLIP))) if not np.isnan(psi) else np.nan

def psi_at(pkl_dict, fasta_ref, i1, i2):
    if fasta_ref not in pkl_dict:
        return np.nan, 0, 0
    jd  = pkl_dict[fasta_ref]
    inc = min(jd.get(i1, 0), jd.get(i2, 0))
    exc = jd.get(E_JXN, 0)
    tot = inc + exc
    if tot < MINCOV:
        return np.nan, inc, exc
    return inc / tot, inc, exc

def compute_cell(fasta_ref, i1, i2, is_wt, cell):
    rep_names  = HEK_WT_REPS if (cell == "HEK" and is_wt) else CELL_REPS[cell]
    out        = {}
    pool_pairs = []
    for ri, rname in enumerate(rep_names, 1):
        p, inc, exc = psi_at(pkls[rname], fasta_ref, i1, i2)
        out[f"{cell}_rep{ri}_included"]    = inc + exc
        out[f"{cell}_rep{ri}_excluded"]    = exc
        out[f"{cell}_rep{ri}_psi_raw"]     = p
        out[f"{cell}_rep{ri}_psi_clipped"] = float(np.clip(p, CLIP, 1-CLIP)) if not np.isnan(p) else np.nan
        out[f"{cell}_rep{ri}_logit"]       = clip_logit(p)
        if not np.isnan(p):
            pool_pairs.append((inc, exc))

    if len(pool_pairs) >= 2:
        ti  = sum(v[0] for v in pool_pairs)
        te  = sum(v[1] for v in pool_pairs)
        tot = ti + te
        pool = ti / tot if tot >= MINCOV else np.nan
        out[f"{cell}_pooled_included"] = ti
        out[f"{cell}_pooled_excluded"] = te
        out[f"{cell}_total_pooled"]    = tot
    else:
        pool = np.nan
        out[f"{cell}_pooled_included"] = np.nan
        out[f"{cell}_pooled_excluded"] = np.nan
        out[f"{cell}_total_pooled"]    = np.nan

    out[f"{cell}_pooled_psi_raw"]     = pool
    out[f"{cell}_pooled_psi_clipped"] = float(np.clip(pool, CLIP, 1-CLIP)) if not np.isnan(pool) else np.nan
    out[f"{cell}_pooled_logit"]       = clip_logit(pool)
    return out


print("Loading data...")
main = pd.read_csv(MAIN_CSV, low_memory=False)
bm   = pd.read_csv(BM_CSV)

ref_jxn_map = {}
for eid in bm["event_id"].unique():
    row = bm[bm["event_id"] == eid].iloc[0]
    ref_jxn_map[eid] = {
        "intron1_len":   int(row["ref_intron1_len"]),
        "exon_len":      int(row["ref_exon_len"]),
        "ref_Reference": int(row["ref_Reference"]),
    }

b_var_refs = set(bm["Reference"].astype(int))
b_ref_refs = set(bm["ref_Reference"].dropna().astype(int))
b_all_refs = b_var_refs | b_ref_refs

ref_rows = main[main["Reference"].isin(b_ref_refs) & (main["snp"] == "none")]
ref_seqs  = ref_rows.set_index("event_id")[["intron1", "exon", "intron2"]]

print("Loading pickles...")
pkls = {}
for name, path in REP_PKLS.items():
    with open(path, "rb") as f:
        pkls[name] = pickle.load(f)
    print(f"  {name}: {len(pkls[name]):,} refs")

print("\nRecomputing PSI at reference junction...")
affected_rows = main[main["Reference"].isin(b_all_refs)].copy()
records = []

for _, row in affected_rows.iterrows():
    eid   = row["event_id"]
    if eid not in ref_jxn_map:
        continue
    rj    = ref_jxn_map[eid]
    fref  = int(row["Reference"])
    is_wt = (row["snp"] == "none")
    i1, i2 = junctions(rj["intron1_len"], rj["exon_len"])

    ref_seq = ref_seqs.loc[eid] if eid in ref_seqs.index else None

    rec = {"Reference": row["Reference"],
           "event_id":  eid,
           "gene_exon": row["gene_exon"],
           "snp":       row["snp"],
           "source":    row["source"],
           "seq_type":  row["seq_type"],
           "intron1":   ref_seq["intron1"]  if ref_seq is not None else row["intron1"],
           "exon":      ref_seq["exon"]     if ref_seq is not None else row["exon"],
           "intron2":   ref_seq["intron2"]  if ref_seq is not None else row["intron2"],
           "full_seq":  row["full_seq"],
           }
    for cell in CELLS:
        rec.update(compute_cell(fref, i1, i2, is_wt, cell))
    records.append(rec)

corrected = pd.DataFrame(records)

wt_lookup = corrected[corrected["snp"] == "none"]
for cell in CELLS:
    wt_psi   = wt_lookup.groupby("event_id")[f"{cell}_pooled_psi_raw"].mean()
    wt_logit = wt_lookup.groupby("event_id")[f"{cell}_pooled_logit"].mean()
    corrected[f"{cell}_wt_pooled_psi_raw"]  = corrected["event_id"].map(wt_psi)
    corrected[f"{cell}_wt_pooled_logit"]    = corrected["event_id"].map(wt_logit)
    corrected[f"{cell}_dpsi_pooled"] = np.where(
        corrected["snp"] == "none", np.nan,
        corrected[f"{cell}_pooled_psi_raw"] - corrected[f"{cell}_wt_pooled_psi_raw"]
    )
    corrected[f"{cell}_delta_logit_pooled"] = np.where(
        corrected["snp"] == "none", np.nan,
        corrected[f"{cell}_pooled_logit"] - corrected[f"{cell}_wt_pooled_logit"]
    )

psi_cols = [f"{c}_pooled_psi_raw" for c in CELLS]
si_df = corrected[corrected[psi_cols].notna().any(axis=1)].copy()

print(f"SI rows: {len(si_df):,}  "
      f"({si_df['event_id'].nunique()} events, "
      f"{(si_df['snp']=='none').sum()} refs, "
      f"{(si_df['snp']!='none').sum()} variants)")

si_df.to_csv(SI_OUT, index=False)
print(f"Saved SI: {SI_OUT}")
