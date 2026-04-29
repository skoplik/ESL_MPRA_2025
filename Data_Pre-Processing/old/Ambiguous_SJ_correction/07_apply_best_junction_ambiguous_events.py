"""
For each ambiguous event (multiple transcript isoforms with different splice junctions),
determine the best junction to use for PSI calculation:
  - Check the primary reference construct's PSI at each transcript's junction
  - Among junctions with sufficient reads, prefer MANE Select > MANE Plus Clinical > lowest Reference
  - Apply that junction uniformly to ALL rows in the event (all refs and all variants)

This fixes cases like KCTD10 exon 4 where:
  - The primary reference's annotated junction has no reads
  - But the MANE junction has abundant reads on the same construct

Updates:
  1. Main NO_DELTAS CSV: recomputes all PSI/logit/delta columns for ambiguous event rows
  2. Supertable: updates transcript_id, mane_status, exon_start_hg38, exon_end_hg38
     to reflect the chosen transcript for all rows in each ambiguous event
     (intron1/exon/full_seq sequences are left as-is — they reflect the actual construct)
  3. WITH_WT CSV: rebuilt from corrected NO_DELTAS

Outputs to: /ESL/Figures_SK/General_preprocessing/output_04_24_2026/
"""

import pickle
import os
import numpy as np
import pandas as pd
from scipy.special import logit as scipy_logit

# ── Paths ──────────────────────────────────────────────────────────────────
SUPERTABLE    = "/ESL/Figures_SK/General_preprocessing/output_04_24_2026/st_working_04_24_2026.csv"
MAIN_CSV      = "/ESL/Figures_SK/General_preprocessing/output_03_16_2026/03_16_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"
OUT_DIR       = "/ESL/Figures_SK/General_preprocessing/output_04_24_2026"
OUT_PREFIX    = "04_24_2026_1e-2"
ST_OUT        = "/ESL/Figures_SK/General_preprocessing/output_04_24_2026/st_04_24_2026.csv"

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

CELLS = ["HeLa", "K562", "MCF7", "HMC3", "HEK"]
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


# ── Helpers ────────────────────────────────────────────────────────────────
def junctions(i1_len, ex_len):
    i1 = (26, SHARED_5P + i1_len)
    i2 = (SHARED_5P + i1_len + ex_len + 1, 871)
    return i1, i2

def clip_logit(psi):
    if np.isnan(psi):
        return np.nan
    return float(scipy_logit(np.clip(psi, CLIP, 1 - CLIP)))

def psi_at(pkl_dict, fasta_ref, i1, i2):
    """Return (psi, included_reads, excluded_reads)."""
    if fasta_ref not in pkl_dict:
        return np.nan, 0, 0
    jd  = pkl_dict[fasta_ref]
    inc = min(jd.get(i1, 0), jd.get(i2, 0))
    exc = jd.get(E_JXN, 0)
    tot = inc + exc
    if tot < MINCOV:
        return np.nan, inc, exc
    return inc / tot, inc, exc

def mane_priority(mane_status):
    if mane_status == "MANE Select":
        return 0
    if mane_status == "MANE Plus Clinical":
        return 1
    return 2


# ── Load data ──────────────────────────────────────────────────────────────
print("Loading supertable...")
st = pd.read_csv(SUPERTABLE, low_memory=False)
st["intron1_len"] = st["intron1"].str.len()
st["exon_len"]    = st["exon"].str.len()

print("Loading main CSV...")
main = pd.read_csv(MAIN_CSV, low_memory=False)
print(f"  Rows: {len(main):,}")

# Identify ambiguous events: >1 distinct intron1 length in supertable
intron_counts = st.groupby("event_id")["intron1_len"].nunique()
ambig_eids = set(intron_counts[intron_counts > 1].index)
print(f"Ambiguous events: {len(ambig_eids)}")

print("Loading pickles...")
pkls = {}
for name, path in REP_PKLS.items():
    with open(path, "rb") as f:
        pkls[name] = pickle.load(f)
    print(f"  {name}: {len(pkls[name]):,} refs")


# ── For each ambiguous event: determine best junction ─────────────────────
print("\nDetermining best junction per ambiguous event...")

# transcript info per event: list of (mane_priority, Reference_0idx, intron1_len, exon_len,
#                                      transcript_id, mane_status, exon_start, exon_end)
event_transcripts = {}
for event_id in ambig_eids:
    sub = (st[st["event_id"] == event_id]
           .drop_duplicates("transcript_id")
           .copy())
    sub["_mp"] = sub["mane_status"].map(mane_priority)
    sub = sub.sort_values(["_mp", "Reference"])
    txs = []
    for _, row in sub.iterrows():
        txs.append({
            "mane_priority":   mane_priority(str(row.get("mane_status", ""))),
            "Reference_0":     int(row["Reference"]),
            "Reference_1":     int(row["Reference"]) + 1,
            "intron1_len":     int(row["intron1_len"]),
            "exon_len":        int(row["exon_len"]),
            "transcript_id":   row["transcript_id"],
            "mane_status":     str(row.get("mane_status", "")),
            "exon_start_hg38": row["exon_start_hg38"],
            "exon_end_hg38":   row["exon_end_hg38"],
        })
    event_transcripts[event_id] = txs

# Find primary reference (snp==none, lowest Reference) for each event
primary_refs = {}
for eid in ambig_eids:
    sub = st[(st["event_id"] == eid) & (st["snp"] == "none")].sort_values("Reference")
    if len(sub):
        r = sub.iloc[0]
        primary_refs[eid] = int(r["Reference"]) + 1  # 1-indexed for pickle lookup

# For the primary reference construct, check PSI at each transcript's junction
# across all reps; pick best junction (MANE preferred, must have reads in >= 2 reps)
chosen_junction = {}   # event_id -> {i1, i2, transcript_id, mane_status, exon_start, exon_end}

for eid, txs in event_transcripts.items():
    pref_ref = primary_refs.get(eid)
    if pref_ref is None:
        continue

    candidates = []
    for tx in txs:
        i1, i2 = junctions(tx["intron1_len"], tx["exon_len"])
        # Count reps with sufficient coverage across all cell lines
        valid_reps = 0
        for rep_name, pkl in pkls.items():
            p, _, _ = psi_at(pkl, pref_ref, i1, i2)
            if not np.isnan(p):
                valid_reps += 1
        candidates.append({
            **tx,
            "i1": i1,
            "i2": i2,
            "valid_reps": valid_reps,
        })

    # Among junctions with >=2 valid reps, pick MANE > non-MANE > first by Reference
    valid = [c for c in candidates if c["valid_reps"] >= 2]
    if not valid:
        # Fall back to any junction with >=1 rep
        valid = [c for c in candidates if c["valid_reps"] >= 1]
    if not valid:
        print(f"  WARNING: {eid} — no junction with reads; skipping")
        continue

    best = sorted(valid, key=lambda c: (c["mane_priority"], c["Reference_0"]))[0]
    chosen_junction[eid] = best
    print(f"  {eid}: chose {best['transcript_id']} "
          f"(MANE={best['mane_status'] or 'none'}) "
          f"i1={best['i1']} valid_reps={best['valid_reps']}")


# ── Recompute PSI for all ambiguous-event rows at chosen junction ──────────
print("\nRecomputing PSI for ambiguous event rows...")

ambig_mask = main["event_id"].isin(chosen_junction.keys())
ambig_rows = main[ambig_mask].copy()
print(f"  Rows to recompute: {len(ambig_rows):,}")

def compute_row_psijs(fasta_ref, i1, i2, is_wt, cell):
    """Return dict of per-rep and pooled PSI columns for one cell line."""
    rep_names = HEK_WT_REPS if (cell == "HEK" and is_wt) else CELL_REPS[cell]
    out = {}
    pool_pairs = []

    for ri, rname in enumerate(rep_names, 1):
        p, inc, exc = psi_at(pkls[rname], fasta_ref, i1, i2)
        out[f"{cell}_rep{ri}_included"]    = float(inc) if not np.isnan(p) else np.nan
        out[f"{cell}_rep{ri}_excluded"]    = float(exc) if not np.isnan(p) else np.nan
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
        out[f"{cell}_pooled_included"] = float(ti)
        out[f"{cell}_pooled_excluded"] = float(te)
        out[f"{cell}_total_pooled"]    = float(tot)
    else:
        pool = np.nan
        out[f"{cell}_pooled_included"] = np.nan
        out[f"{cell}_pooled_excluded"] = np.nan
        out[f"{cell}_total_pooled"]    = np.nan

    out[f"{cell}_pooled_psi_raw"]     = pool
    out[f"{cell}_pooled_psi_clipped"] = float(np.clip(pool, CLIP, 1-CLIP)) if not np.isnan(pool) else np.nan
    out[f"{cell}_pooled_logit"]       = clip_logit(pool)
    return out

records = []
for _, row in ambig_rows.iterrows():
    eid = row["event_id"]
    cj  = chosen_junction[eid]
    i1, i2 = cj["i1"], cj["i2"]
    fasta_ref = int(row["Reference"])   # main CSV is already 1-indexed
    is_wt = (row["snp"] == "none")

    rec = {"Reference": row["Reference"]}
    for cell in CELLS:
        rec.update(compute_row_psijs(fasta_ref, i1, i2, is_wt, cell))
    records.append(rec)

corrected = pd.DataFrame(records)

# Compute WT pooled PSI per event/cell and delta columns
for cell in CELLS:
    wt_rows = corrected[corrected["Reference"].isin(
        main.loc[(main["event_id"].isin(chosen_junction.keys())) & (main["snp"] == "none"), "Reference"]
    )]
    wt_rows = wt_rows.copy()
    wt_rows["event_id"] = wt_rows["Reference"].map(
        main.set_index("Reference")["event_id"]
    )
    wt_psi_map   = wt_rows.groupby("event_id")[f"{cell}_pooled_psi_raw"].mean()
    wt_logit_map = wt_rows.groupby("event_id")[f"{cell}_pooled_logit"].mean()

    corrected["_event_id"] = corrected["Reference"].map(main.set_index("Reference")["event_id"])
    corrected[f"{cell}_wt_pooled_psi_raw"] = corrected["_event_id"].map(wt_psi_map)
    corrected[f"{cell}_wt_pooled_logit"]   = corrected["_event_id"].map(wt_logit_map)

    is_wt_mask = corrected["Reference"].isin(
        main.loc[main["snp"] == "none", "Reference"]
    )
    corrected[f"{cell}_dpsi_pooled"] = np.where(
        is_wt_mask, np.nan,
        corrected[f"{cell}_pooled_psi_raw"] - corrected[f"{cell}_wt_pooled_psi_raw"]
    )
    corrected[f"{cell}_delta_logit_pooled"] = np.where(
        is_wt_mask, np.nan,
        corrected[f"{cell}_pooled_logit"] - corrected[f"{cell}_wt_pooled_logit"]
    )

corrected = corrected.drop(columns=["_event_id"])

# Recompute SD columns per cell
for cell in CELLS:
    rep_psi_cols   = [c for c in corrected.columns if c.startswith(f"{cell}_rep") and c.endswith("_psi_raw")]
    rep_logit_cols = [c for c in corrected.columns if c.startswith(f"{cell}_rep") and c.endswith("_logit")]
    if len(rep_psi_cols) >= 2:
        corrected[f"{cell}_sd_psi"]   = corrected[rep_psi_cols].std(axis=1, ddof=1)
        corrected[f"{cell}_sd_logit"] = corrected[rep_logit_cols].std(axis=1, ddof=1)
    corrected[f"{cell}_sd_dpsi"]         = np.nan  # recomputed below if possible
    corrected[f"{cell}_sd_delta_logit"]  = np.nan


# ── Merge corrections back into main ──────────────────────────────────────
print("Merging corrections into main table...")

# Columns that corrected provides (all PSI-related for ambiguous rows)
psi_cols = [c for c in corrected.columns if c != "Reference"]

main_out = main.copy()
corrected_indexed = corrected.set_index("Reference")

for col in psi_cols:
    if col in main_out.columns:
        main_out.loc[ambig_mask, col] = main_out.loc[ambig_mask, "Reference"].map(
            corrected_indexed[col]
        )

# ── Save corrected NO_DELTAS ───────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
no_deltas_path = os.path.join(OUT_DIR, f"{OUT_PREFIX}_ALL_WTS_VARS_NO_DELTAS.csv")
main_out.to_csv(no_deltas_path, index=False)
print(f"Saved NO_DELTAS: {no_deltas_path}")


# ── Rebuild WITH_WT ────────────────────────────────────────────────────────
print("Rebuilding WITH_WT...")
wt_eids  = set(main_out.loc[main_out["snp"] == "none",  "event_id"])
var_eids = set(main_out.loc[main_out["snp"] != "none",  "event_id"])
both_eids = wt_eids & var_eids
with_wt = main_out[main_out["event_id"].isin(both_eids)].copy()
with_wt_path = os.path.join(OUT_DIR, f"{OUT_PREFIX}_ALL_WITH_WT.csv")
with_wt.to_csv(with_wt_path, index=False)
print(f"Saved WITH_WT:   {with_wt_path}")
print(f"  Total rows: {len(with_wt):,}  "
      f"WT: {(with_wt['snp']=='none').sum():,}  "
      f"Var: {(with_wt['snp']!='none').sum():,}")


# ── Update supertable ──────────────────────────────────────────────────────
print("\nUpdating supertable...")
# Per-row transcript_id/mane_status/exon coordinates are PRESERVED (reflect
# the original construct annotation and are needed by the alt transcript SI table).
# New per-event columns record the chosen junction for PSI analysis:
#   chosen_junction_transcript_id, chosen_junction_mane_status,
#   chosen_junction_exon_start_hg38, chosen_junction_exon_end_hg38
st_out = st.copy()
st_out["chosen_junction_transcript_id"]   = st_out["transcript_id"]
st_out["chosen_junction_mane_status"]     = st_out.get("mane_status", "")
st_out["chosen_junction_exon_start_hg38"] = st_out["exon_start_hg38"]
st_out["chosen_junction_exon_end_hg38"]   = st_out["exon_end_hg38"]

changed_events = 0
for eid, cj in chosen_junction.items():
    mask = st_out["event_id"] == eid
    orig_primary_tx = st_out.loc[mask & (st_out["snp"] == "none"), "transcript_id"].iloc[0] if mask.any() else None

    st_out.loc[mask, "chosen_junction_transcript_id"]   = cj["transcript_id"]
    st_out.loc[mask, "chosen_junction_mane_status"]     = cj["mane_status"] if cj["mane_status"] else ""
    st_out.loc[mask, "chosen_junction_exon_start_hg38"] = cj["exon_start_hg38"]
    st_out.loc[mask, "chosen_junction_exon_end_hg38"]   = cj["exon_end_hg38"]

    if orig_primary_tx != cj["transcript_id"]:
        changed_events += 1
        print(f"  {eid}: junction chosen = {cj['transcript_id']} "
              f"(MANE={cj['mane_status'] or 'none'}, was primary={orig_primary_tx})")

print(f"Events with non-primary junction chosen: {changed_events}")
st_out.to_csv(ST_OUT, index=False)
print(f"Saved supertable: {ST_OUT}")

print("\nDone.")
