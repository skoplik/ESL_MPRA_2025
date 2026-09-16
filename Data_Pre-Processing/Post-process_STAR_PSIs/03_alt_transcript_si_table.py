"""
03_alt_transcript_si_table.py — 2026-05-01 rewrite.

Builds the SI alt-transcript PSI table for **Gencode-only alts** — Gencode
transcripts whose junctions fit the construct window AND have read coverage
but were NOT in the original supertable design.

Supertable-design alts (the duplicate rows of ambiguous events, e.g. KCTD10
84926, 84927) live in the main table now under their own re-keyed event_ids
and are NOT included here.

Inputs:
  - st_corrected.csv from Stage 1 (carries event_id_161 + new columns)
  - st_alt_junctions.csv from Stage 1 (Gencode-only alts; one row per
    (canonical_reference, alt_transcript_id) with junction coords)
  - 1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz from Stage 2 (per-row main data)
  - 12 per-replicate junction-count pickles
"""

import os
import pickle
import re
import pandas as pd
import numpy as np
from scipy.special import logit as scipy_logit

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_OUT           = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output"
CORRECTED_ST       = os.path.join(BASE_OUT, "st_corrected.csv")
ST_ALT_JUNCTIONS   = os.path.join(BASE_OUT, "st_alt_junctions.csv")
MAIN_CSV           = os.path.join(BASE_OUT, "1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz")
GTF                = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/gencode.v48.annotation.gtf"
OUTPUT_DIR         = os.path.join(BASE_OUT, "ambiguous_sjs")
OUTPUT_BASENAME    = "SI_alt_transcript_psi.csv"

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


# ── Helpers ────────────────────────────────────────────────────────────────
def junctions(i1_len, ex_len):
    return (26, SHARED_5P + i1_len), (SHARED_5P + i1_len + ex_len + 1, 871), (26, 871)


def get_psi(pkl, ref, i1, i2, e=(26, 871)):
    if ref not in pkl:
        return np.nan, np.nan, 0
    jd  = pkl[ref]
    inc = min(jd.get(i1, 0), jd.get(i2, 0))
    exc = jd.get(e, 0)
    tot = inc + exc
    if tot < MINCOV:
        return np.nan, np.nan, tot
    return inc / tot, inc / tot, tot


def clip_logit(psi):
    if np.isnan(psi):
        return np.nan
    return float(scipy_logit(np.clip(psi, CLIP, 1 - CLIP)))


# ── Parse MANE Select from GTF ────────────────────────────────────────────
print("Parsing MANE Select from GTF...")
mane_select = set()
mane_plus   = set()
with open(GTF) as _f:
    for _line in _f:
        if '\ttranscript\t' not in _line:
            continue
        _m = re.search(r'transcript_id "([^"]+)"', _line)
        if not _m:
            continue
        _tx = _m.group(1).split('.')[0]
        if 'MANE_Select' in _line:
            mane_select.add(_tx)
        if 'MANE_Plus_Clinical' in _line:
            mane_plus.add(_tx)


def mane_label(transcript_id):
    tx = str(transcript_id).split('.')[0]
    if tx in mane_select:
        return 'MANE Select'
    if tx in mane_plus:
        return 'MANE Plus Clinical'
    return ''


# ── Load corrected supertable + Gencode-only alt junctions ────────────────
print("Loading corrected supertable...")
st_corr = pd.read_csv(CORRECTED_ST, low_memory=False)

# Per full_seq → sorted list of supertable Refs, converted to NO_DELTAS-Ref
# space (+1 systematic offset). Used in SI to expose other supertable rows
# sharing the same sequence (MANE-preferred row is `Reference`; others list
# in `alt_supertable_refs`).
supertable_refs_by_fseq = (
    st_corr.groupby("full_seq")["Reference"]
           .agg(lambda s: sorted((int(x) + 1) for x in s.dropna().unique()))
           .to_dict()
)

print("Loading Gencode-only alt junctions table...")
alt_jxns_df = pd.read_csv(ST_ALT_JUNCTIONS)
print(f"  Gencode-only alts: {len(alt_jxns_df)}")

# MANE priority helper (used both for picking the dedup row per full_seq and
# for picking which transcript_id labels each candidate junction).
def _mane_priority(tx_id):
    base = str(tx_id).split(".")[0]
    if base in mane_select:
        return 0
    if base in mane_plus:
        return 1
    return 2

# Build candidate alt junctions per event_id_161. Each unique (intron1_len,
# exon_len) pair becomes one candidate, labeled by its preferred transcript:
# supertable transcripts beat Gencode-only alts; within supertable, MANE Select
# beats MANE Plus Clinical beats other (lowest Reference tiebreak).
candidates_per_event = {}
for ev, grp in st_corr.groupby("event_id_161"):
    g = grp.drop_duplicates("transcript_id").copy()
    g["_mp"] = g["transcript_id"].apply(_mane_priority)
    g = g.sort_values(["_mp", "Reference"])
    cands = {}
    for _, r in g.iterrows():
        i1 = r.get("intron1_len")
        ex = r.get("exon_len")
        if pd.notna(i1) and pd.notna(ex):
            key = (int(i1), int(ex))
            if key not in cands:
                cands[key] = {
                    "intron1_len":     int(i1),
                    "exon_len":        int(ex),
                    "transcript_id":   r["transcript_id"],
                    "exon_start_hg38": r.get("exon_start_hg38"),
                    "exon_end_hg38":   r.get("exon_end_hg38"),
                    "alt_mane_status": r.get("mane_status", "") or mane_label(r["transcript_id"]),
                    "in_supertable":   True,
                    "canonical_reference": int(grp["Reference"].min()),
                }
    candidates_per_event[ev] = cands

# Layer in Gencode-only alts from st_alt_junctions.csv (skip if the junction
# already has a supertable representative — supertable wins).
n_gencode_added = 0
n_gencode_skipped = 0
for _, row in alt_jxns_df.iterrows():
    eid_161 = row["event_id_161"]
    key = (int(row["alt_intron1_len"]), int(row["alt_exon_len"]))
    cands = candidates_per_event.setdefault(eid_161, {})
    if key in cands:
        n_gencode_skipped += 1
        continue
    cands[key] = {
        "intron1_len":     key[0],
        "exon_len":        key[1],
        "transcript_id":   row["alt_transcript_id"],
        "exon_start_hg38": row["alt_exon_start_hg38"],
        "exon_end_hg38":   row["alt_exon_end_hg38"],
        "alt_mane_status": row.get("alt_mane_status", ""),
        "in_supertable":   False,
        "canonical_reference": int(row["canonical_reference"]),
    }
    n_gencode_added += 1
print(f"  Gencode-only alts added: {n_gencode_added}; skipped (junction already in supertable): {n_gencode_skipped}")
# Restrict to events that have >1 candidate junction (else there's nothing alt
# to score — the construct only has one possible junction).
candidates_per_event = {ev: c for ev, c in candidates_per_event.items() if len(c) > 1}
print(f"  Events with multiple candidate junctions: {len(candidates_per_event)}")

# Build full_seq → canonical Reference (0-indexed) map from supertable
canonical_ref_by_fseq = (
    st_corr.groupby("full_seq")["Reference"].min().astype(int).to_dict()
)


# ── Load main data ────────────────────────────────────────────────────────
print("Loading main data...")
main = pd.read_csv(MAIN_CSV, low_memory=False)
main["Reference_0"] = main["Reference"] - 1

# For each row in main: look up its canonical_reference (= pkl key - 1)
main["canonical_reference"] = main["full_seq"].map(canonical_ref_by_fseq)
main["fasta_ref"] = main["canonical_reference"] + 1

# Junctions already represented in the data file per full_seq. Used to skip
# candidates that are already scored (don't double-count what's in NO_DELTAS).
main["_jx_pair"] = list(zip(
    main["intron1"].fillna("").astype(str).str.len(),
    main["exon"].fillna("").astype(str).str.len(),
))
nd_jxns_per_fseq = (
    main.groupby("full_seq")["_jx_pair"]
        .agg(lambda s: set(p for p in s if p[0] > 0 and p[1] > 0))
        .to_dict()
)

# Restrict to rows in events that have multiple candidate junctions, and
# collapse to one row per unique full_seq (MANE Select > MANE Plus > lowest Ref).
ambig_main = main[main["event_id_161"].isin(candidates_per_event.keys())].copy()
n_before = len(ambig_main)
ambig_main["_mane_priority"] = ambig_main["transcript_id"].apply(_mane_priority)
ambig_main = (ambig_main
              .sort_values(["_mane_priority", "Reference"])
              .drop_duplicates(subset=["full_seq"], keep="first")
              .drop(columns=["_mane_priority"]))
print(f"Rows in events with candidate alts: {n_before:,}  →  unique sequences: {len(ambig_main):,}")


# ── Load pickles ──────────────────────────────────────────────────────────
print("Loading pickles...")
pkls = {}
for name, path in REP_PKLS.items():
    with open(path, "rb") as f:
        pkls[name] = pickle.load(f)
    print(f"  {name}: {len(pkls[name])} refs")


# ── Compute alt PSI per row per replicate ──────────────────────────────────
print("Computing alt junction PSI for all unique sequences × candidate alts...")
records = []
n_candidates_skipped_in_data = 0
n_candidates_skipped_main_jx = 0

for _, row in ambig_main.iterrows():
    eid_161 = row["event_id_161"]
    cands = candidates_per_event.get(eid_161, {})
    if not cands:
        continue
    ref   = int(row["fasta_ref"])
    is_wt = (row["snp"] == "none")
    fseq  = row["full_seq"]
    main_i1_len = len(str(row["intron1"])) if pd.notna(row["intron1"]) else 0
    main_ex_len = len(str(row["exon"]))    if pd.notna(row["exon"])    else 0
    main_pair = (main_i1_len, main_ex_len)
    represented = nd_jxns_per_fseq.get(fseq, set())

    for key, aj in cands.items():
        # Skip the row's own junction (this is the "main" measurement, already in the data file).
        if key == main_pair:
            n_candidates_skipped_main_jx += 1
            continue
        # For variants: skip if the junction is already in NO_DELTAS for this full_seq.
        # For WTs: keep — needed as the dPSI/delta-logit baseline for variant SI rows.
        if (not is_wt) and key in represented:
            n_candidates_skipped_in_data += 1
            continue

        i1, i2, e = junctions(aj["intron1_len"], aj["exon_len"])

        # All supertable Refs for this full_seq, excluding this row's Reference
        # (which is the MANE-preferred one). Empty if the full_seq has only the
        # one supertable row.
        all_st_refs = supertable_refs_by_fseq.get(fseq, [])
        other_refs = [r for r in all_st_refs if r != int(row["Reference"])]
        alt_refs_str = ";".join(str(r) for r in other_refs)

        # Composite row id, distinct from the main-file Reference. Same
        # variant/Reference can appear in multiple SI rows (one per alt
        # junction), so the bare Reference is not unique here. Format:
        # "<Reference>__alt_<i1>-<ex>__<alt_transcript_id>".
        si_row_id = f"{int(row['Reference'])}__alt_{aj['intron1_len']}-{aj['exon_len']}__{aj['transcript_id']}"

        rec = {
            "si_row_id":            si_row_id,
            "Reference":            row["Reference"],   # MANE-preferred Reference (1-indexed); links to ALL_WITH_WT.csv
            "alt_supertable_refs":  alt_refs_str,        # other supertable Refs for the same full_seq
            "event_id_161":         eid_161,
            "event_id":             row["event_id"],
            "gene_exon":            row["gene_exon"],
            "snp":                  row["snp"],
            "source":               row["source"],
            "seq_type":             row["seq_type"],
            "main_transcript_id":   row["transcript_id"],
            "main_intron1_len":     main_i1_len,
            "main_exon_len":        main_ex_len,
            "alt_transcript_id":    aj["transcript_id"],
            "alt_mane_status":      aj["alt_mane_status"] or mane_label(aj["transcript_id"]),
            "alt_intron1_len":      aj["intron1_len"],
            "alt_exon_len":         aj["exon_len"],
            "alt_exon_start_hg38":  aj["exon_start_hg38"],
            "alt_exon_end_hg38":    aj["exon_end_hg38"],
            "alt_in_supertable":    aj["in_supertable"],
        }

        for cell in CELLS:
            rep_names = CELL_REPS[cell]
            if cell == "HEK" and is_wt:
                rep_names = HEK_WT_REPS

            rep_psijs = []
            for ri, rname in enumerate(rep_names, 1):
                p, _, tot_rep = get_psi(pkls[rname], ref, i1, i2, e)
                # Raw inc/exc counts at this junction in this rep (regardless of MINCOV)
                if ref in pkls[rname]:
                    jd  = pkls[rname][ref]
                    inc = min(jd.get(i1, 0), jd.get(i2, 0))
                    exc = jd.get(e, 0)
                else:
                    inc, exc = 0, 0
                rec[f"{cell}_rep{ri}_included_alt"]    = inc
                rec[f"{cell}_rep{ri}_excluded_alt"]    = exc
                rec[f"{cell}_rep{ri}_psi_raw_alt"]     = p
                rec[f"{cell}_rep{ri}_psi_clipped_alt"] = float(np.clip(p, CLIP, 1-CLIP)) if not np.isnan(p) else np.nan
                rec[f"{cell}_rep{ri}_logit_alt"]       = clip_logit(p)
                if not np.isnan(p):
                    rep_psijs.append((inc, exc))

            valid = [(i, ex) for i, ex in rep_psijs if not np.isnan(i)]
            if len(valid) >= 2:
                tot_inc = sum(v[0] for v in valid)
                tot_exc = sum(v[1] for v in valid)
                tot     = tot_inc + tot_exc
                pool    = tot_inc / tot if tot >= MINCOV else np.nan
            else:
                tot_inc, tot_exc, tot, pool = 0, 0, 0, np.nan
            rec[f"{cell}_pooled_included_alt"]    = tot_inc
            rec[f"{cell}_pooled_excluded_alt"]    = tot_exc
            rec[f"{cell}_total_pooled_alt"]       = tot
            rec[f"{cell}_pooled_psi_raw_alt"]     = pool
            rec[f"{cell}_pooled_psi_clipped_alt"] = float(np.clip(pool, CLIP, 1-CLIP)) if not np.isnan(pool) else np.nan
            rec[f"{cell}_pooled_logit_alt"]       = clip_logit(pool)

        records.append(rec)

alt_df = pd.DataFrame(records)
print(f"  Rows computed: {len(alt_df):,}")
print(f"  Candidates skipped (own junction = main row's): {n_candidates_skipped_main_jx:,}")
print(f"  Candidates skipped (already in NO_DELTAS for this full_seq): {n_candidates_skipped_in_data:,}")


# ── Compute WT PSI at alt junction per (event_id_161, alt_transcript) per cell ──
# Coverage flag (not a filter — Gabriel needs all candidate measurements,
# including low-coverage ones, with counts so downstream can filter):
# add `low_coverage_rescue=True` for (event, alt_transcript) pairs without
# ≥1 WT AND ≥1 variant having a valid pooled PSI in any cell.
print("Computing dPSI and delta logit; flagging low-coverage pairs...")
psi_cols_per_cell = {c: f"{c}_pooled_psi_raw_alt" for c in CELLS}

wt_df  = alt_df[alt_df["snp"] == "none"]
var_df = alt_df[alt_df["snp"] != "none"]

# Pairs with ≥1 WT (resp. variant) having a valid pooled PSI in any cell
def any_valid(df, cell_cols):
    return df[list(cell_cols.values())].notna().any(axis=1)

wt_ok  = wt_df.assign(_ok=any_valid(wt_df, psi_cols_per_cell))
var_ok = var_df.assign(_ok=any_valid(var_df, psi_cols_per_cell))

wt_pairs  = set(map(tuple,
    wt_ok.loc[wt_ok["_ok"], ["event_id_161", "alt_transcript_id"]].values.tolist()))
var_pairs = set(map(tuple,
    var_ok.loc[var_ok["_ok"], ["event_id_161", "alt_transcript_id"]].values.tolist()))
ok_pairs  = wt_pairs & var_pairs

alt_df["_pair"] = list(zip(alt_df["event_id_161"], alt_df["alt_transcript_id"]))
alt_df["low_coverage_rescue"] = ~alt_df["_pair"].isin(ok_pairs)
alt_df = alt_df.drop(columns=["_pair"])
n_rescue = int(alt_df["low_coverage_rescue"].sum())
print(f"  Rows kept: {len(alt_df):,}  (low_coverage_rescue=True: {n_rescue:,})")

wt_df = alt_df[alt_df["snp"] == "none"].copy()
for cell in CELLS:
    wt_map = (wt_df.groupby(["event_id_161", "alt_transcript_id"])[f"{cell}_pooled_psi_raw_alt"].mean()
              .rename(f"wt_psi_alt_{cell}"))
    wt_logit_map = (wt_df.groupby(["event_id_161", "alt_transcript_id"])[f"{cell}_pooled_logit_alt"].mean()
                    .rename(f"wt_logit_alt_{cell}"))
    alt_df = alt_df.join(wt_map, on=["event_id_161", "alt_transcript_id"])
    alt_df = alt_df.join(wt_logit_map, on=["event_id_161", "alt_transcript_id"])
    alt_df[f"{cell}_wt_pooled_psi_raw_alt"]  = alt_df[f"wt_psi_alt_{cell}"]
    alt_df[f"{cell}_wt_pooled_logit_alt"]    = alt_df[f"wt_logit_alt_{cell}"]
    alt_df[f"{cell}_dpsi_pooled_alt"] = np.where(
        alt_df["snp"] == "none", np.nan,
        alt_df[f"{cell}_pooled_psi_raw_alt"] - alt_df[f"wt_psi_alt_{cell}"]
    )
    alt_df[f"{cell}_delta_logit_pooled_alt"] = np.where(
        alt_df["snp"] == "none", np.nan,
        alt_df[f"{cell}_pooled_logit_alt"] - alt_df[f"wt_logit_alt_{cell}"]
    )
    alt_df = alt_df.drop(columns=[f"wt_psi_alt_{cell}", f"wt_logit_alt_{cell}"])

print(f"  Final rows: {len(alt_df):,}")
print(f"  event_id_161 events: {alt_df['event_id_161'].nunique()}")
print(f"  Unique alt transcripts: {alt_df['alt_transcript_id'].nunique()}")
print(f"  WTs: {(alt_df['snp']=='none').sum():,}   Variants: {(alt_df['snp']!='none').sum():,}")

# ── Re-key Reference so each SI row has a unique number ───────────────────
# In the original schema, SI's `Reference` was the MANE-row's Reference (the
# row in NO_DELTAS this SI row attaches to), so two SI rows could share the
# same `Reference` if a construct had multiple alt junctions. That made the
# merged files awkward (multiple rows with the same `Reference`). Move the
# MANE-row Reference to `original_reference` and assign each SI row its own
# Reference starting at SUPERTABLE_SIZE + 1 = 244,001 (the supertable ends at
# data-Ref 244,000). `si_row_id` (already in the table) remains a stable
# identifier across pipeline runs; the synthetic Reference is recomputed on
# every run.
SUPERTABLE_SIZE = 244_000
alt_df = alt_df.reset_index(drop=True)
alt_df.insert(alt_df.columns.get_loc("Reference"),
              "original_reference",
              alt_df["Reference"].astype(int).values)
alt_df["Reference"] = pd.RangeIndex(SUPERTABLE_SIZE + 1,
                                     SUPERTABLE_SIZE + 1 + len(alt_df))
print(f"  Re-keyed Reference to {alt_df['Reference'].min()}..{alt_df['Reference'].max()}; original_reference column added")

# ── Add wt_reference (= Reference of WT row at same alt junction) ─────────
# Done AFTER re-keying so wt_reference points to the synthetic SI Reference
# of the WT row, not the old MANE-row Reference.
wt = alt_df[alt_df["snp"] == "none"][["Reference", "event_id_161", "alt_intron1_len", "alt_exon_len"]]
wt_jx = (wt.sort_values("Reference")
           .drop_duplicates(subset=["event_id_161", "alt_intron1_len", "alt_exon_len"], keep="first")
           .set_index(["event_id_161", "alt_intron1_len", "alt_exon_len"])["Reference"])
wt_ev = (wt.sort_values("Reference")
           .drop_duplicates(subset=["event_id_161"], keep="first")
           .set_index("event_id_161")["Reference"])
key_jx = list(zip(alt_df["event_id_161"], alt_df["alt_intron1_len"], alt_df["alt_exon_len"]))
wt_jx_map = wt_jx.to_dict()
wt_ev_map = wt_ev.to_dict()
wt_ref = [wt_jx_map.get(k, wt_ev_map.get(k[0])) for k in key_jx]
alt_df["wt_reference"] = wt_ref
is_wt = alt_df["snp"] == "none"
alt_df.loc[is_wt, "wt_reference"] = alt_df.loc[is_wt, "Reference"].values
alt_df["wt_reference"] = alt_df["wt_reference"].astype("Int64")
# Move wt_reference right after Reference
cols = list(alt_df.columns)
cols.remove("wt_reference")
cols.insert(cols.index("Reference") + 1, "wt_reference")
alt_df = alt_df[cols]
print(f"  wt_reference filled: {alt_df['wt_reference'].notna().sum():,}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
out = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME)
alt_df.to_csv(out, index=False)
print(f"\nSaved: {out}")
