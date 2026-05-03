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

print("Loading Gencode-only alt junctions table...")
alt_jxns_df = pd.read_csv(ST_ALT_JUNCTIONS)
print(f"  Gencode-only alts: {len(alt_jxns_df)}")

# Map: event_id_161 → list of alt transcript dicts.
# st_alt_junctions.csv is now deduped per (event_id_161, alt_transcript_id).
alt_jxns = {}
for _, row in alt_jxns_df.iterrows():
    eid_161 = row["event_id_161"]
    alt_jxns.setdefault(eid_161, []).append({
        "intron1_len":        int(row["alt_intron1_len"]),
        "exon_len":           int(row["alt_exon_len"]),
        "transcript_id":      row["alt_transcript_id"],
        "exon_start_hg38":    row["alt_exon_start_hg38"],
        "exon_end_hg38":      row["alt_exon_end_hg38"],
        "alt_mane_status":    row.get("alt_mane_status", ""),
        "main_transcript_id": row.get("main_transcript_id", ""),
        "canonical_reference":int(row["canonical_reference"]),
    })

# Build event_id_161 → set of supertable transcript_id bases (any row sharing
# the same construct). Used to flag whether each alt is also a supertable alt.
st_tx_bases_by_event = {}
for eid_161, grp in st_corr.groupby("event_id_161"):
    bases = {str(t).split(".")[0] for t in grp["transcript_id"].dropna()}
    st_tx_bases_by_event[eid_161] = bases
print(f"  Events (event_id_161) with alts: {len(alt_jxns)}")

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

# Restrict to rows whose event_id_161 has Gencode-only alts
ambig_main = main[main["event_id_161"].isin(alt_jxns.keys())].copy()
print(f"Rows in events with Gencode-only alts: {len(ambig_main):,}")


# ── Load pickles ──────────────────────────────────────────────────────────
print("Loading pickles...")
pkls = {}
for name, path in REP_PKLS.items():
    with open(path, "rb") as f:
        pkls[name] = pickle.load(f)
    print(f"  {name}: {len(pkls[name])} refs")


# ── Compute alt PSI per row per replicate ──────────────────────────────────
print("Computing alt junction PSI for all rows × Gencode-only alts...")
records = []

for _, row in ambig_main.iterrows():
    eid_161 = row["event_id_161"]
    if eid_161 not in alt_jxns:
        continue
    ref   = int(row["fasta_ref"])
    is_wt = (row["snp"] == "none")

    for aj in alt_jxns[eid_161]:
        i1, i2, e = junctions(aj["intron1_len"], aj["exon_len"])

        rec = {
            "Reference":            row["Reference"],   # 1-indexed to match main table
            "canonical_reference":  aj["canonical_reference"],
            "event_id_161":         eid_161,
            "event_id":             row["event_id"],
            "gene_exon":            row["gene_exon"],
            "snp":                  row["snp"],
            "source":               row["source"],
            "seq_type":             row["seq_type"],
            "main_transcript_id":   row["transcript_id"],
            "main_intron1_len":     len(str(row["intron1"])) if pd.notna(row["intron1"]) else "",
            "main_exon_len":        len(str(row["exon"]))    if pd.notna(row["exon"])    else "",
            "alt_transcript_id":    aj["transcript_id"],
            "alt_mane_status":      aj["alt_mane_status"] or mane_label(aj["transcript_id"]),
            "alt_intron1_len":      aj["intron1_len"],
            "alt_exon_len":         aj["exon_len"],
            "alt_exon_start_hg38":  aj["exon_start_hg38"],
            "alt_exon_end_hg38":    aj["exon_end_hg38"],
            "alt_in_supertable":    str(aj["transcript_id"]).split(".")[0] in
                                       st_tx_bases_by_event.get(eid_161, set()),
        }

        for cell in CELLS:
            rep_names = CELL_REPS[cell]
            if cell == "HEK" and is_wt:
                rep_names = HEK_WT_REPS

            rep_psijs = []
            for ri, rname in enumerate(rep_names, 1):
                p, _, tot = get_psi(pkls[rname], ref, i1, i2, e)
                rec[f"{cell}_rep{ri}_psi_raw_alt"]     = p
                rec[f"{cell}_rep{ri}_psi_clipped_alt"] = float(np.clip(p, CLIP, 1-CLIP)) if not np.isnan(p) else np.nan
                rec[f"{cell}_rep{ri}_logit_alt"]       = clip_logit(p)
                if not np.isnan(p) and ref in pkls[rname]:
                    jd  = pkls[rname][ref]
                    inc = min(jd.get(i1, 0), jd.get(i2, 0))
                    exc = jd.get(e, 0)
                    rep_psijs.append((inc, exc))

            valid = [(i, ex) for i, ex in rep_psijs if not np.isnan(i)]
            if len(valid) >= 2:
                tot_inc = sum(v[0] for v in valid)
                tot_exc = sum(v[1] for v in valid)
                tot     = tot_inc + tot_exc
                pool    = tot_inc / tot if tot >= MINCOV else np.nan
            else:
                pool = np.nan
            rec[f"{cell}_pooled_psi_raw_alt"]     = pool
            rec[f"{cell}_pooled_psi_clipped_alt"] = float(np.clip(pool, CLIP, 1-CLIP)) if not np.isnan(pool) else np.nan
            rec[f"{cell}_pooled_logit_alt"]       = clip_logit(pool)

        records.append(rec)

alt_df = pd.DataFrame(records)
print(f"  Rows computed: {len(alt_df):,}")


# ── Compute WT PSI at alt junction per (event_id_161, alt_transcript) per cell ──
# Coverage filter: keep only (event_id_161, alt_transcript) pairs where ≥1 WT
# and ≥1 variant have valid pooled PSI in at least one cell line.
print("Computing dPSI and delta logit; applying WT+variant coverage filter...")
psi_cols_per_cell = {c: f"{c}_pooled_psi_raw_alt" for c in CELLS}

wt_df  = alt_df[alt_df["snp"] == "none"]
var_df = alt_df[alt_df["snp"] != "none"]

# Pairs with ≥1 WT having a valid pooled PSI in any cell
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
alt_df = alt_df[alt_df["_pair"].isin(ok_pairs)].copy()
alt_df = alt_df.drop(columns=["_pair"])
print(f"  Rows after coverage filter: {len(alt_df):,}")

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

os.makedirs(OUTPUT_DIR, exist_ok=True)
out = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME)
alt_df.to_csv(out, index=False)
print(f"\nSaved: {out}")
