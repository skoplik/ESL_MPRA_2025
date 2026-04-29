"""
Build SI table for ambiguous SJ events.

For each ambiguous event, computes PSI at EVERY non-primary transcript's
junction for ALL rows (every WT and variant), regardless of which transcript
that row was originally annotated to.

For events with N transcripts, this produces N-1 alt rows per reference row,
one per alternative transcript. Each row is distinguished by alt_transcript_id.

Junction formula (0-indexed construct coordinates):
  i1 = (26, 286 + alt_intron1_len)
  i2 = (286 + alt_intron1_len + alt_exon_len + 1, 871)
  e  = (26, 871)

Output: SI_alt_transcript_psi.csv
  One row per (Reference, alt_transcript). Columns mirror the main
  ALL_WTS_VARS_NO_DELTAS.csv structure with _alt suffix.
"""

import pickle
import re
import pandas as pd
import numpy as np
from scipy.special import logit as scipy_logit

# ── Paths ──────────────────────────────────────────────────────────────────
SUPERTABLE = "/ESL/Figures_SK/General_preprocessing/output_04_24_2026/st_working_04_24_2026.csv"
MAIN_CSV   = "/ESL/Figures_SK/General_preprocessing/output_04_24_2026/04_24_2026_1e-2_ALL_WTS_VARS_NO_DELTAS_seqfix.csv"
GTF        = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/gencode.v48.annotation.gtf"
OUTPUT_DIR = "/ESL/Figures_SK/ambiguous_sjs"

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

def mane_priority(transcript_id):
    tx = str(transcript_id).split('.')[0]
    if tx in mane_select:
        return 0
    if tx in mane_plus:
        return 1
    return 2


# ── Load supertable ────────────────────────────────────────────────────────
print("Loading supertable...")
st = pd.read_csv(SUPERTABLE, low_memory=False)

# Detect ambiguity by intron1 length diversity — robust to supertable coordinate
# updates (which preserve the actual intron1/exon sequences).
st["_intron1_len"] = st["intron1"].str.len()
intron_counts = st.groupby("event_id")["_intron1_len"].nunique()
ambig_eids = set(intron_counts[intron_counts > 1].index)
print(f"Ambiguous events: {len(ambig_eids)}")

# Primary Reference per event (lowest Reference with snp==none)
primary_ref = (st[(st["event_id"].isin(ambig_eids)) & (st["snp"] == "none")]
               .sort_values("Reference")
               .drop_duplicates("event_id")
               .set_index("event_id")["Reference"])

# All alt junction coords per event: all non-primary intron1 lengths,
# sorted MANE Select > MANE Plus Clinical > lowest Reference.
# Deduplicate by intron1_len (not transcript_id) to capture all distinct junctions
# even after supertable transcript annotation updates.
alt_jxns = {}   # event_id -> list of alt transcript dicts
for event_id in ambig_eids:
    sub = (st[(st["event_id"] == event_id)]
           .copy())
    sub["_mane_priority"] = sub["transcript_id"].map(mane_priority)
    sub = sub.sort_values(["_mane_priority", "Reference"])
    sub = sub.drop_duplicates("_intron1_len")   # one entry per distinct junction
    pref_i1_len = sub.iloc[0]["_intron1_len"]   # primary = first after sort (lowest ref, MANE preferred)
    # But primary_ref is actually the lowest Reference snp==none; use its intron1_len
    pref_ref = primary_ref.get(event_id)
    if pref_ref is not None:
        pref_row = st[st["Reference"] == pref_ref]
        pref_i1_len = int(pref_row["_intron1_len"].iloc[0]) if len(pref_row) else pref_i1_len

    alts = []
    for _, row in sub.iterrows():
        if int(row["_intron1_len"]) != pref_i1_len:
            alts.append({
                "intron1_len":   int(row["_intron1_len"]),
                "exon_len":      len(str(row["exon"])),
                "transcript_id": row["transcript_id"],
                "exon_start_hg38": row["exon_start_hg38"],
                "exon_end_hg38":   row["exon_end_hg38"],
            })
    if alts:
        alt_jxns[event_id] = alts

# Pickle keys are 1-indexed; supertable Reference is 0-indexed → add 1
seen_seq = {}
for _, r in st.iterrows():
    if r["full_seq"] not in seen_seq:
        seen_seq[r["full_seq"]] = int(r["Reference"]) + 1

# ── Load main data: all rows for ambiguous events ──────────────────────────
print("Loading main data...")
main = pd.read_csv(MAIN_CSV, low_memory=False)
main["Reference_0"] = main["Reference"] - 1

ambig_main = main[main["event_id"].isin(ambig_eids)].copy()
print(f"Rows in ambiguous events: {len(ambig_main):,}")

ref0_to_seq = st.set_index("Reference")["full_seq"]
ambig_main["full_seq_st"]  = ambig_main["Reference_0"].map(ref0_to_seq)
ambig_main["fasta_ref"]    = ambig_main["full_seq_st"].map(seen_seq)

# ── Load pickles ───────────────────────────────────────────────────────────
print("Loading pickles...")
pkls = {}
for name, path in REP_PKLS.items():
    with open(path, "rb") as f:
        pkls[name] = pickle.load(f)
    print(f"  {name}: {len(pkls[name])} refs")

# ── Compute alt PSI per row per replicate ──────────────────────────────────
print("Computing alt junction PSI for all rows (all alt transcripts)...")
records = []

for _, row in ambig_main.iterrows():
    event_id = row["event_id"]
    if event_id not in alt_jxns:
        continue
    ref   = row["fasta_ref"]
    is_wt = (row["snp"] == "none")

    for aj in alt_jxns[event_id]:
        i1, i2, e = junctions(aj["intron1_len"], aj["exon_len"])

        rec = {
            "Reference":            row["Reference"],   # 1-indexed to match main table
            "event_id":             event_id,
            "gene_exon":            row["gene_exon"],
            "snp":                  row["snp"],
            "source":               row["source"],
            "seq_type":             row["seq_type"],
            "alt_transcript_id":    aj["transcript_id"],
            "alt_mane_status":      mane_label(aj["transcript_id"]),
            "alt_exon_start_hg38":  aj["exon_start_hg38"],
            "alt_exon_end_hg38":    aj["exon_end_hg38"],
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

# ── Compute WT PSI at alt junction per event per cell ──────────────────────
print("Computing dPSI and delta logit...")
# Group by (event_id, alt_transcript_id) so each alt transcript gets its own WT reference
wt_df = alt_df[alt_df["snp"] == "none"].copy()

for cell in CELLS:
    wt_map = (wt_df.groupby(["event_id", "alt_transcript_id"])[f"{cell}_pooled_psi_raw_alt"].mean()
              .rename(f"wt_psi_alt_{cell}"))
    wt_logit_map = (wt_df.groupby(["event_id", "alt_transcript_id"])[f"{cell}_pooled_logit_alt"].mean()
                    .rename(f"wt_logit_alt_{cell}"))
    alt_df = alt_df.join(wt_map, on=["event_id", "alt_transcript_id"])
    alt_df = alt_df.join(wt_logit_map, on=["event_id", "alt_transcript_id"])
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

psi_cols = [f"{c}_pooled_psi_raw_alt" for c in CELLS]
alt_df = alt_df[alt_df[psi_cols].notna().any(axis=1)].copy()
print(f"  Rows with ≥1 valid alt PSI: {len(alt_df):,}")
print(f"  Events: {alt_df['event_id'].nunique()}")
print(f"  Unique alt transcripts: {alt_df['alt_transcript_id'].nunique()}")
print(f"  WTs: {(alt_df['snp']=='none').sum():,}   Variants: {(alt_df['snp']!='none').sum():,}")

out = f"{OUTPUT_DIR}/SI_alt_transcript_psi.csv"
alt_df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(f"Columns: {list(alt_df.columns)}")
