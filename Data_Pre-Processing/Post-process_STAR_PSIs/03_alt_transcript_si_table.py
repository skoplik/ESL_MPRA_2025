"""
03_alt_transcript_si_table.py.

For every ambiguous event, computes PSI at every NON-CHOSEN Gencode-annotated
transcript's junction for all rows (each WT + variant). One output row per
(Reference, alt_transcript_id) with `_alt` suffixed PSI / logit / dPSI /
delta-logit columns mirroring the main table schema.

Junction formula (0-indexed construct coordinates):
  i1 = (26, 286 + alt_intron1_len)
  i2 = (286 + alt_intron1_len + alt_exon_len + 1, 871)
  e  = (26, 871)

Inputs:
  - source supertable (gz) — to enumerate alt transcripts per event
  - corrected supertable from Stage 1 — to identify the chosen transcript to skip
  - merged main CSV from Stage 2 — for the per-row context
  - 12 per-replicate junction-count pickles
"""

import gzip
import os
import pickle
import re
import pandas as pd
import numpy as np
from scipy.special import logit as scipy_logit

# ── Paths ──────────────────────────────────────────────────────────────────
SOURCE_GZ          = "/ESL/ESL_MPRA/Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25.csv.gz"
CORRECTED_ST       = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/st_corrected.csv"
MAIN_CSV           = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/ALL_WTS_VARS_NO_DELTAS.csv"
GTF                = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/gencode.v48.annotation.gtf"
OUTPUT_DIR         = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/ambiguous_sjs"
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

def mane_priority(transcript_id):
    tx = str(transcript_id).split('.')[0]
    if tx in mane_select:
        return 0
    if tx in mane_plus:
        return 1
    return 2


# ── Load source + corrected supertables ────────────────────────────────────
# Source carries one row per (Reference, transcript) — needed to enumerate
# alts. Corrected has one row per (Reference, event) at the chosen MANE
# junction — used to identify which transcript to skip per event.
print("Loading source supertable from gz...")
with gzip.open(SOURCE_GZ, "rt") as f:
    st_source = pd.read_csv(f, low_memory=False)
st_source["_intron1_len"] = st_source["intron1"].str.len()

print("Loading corrected supertable...")
st_corr = pd.read_csv(CORRECTED_ST, low_memory=False)

# Ambiguity: source supertable rows per event with >1 distinct intron1_len.
intron_counts = st_source.groupby("event_id")["_intron1_len"].nunique()
ambig_eids = set(intron_counts[intron_counts > 1].index)
print(f"Ambiguous events (source): {len(ambig_eids)}")

# Chosen transcript per event = the per-row transcript_id of any row in the
# corrected supertable for that event (all rows in an event share it).
chosen_tx_per_event = (st_corr[st_corr["event_id"].isin(ambig_eids)]
                       .drop_duplicates("event_id")
                       .set_index("event_id")["transcript_id"])

# Alt junctions per event: every distinct intron1_len in the source supertable
# whose transcript_id base differs from the chosen transcript.
alt_jxns = {}   # event_id -> list of alt transcript dicts
for event_id in ambig_eids:
    chosen_tx = chosen_tx_per_event.get(event_id)
    if chosen_tx is None:
        # No chosen junction (e.g. one of the 6 events with no reads anywhere) —
        # fall back to skipping the lowest-Reference primary's intron1_len.
        chosen_tx_base = None
    else:
        chosen_tx_base = str(chosen_tx).split(".")[0]

    sub = (st_source[st_source["event_id"] == event_id].copy())
    sub["_mane_priority"] = sub["transcript_id"].map(mane_priority)
    sub = sub.sort_values(["_mane_priority", "Reference"])
    sub = sub.drop_duplicates("_intron1_len")  # one entry per distinct junction

    alts = []
    for _, row in sub.iterrows():
        tx_base = str(row["transcript_id"]).split(".")[0]
        if chosen_tx_base is not None and tx_base == chosen_tx_base:
            continue  # skip the chosen junction
        alts.append({
            "intron1_len":     int(row["_intron1_len"]),
            "exon_len":        len(str(row["exon"])),
            "transcript_id":   row["transcript_id"],
            "exon_start_hg38": row["exon_start_hg38"],
            "exon_end_hg38":   row["exon_end_hg38"],
        })
    if alts:
        alt_jxns[event_id] = alts

# Map full_seq → 1-indexed pickle Reference. Source supertable Reference is
# 0-indexed (matching pickle = Reference + 1).
seen_seq = {}
for _, r in st_source.iterrows():
    if r["full_seq"] not in seen_seq:
        seen_seq[r["full_seq"]] = int(r["Reference"]) + 1

# ── Load main data: all rows for ambiguous events ──────────────────────────
print("Loading main data...")
main = pd.read_csv(MAIN_CSV, low_memory=False)
main["Reference_0"] = main["Reference"] - 1

ambig_main = main[main["event_id"].isin(ambig_eids)].copy()
print(f"Rows in ambiguous events: {len(ambig_main):,}")

ref0_to_seq = st_source.drop_duplicates("Reference").set_index("Reference")["full_seq"]
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

os.makedirs(OUTPUT_DIR, exist_ok=True)
out = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME)
alt_df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(f"Columns: {list(alt_df.columns)}")
