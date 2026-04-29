"""
01_fix_sj_supertable.py.

For ambiguous splice-junction events (event_ids with multiple Gencode
transcripts annotating the same exon at different intron1/exon lengths), pick
one canonical junction and rewrite every supertable row in the event to use it.
Then patch the per-replicate PSI text files so the downstream merge step
(02_merge_psi.py) computes PSI at the chosen junction.

Junction selection per event:
  1. For the primary reference construct (lowest Reference, snp == 'none'),
     count how many replicates have valid PSI (Coverage >= MINCOV) at each
     candidate Gencode-annotated junction. Pick a junction with reads in
     >= 2 reps; tiebreak MANE Select > MANE Plus Clinical > lowest Reference.
  2. If the primary reference is a synthesis dropout (zero reads everywhere),
     fall back: score each candidate junction by the number of *event refs*
     (WT + variants) that pass MINCOV in any rep, then by summed reads, then
     MANE priority, then lowest Reference. Only Gencode-annotated junctions
     are ever considered.
  3. If no junction has reads at any ref in any rep, skip the event.

Consistency rewrite per chosen event:
  - full_seq is preserved (it is the physical synthesized 161-nt construct).
  - intron1/exon/intron2 are re-derived by splitting full_seq at the chosen
    junction's intron1_len/exon_len.
  - transcript_id, mane_status, exon_start_hg38, exon_end_hg38, intron1_len,
    exon_len are overwritten across every row in the event to the chosen values.

PSI text-file patching:
  - Each of the 12 per-rep files is written to a new sibling directory.
  - For each Reference in an ambig event with a chosen junction, (PSI, Coverage)
    is recomputed from the rep's pickle at the chosen junction with MINCOV=10.
    Rows failing MINCOV are dropped; rows newly passing MINCOV are added.
  - Non-ambig refs are passed through unchanged.
"""

import gzip
import os
import pickle
import re
import sys
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
GZ_IN  = "/ESL/ESL_MPRA/Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25.csv.gz"
GTF    = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/gencode.v48.annotation.gtf"

OUT_DIR    = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output"
ST_OUT     = os.path.join(OUT_DIR, "st_corrected.csv")
PSI_OUT_DIR = os.path.join(OUT_DIR, "recount_PSIs")

PKL_BASE = "/ESL/Analysis/STAR_alignment/separate_concat_2023_09_19_d1c_ms75_from_s3_2023_11_27/recount_SJs"
PKL_2024 = PKL_BASE + "/2024_07_26"
WT_BASE  = "/ESL/Analysis/WT_Library/separate/recount_SJs"

# Per-replicate inputs: (psi_text_file, pickle_file)
# Names match the merge_psi --*_PSI argument names so the wrapper stays in sync.
REP_INPUTS = {
    "HeLa_Rep1":      (PKL_BASE + "/HeLa_Rep1_separate_splicing_profiles/HeLa_Rep1_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/HeLa_Rep1_separate_splicing_profiles/HeLa_Rep1_separate_all_splicing_counts.p"),
    "HeLa_Rep2":      (PKL_BASE + "/HeLa_Rep3_20231031_separate_splicing_profiles/HeLa_Rep3_20231031_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/HeLa_Rep3_20231031_separate_splicing_profiles/HeLa_Rep3_20231031_separate_all_splicing_counts.p"),
    "K562_Rep1":      (PKL_BASE + "/K562_Rep1_separate_splicing_profiles/K562_Rep1_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/K562_Rep1_separate_splicing_profiles/K562_Rep1_separate_all_splicing_counts.p"),
    "K562_Rep2":      (PKL_BASE + "/K562_Rep3_20231031_separate_splicing_profiles/K562_Rep3_20231031_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/K562_Rep3_20231031_separate_splicing_profiles/K562_Rep3_20231031_separate_all_splicing_counts.p"),
    "MCF7_Rep1":      (PKL_2024 + "/MCF7_Rep1_separate_splicing_profiles/MCF7_Rep1_recalc_PSIs_mincov10.txt",
                       PKL_2024 + "/MCF7_Rep1_separate_splicing_profiles/MCF7_Rep1_all_splicing_counts.p"),
    "MCF7_Rep2":      (PKL_2024 + "/MCF7_Rep2_separate_splicing_profiles/MCF7_Rep2_recalc_PSIs_mincov10.txt",
                       PKL_2024 + "/MCF7_Rep2_separate_splicing_profiles/MCF7_Rep2_all_splicing_counts.p"),
    "HMC3_Rep1":      (PKL_2024 + "/HMC3_Rep1_separate_splicing_profiles/HMC3_Rep1_recalc_PSIs_mincov10.txt",
                       PKL_2024 + "/HMC3_Rep1_separate_splicing_profiles/HMC3_Rep1_all_splicing_counts.p"),
    "HMC3_Rep2":      (PKL_2024 + "/HMC3_Rep2_separate_splicing_profiles/HMC3_Rep2_recalc_PSIs_mincov10.txt",
                       PKL_2024 + "/HMC3_Rep2_separate_splicing_profiles/HMC3_Rep2_all_splicing_counts.p"),
    "HEK293_Rep1":    (PKL_BASE + "/HEK293_Rep2_separate_splicing_profiles/HEK293_Rep2_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/HEK293_Rep2_separate_splicing_profiles/HEK293_Rep2_separate_all_splicing_counts.p"),
    "HEK293_Rep2":    (PKL_BASE + "/HEK293_Rep3_separate_splicing_profiles/HEK293_Rep3_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/HEK293_Rep3_separate_splicing_profiles/HEK293_Rep3_separate_all_splicing_counts.p"),
    "HEK293_WT_Rep1": (WT_BASE  + "/HEK_WT_Rep1_20231101_separate_splicing_profiles/HEK_WT_Rep1_20231101_separate_recalc_PSIs_mincov10.txt",
                       WT_BASE  + "/HEK_WT_Rep1_20231101_separate_splicing_profiles/HEK_WT_Rep1_20231101_separate_all_splicing_counts.p"),
    "HEK293_WT_Rep2": (WT_BASE  + "/HEK_WT_Rep2_20231101_separate_splicing_profiles/HEK_WT_Rep2_20231101_separate_recalc_PSIs_mincov10.txt",
                       WT_BASE  + "/HEK_WT_Rep2_20231101_separate_splicing_profiles/HEK_WT_Rep2_20231101_separate_all_splicing_counts.p"),
}

# Construct geometry — fixed by the MPRA library design.
SHARED_5P = 286
MINCOV    = 10
E_JXN     = (26, 871)


# ── Helpers ────────────────────────────────────────────────────────────────
def junctions(i1_len, ex_len):
    i1 = (26, SHARED_5P + i1_len)
    i2 = (SHARED_5P + i1_len + ex_len + 1, 871)
    return i1, i2


def psi_at(pkl_dict, fasta_ref, i1, i2):
    """Return (psi, included_reads, excluded_reads). NaN PSI if total < MINCOV."""
    if fasta_ref not in pkl_dict:
        return np.nan, 0, 0
    jd  = pkl_dict[fasta_ref]
    inc = min(jd.get(i1, 0), jd.get(i2, 0))
    exc = jd.get(E_JXN, 0)
    tot = inc + exc
    if tot < MINCOV:
        return np.nan, inc, exc
    return inc / tot, inc, exc


def parse_mane_from_gtf(gtf_path):
    mane_select = set()
    mane_plus   = set()
    with open(gtf_path) as f:
        for line in f:
            if "\ttranscript\t" not in line:
                continue
            m = re.search(r'transcript_id "([^"]+)"', line)
            if not m:
                continue
            tx = m.group(1).split(".")[0]
            if "MANE_Select" in line:
                mane_select.add(tx)
            if "MANE_Plus_Clinical" in line:
                mane_plus.add(tx)
    return mane_select, mane_plus


def mane_label(transcript_id, mane_select, mane_plus):
    tx = str(transcript_id).split(".")[0]
    if tx in mane_select:
        return "MANE Select"
    if tx in mane_plus:
        return "MANE Plus Clinical"
    return ""


def mane_priority(mane_status):
    if mane_status == "MANE Select":
        return 0
    if mane_status == "MANE Plus Clinical":
        return 1
    return 2


# ── Load source supertable (immutable gz) ──────────────────────────────────
print("Loading source supertable from gz...")
with gzip.open(GZ_IN, "rt") as f:
    st = pd.read_csv(f, low_memory=False)
print(f"  Rows: {len(st):,}")

st["intron1_len"] = st["intron1"].str.len()
st["exon_len"]    = st["exon"].str.len()

# Sanity check on the construct invariant for the source data.
bad_len = (st["intron1_len"] + st["exon_len"] + st["intron2"].str.len() != 161).sum()
if bad_len:
    print(f"  WARNING: {bad_len} source rows do not sum to 161 nt", file=sys.stderr)


# ── Add mane_status from GTF ───────────────────────────────────────────────
print("Parsing MANE status from GTF...")
mane_select, mane_plus = parse_mane_from_gtf(GTF)
print(f"  MANE Select: {len(mane_select):,}  MANE Plus Clinical: {len(mane_plus):,}")
st["mane_status"] = st["transcript_id"].apply(
    lambda t: mane_label(t, mane_select, mane_plus)
)
print("  mane_status counts:")
for status, n in st["mane_status"].value_counts(dropna=False).items():
    print(f"    {status or '(none)':<25} {n:,}")


# ── Detect ambiguous events ────────────────────────────────────────────────
intron_counts = st.groupby("event_id")["intron1_len"].nunique()
ambig_eids = set(intron_counts[intron_counts > 1].index)
print(f"\nAmbiguous events (>1 intron1_len in event): {len(ambig_eids)}")


# ── Load pickles ───────────────────────────────────────────────────────────
print("\nLoading pickles...")
pkls = {}
for rep_name, (_, pkl_path) in REP_INPUTS.items():
    with open(pkl_path, "rb") as f:
        pkls[rep_name] = pickle.load(f)
    print(f"  {rep_name}: {len(pkls[rep_name]):,} refs")


# ── Pick chosen junction per ambiguous event ──────────────────────────────
print("\nPicking chosen junction per ambiguous event...")

# Catalog candidate transcripts per event.
event_transcripts = {}
for eid in ambig_eids:
    sub = (st[st["event_id"] == eid]
           .drop_duplicates("transcript_id")
           .sort_values("Reference"))
    txs = []
    for _, row in sub.iterrows():
        txs.append({
            "mane_priority":   mane_priority(row["mane_status"]),
            "Reference":       int(row["Reference"]),
            "intron1_len":     int(row["intron1_len"]),
            "exon_len":        int(row["exon_len"]),
            "transcript_id":   row["transcript_id"],
            "mane_status":     row["mane_status"],
            "exon_start_hg38": row["exon_start_hg38"],
            "exon_end_hg38":   row["exon_end_hg38"],
        })
    event_transcripts[eid] = txs

# For each event use the primary reference (lowest Reference, snp==none) to
# count how many replicates have reads at each candidate junction.
primary_refs = {}
for eid in ambig_eids:
    sub = st[(st["event_id"] == eid) & (st["snp"] == "none")].sort_values("Reference")
    if len(sub):
        # Pickle keys are 1-indexed; supertable Reference is 0-indexed.
        primary_refs[eid] = int(sub.iloc[0]["Reference"]) + 1

# Map every Reference per event (1-indexed for pickle lookup) for the fallback
# path: if the primary reference is a synthesis dropout (zero reads everywhere),
# pool reads across ALL refs in the event to pick the junction with the most
# data instead of skipping the event entirely.
event_all_refs_1idx = {}
for eid in ambig_eids:
    refs = (st.loc[st["event_id"] == eid, "Reference"].astype(int) + 1).tolist()
    event_all_refs_1idx[eid] = refs

chosen_junction = {}
fallback_events = set()
for eid, txs in event_transcripts.items():
    pref_ref = primary_refs.get(eid)
    if pref_ref is None:
        continue

    # First pass: rate each junction by primary-ref evidence (2 reps with reads
    # at i1 ∧ i2 ≥ MINCOV is the bar from the 04/24 logic).
    candidates = []
    for tx in txs:
        i1, i2 = junctions(tx["intron1_len"], tx["exon_len"])
        valid_reps = 0
        for rep_name, pkl in pkls.items():
            p, _, _ = psi_at(pkl, pref_ref, i1, i2)
            if not np.isnan(p):
                valid_reps += 1
        candidates.append({**tx, "i1": i1, "i2": i2, "valid_reps": valid_reps})

    valid = [c for c in candidates if c["valid_reps"] >= 2]
    if not valid:
        valid = [c for c in candidates if c["valid_reps"] >= 1]

    if valid:
        best = sorted(valid, key=lambda c: (c["mane_priority"], c["Reference"]))[0]
        chosen_junction[eid] = best
        continue

    # Fallback: primary is a dropout. Score each junction by the number of
    # *event refs* that pass MINCOV in any rep, then by summed (inc + exc),
    # then by MANE priority.
    refs = event_all_refs_1idx.get(eid, [])
    fallback_candidates = []
    for tx in txs:
        i1, i2 = junctions(tx["intron1_len"], tx["exon_len"])
        refs_passing = 0
        summed_total = 0
        for ref in refs:
            ref_passes = False
            for rep_name, pkl in pkls.items():
                p, inc, exc = psi_at(pkl, ref, i1, i2)
                summed_total += inc + exc
                if not np.isnan(p):
                    ref_passes = True
            if ref_passes:
                refs_passing += 1
        fallback_candidates.append({
            **tx, "i1": i1, "i2": i2,
            "refs_passing": refs_passing, "summed_total": summed_total,
        })

    has_data = [c for c in fallback_candidates if c["refs_passing"] >= 1]
    if not has_data:
        print(f"  SKIP {eid} — no junction with reads at any ref in any rep")
        continue

    # Highest refs_passing → highest summed_total → MANE > non-MANE → lowest Reference.
    best = sorted(has_data, key=lambda c: (
        -c["refs_passing"], -c["summed_total"], c["mane_priority"], c["Reference"]
    ))[0]
    # Reuse the same downstream rewrite path; carry an extra flag so we know it
    # came from the fallback.
    best["valid_reps"] = 0  # primary ref is a dropout, but the event has data.
    chosen_junction[eid] = best
    fallback_events.add(eid)
    print(f"  FALLBACK {eid} — primary dropout, chose {best['transcript_id']} "
          f"(MANE={best['mane_status'] or 'none'}) from {best['refs_passing']} "
          f"refs / {best['summed_total']} reads")

print(f"  Events with a chosen junction: {len(chosen_junction)}")
print(f"    via primary-ref vote: {len(chosen_junction) - len(fallback_events)}")
print(f"    via variant-pool fallback: {len(fallback_events)}")
print(f"  Skipped (no reads anywhere): {len(ambig_eids) - len(chosen_junction)}")


# ── Cross-check chosen lengths against GTF (fail loudly on mismatch) ───────
print("\nCross-checking chosen junctions against GTF...")
# Parse GTF exons for the chosen transcripts.
chosen_tx_versions = {cj["transcript_id"] for cj in chosen_junction.values()}
chosen_tx_bases    = {tx.split(".")[0] for tx in chosen_tx_versions}
gtf_exon_lens = {}  # transcript_id_base -> set of (start, end, length) tuples
with open(GTF) as f:
    for line in f:
        if "\texon\t" not in line:
            continue
        m = re.search(r'transcript_id "([^"]+)"', line)
        if not m:
            continue
        tx_base = m.group(1).split(".")[0]
        if tx_base not in chosen_tx_bases:
            continue
        parts = line.split("\t")
        try:
            start = int(parts[3])
            end   = int(parts[4])
        except (IndexError, ValueError):
            continue
        gtf_exon_lens.setdefault(tx_base, set()).add((start, end, end - start + 1))

mismatches = 0
for eid, cj in chosen_junction.items():
    tx_base   = cj["transcript_id"].split(".")[0]
    expected  = cj["exon_len"]
    candidates = gtf_exon_lens.get(tx_base, set())
    # exon_len in the supertable is the synthesized exon length; the GTF
    # lists multiple exons per transcript. Require at least one GTF exon
    # of matching length.
    if not any(L == expected for (_, _, L) in candidates):
        mismatches += 1
        print(f"  WARN {eid}: chosen tx {cj['transcript_id']} GTF exon lens "
              f"{sorted({L for (_,_,L) in candidates})} do not include {expected}",
              file=sys.stderr)
if mismatches:
    print(f"  {mismatches} chosen junctions had no GTF exon of matching length", file=sys.stderr)


# ── Apply consistency rewrite to the supertable ───────────────────────────
print("\nApplying consistency rewrite to ambiguous events...")
st_out = st.copy()
rewrites = {"events": 0, "rows": 0, "resplit_rows": 0}

for eid, cj in chosen_junction.items():
    mask = st_out["event_id"] == eid
    n_rows = int(mask.sum())
    if n_rows == 0:
        continue
    rewrites["events"] += 1
    rewrites["rows"]   += n_rows

    i1_target = cj["intron1_len"]
    ex_target = cj["exon_len"]

    # Re-split full_seq for every row in the event.
    full = st_out.loc[mask, "full_seq"].astype(str)
    new_intron1 = full.str.slice(0, i1_target)
    new_exon    = full.str.slice(i1_target, i1_target + ex_target)
    new_intron2 = full.str.slice(i1_target + ex_target)

    # Track how many rows actually changed (sequence-wise).
    changed_seq = (
        (st_out.loc[mask, "intron1"] != new_intron1) |
        (st_out.loc[mask, "exon"]    != new_exon)    |
        (st_out.loc[mask, "intron2"] != new_intron2)
    ).sum()
    rewrites["resplit_rows"] += int(changed_seq)

    st_out.loc[mask, "intron1"]         = new_intron1
    st_out.loc[mask, "exon"]            = new_exon
    st_out.loc[mask, "intron2"]         = new_intron2
    st_out.loc[mask, "intron1_len"]     = i1_target
    st_out.loc[mask, "exon_len"]        = ex_target
    st_out.loc[mask, "transcript_id"]   = cj["transcript_id"]
    st_out.loc[mask, "mane_status"]     = cj["mane_status"]
    st_out.loc[mask, "exon_start_hg38"] = cj["exon_start_hg38"]
    st_out.loc[mask, "exon_end_hg38"]   = cj["exon_end_hg38"]

    # Sanity: invariants must hold for every row in the event after rewrite.
    assert (st_out.loc[mask, "intron1"].str.len() == i1_target).all(), eid
    assert (st_out.loc[mask, "exon"].str.len()    == ex_target).all(), eid
    assert (st_out.loc[mask, "intron2"].str.len() == 20).all(), eid

print(f"  Events rewritten: {rewrites['events']}")
print(f"  Rows in those events: {rewrites['rows']}")
print(f"  Rows whose intron1/exon/intron2 actually changed: {rewrites['resplit_rows']}")


# ── Save corrected supertable (preserve source row order → Reference stable) ──
os.makedirs(OUT_DIR, exist_ok=True)
st_out.to_csv(ST_OUT, index=False)
print(f"\nSaved corrected supertable: {ST_OUT}")
print(f"  Rows: {len(st_out):,}   Columns: {len(st_out.columns)}")


# ── Patch per-replicate PSI text files ────────────────────────────────────
print(f"\nPatching per-replicate PSI text files into {PSI_OUT_DIR}/")
os.makedirs(PSI_OUT_DIR, exist_ok=True)

# Build event_id → (chosen_i1, chosen_i2) for ambiguous events with a chosen junction.
event_chosen_jxn = {eid: (cj["i1"], cj["i2"]) for eid, cj in chosen_junction.items()}

# Build set of ambiguous-event Reference IDs (1-indexed for pickle/PSI files).
ambig_refs_1idx_to_event = {}
for _, row in st.iterrows():
    eid = row["event_id"]
    if eid in event_chosen_jxn:
        ambig_refs_1idx_to_event[int(row["Reference"]) + 1] = eid

print(f"  {len(ambig_refs_1idx_to_event):,} References in ambiguous events with a chosen junction")

patch_summary = {}
for rep_name, (psi_in, _) in REP_INPUTS.items():
    pkl = pkls[rep_name]
    out_path = os.path.join(PSI_OUT_DIR, os.path.basename(psi_in))

    # Original file: 3 cols, tab-separated, header: Reference, PSI, Coverage.
    orig = pd.read_csv(psi_in, sep="\t", engine="python")
    orig_refs = set(orig["Reference"].astype(int))
    orig_indexed = orig.set_index("Reference")

    # Recompute (PSI, Coverage) at the chosen junction for ambig refs.
    new_rows = {}  # ref(1-idx) -> (psi, coverage)
    for ref_1idx, eid in ambig_refs_1idx_to_event.items():
        i1, i2 = event_chosen_jxn[eid]
        p, inc, exc = psi_at(pkl, ref_1idx, i1, i2)
        if np.isnan(p):
            continue  # drop — does not pass MINCOV at chosen junction
        new_rows[ref_1idx] = (float(p), float(inc + exc))

    # Build patched dataframe: non-ambig rows from original + recomputed ambig rows.
    keep_orig = orig[~orig["Reference"].isin(ambig_refs_1idx_to_event)]
    if new_rows:
        new_df = pd.DataFrame(
            [(r, p, c) for r, (p, c) in new_rows.items()],
            columns=["Reference", "PSI", "Coverage"],
        )
    else:
        new_df = pd.DataFrame(columns=["Reference", "PSI", "Coverage"])

    patched = pd.concat([keep_orig, new_df], ignore_index=True)
    patched = patched.sort_values("Reference").reset_index(drop=True)
    patched.to_csv(out_path, sep="\t", index=False)

    # Patch summary
    n_added   = sum(1 for r in new_rows if r not in orig_refs)
    n_dropped = sum(1 for r in ambig_refs_1idx_to_event if r in orig_refs and r not in new_rows)
    n_changed = sum(1 for r in new_rows if r in orig_refs)
    patch_summary[rep_name] = {
        "in_rows": len(orig),
        "out_rows": len(patched),
        "added":   n_added,
        "dropped": n_dropped,
        "changed": n_changed,
    }
    print(f"  {rep_name:<18}  in={len(orig):>7,}  out={len(patched):>7,}  "
          f"changed={n_changed:>5}  added={n_added:>5}  dropped={n_dropped:>5}")


# ── Summary ────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"Source rows: {len(st):,}")
print(f"Ambiguous events: {len(ambig_eids)}")
print(f"  With chosen junction: {len(chosen_junction)}")
print(f"  Skipped (no reads):   {len(ambig_eids) - len(chosen_junction)}")
print(f"Rows in events with chosen junction: {rewrites['rows']:,}")
print(f"Rows whose intron1/exon/intron2 changed: {rewrites['resplit_rows']:,}")
print(f"\nOutputs:")
print(f"  Corrected supertable: {ST_OUT}")
print(f"  Patched PSI dir:      {PSI_OUT_DIR}/")
print("\nNext: run /ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/02_run_merge_psi.sh")
