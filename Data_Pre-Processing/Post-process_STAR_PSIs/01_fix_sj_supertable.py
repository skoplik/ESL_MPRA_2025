"""
01_fix_sj_supertable.py — 2026-05-01 rewrite.

Fixes ambiguous-event handling in the supertable.

Background:
  The DNA-clustering FASTA assembler keeps only the FIRST occurring supertable
  row per unique 161-nt full_seq (line 192 of assemble_FASTA_…py uses
  `agilent_seqID_desc_dict[curr_seq][0]`). Its `index_1_count = Reference + 1`
  becomes the FASTA id, and all STAR reads for that 161 nt live under that id.
  Subsequent supertable rows with the same full_seq have NO separate FASTA
  entry — their reads physically commingle with the canonical row's barcodes.

Behavior:
  1. Canonical = first occurring row per unique full_seq (matches FASTA).
  2. MANE rewrite (canonical row only) with read-coverage check + fallback:
       - If MANE Select fits the construct window AND has reads in ≥2 reps
         at pkl[canonical_Reference+1] → rewrite canonical row's annotation
         fields to MANE; recompute PSI at MANE junction.
       - Else pick the alt (supertable duplicate or Gencode-window candidate)
         with the most reads (summed across reps) and rewrite to it.
       - If canonical's original transcript is already MANE Select → no rewrite.
  3. Duplicate rows (same full_seq, occurring after canonical) keep their
     ORIGINAL sequence fields untouched. PSI computed against canonical's
     pkl key at the duplicate's own junction.
  4. event_id_161 = original SE:chr:… event_id (preserved).
     event_id      = chr:exon_start_hg38-exon_end_hg38:strand (new).
  5. transcript_class column: MANE / alt / duplicate.
  6. alt_transcripts_in_supertable: other supertable rows sharing this full_seq.
     alt_transcripts_gencode_only: Gencode transcripts in window with reads,
     NOT in original supertable design.
  7. Per-rep PSI text files: one row per supertable Reference, computed at
     each row's own junction against canonical_Reference+1 pkl key.
"""

import gzip
import os
import pickle
import re
import sys
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
# Use the strand-corrected supertable. The original
# st_final_with_snp_and_coords_05_30_25.csv.gz had exon_start_hg38 /
# exon_end_hg38 mirrored across the construct window for - strand events
# (bug in make_new_st_2.py:264). The strandfix file has correct genomic
# coords matching gencode v48.
GZ_IN  = "/ESL/ESL_MPRA/Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25_strandfix.csv"
GTF    = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/gencode.v48.annotation.gtf"

OUT_DIR      = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output"
ST_OUT       = os.path.join(OUT_DIR, "st_corrected.csv")
ST_ALT_OUT   = os.path.join(OUT_DIR, "st_alt_junctions.csv")
PSI_OUT_DIR  = os.path.join(OUT_DIR, "recount_PSIs")

PKL_BASE = "/ESL/Analysis/STAR_alignment/separate_concat_2023_09_19_d1c_ms75_from_s3_2023_11_27/recount_SJs"
PKL_2024 = PKL_BASE + "/2024_07_26"
WT_BASE  = "/ESL/Analysis/WT_Library/separate/recount_SJs"

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

SHARED_5P = 286
MINCOV    = 10
E_JXN     = (26, 871)


# ── Helpers ────────────────────────────────────────────────────────────────
def junctions(i1_len, ex_len):
    i1 = (26, SHARED_5P + i1_len)
    i2 = (SHARED_5P + i1_len + ex_len + 1, 871)
    return i1, i2


def psi_at(pkl_dict, fasta_ref, i1, i2):
    if fasta_ref not in pkl_dict:
        return np.nan, 0, 0
    jd = pkl_dict[fasta_ref]
    inc_i1 = jd.get(i1, 0)
    inc_i2 = jd.get(i2, 0)
    inc = min(inc_i1, inc_i2)
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


def parse_gtf_gene_transcripts_exons(gtf_path):
    gene_txs    = {}
    tx_versioned = {}
    tx_exons    = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            feat = line.split("\t")
            if len(feat) < 9:
                continue
            feature_type = feat[2]
            attrs = feat[8]
            m_tx   = re.search(r'transcript_id "([^"]+)"', attrs)
            m_gene = re.search(r'gene_name "([^"]+)"', attrs)
            if not m_tx:
                continue
            tx_full = m_tx.group(1)
            tx_base = tx_full.split(".")[0]
            if feature_type == "transcript":
                tx_versioned[tx_base] = tx_full
                if m_gene:
                    gene_txs.setdefault(m_gene.group(1), set()).add(tx_base)
            elif feature_type == "exon":
                try:
                    start = int(feat[3])
                    end   = int(feat[4])
                except (IndexError, ValueError):
                    continue
                tx_exons.setdefault(tx_base, []).append((start, end))
    return gene_txs, tx_versioned, tx_exons


def mane_label(tx_id, mane_select, mane_plus):
    tx = str(tx_id).split(".")[0]
    if tx in mane_select:
        return "MANE Select"
    if tx in mane_plus:
        return "MANE Plus Clinical"
    return ""


def junction_in_window(exon_start_hg38, exon_end_hg38, ev_start, ev_end, strand="+"):
    """Compute (intron1_len, exon_len) in TRANSCRIPT order. The supertable's
    intron1 is the 5' intronic flank of the mRNA, so for - strand events
    intron1 corresponds genomically to the region AFTER exon_end_hg38."""
    if strand == "+":
        i1 = exon_start_hg38 - ev_start
    else:
        i1 = ev_end - exon_end_hg38
    ex = exon_end_hg38 - exon_start_hg38 + 1
    if i1 < 0 or ex <= 0 or i1 + ex > 141:
        return None
    return i1, ex


def parse_event_coords(eid):
    """Parse event_id → (chrom, ev_start, ev_end, strand).

    Accepts two formats:
      1. SE:chr:up_start-up_end:dn_start-dn_end:strand  (construct window
         derived as ev_start = up_end+1, ev_end = dn_start-1).
      2. chr12:ev_start-ev_end:strand  (already the construct window).
    Returns None if eid doesn't match either format.
    """
    s = str(eid)
    m = re.match(r'SE:([^:]+):(\d+)-(\d+):(\d+)-(\d+):([+-])$', s)
    if m:
        chrom = m.group(1)
        if not chrom.startswith("chr"):
            chrom = "chr" + chrom
        return chrom, int(m.group(3)) + 1, int(m.group(4)) - 1, m.group(6)
    m = re.match(r'(chr[^:]+):(\d+)-(\d+):([+-])$', s)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    return None


def reads_at_junction(pkl_key, i1_len, ex_len, pkls):
    """Return (valid_reps, total_reads) at this junction across all replicates.

    valid_reps counts reps where Coverage >= MINCOV at this junction.
    total_reads sums inc+exc across all reps (irrespective of MINCOV).
    """
    i1, i2 = junctions(i1_len, ex_len)
    valid = 0
    total = 0
    for pkl in pkls.values():
        p, inc, exc = psi_at(pkl, pkl_key, i1, i2)
        if not np.isnan(p):
            valid += 1
        total += inc + exc
    return valid, total


# ── Load source supertable (strand-corrected) ─────────────────────────────
print("Loading strand-corrected supertable...")
st = pd.read_csv(GZ_IN, low_memory=False)
print(f"  Rows: {len(st):,}")

st["intron1_len"] = st["intron1"].str.len()
st["exon_len"]    = st["exon"].str.len()
bad_len = (st["intron1_len"] + st["exon_len"] + st["intron2"].str.len() != 161).sum()
if bad_len:
    print(f"  WARNING: {bad_len} source rows do not sum to 161 nt", file=sys.stderr)


# ── Parse GTF ──────────────────────────────────────────────────────────────
print("Parsing MANE status from GTF...")
mane_select, mane_plus = parse_mane_from_gtf(GTF)
print(f"  MANE Select: {len(mane_select):,}  MANE Plus Clinical: {len(mane_plus):,}")

print("Parsing GTF gene/transcript/exon structure...")
gene_txs, tx_versioned, tx_exons = parse_gtf_gene_transcripts_exons(GTF)
print(f"  Genes: {len(gene_txs):,}  Transcripts: {len(tx_versioned):,}")

# Extend tx_versioned with any source-supertable transcript_ids not in GTF
for tx in st["transcript_id"].dropna().unique():
    tx_str = str(tx)
    base   = tx_str.split(".")[0]
    if base not in tx_versioned:
        tx_versioned[base] = tx_str

st["mane_status"] = st["transcript_id"].apply(
    lambda t: mane_label(t, mane_select, mane_plus)
)
print("  mane_status counts:")
for status, n in st["mane_status"].value_counts(dropna=False).items():
    print(f"    {status or '(none)':<25} {n:,}")


# ── Load pickles ───────────────────────────────────────────────────────────
print("\nLoading pickles...")
pkls = {}
for rep_name, (_, pkl_path) in REP_INPUTS.items():
    with open(pkl_path, "rb") as f:
        pkls[rep_name] = pickle.load(f)
    print(f"  {rep_name}: {len(pkls[rep_name]):,} refs")


# ── Identify ambiguous vs. duplicate full_seqs ─────────────────────────────
# Ambiguous full_seq = multiple rows with DIFFERENT intron1_len (genuinely
#   different transcript annotations of the same 161 nt). These are the rows
#   that need MANE rewrite + alt processing.
# Duplicate full_seq  = multiple rows with the SAME intron1_len (true
#   duplicates from supertable construction). No rewrite needed; flagged with
#   transcript_class="duplicate".
print("\nIdentifying ambiguous vs. duplicate full_seqs...")
# A full_seq is "ambiguous" iff its rows have multiple distinct
# (intron1_len, exon_len) pairs — i.e. genuinely different SJ splits.
fs_count = st.groupby("full_seq").size()
sj_uniq  = (st.assign(_sj=list(zip(st["intron1_len"], st["exon_len"])))
              .groupby("full_seq")["_sj"].nunique())
ambig_fseqs    = set(sj_uniq[sj_uniq > 1].index)
dup_only_fseqs = set(fs_count[fs_count > 1].index) - ambig_fseqs
n_ambig_rows = int(fs_count.loc[list(ambig_fseqs)].sum()) if ambig_fseqs else 0
n_dup_rows   = int(fs_count.loc[list(dup_only_fseqs)].sum()) if dup_only_fseqs else 0
print(f"  Unique full_seqs total:                          {len(fs_count):,}")
print(f"  Ambiguous full_seqs (diff (intron1,exon) split): {len(ambig_fseqs):,}")
print(f"    Rows in ambiguous full_seqs:                   {n_ambig_rows:,}")
print(f"  Duplicate-only full_seqs (same SJ, repeated):    {len(dup_only_fseqs):,}")
print(f"    Rows in duplicate-only full_seqs:              {n_dup_rows:,}")


# ── Canonical Reference per full_seq (= first occurring = lowest Reference) ──
canonical_ref_by_fseq = (
    st.groupby("full_seq")["Reference"].min().astype(int).to_dict()
)


# ── Helpers for MANE rewrite ───────────────────────────────────────────────
def find_mane_split_for_gene(gene_name, ev_start, ev_end, strand="+"):
    """Look up gene_name's MANE Select transcript and try to fit its exon
    into the construct window. Returns dict with i1/ex/transcript_id/coords
    if a valid MANE junction is found, else None."""
    if gene_name not in gene_txs:
        return None
    for tx_base in gene_txs[gene_name]:
        if tx_base not in mane_select:
            continue
        for (ex_start, ex_end) in tx_exons.get(tx_base, []):
            r = junction_in_window(ex_start, ex_end, ev_start, ev_end, strand)
            if r is None:
                continue
            i1, ex = r
            return {
                "intron1_len": i1, "exon_len": ex,
                "transcript_id": tx_versioned.get(tx_base, tx_base),
                "mane_status": "MANE Select",
                "exon_start_hg38": ex_start,
                "exon_end_hg38": ex_end,
            }
    return None


def gather_alt_candidates(full_seq, gene_name, ev_start, ev_end, source_intron1_lens, strand="+"):
    """Collect all alt junction candidates (other supertable rows + Gencode
    transcripts in window) for a given canonical row.

    `source_intron1_lens` is the set of intron1_lens already in the supertable
    for this full_seq (so we mark which alts came from the supertable design).

    Returns list of dicts with intron1_len/exon_len/transcript_id/etc., each
    annotated with `from_supertable` (bool).
    """
    candidates = []
    seen_i1 = set()

    # From other supertable rows sharing this full_seq
    sub = st[st["full_seq"] == full_seq]
    for _, row in sub.iterrows():
        i1 = int(row["intron1_len"])
        if i1 in seen_i1:
            continue
        seen_i1.add(i1)
        candidates.append({
            "intron1_len": i1,
            "exon_len": int(row["exon_len"]),
            "transcript_id": row["transcript_id"],
            "mane_status": mane_label(row["transcript_id"], mane_select, mane_plus),
            "exon_start_hg38": row["exon_start_hg38"],
            "exon_end_hg38":   row["exon_end_hg38"],
            "from_supertable": True,
        })

    # From Gencode transcripts for this gene. Dedupe by intron1_len only —
    # a Gencode entry for a transcript_id already in the supertable but at a
    # different splice site is still a valid alt to report (different annotation
    # of the same transcript by different gencode versions, or genuinely
    # alternative splice sites).
    if gene_name in gene_txs:
        for tx_base in gene_txs[gene_name]:
            for (ex_start, ex_end) in tx_exons.get(tx_base, []):
                r = junction_in_window(ex_start, ex_end, ev_start, ev_end, strand)
                if r is None:
                    continue
                i1, ex = r
                if i1 in seen_i1:
                    continue
                seen_i1.add(i1)
                tx_full = tx_versioned.get(tx_base, tx_base)
                candidates.append({
                    "intron1_len": i1, "exon_len": ex,
                    "transcript_id": tx_full,
                    "mane_status": mane_label(tx_full, mane_select, mane_plus),
                    "exon_start_hg38": ex_start,
                    "exon_end_hg38": ex_end,
                    "from_supertable": False,
                })
    return candidates


# ── Per-full_seq alt-transcript bookkeeping (no rewrite) ──────────────────
# Builds alt_supertable_by_ref + alt_gencode_by_ref so Stage 3 can find them.
# MANE-canonical rewrite happens later, per-event, not per-full_seq.
print("\nBuilding per-row alt-transcript columns...")

alt_supertable_by_ref = {}
alt_gencode_by_ref    = {}

fs_groups = st.groupby("full_seq", sort=False)
for full_seq, group in fs_groups:
    canonical_ref = int(group["Reference"].min())
    crow_match = group[group["Reference"].astype(int) == canonical_ref]
    if crow_match.empty:
        continue
    crow = crow_match.iloc[0]
    pkl_key = canonical_ref + 1
    coords = parse_event_coords(crow["event_id"])
    gene = crow.get("gene_name", "")

    fs_all_tx_ids = [str(t) for t in group["transcript_id"].dropna().unique().tolist()]

    # Gencode-only alts: discover only for genuinely ambiguous full_seqs.
    gencode_only = []
    if coords is not None and full_seq in ambig_fseqs:
        chrom, ev_start, ev_end, strand = coords
        source_i1s = set(int(i) for i in group["intron1_len"].tolist())
        all_alts = gather_alt_candidates(full_seq, gene, ev_start, ev_end, source_i1s, strand)
        for a in all_alts:
            if a["from_supertable"]:
                continue
            valid_reps, _ = reads_at_junction(pkl_key, a["intron1_len"], a["exon_len"], pkls)
            if valid_reps >= 1:
                gencode_only.append(a["transcript_id"])
    alt_gen_str = ";".join(str(t) for t in gencode_only)

    for ref, own_tx in zip(group["Reference"].astype(int),
                            group["transcript_id"].astype(str)):
        own_alts = [t for t in fs_all_tx_ids if t != own_tx]
        alt_supertable_by_ref[ref] = ";".join(own_alts)
        alt_gencode_by_ref[ref]    = alt_gen_str


# ── Per-event MANE-canonical SJ assignment ────────────────────────────────
# For each ambiguous event_id_161, pick a canonical SJ (MANE Select preferred,
# else alt with most WT reads). For each unique full_seq in the event, if no
# row already uses the canonical SJ, rewrite ONE of its rows (the per-fullseq
# canonical = lowest Reference) to the canonical SJ.
print("\nProcessing ambiguous events for MANE-canonical SJ assignment...")

canonical_rewrites = {}

n_events_mane          = 0
n_events_alt_fallback  = 0
n_events_no_canonical  = 0
n_fullseqs_already_at_canonical = 0
n_fullseqs_rewritten   = 0

# Build event_id_161 → group lookup once
event_groups = st.groupby("event_id", sort=False)

for eid_161, ev_group in event_groups:
    # Identify unique SJs across WT rows in this event
    wt_rows = ev_group[ev_group["snp"] == "none"]
    if wt_rows.empty:
        continue
    wt_sjs = wt_rows[["intron1_len", "exon_len"]].drop_duplicates()
    if len(wt_sjs) <= 1:
        # Not an ambiguous event (only one WT split)
        continue

    coords = parse_event_coords(eid_161)
    if coords is None:
        n_events_no_canonical += 1
        continue
    chrom, ev_start, ev_end, strand = coords
    gene = ev_group["gene_name"].iloc[0]

    # WT pkl key for read-coverage check: pick the lowest-Reference WT canonical
    wt_full_seqs = wt_rows["full_seq"].unique()
    wt_canonical_refs = [
        canonical_ref_by_fseq[fs] for fs in wt_full_seqs
        if fs in canonical_ref_by_fseq
    ]
    if not wt_canonical_refs:
        n_events_no_canonical += 1
        continue
    wt_pkl_key = min(wt_canonical_refs) + 1

    # Pick event canonical SJ — MANE Select if it has reads, else alt-with-most-reads
    canonical_sj_info = None

    mane_split = find_mane_split_for_gene(gene, ev_start, ev_end, strand)
    if mane_split is not None:
        valid_reps, _ = reads_at_junction(
            wt_pkl_key, mane_split["intron1_len"], mane_split["exon_len"], pkls
        )
        if valid_reps >= 2:
            canonical_sj_info = mane_split
            canonical_sj_info["_is_mane"] = True
            n_events_mane += 1

    if canonical_sj_info is None:
        # Fallback: collect alt candidates from any WT full_seq, pick most-reads
        candidates = []
        seen_keys = set()
        for fs in wt_full_seqs:
            source_i1s = set(int(i) for i in ev_group[ev_group["full_seq"] == fs]["intron1_len"].tolist())
            for c in gather_alt_candidates(fs, gene, ev_start, ev_end, source_i1s, strand):
                key = (int(c["intron1_len"]), int(c["exon_len"]))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                candidates.append(c)

        best = None
        best_total = -1
        for c in candidates:
            _, total = reads_at_junction(
                wt_pkl_key, c["intron1_len"], c["exon_len"], pkls
            )
            if total > best_total:
                best_total = total
                best = c

        if best is not None and best_total > 0:
            canonical_sj_info = best
            canonical_sj_info["_is_mane"] = False
            n_events_alt_fallback += 1

    if canonical_sj_info is None:
        n_events_no_canonical += 1
        continue

    canon_i1 = int(canonical_sj_info["intron1_len"])
    canon_ex = int(canonical_sj_info["exon_len"])

    # For each unique full_seq in this event, ensure a row at canonical SJ
    fs_in_event = ev_group["full_seq"].unique()
    for fs in fs_in_event:
        fs_rows = ev_group[ev_group["full_seq"] == fs]
        has_canonical = (
            (fs_rows["intron1_len"].astype(int) == canon_i1) &
            (fs_rows["exon_len"].astype(int) == canon_ex)
        ).any()
        if has_canonical:
            n_fullseqs_already_at_canonical += 1
            continue

        # Rewrite the per-fullseq canonical row (lowest Reference)
        canonical_ref = canonical_ref_by_fseq.get(fs)
        if canonical_ref is None:
            continue

        new_intron1 = fs[:canon_i1]
        new_exon    = fs[canon_i1:canon_i1 + canon_ex]
        new_intron2 = fs[canon_i1 + canon_ex:]
        canonical_rewrites[canonical_ref] = {
            "intron1": new_intron1,
            "exon": new_exon,
            "intron2": new_intron2,
            "intron1_len": canon_i1,
            "exon_len": canon_ex,
            "transcript_id": canonical_sj_info["transcript_id"],
            "transcript_name": canonical_sj_info["transcript_id"],
            "mane_status": canonical_sj_info.get("mane_status", ""),
            "exon_start_hg38": canonical_sj_info["exon_start_hg38"],
            "exon_end_hg38": canonical_sj_info["exon_end_hg38"],
        }
        n_fullseqs_rewritten += 1

print(f"  Events using MANE canonical:                  {n_events_mane:,}")
print(f"  Events using alt-with-most-reads fallback:    {n_events_alt_fallback:,}")
print(f"  Events skipped (no canonical SJ chosen):      {n_events_no_canonical:,}")
print(f"  full_seqs already at canonical SJ (no rewrite): {n_fullseqs_already_at_canonical:,}")
print(f"  full_seqs rewritten to canonical SJ:           {n_fullseqs_rewritten:,}")


# ── Apply rewrites to st_out ───────────────────────────────────────────────
print("\nApplying canonical rewrites...")
st_out = st.copy()
for ref, rewrite in canonical_rewrites.items():
    mask = st_out["Reference"].astype(int) == ref
    for col, val in rewrite.items():
        st_out.loc[mask, col] = val


# ── Add event_id_161 + new event_id ────────────────────────────────────────
print("Building event_id_161 + new event_id (chr:exon_start-exon_end:strand)...")
st_out["event_id_161"] = st_out["event_id"]

# Pre-compute (chrom, strand) per unique event_id_161 once
unique_eids = st_out["event_id_161"].dropna().unique()
eid_chrom_strand = {}
for eid in unique_eids:
    coords = parse_event_coords(eid)
    if coords is not None:
        eid_chrom_strand[eid] = (coords[0], coords[3])

chrom_series  = st_out["event_id_161"].map(lambda e: eid_chrom_strand.get(e, (None, None))[0])
strand_series = st_out["event_id_161"].map(lambda e: eid_chrom_strand.get(e, (None, None))[1])

# Coerce exon_start/end to numeric (NaN-safe)
es_num = pd.to_numeric(st_out["exon_start_hg38"], errors="coerce")
ee_num = pd.to_numeric(st_out["exon_end_hg38"],   errors="coerce")

new_eid = (
    chrom_series.astype(str) + ":" +
    es_num.astype("Int64").astype(str) + "-" +
    ee_num.astype("Int64").astype(str) + ":" +
    strand_series.astype(str)
)
# Fall back to original event_id_161 wherever any field is missing
fallback_mask = (
    chrom_series.isna() | strand_series.isna() | es_num.isna() | ee_num.isna()
)
new_eid = new_eid.where(~fallback_mask, st_out["event_id_161"])
st_out["event_id"] = new_eid


# ── transcript_class column (MANE / alt / duplicate) ───────────────────────
# - MANE: row's mane_status == "MANE Select" AND it's the primary row for its
#         (full_seq, intron1_len, exon_len) combination (lowest Reference).
# - alt:  primary row for its (full_seq, SJ) combination, not MANE.
# - duplicate: non-primary row (i.e., another supertable row already has the
#              same full_seq + same SJ at a lower Reference — barcode-cluster
#              replicate of the same isoform).
print("Assigning transcript_class (vectorized)...")

st_out["intron1_len"] = st_out["intron1_len"].astype(int)
st_out["exon_len"]    = st_out["exon_len"].astype(int)
st_out["Reference"]   = st_out["Reference"].astype(int)

# Primary row per (full_seq, intron1_len, exon_len) = lowest Reference
primary_ref = (
    st_out.groupby(["full_seq", "intron1_len", "exon_len"])["Reference"]
          .transform("min")
)
is_primary = st_out["Reference"] == primary_ref
is_mane    = st_out["mane_status"].astype(str) == "MANE Select"

st_out["transcript_class"] = np.where(
    is_primary & is_mane, "MANE",
    np.where(is_primary, "alt", "duplicate")
)
print("  transcript_class counts:")
for cls, n in st_out["transcript_class"].value_counts(dropna=False).items():
    print(f"    {cls or '(none)':<15} {n:,}")


# ── n_rows_per_full_seq column ────────────────────────────────────────────
# How many supertable rows share each row's full_seq. = 1 means single-SJ,
# single-barcode-cluster (clean). > 1 means either multi-isoform annotation
# or barcode-cluster duplicates of the same isoform (or both).
st_out["n_rows_per_full_seq"] = st_out.groupby("full_seq")["full_seq"].transform("size").astype(int)


# ── alt_transcripts_in_supertable + alt_transcripts_gencode_only ──────────
print("Adding alt_transcripts_in_supertable and alt_transcripts_gencode_only...")
ref_int = st_out["Reference"].astype(int)
st_out["alt_transcripts_in_supertable"] = ref_int.map(alt_supertable_by_ref).fillna("")
st_out["alt_transcripts_gencode_only"]  = ref_int.map(alt_gencode_by_ref).fillna("")

# Count columns — easy filtering for rows with alt-transcript annotations
def _count_semi(s):
    s = str(s)
    return 0 if not s else s.count(";") + 1
st_out["n_alt_transcripts_in_supertable"] = st_out["alt_transcripts_in_supertable"].apply(_count_semi)
st_out["n_alt_transcripts_gencode_only"]  = st_out["alt_transcripts_gencode_only"].apply(_count_semi)


# ── Save corrected supertable ──────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
st_out.to_csv(ST_OUT, index=False)
print(f"\nSaved corrected supertable: {ST_OUT}")
print(f"  Rows: {len(st_out):,}   Columns: {len(st_out.columns)}")


# ── Save st_alt_junctions.csv (Gencode-only alts for SI table) ────────────
# One row per (event_id_161, alt_transcript_id). Lookup table for Stage 3.
# canonical_reference picks the WT MANE row when one exists in the event,
# else the lowest WT Reference. Variant Refs are never used as anchors.
print("\nBuilding st_alt_junctions.csv (Gencode-only alts, deduped per event)...")

# Per-event canonical Reference: WT MANE row preferred, else lowest WT Reference.
wt_st_out = st_out[st_out["snp"] == "none"].copy()
wt_st_out["_is_mane"] = (wt_st_out["mane_status"].astype(str) == "MANE Select").astype(int)
# Sort so MANE rows come first, then lowest Reference
wt_sorted = wt_st_out.sort_values(["event_id_161", "_is_mane", "Reference"],
                                   ascending=[True, False, True])
canon_per_event = wt_sorted.drop_duplicates("event_id_161", keep="first")
canon_lookup = canon_per_event.set_index("event_id_161")

alt_rows = []
seen_pairs = set()
# Pre-build event-level set of supertable transcript bases (for alt_in_supertable flag)
event_tx_bases = (
    st_out.assign(_tx_base=st_out["transcript_id"].astype(str).str.split(".").str[0])
          .groupby("event_id_161")["_tx_base"]
          .apply(lambda s: set(s.dropna()))
          .to_dict()
)

for ref, gen_alts in alt_gencode_by_ref.items():
    if not gen_alts:
        continue
    fs = st.loc[st["Reference"].astype(int) == ref, "full_seq"]
    if fs.empty:
        continue
    # Find which event_id_161 this row belongs to
    eid_match = st_out.loc[st_out["Reference"].astype(int) == ref, "event_id_161"]
    if eid_match.empty:
        continue
    eid_161 = eid_match.iloc[0]
    if eid_161 not in canon_lookup.index:
        continue
    cr = canon_lookup.loc[eid_161]
    if isinstance(cr, pd.DataFrame):
        cr = cr.iloc[0]
    canonical_ref = int(cr["Reference"])
    coords = parse_event_coords(eid_161)
    if coords is None:
        continue
    chrom, ev_start, ev_end, strand = coords
    gene = cr.get("gene_name", "")
    st_tx_bases = event_tx_bases.get(eid_161, set())
    for tx_id in gen_alts.split(";"):
        if not tx_id:
            continue
        key = (eid_161, tx_id)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        tx_base = tx_id.split(".")[0]
        if tx_base not in tx_exons:
            continue
        for (ex_start, ex_end) in tx_exons[tx_base]:
            r = junction_in_window(ex_start, ex_end, ev_start, ev_end, strand)
            if r is None:
                continue
            i1, ex = r
            alt_rows.append({
                "event_id_161":         eid_161,
                "canonical_reference":  canonical_ref,
                "gene_name":            gene,
                "main_transcript_id":   cr["transcript_id"],
                "main_intron1_len":     int(cr["intron1_len"]),
                "main_exon_len":        int(cr["exon_len"]),
                "alt_transcript_id":    tx_id,
                "alt_mane_status":      mane_label(tx_id, mane_select, mane_plus),
                "alt_intron1_len":      i1,
                "alt_exon_len":         ex,
                "alt_exon_start_hg38":  ex_start,
                "alt_exon_end_hg38":    ex_end,
                "alt_in_supertable":    tx_id.split(".")[0] in st_tx_bases,
            })
            break

pd.DataFrame(alt_rows).to_csv(ST_ALT_OUT, index=False)
print(f"Saved: {ST_ALT_OUT}  ({len(alt_rows)} rows, deduped per (event_id_161, alt_transcript_id))")


# ── Patch per-replicate PSI text files ────────────────────────────────────
# Write fresh PSI files: one row per supertable Reference (1-indexed),
# with PSI computed at the row's own (intron1_len, exon_len) junction
# against pkl[canonical_Reference + 1].
print(f"\nWriting per-replicate PSI text files into {PSI_OUT_DIR}/")
os.makedirs(PSI_OUT_DIR, exist_ok=True)

# Vectorized: build numpy arrays for all rows at once
row_ref_1idx = (st_out["Reference"].astype(int) + 1).to_numpy()
row_pkl_key  = (st_out["full_seq"].map(canonical_ref_by_fseq).astype(int) + 1).to_numpy()
row_intron1  = st_out["intron1_len"].astype(int).to_numpy()
row_exon     = st_out["exon_len"].astype(int).to_numpy()

# Pre-compute (i1, i2) tuples for each row (small numpy arrays)
i1_left  = np.full(len(st_out), 26, dtype=np.int64)
i1_right = SHARED_5P + row_intron1
i2_left  = SHARED_5P + row_intron1 + row_exon + 1
i2_right = np.full(len(st_out), 871, dtype=np.int64)

for rep_name, (psi_in, _) in REP_INPUTS.items():
    pkl = pkls[rep_name]
    out_path = os.path.join(PSI_OUT_DIR, os.path.basename(psi_in))

    rows_out = []
    for k in range(len(st_out)):
        pk = int(row_pkl_key[k])
        if pk not in pkl:
            continue
        jd = pkl[pk]
        i1 = (int(i1_left[k]),  int(i1_right[k]))
        i2 = (int(i2_left[k]),  int(i2_right[k]))
        inc_i1 = jd.get(i1, 0)
        inc_i2 = jd.get(i2, 0)
        inc = inc_i1 if inc_i1 < inc_i2 else inc_i2
        exc = jd.get(E_JXN, 0)
        tot = inc + exc
        if tot < MINCOV:
            continue
        rows_out.append((int(row_ref_1idx[k]), inc / tot, tot))

    out_df = pd.DataFrame(rows_out, columns=["Reference", "PSI", "Coverage"])
    out_df = out_df.sort_values("Reference").reset_index(drop=True)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"  {rep_name:<18}  rows={len(out_df):>7,}  -> {out_path}")


# ── Summary ────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"Source rows: {len(st):,}")
print(f"Unique full_seqs: {len(fs_count):,}")
print(f"Ambiguous full_seqs (>1 row): {len(ambig_fseqs):,}")
print(f"Rows in ambiguous full_seqs: {n_ambig_rows:,}")
print(f"Canonical rewrites applied: {len(canonical_rewrites):,}")
print(f"  Events using MANE canonical: {n_events_mane:,}")
print(f"  Events using alt fallback:   {n_events_alt_fallback:,}")
print(f"  Events skipped:              {n_events_no_canonical:,}")
print(f"Output supertable: {ST_OUT}")
print(f"Output Gencode-only alts: {ST_ALT_OUT}")
print(f"Output per-rep PSI dir: {PSI_OUT_DIR}/")
