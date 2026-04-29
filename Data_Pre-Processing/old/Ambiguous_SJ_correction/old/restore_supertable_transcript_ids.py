"""
Restore per-row transcript_id, mane_status, exon_start_hg38, exon_end_hg38
for events that were incorrectly homogenized by a previous run of script 07.

For each ambiguous event (detected by intron1_len diversity), maps each
distinct intron1_len back to its original transcript via the Gencode GTF,
then restores the correct annotation for every row.

Logic: for a given event_id (chr:start-end:strand) and intron1_len, the
exon start/end in the genome can be derived from the construct geometry,
and the matching Gencode transcript identified.

Construct geometry (0-indexed, shared 5' context = 286 nt):
  For + strand: exon_start_hg38 = event_start - intron1_len (approx)
  For - strand: exon_end_hg38   = event_end   + intron1_len (approx)
  (exact: derived from the supertable reference rows for non-affected events)

More robustly: parse GTF for each affected event's gene, find transcripts
whose exon boundaries match the event family, then match each intron1_len
to a transcript by reconstructing the MPRA intron1 length.
"""

import re
import pandas as pd
import numpy as np

SUPERTABLE = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv"
GTF        = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/gencode.v48.annotation.gtf"

print("Loading supertable...")
st = pd.read_csv(SUPERTABLE, low_memory=False)
st["_intron1_len"] = st["intron1"].str.len()
st["_exon_len"]    = st["exon"].str.len()

# Detect ambiguous events by intron1_len diversity
intron_counts = st.groupby("event_id")["_intron1_len"].nunique()
ambig_eids = set(intron_counts[intron_counts > 1].index)
print(f"Ambiguous events: {len(ambig_eids)}")

# Detect affected events: all rows have same transcript_id but multiple intron1_lens
affected = []
for eid in ambig_eids:
    sub = st[st["event_id"] == eid]
    if sub["transcript_id"].nunique() == 1 and sub["_intron1_len"].nunique() > 1:
        affected.append(eid)
print(f"Events needing transcript_id restoration: {len(affected)}")
for e in affected:
    print(f"  {e}")

if not affected:
    print("Nothing to restore.")
    import sys; sys.exit(0)

# Parse GTF: build map of transcript_id -> list of exon (start, end, chrom, strand)
print("\nParsing GTF for exon features...")
tx_exons = {}   # transcript_id (no version) -> list of (chrom, start, end, strand)
mane_select = set()
mane_plus   = set()

with open(GTF) as f:
    for line in f:
        if line.startswith('#'):
            continue
        if '\texon\t' not in line:
            continue
        fields = line.strip().split('\t')
        chrom, ftype, start, end, strand = fields[0], fields[2], int(fields[3]), int(fields[4]), fields[6]
        m = re.search(r'transcript_id "([^"]+)"', line)
        if not m:
            continue
        tx = m.group(1)
        tx_novers = tx.split('.')[0]
        if tx_novers not in tx_exons:
            tx_exons[tx_novers] = []
        tx_exons[tx_novers].append((chrom, start, end, strand))

# Parse MANE status from transcript lines
with open(GTF) as f:
    for line in f:
        if '\ttranscript\t' not in line:
            continue
        m = re.search(r'transcript_id "([^"]+)"', line)
        if not m:
            continue
        tx = m.group(1).split('.')[0]
        if 'MANE_Select' in line:
            mane_select.add(tx)
        if 'MANE_Plus_Clinical' in line:
            mane_plus.add(tx)

def mane_label(tx_id):
    tx = str(tx_id).split('.')[0]
    if tx in mane_select:
        return 'MANE Select'
    if tx in mane_plus:
        return 'MANE Plus Clinical'
    return ''

# For each affected event, find transcripts and match intron1_len
SHARED_5P = 286   # construct 5' context length

print("\nRestoring transcript_id per row...")
st_out = st.copy()

for eid in affected:
    # Parse event_id: chr:start-end:strand (0-based exon family boundary)
    parts = eid.rsplit(':', 1)
    strand = parts[1]
    coord_part = parts[0]
    chrom_part, coords = coord_part.rsplit(':', 1)
    ev_start, ev_end = [int(x) for x in coords.split('-')]

    sub = st[st["event_id"] == eid]
    gene_name = sub["gene_exon"].iloc[0].split()[0] if len(sub) else "?"
    unique_i1_lens = sorted(sub["_intron1_len"].unique())
    print(f"\n  {eid} ({gene_name}): intron1_lens = {unique_i1_lens}")

    # For each intron1_len, derive expected exon_start/end
    # The construct has: [286 nt shared 5'] [intron1] [exon] [20 nt intron2] [shared 3']
    # exon_start_hg38 and exon_end_hg38 reflect genomic exon boundaries
    # For + strand: genomic exon starts = event_region_start + offset by intron1
    # For - strand: genomic exon ends   = event_region_end   - offset by intron1
    # More precisely: look at non-affected events with same family to derive formula,
    # or use the supertable's surviving ref rows for non-homogenized events.

    # Strategy: match each intron1_len to a Gencode transcript by finding transcripts
    # overlapping the event region and computing their intron1_len in the construct.
    # For a transcript with exon at (tx_ex_start, tx_ex_end):
    #   if + strand: intron1_len = tx_ex_start - (event_region_start)   [approx]
    #   if - strand: intron1_len = (event_region_end) - tx_ex_end        [approx]
    # where event_region = ev_start..ev_end (hg38, 1-based GTF coords)

    # Collect candidate transcripts from GTF that have an exon overlapping this region
    candidates = {}   # intron1_len -> {transcript_id, exon_start, exon_end, mane_status}
    for tx_id, exons in tx_exons.items():
        for (chrom, ex_s, ex_e, ex_strand) in exons:
            if chrom != chrom_part or ex_strand != strand:
                continue
            # Check overlap with event region (GTF is 1-based)
            ev_s1 = ev_start + 1   # convert 0-based to 1-based
            ev_e1 = ev_end
            if ex_e < ev_s1 or ex_s > ev_e1:
                continue
            # Compute intron1_len from this transcript's exon boundary
            if strand == '+':
                i1_len = ex_s - ev_s1   # bases between event start and exon start
            else:
                i1_len = ev_e1 - ex_e   # bases between exon end and event end
            if i1_len < 0:
                continue
            # Match to a known intron1_len in the supertable
            if i1_len in unique_i1_lens:
                # Prefer MANE transcripts when multiple match the same length
                existing = candidates.get(i1_len)
                ml = mane_label(tx_id)
                if existing is None or (ml and not existing["mane_status"]):
                    # Find full versioned transcript_id
                    full_tx = None
                    for tx_full, exons2 in tx_exons.items():
                        if tx_full == tx_id:
                            # Find a Gencode line with this tx to get versioned ID
                            break
                    candidates[i1_len] = {
                        "transcript_id":   tx_id,
                        "mane_status":     ml,
                        "exon_start_hg38": ex_s - 1,   # convert back to 0-based
                        "exon_end_hg38":   ex_e,
                    }

    print(f"    GTF matches: {candidates}")

    if len(candidates) < len(unique_i1_lens):
        print(f"    WARNING: only matched {len(candidates)}/{len(unique_i1_lens)} intron1_lens — skipping")
        continue

    # Look up versioned transcript_id from original supertable data (other ambig events)
    # by matching unversioned tx_id. Use the chosen_junction_transcript_id if available.
    # Fallback: use the unversioned tx_id as-is.

    # Apply per-row restoration
    mask = st_out["event_id"] == eid
    rows_fixed = 0
    for i1_len, info in candidates.items():
        row_mask = mask & (st_out["_intron1_len"] == i1_len)
        n = row_mask.sum()
        # Try to find versioned transcript_id from supertable (non-affected events)
        versioned_tx = info["transcript_id"]
        st_match = st[
            (st["transcript_id"].str.startswith(info["transcript_id"])) &
            ~(st["event_id"].isin(affected))
        ]
        if len(st_match):
            versioned_tx = st_match["transcript_id"].iloc[0]

        st_out.loc[row_mask, "transcript_id"]   = versioned_tx
        st_out.loc[row_mask, "mane_status"]     = info["mane_status"]
        st_out.loc[row_mask, "exon_start_hg38"] = info["exon_start_hg38"]
        st_out.loc[row_mask, "exon_end_hg38"]   = info["exon_end_hg38"]
        rows_fixed += n
        print(f"    i1_len={i1_len}: {versioned_tx} ({info['mane_status'] or 'none'}) → {n} rows")

# Drop temp columns
st_out = st_out.drop(columns=["_intron1_len", "_exon_len"])

st_out.to_csv(SUPERTABLE, index=False)
print(f"\nSaved: {SUPERTABLE}")
