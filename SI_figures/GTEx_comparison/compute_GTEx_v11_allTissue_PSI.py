"""
compute_GTEx_v11_allTissue_PSI.py

Computes pan-tissue GTEx v11 PSI for MPRA reference exons.
Identical logic to compute_GTEx_v8_allTissue_PSI.py, updated for v11.

Steps:
1. Parse hg38.knownGene.gtf → transcript_dict[transcript_id]
2. Load MPRA WT rows; parse exon_num from gene_exon, strand from event_id
3. For each WT row look up exon in transcript_dict → find prev/next exon
   → derive junction triplets (j1, j2, excl)
4. Convert junctions to GTEx key format (chr_start_end)
5. Stream GTEx v11 GCT → accumulate pooled counts for target junctions
6. Compute per-Reference PSI (coverage >= 20, both inclusion jxns > 0)
7. Apply logit transform (clip 0.01-0.99) and write output

Output:
  output_GTEx_v11_WT_comparison/GTEx_v11_allTissue_avg_STAR_ref_PSIs.txt
  output_GTEx_v11_WT_comparison/mpra_wt_knownGene_jxn_STAR_ref.txt
"""

import os
import re
import gzip
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--mincov", type=int, default=20,
                    help="Minimum pooled read coverage to include an exon (default: 20)")
args = parser.parse_args()
MINCOV = args.mincov

# ── Paths ────────────────────────────────────────────────────────────────────
KNOWNGENE_GTF = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/hg38.knownGene.gtf"
MPRA_CSV      = "/ESL/Figures_SK/General_preprocessing/output_03_16_2026/03_16_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"
GTEX_GCT      = "/ESL/Data/GTEx/GTEx_Analysis_2025-08-22_v11_STARv2.7.11b_junctions.gct.gz"
OUTPUT_DIR    = f"/ESL/Analysis/Native/output_GTEx_v11_WT_comparison_mincov{MINCOV}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Parse knownGene GTF ──────────────────────────────────────────────
print("Parsing hg38.knownGene.gtf ...", flush=True)

transcript_dict = defaultdict(list)
attr_re = re.compile(r'(\w+)\s+"([^"]+)"')

with open(KNOWNGENE_GTF, "r") as f:
    for line in f:
        if line[0] == "#":
            continue
        var = line.rstrip("\n").split("\t")
        if len(var) < 9 or var[2] != "exon":
            continue
        attrs = dict(attr_re.findall(var[8]))
        if "transcript_id" not in attrs or "exon_number" not in attrs:
            continue

        transcript_id = attrs["transcript_id"]
        exon_number   = int(attrs["exon_number"])
        chrom         = var[0]
        start         = int(var[3])
        end           = int(var[4])
        strand        = var[6]

        transcript_dict[transcript_id].append((exon_number, chrom, start, end, strand))

for tid in transcript_dict:
    transcript_dict[tid].sort(key=lambda x: x[0])

print(f"  Loaded {len(transcript_dict):,} transcripts", flush=True)

# ── Step 2: Load MPRA WT rows ────────────────────────────────────────────────
print("Loading MPRA data ...", flush=True)
df = pd.read_csv(MPRA_CSV, low_memory=False)
wt = df[df["snp"] == "none"].copy()
print(f"  {len(wt)} WT rows, {wt['Reference'].nunique()} unique References", flush=True)

wt["exon_num"] = wt["gene_exon"].str.extract(r"exon\s+(\d+)").astype(float)
wt["strand"]   = wt["event_id"].str.rsplit(":", n=1).str[-1]

# ── Step 3: Derive junction triplets via knownGene lookup ───────────────────
print("Deriving junction triplets from knownGene GTF ...", flush=True)

star_ref_jxns    = {}
no_transcript    = []
no_exon_num      = []
first_last_exons = []
no_neighbors     = []
jxn_rows         = []

for _, row in wt.iterrows():
    ref           = int(row["Reference"])
    transcript_id = str(row["transcript_id"])
    strand        = row["strand"]

    if pd.isna(row["exon_num"]):
        no_exon_num.append(ref)
        continue

    exon_num = int(row["exon_num"])

    if transcript_id not in transcript_dict:
        no_transcript.append(ref)
        continue

    all_exons    = transcript_dict[transcript_id]
    exon_numbers = [e[0] for e in all_exons]
    max_exon_num = max(exon_numbers)
    min_exon_num = min(exon_numbers)

    if exon_num == min_exon_num or exon_num == max_exon_num:
        first_last_exons.append(ref)
        continue

    this_exons = [e for e in all_exons if e[0] == exon_num]
    if not this_exons:
        no_exon_num.append(ref)
        continue

    prev_exons = [e for e in all_exons if e[0] == exon_num - 1]
    next_exons = [e for e in all_exons if e[0] == exon_num + 1]

    if not prev_exons or not next_exons:
        no_neighbors.append(ref)
        continue

    jxns_for_ref      = []
    seen_jxn_triplets = set()

    for this_e in this_exons:
        _, chrom, ex_start, ex_end, ex_strand = this_e
        chr_num = chrom[3:] if chrom.startswith("chr") else chrom

        for prev_e in prev_exons:
            for next_e in next_exons:
                if strand == "-":
                    j1   = "chr%s:%s-%s:%s" % (chr_num, ex_end + 1,    prev_e[2] - 1, strand)
                    j2   = "chr%s:%s-%s:%s" % (chr_num, next_e[3] + 1, ex_start - 1,  strand)
                    excl = "chr%s:%s-%s:%s" % (chr_num, next_e[3] + 1, prev_e[2] - 1, strand)
                else:
                    j1   = "chr%s:%s-%s:%s" % (chr_num, prev_e[3] + 1, ex_start - 1, strand)
                    j2   = "chr%s:%s-%s:%s" % (chr_num, ex_end + 1,    next_e[2] - 1, strand)
                    excl = "chr%s:%s-%s:%s" % (chr_num, prev_e[3] + 1, next_e[2] - 1, strand)

                triplet = (j1, j2, excl)
                if triplet not in seen_jxn_triplets:
                    seen_jxn_triplets.add(triplet)
                    jxns_for_ref.append(triplet)

    if jxns_for_ref:
        star_ref_jxns[ref] = jxns_for_ref
        for j1, j2, excl in jxns_for_ref:
            jxn_rows.append(f"{ref}\t{j1},{j2},{excl}")

print(f"  Derived junctions for {len(star_ref_jxns):,} References", flush=True)
print(f"  No transcript found in knownGene: {len(no_transcript)}", flush=True)
print(f"  No/bad exon_num: {len(no_exon_num)}", flush=True)
print(f"  First/last exon (skipped): {len(set(first_last_exons))}", flush=True)
print(f"  No neighbors in transcript: {len(set(no_neighbors))}", flush=True)

jxn_ref_path = os.path.join(OUTPUT_DIR, "mpra_wt_knownGene_jxn_STAR_ref.txt")
with open(jxn_ref_path, "w") as f:
    f.write("STAR_ref\tj1,j2,excl\n")
    f.write("\n".join(jxn_rows) + "\n")
print(f"  Wrote {jxn_ref_path}", flush=True)

# ── Step 4: Build target junction set in GTEx v11 format ─────────────────────
# GTEx v11 GCT uses 'chrN:start-end:strand' — identical to our internal format.
# No conversion needed (unlike v8 which used 'chrN_start_end').

gtex_key_to_refs = defaultdict(list)

for ref, jxn_list in star_ref_jxns.items():
    for j1, j2, excl in jxn_list:
        gtex_key_to_refs[j1  ].append((ref, "j1"))
        gtex_key_to_refs[j2  ].append((ref, "j2"))
        gtex_key_to_refs[excl].append((ref, "excl"))

target_gtex_keys = set(gtex_key_to_refs.keys())
print(f"  Targeting {len(target_gtex_keys):,} unique GTEx junction keys", flush=True)

# ── Step 5: Stream GTEx v11 GCT, accumulate pooled counts ───────────────────
print("Streaming GTEx v11 GCT (this may take several minutes) ...", flush=True)
pooled_counts = {}
n_samples     = None
found_count   = 0

with gzip.open(GTEX_GCT, "rt") as f:
    for i, line in enumerate(f):
        if i == 0:
            continue   # "#1.2"
        if i == 1:
            parts     = line.strip().split("\t")
            n_samples = int(parts[1])
            print(f"  GCT: {parts[0]} junctions × {n_samples} samples", flush=True)
            continue
        if i == 2:
            continue   # header row (Name, Description, sample1, ...)
        parts = line.rstrip("\n").split("\t")
        key   = parts[0]
        if key in target_gtex_keys:
            pooled_counts[key] = sum(float(x) for x in parts[2:])
            found_count += 1
            if found_count % 200 == 0:
                print(f"  Found {found_count} / {len(target_gtex_keys)} target junctions ...",
                      flush=True)

print(f"  Done. Found {len(pooled_counts):,} of {len(target_gtex_keys):,} target junctions in GTEx",
      flush=True)

# ── Step 6: Compute PSI per Reference ────────────────────────────────────────
print("Computing PSI per Reference ...", flush=True)

def logit_clip(p, lo=0.01, hi=0.99):
    p = max(lo, min(hi, p))
    return np.log(p / (1 - p))

gtex_psi_results = {}

for ref, jxn_list in star_ref_jxns.items():
    valid_psis = []
    for j1, j2, excl in jxn_list:
        k1   = j1
        k2   = j2
        k_ex = excl

        c_j1   = pooled_counts.get(k1,   0.0)
        c_j2   = pooled_counts.get(k2,   0.0)
        c_excl = pooled_counts.get(k_ex, 0.0)

        incl     = (c_j1 + c_j2) / 2.0
        coverage = incl + c_excl

        if coverage >= MINCOV and c_j1 > 0 and c_j2 > 0:
            psi = incl / coverage
            valid_psis.append(psi)

    if valid_psis:
        gtex_psi_results[ref] = float(np.mean(valid_psis))

print(f"  Computed PSI for {len(gtex_psi_results):,} References", flush=True)

# ── Step 7: Write output ─────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "GTEx_v11_allTissue_avg_STAR_ref_PSIs.txt")
with open(out_path, "w") as f:
    f.write("STAR_ref\tavg_PSI\tlogit_PSI\n")
    for ref in sorted(gtex_psi_results.keys()):
        psi       = gtex_psi_results[ref]
        logit_psi = logit_clip(psi)
        f.write(f"{ref}\t{psi:.6f}\t{logit_psi:.6f}\n")

print(f"Wrote {out_path}", flush=True)
print("Done.", flush=True)
