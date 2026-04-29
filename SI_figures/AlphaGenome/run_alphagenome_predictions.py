"""
AlphaGenome splice site prediction — local model (alphagenome_research).

Test run: 10 reference exons, each with top-N variants by |delta_logit| that
fall at canonical splice junctions (5'SS: first 6 nt of intron2;
3'SS: last 6 nt of intron1).

Install on GPU instance:
    git clone https://github.com/google-deepmind/alphagenome_research.git
    pip install -e ./alphagenome_research

Weights (accept non-commercial license first):
    model = dna_model.create_from_kaggle('all_folds')
    # or: dna_model.create_from_huggingface('all_folds')
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.special import logit
from matplotlib import pylab as plt
from tqdm import tqdm

from alphagenome_research.model import dna_model

# === CONFIG ===
DATA_PATH  = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz"
OUT_DIR    = "/ESL/ESL_MPRA/SI_figures/AlphaGenome/Out"
OUTPUT_TSV = os.path.join(OUT_DIR, "alphagenome_test_sj_variants.tsv")
os.makedirs(OUT_DIR, exist_ok=True)

N_REFS     = 10    # number of reference exons to test
VARS_PER_REF = 3   # top variants per ref (by mean |delta_logit| across cell lines)
SJ_FLANK   = 6     # how many nt into intron to consider a SJ variant

CELL_LINES = ['HeLa', 'K562', 'MCF7', 'HMC3', 'HEK']
DELTA_LOGIT_COLS = [f"{cl}_delta_logit_pooled" for cl in CELL_LINES]

# SMN2/Citrine flanking sequences (same as construct design)
CITRINE_EXON1 = "ATGGTGTCCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTCAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAACTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGATGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAAGTGAAGTTCGAGGGCGACACCCTCGTGAACCGCATCGAGCTAAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAAGTGAACTTCAAGATCCGCCACAACATCGAG"
CITRINE_EXON2 = "GACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAG"
SMN2_INTRON6  = "GTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA"
SMN2_INTRON7  = "AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAAAAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCCAAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTGTGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTAGGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTCTGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAATTCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAG"

PREFIX = CITRINE_EXON1 + SMN2_INTRON6
SUFFIX = SMN2_INTRON7 + CITRINE_EXON2

# === LOAD MODEL ===
print(f"{datetime.datetime.now()}: Loading AlphaGenome model...")
model = dna_model.create_from_kaggle('all_folds')
# model = dna_model.create_from_huggingface('all_folds')  # alternative

SEQ_LEN = 16_384  # 16KB — smallest supported length, sufficient for synthetic ~1kb constructs
CENTER  = SEQ_LEN // 2

# SPLICE_SITES track layout (confirmed):
# index 0 = donor    +
# index 1 = acceptor +
# index 2 = donor    -
# index 3 = acceptor -
pos_donor_idx    = [0]
pos_acceptor_idx = [1]


# === HELPERS ===
def get_splice_scores(seq):
    """Pad sequence to 1MB centered and return splice site values + offset."""
    padded = seq.center(SEQ_LEN, 'N')
    offset = CENTER - len(seq) // 2
    out = model.predict_sequence(
        sequence=padded,
        organism=dna_model.Organism.HOMO_SAPIENS,
        requested_outputs={dna_model.OutputType.SPLICE_SITES},
        ontology_terms=['UBERON:0000955'],
    )
    return out.splice_sites.values, offset


def safe_logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return logit(x)


def compute_psi_scores(values, offset, exon_start, exon_end):
    """
    Extract SA (acceptor) at 3'SS and SD (donor) at 5'SS, return PSI = SA * SD.
    exon_start/exon_end are 1-indexed relative to the full padded sequence start.
    AlphaGenome peaks at -1 relative to the exon boundary (i.e. the AG/GT
    dinucleotide last base in the intron), confirmed by position scan.
    """
    pos_3ss = offset + exon_start - 2  # -1 offset: G of AG at 3'SS
    pos_5ss = offset + exon_end - 2    # -1 offset: T of GT at 5'SS
    sa = float(values[pos_3ss, pos_acceptor_idx[0]])
    sd = float(values[pos_5ss, pos_donor_idx[0]])
    psi = sa * sd
    return sa, sd, psi


def is_sj_variant(snp, len_intron1, len_exon):
    """
    Returns True if variant falls within SJ_FLANK nt of either splice junction.
    snp format: 'pos:REF>ALT' (0-indexed within intron1+exon+intron2)
    5'SS = first SJ_FLANK nt of intron2 (positions len_intron1+len_exon to +SJ_FLANK)
    3'SS = last SJ_FLANK nt of intron1  (positions len_intron1-SJ_FLANK to len_intron1-1)
    """
    try:
        pos = int(snp.split(':')[0])
    except (ValueError, IndexError):
        return False
    at_5ss = (len_intron1 + len_exon) <= pos < (len_intron1 + len_exon + SJ_FLANK)
    at_3ss = (len_intron1 - SJ_FLANK) <= pos < len_intron1
    return at_5ss or at_3ss


# === SELECT TEST VARIANTS ===
print(f"{datetime.datetime.now()}: Loading data and selecting test variants...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Only single variants with all required columns present
vars_df = df[(df['snp'] != 'none') & (df['seq_type'] == 'single')].copy()

# Compute mean |delta_logit| across cell lines
vars_df['mean_abs_dlogit'] = vars_df[DELTA_LOGIT_COLS].apply(
    pd.to_numeric, errors='coerce'
).abs().mean(axis=1)

# Keep only SJ variants
ref_df = df[df['snp'] == 'none'][['event_id', 'intron1', 'exon', 'intron2']].drop_duplicates('event_id')
vars_df = vars_df.merge(ref_df, on='event_id', suffixes=('', '_ref'))
vars_df['is_sj'] = vars_df.apply(
    lambda r: is_sj_variant(r['snp'], len(str(r['intron1_ref'])), len(str(r['exon_ref']))),
    axis=1
)
sj_vars = vars_df[vars_df['is_sj']].copy()
print(f"  SJ variants found: {len(sj_vars)}")

# Pick top VARS_PER_REF variants per event_id, then pick N_REFS event_ids with largest max effect
sj_vars_sorted = (
    sj_vars
    .sort_values('mean_abs_dlogit', ascending=False)
    .groupby('event_id')
    .head(VARS_PER_REF)
)
top_event_ids = (
    sj_vars_sorted
    .groupby('event_id')['mean_abs_dlogit']
    .max()
    .nlargest(N_REFS)
    .index
    .tolist()
)
test_vars = sj_vars_sorted[sj_vars_sorted['event_id'].isin(top_event_ids)].copy()
print(f"  Selected {len(top_event_ids)} event_ids, {len(test_vars)} total test variants:")
for eid in top_event_ids:
    rows = test_vars[test_vars['event_id'] == eid]
    print(f"    {eid}: {rows['snp'].tolist()}, |dlogit|={rows['mean_abs_dlogit'].round(2).tolist()}")


# === MAIN LOOP ===
if os.path.exists(OUTPUT_TSV):
    os.remove(OUTPUT_TSV)
header_written = False

for event_id in tqdm(top_event_ids, desc="Processing exon families"):
    ref_row  = df[(df['event_id'] == event_id) & (df['snp'] == 'none')].iloc[0]
    alt_rows = test_vars[test_vars['event_id'] == event_id]

    ref_core  = ref_row['intron1'] + ref_row['exon'] + ref_row['intron2']
    ref_full  = PREFIX + ref_core.upper().replace('U', 'T') + SUFFIX
    exon_start = len(PREFIX) + len(ref_row['intron1']) + 1
    exon_end   = exon_start + len(ref_row['exon']) - 1

    ref_values, offset = get_splice_scores(ref_full)

    # Print full splice site score array over the variable region
    print(f"\n--- REF scores for {event_id} (variable region, shape={ref_values[offset:offset+len(ref_full)].shape}) ---")
    print(f"  Acceptor tracks: {pos_acceptor_idx}")
    print(f"  Donor tracks:    {pos_donor_idx}")
    print(ref_values[offset:offset + len(ref_full)])

    # Diagnostic: print scores ±5 positions around the 3'SS and 5'SS
    # to verify which position AlphaGenome peaks at (may differ from SpliceAI)
    window = 5
    # === POSITION SCAN: find where AlphaGenome peaks around each junction ===
    # Tests SA at offsets [-SCAN_WINDOW, +SCAN_WINDOW] from exon_start (3'SS)
    # and SD at offsets [-SCAN_WINDOW, +SCAN_WINDOW] from exon_end (5'SS)
    # Prints raw scores so you can confirm which position to use
    SCAN_WINDOW = 5
    pos_3ss_base = offset + exon_start - 1
    pos_5ss_base = offset + exon_end - 1

    print(f"\n--- Position scan for {event_id} ---")
    print(f"  3'SS acceptor scores (rel to exon_start, i.e. first exon base):")
    print(f"  {'offset':>8}  {'abs_pos':>8}  {'base':>5}  {'SA':>10}")
    for rel in range(-SCAN_WINDOW, SCAN_WINDOW + 1):
        p = pos_3ss_base + rel
        base = ref_full[p - offset] if 0 <= (p - offset) < len(ref_full) else '?'
        sa = ref_values[p, pos_acceptor_idx[0]]
        print(f"  {rel:>+8d}  {p:>8d}  {base:>5}  {sa:>10.6f}")

    print(f"\n  5'SS donor scores (rel to exon_end, i.e. last exon base):")
    print(f"  {'offset':>8}  {'abs_pos':>8}  {'base':>5}  {'SD':>10}")
    for rel in range(-SCAN_WINDOW, SCAN_WINDOW + 1):
        p = pos_5ss_base + rel
        base = ref_full[p - offset] if 0 <= (p - offset) < len(ref_full) else '?'
        sd = ref_values[p, pos_donor_idx[0]]
        print(f"  {rel:>+8d}  {p:>8d}  {base:>5}  {sd:>10.6f}")

    # Also print SA×SD product for every combination of offsets
    print(f"\n  SA×SD product grid (rows=SA offset, cols=SD offset):")
    sa_offsets = list(range(-SCAN_WINDOW, SCAN_WINDOW + 1))
    sd_offsets = list(range(-SCAN_WINDOW, SCAN_WINDOW + 1))
    header_row = "SA\\SD  " + "  ".join(f"{o:>+4d}" for o in sd_offsets)
    print(f"  {header_row}")
    for sa_rel in sa_offsets:
        sa_val = ref_values[pos_3ss_base + sa_rel, pos_acceptor_idx[0]]
        row_vals = []
        for sd_rel in sd_offsets:
            sd_val = ref_values[pos_5ss_base + sd_rel, pos_donor_idx[0]]
            row_vals.append(f"{sa_val * sd_val:>6.4f}")
        print(f"  {sa_rel:>+4d}   " + "  ".join(row_vals))

    for _, alt_row in alt_rows.iterrows():
        alt_core   = alt_row['intron1'] + alt_row['exon'] + alt_row['intron2']
        alt_full   = PREFIX + alt_core.upper().replace('U', 'T') + SUFFIX
        alt_values, _ = get_splice_scores(alt_full)

        ref_sa, ref_sd, ref_psi = compute_psi_scores(ref_values, offset, exon_start, exon_end)
        alt_sa, alt_sd, alt_psi = compute_psi_scores(alt_values, offset, exon_start, exon_end)

        delta_psi   = alt_psi - ref_psi
        delta_logit = safe_logit(alt_psi) - safe_logit(ref_psi)

        exp_logit_avg = np.nanmean(alt_row[DELTA_LOGIT_COLS].values.astype(float))
        exp_dpsi_avg  = np.nanmean(
            alt_row[[f"{cl}_dpsi_pooled" for cl in CELL_LINES]].values.astype(float)
        )

        # Also output SA/SD at all scanned offsets for post-hoc analysis
        sa_by_offset = {f'ref_sa_rel{rel:+d}': float(ref_values[pos_3ss_base + rel, pos_acceptor_idx[0]])
                        for rel in sa_offsets}
        sd_by_offset = {f'ref_sd_rel{rel:+d}': float(ref_values[pos_5ss_base + rel, pos_donor_idx[0]])
                        for rel in sd_offsets}
        alt_sa_by_offset = {f'alt_sa_rel{rel:+d}': float(alt_values[pos_3ss_base + rel, pos_acceptor_idx[0]])
                            for rel in sa_offsets}
        alt_sd_by_offset = {f'alt_sd_rel{rel:+d}': float(alt_values[pos_5ss_base + rel, pos_donor_idx[0]])
                            for rel in sd_offsets}

        row = pd.DataFrame([{
            'event_id':                        str(event_id),
            'Reference':                       str(alt_row['Reference']),
            'snp':                             str(alt_row['snp']),
            'gene_exon':                       str(alt_row['gene_exon']),
            'ref_sa':                          ref_sa,
            'ref_sd':                          ref_sd,
            'ref_psi':                         ref_psi,
            'alt_sa':                          alt_sa,
            'alt_sd':                          alt_sd,
            'alt_psi':                         alt_psi,
            'alphagenome_delta_psi':           delta_psi,
            'alphagenome_delta_logit':         delta_logit,
            'experimental_delta_logit_pooled': float(exp_logit_avg),
            'experimental_dpsi_pooled':        float(exp_dpsi_avg),
            **sa_by_offset,
            **sd_by_offset,
            **alt_sa_by_offset,
            **alt_sd_by_offset,
        }])
        row.to_csv(OUTPUT_TSV, sep='\t', mode='a', index=False, header=not header_written)
        header_written = True

print(f"{datetime.datetime.now()}: Done. Results written to {OUTPUT_TSV}")

# === QUICK SANITY PLOT ===
results = pd.read_csv(OUTPUT_TSV, sep='\t')
for score_type, xcol, ycol, xlabel, ylabel in [
    ('delta_logit', 'alphagenome_delta_logit', 'experimental_delta_logit_pooled',
     'AlphaGenome Δlogit(SA×SD)', 'Experimental Δlogit(ψ)'),
    ('delta_psi',   'alphagenome_delta_psi',   'experimental_dpsi_pooled',
     'AlphaGenome Δ(SA×SD)',      'Experimental ΔPSI'),
]:
    x = results[xcol].astype(float)
    y = results[ycol].astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    r, _ = pearsonr(x, y)
    n = len(x)

    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=30, alpha=0.7, color='black')
    lim = max(abs(x).max(), abs(y).max()) * 1.1
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.axhline(0, color='gray', lw=0.5); plt.axvline(0, color='gray', lw=0.5)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f'n={n}  r={r:.2f}')
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"alphagenome_test_{score_type}.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"  {score_type}: r={r:.3f}, n={n} → {out_path}")
