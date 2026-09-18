"""
AlphaGenome splice site prediction — full run, all single + double variants.

Scoring: PSI = SA * SD at confirmed -1 offset from exon boundaries.
Output: TSV with ref/alt SA, SD, PSI, delta_psi, delta_logit vs experimental.

Install:
    git clone https://github.com/google-deepmind/alphagenome_research.git
    pip install -e ./alphagenome_research alphagenome
    export KAGGLE_USERNAME=... KAGGLE_KEY=...
"""

import os
import datetime
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.special import logit
from matplotlib import pylab as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from alphagenome_research.model import dna_model

# === CONFIG ===
DATA_PATH  = "/content/drive/MyDrive/AlphaGenome/Data_files/03_16_2026_1e-2_ALL_WTS_VARS_NO_DELTAS_copy.csv"
OUT_DIR    = "/content/drive/MyDrive/AlphaGenome/Out"
OUTPUT_TSV = os.path.join(OUT_DIR, "alphagenome_all_variants.tsv")
os.makedirs(OUT_DIR, exist_ok=True)

CELL_LINES       = ['HeLa', 'K562', 'MCF7', 'HMC3', 'HEK']
DELTA_LOGIT_COLS = [f"{cl}_delta_logit_pooled" for cl in CELL_LINES]
DPSI_COLS        = [f"{cl}_dpsi_pooled"        for cl in CELL_LINES]
WT_PSI_COLS      = [f"{cl}_wt_pooled_psi_raw"  for cl in CELL_LINES]

CITRINE_EXON1 = "ATGGTGTCCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTCAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAACTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGATGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAAGTGAAGTTCGAGGGCGACACCCTCGTGAACCGCATCGAGCTAAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAAGTGAACTTCAAGATCCGCCACAACATCGAG"
CITRINE_EXON2 = "GACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAG"
SMN2_INTRON6  = "GTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA"
SMN2_INTRON7  = "AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAAAAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCCAAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTGTGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTAGGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTCTGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAATTCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAG"

PREFIX = CITRINE_EXON1 + SMN2_INTRON6
SUFFIX = SMN2_INTRON7 + CITRINE_EXON2

SEQ_LEN = 16_384
CENTER  = SEQ_LEN // 2

# SPLICE_SITES track layout (confirmed by position scan):
# index 0 = donor + strand, index 1 = acceptor + strand
# AlphaGenome peaks at -1 from exon boundary (last base of AG/GT dinucleotide)
DONOR_IDX    = 0
ACCEPTOR_IDX = 1

# === LOAD MODEL ===
print(f"{datetime.datetime.now()}: Loading AlphaGenome model...")
model = dna_model.create_from_kaggle('all_folds')
# model = dna_model.create_from_huggingface('all_folds')


# === HELPERS ===
def get_splice_scores(seq):
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
    return logit(np.clip(x, 1e-4, 1 - 1e-4))


def compute_psi(values, offset, exon_start, exon_end):
    """PSI = SA * SD. AlphaGenome peaks at -1 from exon boundary (confirmed)."""
    sa = float(values[offset + exon_start - 2, ACCEPTOR_IDX])
    sd = float(values[offset + exon_end   - 2, DONOR_IDX])
    return sa, sd, sa * sd


def build_seq(row, is_ref=False):
    """Build full construct sequence for a row. CPU-only, safe to thread."""
    core = row['intron1'] + row['exon'] + row['intron2']
    return PREFIX + core.upper().replace('U', 'T') + SUFFIX


# === LOAD DATA ===
print(f"{datetime.datetime.now()}: Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

vars_df = df[df['snp'] != 'none'].copy()
print(f"  All variants (single + double): {len(vars_df):,}")

# Get ref sequences for exon coordinate lookup
ref_seqs = (
    df[df['snp'] == 'none'][['event_id', 'intron1', 'exon', 'intron2']]
    .drop_duplicates('event_id')
    .set_index('event_id')
)

# === CHECKPOINT: skip already-done event_ids ===
done_event_ids = set()
if os.path.exists(OUTPUT_TSV):
    done_df = pd.read_csv(OUTPUT_TSV, sep='\t', usecols=['event_id'])
    done_event_ids = set(done_df['event_id'].unique())
    print(f"  Resuming — {len(done_event_ids)} event_ids already done")

header_written = os.path.exists(OUTPUT_TSV)

# === MAIN LOOP ===
# Strategy: use a thread pool to pre-build sequences for the NEXT event_id
# while the GPU runs inference on the current one. CPU/GPU overlap.
N_PREFETCH_THREADS = 4

all_event_ids = [
    eid for eid in vars_df['event_id'].dropna().unique()
    if eid not in done_event_ids and eid in ref_seqs.index
]

def prefetch_seqs(event_id):
    """Pre-build all sequences for an event_id. Returns (event_id, ref_full, exon_start, exon_end, alt_rows_list)."""
    ref_seq = ref_seqs.loc[event_id]
    ref_full = build_seq(ref_seq)
    exon_start = len(PREFIX) + len(ref_seq['intron1']) + 1
    exon_end   = exon_start + len(ref_seq['exon']) - 1
    alt_rows = vars_df[vars_df['event_id'] == event_id]
    alt_list = [(row, build_seq(row)) for _, row in alt_rows.iterrows()]
    return event_id, ref_full, exon_start, exon_end, alt_list

with ThreadPoolExecutor(max_workers=N_PREFETCH_THREADS) as executor:
    futures = {executor.submit(prefetch_seqs, eid): eid for eid in all_event_ids}

    for future in tqdm(futures, desc="event_ids", total=len(all_event_ids)):
        try:
            event_id, ref_full, exon_start, exon_end, alt_list = future.result()
        except Exception as e:
            print(f"  ERROR building seqs: {e}")
            continue

        try:
            ref_values, offset = get_splice_scores(ref_full)
            ref_sa, ref_sd, ref_psi = compute_psi(ref_values, offset, exon_start, exon_end)
        except Exception as e:
            print(f"  ERROR ref {event_id}: {e}")
            continue

        batch = []
        for alt_row, alt_full in alt_list:
            try:
                alt_values, _ = get_splice_scores(alt_full)
                alt_sa, alt_sd, alt_psi = compute_psi(alt_values, offset, exon_start, exon_end)
            except Exception as e:
                print(f"  ERROR alt {alt_row['Reference']}: {e}")
                continue

            delta_psi   = alt_psi - ref_psi
            delta_logit = safe_logit(alt_psi) - safe_logit(ref_psi)

            batch.append({
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
                'experimental_delta_logit_pooled': float(np.nanmean(alt_row[DELTA_LOGIT_COLS].values.astype(float))),
                'experimental_dpsi_pooled':        float(np.nanmean(alt_row[DPSI_COLS].values.astype(float))),
            })

        if batch:
            pd.DataFrame(batch).to_csv(OUTPUT_TSV, sep='\t', mode='a', index=False, header=not header_written)
            header_written = True

print(f"{datetime.datetime.now()}: Done. Results written to {OUTPUT_TSV}")

# === PLOT ===
results = pd.read_csv(OUTPUT_TSV, sep='\t')

# WT PSI edge filter: exclude variants whose ref has PSI == 0 or 1 in any cell line
df[WT_PSI_COLS] = df[WT_PSI_COLS].apply(pd.to_numeric, errors='coerce')
wt_rows = df[df['snp'] == 'none'].copy()
wt_rows['is_edge'] = wt_rows[WT_PSI_COLS].apply(lambda r: any(r == 0) or any(r == 1), axis=1)
non_edge_refs = set(wt_rows[~wt_rows['is_edge']]['Reference'])
results_filtered = results[results['Reference'].isin(non_edge_refs)].copy()

def make_plot(data, xcol, ycol, xlabel, ylabel, out_path):
    x = pd.to_numeric(data[xcol], errors='coerce')
    y = pd.to_numeric(data[ycol], errors='coerce')
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        print(f"  Skipping {out_path} — not enough data")
        return
    r, _ = pearsonr(x, y)
    n = len(x)
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=5, alpha=0.2, color='black', rasterized=True)
    lim = max(abs(x).max(), abs(y).max()) * 1.1
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.axhline(0, color='gray', lw=0.5); plt.axvline(0, color='gray', lw=0.5)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f'n={n:,}  r={r:.2f}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  r={r:.3f}, n={n:,} → {out_path}")

plot_tasks = [
    ('delta_logit', 'alphagenome_delta_logit', 'experimental_delta_logit_pooled',
     'AlphaGenome Δlogit(SA×SD)', 'Experimental Δlogit(ψ)'),
    ('delta_psi',   'alphagenome_delta_psi',   'experimental_dpsi_pooled',
     'AlphaGenome Δ(SA×SD)',      'Experimental ΔPSI'),
]

print("Plotting all variants...")
for score_type, xcol, ycol, xlabel, ylabel in plot_tasks:
    make_plot(results, xcol, ycol, xlabel, ylabel,
              os.path.join(OUT_DIR, f"alphagenome_all_{score_type}.pdf"))

print("Plotting filtered (no WT PSI == 0 or 1)...")
for score_type, xcol, ycol, xlabel, ylabel in plot_tasks:
    make_plot(results_filtered, xcol, ycol, xlabel, ylabel,
              os.path.join(OUT_DIR, f"alphagenome_all_{score_type}_no_edge_wtpsi.pdf"))
