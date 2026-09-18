import os
import re
import datetime
import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from pkg_resources import resource_filename
from pangolin.model import Pangolin, L, W, AR
from tqdm import tqdm
import json

# ============================================================
# Pangolin re-run on MAY 2026 reprocessed data, saving FULL
# per-position score tracks per construct so SA x SD can be
# re-extracted at any junction later without re-running.
#
# NEW file (does not overwrite run_pangolin_all.py).
# Extraction convention is preserved EXACTLY from the original
# (score at s[exon_start-1] / s[exon_end-1]); the full padded
# track `s` is saved verbatim alongside, plus n_pad / junction
# metadata, so future re-extraction stays self-consistent.
# ============================================================

# === CONFIG (override via env) ===
SCORE_TYPE = os.environ.get("SCORE_TYPE", "P(splice)")   # "Usage" or "P(splice)"
INPUT_DF   = os.environ.get("INPUT_DF", "/home/ubuntu/Pangolin/09_11_2026_1e-2_ALL_WITH_WT.csv.gz")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/ubuntu/Pangolin/pangolin_MAY_fullpos_20260911")
RUN_ALL    = os.environ.get("RUN_ALL", "0") == "1"       # default: only the 8 affected exons
N_PAD      = int(os.environ.get("N_PAD", "5000"))

# The 8 exons whose junction boundaries changed in the May reprocessing.
AFFECTED_GENE_EXONS = [
    "ACAD9 exon 2", "PUF60 exon 2", "UBQLN1 exon 8", "KCTD10 exon 4",
    "CES5A exon 9", "REPS1 exon 10", "EPB41L1 exon 20", "PPM1N exon 3",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
NPZ_DIR = os.path.join(OUTPUT_DIR, "fullpos_npz")
os.makedirs(NPZ_DIR, exist_ok=True)
OUTPUT_TSV = os.path.join(OUTPUT_DIR, "pangolin_variant_deltas_MAY_fullpos.tsv")

CITRINE_EXON1 = "ATGGTGTCCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTCAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAACTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGATGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAAGTGAAGTTCGAGGGCGACACCCTCGTGAACCGCATCGAGCTAAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAAGTGAACTTCAAGATCCGCCACAACATCGAG"
CITRINE_EXON2 = "GACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAG"
SMN2_INTRON6 = "GTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA"
SMN2_INTRON7 = "AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAAAAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCCAAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTGTGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTAGGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTCTGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAATTCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAG"
prefix = CITRINE_EXON1 + SMN2_INTRON6
suffix = SMN2_INTRON7 + CITRINE_EXON2

if SCORE_TYPE == "Usage":
    model_nums = [1, 3, 5, 7]
else:
    model_nums = [0, 2, 4, 6]

IN_MAP = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
INDEX_MAP = {0:1, 1:2, 2:4, 3:5, 4:7, 5:8, 6:10, 7:11}

def get_input_sequence(seq):
    return re.sub('[^ACGTN]+', '', seq.upper().replace('U', 'T').replace(' ', ''))

def one_hot_encode(seq):
    seq = seq.replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('N', '0')
    return IN_MAP[np.array(list(map(int, list(seq))), dtype='int8')]

def load_models():
    using_gpu = 0
    models = []
    for i in model_nums:
        for j in range(1, 5+1):
            model = Pangolin(L, W, AR)
            weights = torch.load(resource_filename("pangolin", f"models/final.{j}.{i}.3"),
                                 map_location="cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(weights)
            model.eval()
            if torch.cuda.is_available():
                using_gpu = 1
                model.cuda()
            models.append(model)
    print("Model loaded on GPU" if using_gpu else "Model loaded on CPU")
    return models

def run_models(seq, models):
    seq = one_hot_encode(seq).T
    seq = torch.from_numpy(np.expand_dims(seq, axis=0)).float()
    if torch.cuda.is_available():
        seq = seq.cuda()
    scores = []
    for j, model_num in enumerate(model_nums):
        preds = []
        for model in models[5*j:5*j+5]:
            with torch.no_grad():
                preds.append(model(seq)[0][INDEX_MAP[model_num], :].cpu().numpy())
        scores.append(np.mean(preds, axis=0))
    return scores   # list of `len(model_nums)` per-position tracks (each length == len(seq))

def safe_logit(x):
    x = np.clip(x, 1e-2, 1 - 1e-2)
    return logit(x)

print(f"{datetime.datetime.now()}: Loading models ({SCORE_TYPE}, model_nums={model_nums})")
models = load_models()

print(f"{datetime.datetime.now()}: Reading input {INPUT_DF}")
df = pd.read_csv(INPUT_DF)

if not RUN_ALL:
    if 'gene_exon' not in df.columns:
        raise SystemExit("gene_exon column required to filter affected exons; set RUN_ALL=1 to run everything")
    df = df[df['gene_exon'].isin(AFFECTED_GENE_EXONS)].copy()
    print(f"Filtered to {len(AFFECTED_GENE_EXONS)} affected exons; "
          f"present: {sorted(df['gene_exon'].dropna().unique().tolist())}")

all_event_ids = df['event_id'].dropna().unique()
print(f"{datetime.datetime.now()}: {len(all_event_ids)} event_ids to process")

if os.path.exists(OUTPUT_TSV):
    raise SystemExit(f"Refusing to overwrite existing {OUTPUT_TSV}; choose a fresh OUTPUT_DIR")

header_written = False
CELLS = ['HeLa', 'K562', 'MCF7', 'HMC3', 'HEK']

for event_id in tqdm(all_event_ids, desc="Processing exon families"):
    group = df[df['event_id'] == event_id]
    if group['snp'].eq('none').sum() != 1:
        continue
    ref_row = group[group['snp'] == 'none'].iloc[0]
    alt_rows = group[group['snp'] != 'none']

    gene_exon = str(ref_row.get('gene_exon', event_id))
    ref_seq = prefix + get_input_sequence(ref_row['intron1'] + ref_row['exon'] + ref_row['intron2']) + suffix
    exon_start = len(prefix) + len(ref_row['intron1']) + 1     # 1-based, in ref_seq coords (original convention)
    exon_end = exon_start + len(ref_row['exon']) - 1

    padded_ref = 'N'*N_PAD + ref_seq + 'N'*N_PAD
    ref_scores = run_models(padded_ref, models)               # list[len(model_nums)] of arrays len(padded_ref)
    ref_arr = np.asarray(ref_scores, dtype=np.float32)         # [n_models, L_padded]
    ref_3ss_raw = [s[exon_start-1] for s in ref_scores]        # original extraction, verbatim
    ref_5ss_raw = [s[exon_end-1] for s in ref_scores]
    ref_3ss_logit = safe_logit(np.array(ref_3ss_raw))
    ref_5ss_logit = safe_logit(np.array(ref_5ss_raw))

    alt_arrs, alt_refs, alt_snps = [], [], []

    for _, alt_row in alt_rows.iterrows():
        alt_seq = prefix + get_input_sequence(alt_row['intron1'] + alt_row['exon'] + alt_row['intron2']) + suffix
        alt_scores = run_models('N'*N_PAD + alt_seq + 'N'*N_PAD, models)
        alt_arr = np.asarray(alt_scores, dtype=np.float32)
        alt_3ss_raw = [s[exon_start-1] for s in alt_scores]
        alt_5ss_raw = [s[exon_end-1] for s in alt_scores]
        alt_3ss_logit = safe_logit(np.array(alt_3ss_raw))
        alt_5ss_logit = safe_logit(np.array(alt_5ss_raw))

        alt_arrs.append(alt_arr)
        alt_refs.append(str(alt_row['Reference']))
        alt_snps.append(str(alt_row['snp']))

        delta_3_logit = alt_3ss_logit - ref_3ss_logit
        delta_5_logit = alt_5ss_logit - ref_5ss_logit
        delta_3_psi = np.array(alt_3ss_raw) - np.array(ref_3ss_raw)
        delta_5_psi = np.array(alt_5ss_raw) - np.array(ref_5ss_raw)

        max_d3 = delta_3_logit[np.argmax(np.abs(delta_3_logit))]
        max_d5 = delta_5_logit[np.argmax(np.abs(delta_5_logit))]
        delta_score_logit = np.mean([max_d3, max_d5])
        max_dp3 = delta_3_psi[np.argmax(np.abs(delta_3_psi))]
        max_dp5 = delta_5_psi[np.argmax(np.abs(delta_5_psi))]
        delta_score_psi = np.mean([max_dp3, max_dp5])

        pooled_logit_cols = [f"{cl}_delta_logit_pooled" for cl in CELLS]
        pooled_dpsi_cols = [f"{cl}_dpsi_pooled" for cl in CELLS]
        pooled_logit_avg = np.nanmean(alt_row[pooled_logit_cols].values.astype(float))
        pooled_dpsi_avg = np.nanmean(alt_row[pooled_dpsi_cols].values.astype(float))

        row = pd.DataFrame([{
            'Reference': str(alt_row['Reference']),
            'snp': str(alt_row['snp']),
            'event_id': str(event_id),
            'gene_exon': gene_exon,
            'exon_start': int(exon_start),
            'exon_end': int(exon_end),
            'n_pad': int(N_PAD),
            'pangolin_delta_logit': float(delta_score_logit),
            'pangolin_dpsi': float(delta_score_psi),
            'experimental_delta_logit_pooled': float(pooled_logit_avg),
            'experimental_dpsi_pooled': float(pooled_dpsi_avg),
            'ref_3ss_raw': json.dumps([float(x) for x in ref_3ss_raw]),
            'ref_5ss_raw': json.dumps([float(x) for x in ref_5ss_raw]),
            'alt_3ss_raw': json.dumps([float(x) for x in alt_3ss_raw]),
            'alt_5ss_raw': json.dumps([float(x) for x in alt_5ss_raw]),
            'ref_3ss_logit': json.dumps([float(x) for x in ref_3ss_logit]),
            'ref_5ss_logit': json.dumps([float(x) for x in ref_5ss_logit]),
            'alt_3ss_logit': json.dumps([float(x) for x in alt_3ss_logit]),
            'alt_5ss_logit': json.dumps([float(x) for x in alt_5ss_logit]),
        }])
        row.to_csv(OUTPUT_TSV, sep='\t', mode='a', index=False, header=not header_written)
        header_written = True

    # === Save FULL per-position tracks for this event_id ===
    safe_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(event_id))
    npz_path = os.path.join(NPZ_DIR, f"{safe_id}.npz")
    np.savez_compressed(
        npz_path,
        event_id=str(event_id),
        gene_exon=gene_exon,
        score_type=SCORE_TYPE,
        model_nums=np.asarray(model_nums),
        n_pad=np.int64(N_PAD),
        exon_start=np.int64(exon_start),   # 1-based in ref_seq coords (add nothing for N-pad; original convention)
        exon_end=np.int64(exon_end),
        ref_seq_len=np.int64(len(ref_seq)),
        ref_reference=str(ref_row['Reference']),
        ref_scores=ref_arr,                                   # [n_models, L_padded]
        alt_scores=(np.stack(alt_arrs) if alt_arrs else np.zeros((0,), dtype=np.float32)),  # [n_alts, n_models, L_padded]
        alt_reference=np.asarray(alt_refs),
        alt_snp=np.asarray(alt_snps),
    )

print(f"{datetime.datetime.now()}: TSV -> {OUTPUT_TSV}")
print(f"{datetime.datetime.now()}: full per-position npz -> {NPZ_DIR}/<event_id>.npz")
print("DONE")
