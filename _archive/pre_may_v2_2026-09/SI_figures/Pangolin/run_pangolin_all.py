import os
import re
import datetime
import numpy as np
import pandas as pd
import torch
from matplotlib import pylab as plt
from scipy.stats import pearsonr
from scipy.special import logit, expit
from pkg_resources import resource_filename
from pangolin.model import Pangolin, L, W, AR
from tqdm import tqdm
import json

# === CONFIG ===
SCORE_TYPE = "P(splice)"  # "Usage" or "P(splice)"
INPUT_DF = "/home/ubuntu/Pangolin/07_18_2025_1e-2_ALL_WITH_WT.csv.gz"
OUTPUT_TSV = "/home/ubuntu/Pangolin/pangolin_variant_deltas_all_exons_test.tsv"
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
    using_gpu=0
    models = []
    for i in model_nums:
        for j in range(1, 5+1):
            model = Pangolin(L, W, AR)
            weights = torch.load(resource_filename("pangolin", f"models/final.{j}.{i}.3"), map_location="cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(weights)
            model.eval()
            if torch.cuda.is_available():
                using_gpu=1
                model.cuda()
            models.append(model)
    if using_gpu:
        print(f"Model loaded on GPU")
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
    return scores

def safe_logit(x):
    x = np.clip(x, 1e-2, 1 - 1e-2)
    return logit(x)

print(f"{datetime.datetime.now()}: Loading models")
models = load_models()

print(f"{datetime.datetime.now()}: Reading input")
df = pd.read_csv(INPUT_DF)
all_event_ids = df['event_id'].dropna().unique()

if os.path.exists(OUTPUT_TSV):
    os.remove(OUTPUT_TSV)

header_written = False

for event_id in tqdm(all_event_ids, desc="Processing exon families"):
    group = df[df['event_id'] == event_id]
    if group['snp'].eq('none').sum() != 1:
        continue
    ref_row = group[group['snp'] == 'none'].iloc[0]
    alt_rows = group[group['snp'] != 'none']

    ref_seq = prefix + get_input_sequence(ref_row['intron1'] + ref_row['exon'] + ref_row['intron2']) + suffix
    exon_start = len(prefix) + len(ref_row['intron1']) + 1
    exon_end = exon_start + len(ref_row['exon']) - 1

    ref_scores = run_models('N'*5000 + ref_seq + 'N'*5000, models)
    ref_3ss_raw = [s[exon_start-1] for s in ref_scores]
    ref_5ss_raw = [s[exon_end-1] for s in ref_scores]
    ref_3ss_logit = safe_logit(np.array(ref_3ss_raw))
    ref_5ss_logit = safe_logit(np.array(ref_5ss_raw))

    for _, alt_row in alt_rows.iterrows():
        alt_seq = prefix + get_input_sequence(alt_row['intron1'] + alt_row['exon'] + alt_row['intron2']) + suffix
        alt_scores = run_models('N'*5000 + alt_seq + 'N'*5000, models)
        alt_3ss_raw = [s[exon_start-1] for s in alt_scores]
        alt_5ss_raw = [s[exon_end-1] for s in alt_scores]
        alt_3ss_logit = safe_logit(np.array(alt_3ss_raw))
        alt_5ss_logit = safe_logit(np.array(alt_5ss_raw))

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

        pooled_logit_cols = [f"{cl}_delta_logit_pooled" for cl in ['HeLa', 'K562', 'MCF7', 'HMC3', 'HEK']]
        pooled_dpsi_cols = [f"{cl}_dpsi_pooled" for cl in ['HeLa', 'K562', 'MCF7', 'HMC3', 'HEK']]
        pooled_logit_avg = np.nanmean(alt_row[pooled_logit_cols].values.astype(float))
        pooled_dpsi_avg = np.nanmean(alt_row[pooled_dpsi_cols].values.astype(float))

        row = pd.DataFrame([{
            'Reference': str(alt_row['Reference']),
            'snp': str(alt_row['snp']),
            'event_id': str(event_id),
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
            'alt_5ss_logit': json.dumps([float(x) for x in alt_5ss_logit])
        }])

        row.to_csv(OUTPUT_TSV, sep='\t', mode='a', index=False, header=not header_written)
        header_written = True

print(f"{datetime.datetime.now()}: All results written to {OUTPUT_TSV}")

# === Plot results ===
print(f"{datetime.datetime.now()}: Plotting results")

results = pd.read_csv(OUTPUT_TSV, sep='\t')
super_full = pd.read_csv(INPUT_DF)

# Merge WT rows from supertable to results for ref PSI filtering
wt_df = super_full[super_full['snp'] == 'none'][['event_id', 'Reference'] + [f"{cl}_pooled_psi_raw" for cl in ['HeLa', 'K562', 'MCF7', 'HMC3', 'HEK']]]
wt_df = wt_df.rename(columns={f"{cl}_pooled_psi_raw": f"{cl}_ref_psi" for cl in ['HeLa', 'K562', 'MCF7', 'HMC3', 'HEK']})

results = results.merge(wt_df, on=['event_id'], how='left')

# Filter out any row where the ref PSI is exactly 0 or 1 in any cell line
ref_psi_cols = [f"{cl}_ref_psi" for cl in ['HeLa', 'K562', 'MCF7', 'HMC3', 'HEK']]
mask = ~results[ref_psi_cols].isin([0.0, 1.0]).any(axis=1)
results_filtered = results[mask].copy()

# Δlogit(ψ)
x1 = results_filtered['pangolin_delta_logit']
y1 = results_filtered['experimental_delta_logit_pooled']
r1, _ = pearsonr(x1, y1)
n1 = len(x1)

min_val = min(x1.min(), y1.min())
max_val = max(x1.max(), y1.max())

plt.figure(figsize=(5, 5))
plt.scatter(x1, y1, s=5, alpha=0.3, color='black')
plt.plot([min_val, max_val], [min_val, max_val], ls='--', color='gray')
plt.xlabel('Pangolin Δlogit(ψ)')
plt.ylabel('Experimental Δlogit(ψ)')
plt.title(f'Δlogit(ψ)  r = {r1:.2f}, n = {n1}')
plt.tight_layout()
plt.savefig("/home/ubuntu/Pangolin/pangolin_delta_logit_vs_experimental_filtered.pdf")
plt.close()

# ΔPSI
x2 = results_filtered['pangolin_dpsi']
y2 = results_filtered['experimental_dpsi_pooled']
r2, _ = pearsonr(x2, y2)
n2 = len(x2)

min_val = min(x2.min(), y2.min())
max_val = max(x2.max(), y2.max())

plt.figure(figsize=(5, 5))
plt.scatter(x2, y2, s=5, alpha=0.3, color='black')
plt.plot([min_val, max_val], [min_val, max_val], ls='--', color='gray')
plt.xlabel('Pangolin ΔPSI')
plt.ylabel('Experimental ΔPSI')
plt.title(f'ΔPSI  r = {r2:.2f}, n = {n2}')
plt.tight_layout()
plt.savefig("/home/ubuntu/Pangolin/pangolin_dpsi_vs_experimental_filtered.pdf")
plt.close()

print(f"{datetime.datetime.now()}: Pearson r (Δlogit): {r1:.4f}, n = {n1}")
print(f"{datetime.datetime.now()}: Pearson r (ΔPSI):   {r2:.4f}, n = {n2}")
print(f"{datetime.datetime.now()}: Plots saved to /home/ubuntu/Pangolin/")