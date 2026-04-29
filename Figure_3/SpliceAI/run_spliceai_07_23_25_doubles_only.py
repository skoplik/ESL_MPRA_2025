import pandas as pd
import numpy as np
from keras.models import load_model
from spliceai.utils import one_hot_encode
from tqdm import tqdm
import gc
import os

# === Constants ===
CITRINE_EXON1 = ("ATGGTGTCCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTCAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAACTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGATGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAAGTGAAGTTCGAGGGCGACACCCTCGTGAACCGCATCGAGCTAAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAAGTGAACTTCAAGATCCGCCACAACATCGAG")
CITRINE_EXON2 = ("GACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAG")
SMN2_INTRON6 = ("GTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA")
SMN2_INTRON7 = ("AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAAAAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCCAAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTGTGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTAGGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTCTGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAATTCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAG")

# === Parameters ===
context = 10000
input_csv = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz"
raw_output = "/ESL/ESL_MPRA/Figure_3/SpliceAI/output/spliceai_raw_preds_doubles.tsv"
log_output = raw_output + ".log"
model_paths = [f"/ESL/src_download/SpliceAI/spliceai/models/spliceai{i}.h5" for i in range(1, 6)]
batch_size = 32

# === Load models
models = [load_model(p) for p in model_paths]

# === Load input
df = pd.read_csv(input_csv)
df["snp"] = df["snp"].astype(str).str.strip()
df = df[(df["full_seq"].notnull()) & (df["event_id"].notnull())]
df = df[(df["snp"] != "none") & (df["snp"].str.contains(";"))].copy()
print(f"Filtered double variant rows: {len(df)}")

# === Build sequences
prefix = CITRINE_EXON1 + SMN2_INTRON6
suffix = SMN2_INTRON7 + CITRINE_EXON2
pad = "N" * (context // 2)

# === Prepare output
header_written = False
if os.path.exists(raw_output):
    os.remove(raw_output)
if os.path.exists(log_output):
    os.remove(log_output)

# === Predict in batches
print("Running SpliceAI in batches...")
with open(log_output, "w") as log_file:
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i + batch_size].copy()
        batch_meta = []
        batch_encoded = []

        for _, row in batch.iterrows():
            intron1 = row["intron1"]
            exon = row["exon"]
            intron2 = row["intron2"]
            full_seq = intron1 + exon + intron2
            full_construct = pad + prefix + full_seq + suffix + pad
            batch_encoded.append(one_hot_encode(full_construct))
            batch_meta.append((row["Reference"], row["event_id"], row["snp"]))

        batch_encoded = np.array(batch_encoded)
        preds = [model.predict(batch_encoded, verbose=0) for model in models]
        mean_preds = np.mean(preds, axis=0)

        batch_rows = []
        for j, pred in enumerate(mean_preds):
            ref, eid, snp = batch_meta[j]
            batch_rows.append({
                "Reference": ref,
                "event_id": eid,
                "snp": snp,
                "spliceai_scores": pred.tolist()
            })

        pd.DataFrame(batch_rows).to_csv(raw_output, sep="\t", index=False, mode="a", header=not header_written)
        header_written = True

        msg = f"Batch {i // batch_size + 1} complete: rows {i} to {i + len(batch_rows) - 1}"
        print(msg)
        log_file.write(msg + "\n")

        del batch_encoded, preds, mean_preds, batch_rows, batch
        gc.collect()

print(f"Saved SpliceAI scores to {raw_output}")
print(f"Log written to {log_output}")
