import argparse
import os
import gc
import numpy as np
import pandas as pd
from keras.models import load_model
from spliceai.utils import one_hot_encode
from tqdm import tqdm

CITRINE_EXON1 = "ATGGTGTCCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTCAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAACTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGATGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAAGTGAAGTTCGAGGGCGACACCCTCGTGAACCGCATCGAGCTAAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAAGTGAACTTCAAGATCCGCCACAACATCGAG"
CITRINE_EXON2 = "GACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAG"
SMN2_INTRON6 = "GTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA"
SMN2_INTRON7 = "AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAAAAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCCAAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTGTGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTAGGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTCTGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAATTCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAG"

MODEL_PATHS = [f"/ESL/src_download/SpliceAI/spliceai/models/spliceai{i}.h5" for i in range(1, 6)]
CONTEXT = 10000
BATCH_SIZE = 32

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
raw_output = os.path.join(args.output_dir, "spliceai_raw_preds_all.tsv")
log_output = raw_output + ".log"

print("Loading data...")
df = pd.read_csv(args.input_csv, low_memory=False)
df["snp"] = df["snp"].astype(str).str.strip()
df = df[df["full_seq"].notnull() & df["event_id"].notnull()].copy()
print(f"Total sequences to predict: {len(df):,}")

print("Loading SpliceAI models...")
models = [load_model(p) for p in MODEL_PATHS]

prefix = CITRINE_EXON1 + SMN2_INTRON6
suffix = SMN2_INTRON7 + CITRINE_EXON2
pad = "N" * (CONTEXT // 2)

header_written = False
if os.path.exists(raw_output):
    os.remove(raw_output)
if os.path.exists(log_output):
    os.remove(log_output)

print("Running SpliceAI in batches...")
with open(log_output, "w") as log_file:
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i:i + BATCH_SIZE].copy()
        batch_meta = []
        batch_encoded = []

        for _, row in batch.iterrows():
            full_construct = pad + prefix + row["intron1"] + row["exon"] + row["intron2"] + suffix + pad
            batch_encoded.append(one_hot_encode(full_construct))
            batch_meta.append((row["Reference"], row["event_id"], row["snp"]))

        batch_encoded = np.array(batch_encoded)
        preds = [model.predict(batch_encoded, verbose=0) for model in models]
        mean_preds = np.mean(preds, axis=0)

        batch_rows = [
            {"Reference": ref, "event_id": eid, "snp": snp, "spliceai_scores": pred.tolist()}
            for (ref, eid, snp), pred in zip(batch_meta, mean_preds)
        ]

        pd.DataFrame(batch_rows).to_csv(raw_output, sep="\t", index=False, mode="a", header=not header_written)
        header_written = True

        msg = f"Batch {i // BATCH_SIZE + 1}: rows {i}–{i + len(batch_rows) - 1}"
        print(msg)
        log_file.write(msg + "\n")

        del batch_encoded, preds, mean_preds, batch_rows, batch
        gc.collect()

print(f"Done. Raw scores saved to: {raw_output}")
