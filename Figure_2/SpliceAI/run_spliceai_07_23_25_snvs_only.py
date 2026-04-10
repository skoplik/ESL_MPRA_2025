import pandas as pd
import numpy as np
from keras.models import load_model
from spliceai.utils import one_hot_encode
from tqdm import tqdm

# === Constants ===
CITRINE_EXON1 = ("ATGGTGTCCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTCAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAACTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGATGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAAGTGAAGTTCGAGGGCGACACCCTCGTGAACCGCATCGAGCTAAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAAGTGAACTTCAAGATCCGCCACAACATCGAG")
CITRINE_EXON2 = ("GACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAG")
SMN2_INTRON6 = ("GTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA")
SMN2_INTRON7 = ("AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAAAAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCCAAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTGTGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTAGGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTCTGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAATTCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAG")

# === Parameters ===
context = 10000
input_csv = "/ESL/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WITH_WT.csv"
summary_output = "/ESL/Figures_SK/Spliceai/redo_07_23_25/spliceai_variant_splice_probs.csv"
raw_output = "/ESL/Figures_SK/Spliceai/redo_07_23_25/spliceai_raw_preds.tsv"
model_paths = [f"/ESL/src_download/SpliceAI/spliceai/models/spliceai{i}.h5" for i in range(1, 6)]

# === Load models
models = [load_model(p) for p in model_paths]

# === Load input
df = pd.read_csv(input_csv)
df["snp"] = df["snp"].astype(str).str.strip()
df = df[(df["full_seq"].notnull()) & (df["event_id"].notnull())].copy()
df = df[(df["snp"] != "none") & (~df["snp"].str.contains(";"))].copy()
print(len(df))
#df = df.head(5).copy()

results = []
raw_rows = []

prefix = CITRINE_EXON1 + SMN2_INTRON6
suffix = SMN2_INTRON7 + CITRINE_EXON2
prefix_len = len(prefix)
suffix_len = len(suffix)

print("Running SpliceAI...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        intron1 = row["intron1"]
        exon = row["exon"]
        intron2 = row["intron2"]
        full_seq = intron1 + exon + intron2
        full_construct = prefix + full_seq + suffix

        pad_left = "N" * (context // 2)
        pad_right = "N" * (context // 2)
        padded_seq = pad_left + full_construct + pad_right

        x = one_hot_encode(padded_seq)[None, :]
        y = np.mean([model.predict(x, verbose=0) for model in models], axis=0)[0]

        # Correct indexing: relative to full_construct only
        acc_pos = prefix_len + len(intron1) - 1
        don_pos = prefix_len + len(intron1) + len(exon)

        if acc_pos >= len(y) or don_pos >= len(y):
            continue
        # SpliceAI outputs a (L x 3) matrix where each row contains probabilities 
        # for [no_splice, acceptor, donor] at that position. Each row sums to 1.
        p_acc = y[acc_pos, 1]
        p_don = y[don_pos, 2]
        
        p_splice = p_acc * p_don

        results.append({
            "Reference": row["Reference"],
            "event_id": row["event_id"],
            "snp": row["snp"],
            "P_acceptor": p_acc,
            "P_donor": p_don,
            "P_splice": p_splice
        })

        raw_rows.append({
            "Reference": row["Reference"],
            "event_id": row["event_id"],
            "snp": row["snp"],
            "spliceai_scores": y.tolist()
        })

    except Exception as e:
        print(f"Error at index {idx}, event_id {row['event_id']}: {e}")

# === Save outputs
pd.DataFrame(results).to_csv(summary_output, index=False)
pd.DataFrame(raw_rows).to_csv(raw_output, sep="\t", index=False)
print(f"Saved summary to: {summary_output}")
print(f"Saved raw scores to: {raw_output}")
