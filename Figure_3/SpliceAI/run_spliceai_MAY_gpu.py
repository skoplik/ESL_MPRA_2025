"""
SpliceAI Stage-1 for the MAY 2026 reprocessed COMPASS data.

Differences vs run_spliceai_all.py (deliberate, all documented):
  1. Reads the MAY file directly, so `Reference` is internally consistent
     end-to-end (no cross-version Reference join, the root cause of the mess).
  2. Carries the STABLE key (event_id_161, snp, gene_exon, transcript_class)
     alongside Reference so every downstream join can be *verified*.
  3. Extracts the junction SA/SD inline -> compact TSV, instead of dumping
     full per-position tracks as text (~8 GB) and re-parsing them in stage 2.
  4. Still saves FULL per-position tracks, but as sharded float16 .npz
     (~0.8 GB total) per the requirement to keep full-length outputs.
  5. Writes WT and variants in ONE file (old stage-2 expected three).
  6. Resumable: skips References already present in the output TSV.
"""
import argparse, os, gc, glob
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from tqdm import tqdm

CITRINE_EXON1 = "ATGGTGTCCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTCAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAACTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGATGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAAGTGAAGTTCGAGGGCGACACCCTCGTGAACCGCATCGAGCTAAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAAGTGAACTTCAAGATCCGCCACAACATCGAG"
CITRINE_EXON2 = "GACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAG"
SMN2_INTRON6 = "GTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA"
SMN2_INTRON7 = "AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAAAAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCCAAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTGTGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTAGGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTCTGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAATTCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAG"


# one_hot_encode: verbatim semantics of spliceai.utils.one_hot_encode
# (N/other -> all-zero row), reimplemented so the spliceai package isn't required.
IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float32)


def one_hot_encode(seq):
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    idx = np.frombuffer(seq.encode('ascii'), dtype=np.int8) - 48
    idx = np.where((idx < 0) | (idx > 4), 0, idx)
    return IN_MAP[idx]

MODEL_PATHS = ["/home/ubuntu/rerun2026/models/spliceai%d.h5" % i for i in range(1, 6)]
CONTEXT = 10000
LEN_CITRINE1 = len(CITRINE_EXON1)
LEN_SMN2_5 = len(SMN2_INTRON6)


def logit_clip(p, eps=0.01):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


ap = argparse.ArgumentParser()
ap.add_argument("--input_csv", default="/home/ubuntu/rerun2026/data.csv.gz")
ap.add_argument("--output_dir", required=True)
ap.add_argument("--batch_size", type=int, default=64)
ap.add_argument("--limit", type=int, default=0, help="benchmark only: first N rows")
ap.add_argument("--shard_size", type=int, default=5000)
ap.add_argument("--no_npz", action="store_true")
args = ap.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
NPZ_DIR = os.path.join(args.output_dir, "fullpos_npz")
if not args.no_npz:
    os.makedirs(NPZ_DIR, exist_ok=True)
out_tsv = os.path.join(args.output_dir, "spliceai_junction_scores_MAY.tsv")

print("Loading MAY data: %s" % args.input_csv, flush=True)
df = pd.read_csv(args.input_csv, low_memory=False)
df["snp"] = df["snp"].astype(str).str.strip()
for c in ("intron1", "exon", "intron2"):
    df[c] = df[c].astype(str).str.replace(" ", "", regex=False)
df = df[df["full_seq"].notnull() & df["event_id"].notnull()].copy()

# junction coordinates come from the MAY sequences themselves
df["intron1_len"] = df["intron1"].str.len()
df["exon_len"] = df["exon"].str.len()
df["exon_start"] = LEN_CITRINE1 + LEN_SMN2_5 + df["intron1_len"]
df["exon_end"] = df["exon_start"] + df["exon_len"] - 1

if args.limit:
    df = df.head(args.limit).copy()

# resume support
done = set()
if os.path.exists(out_tsv):
    try:
        done = set(pd.read_csv(out_tsv, sep="\t", usecols=["Reference"])["Reference"].astype(str))
    except Exception:
        done = set()
if done:
    before = len(df)
    df = df[~df["Reference"].astype(str).isin(done)].copy()
    print("Resuming: %d already done, %d remaining" % (before - len(df), len(df)), flush=True)

print("Sequences to predict: %d" % len(df), flush=True)
if len(df) == 0:
    raise SystemExit(0)

print("Loading SpliceAI models...", flush=True)
models = [load_model(p) for p in MODEL_PATHS]

prefix = CITRINE_EXON1 + SMN2_INTRON6
suffix = SMN2_INTRON7 + CITRINE_EXON2
pad = "N" * (CONTEXT // 2)

META = ["Reference", "event_id", "event_id_161", "snp", "gene_exon",
        "transcript_class", "seq_type", "exon_start", "exon_end"]
META = [c for c in META if c in df.columns]

header_written = os.path.exists(out_tsv) and os.path.getsize(out_tsv) > 0
npz_buf, shard_idx = {}, 0
existing = glob.glob(os.path.join(NPZ_DIR, "fullpos_*.npz")) if not args.no_npz else []
if existing:
    shard_idx = max(int(os.path.basename(p).split("_")[1].split(".")[0]) for p in existing) + 1

for i in tqdm(range(0, len(df), args.batch_size)):
    batch = df.iloc[i:i + args.batch_size]
    enc, metas = [], []
    for _, row in batch.iterrows():
        construct = pad + prefix + row["intron1"] + row["exon"] + row["intron2"] + suffix + pad
        enc.append(one_hot_encode(construct))
        metas.append(row)

    enc = np.array(enc)
    preds = [m.predict(enc, verbose=0) for m in models]
    mean_preds = np.mean(preds, axis=0)   # (B, L, 3): [null, acceptor, donor]

    rows = []
    for row, track in zip(metas, mean_preds):
        s, e = int(row["exon_start"]), int(row["exon_end"])
        rec = {k: row[k] for k in META}
        if 0 <= s < len(track) and 0 <= e < len(track):
            sa = float(track[s][1])   # acceptor at exon start
            sd = float(track[e][2])   # donor at exon end
            rec["spliceai_acceptor"] = sa
            rec["spliceai_donor"] = sd
            rec["spliceai_product"] = sa * sd
            rec["spliceai_logit"] = float(logit_clip(sa * sd))
        else:
            rec["spliceai_acceptor"] = rec["spliceai_donor"] = np.nan
            rec["spliceai_product"] = rec["spliceai_logit"] = np.nan
        rec["track_len"] = int(len(track))
        rows.append(rec)
        if not args.no_npz:
            npz_buf[str(row["Reference"])] = track.astype(np.float16)

    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False, mode="a",
                              header=not header_written)
    header_written = True

    if not args.no_npz and len(npz_buf) >= args.shard_size:
        np.savez_compressed(os.path.join(NPZ_DIR, "fullpos_%04d.npz" % shard_idx), **npz_buf)
        npz_buf = {}
        shard_idx += 1

    del enc, preds, mean_preds, rows
    gc.collect()

if not args.no_npz and npz_buf:
    np.savez_compressed(os.path.join(NPZ_DIR, "fullpos_%04d.npz" % shard_idx), **npz_buf)

print("Done -> %s" % out_tsv, flush=True)
