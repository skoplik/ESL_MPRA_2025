"""
Pangolin stage 1 for the May 2026 data.

Scores every row (WT and variant) independently and writes per-row junction
scores; the WT subtraction happens in stage 2. Each row's coordinates come from
its own intron1/exon. Batched across the 20 models (4 model_nums x 5 folds),
which takes the run from ~25 h to under an hour.

Scoring math is unchanged: the per-position track per model_num is the mean over
the 5 folds, sampled at s[exon_start-1] and s[exon_end-1] in the model's cropped
output coordinates.
"""
import os, re, argparse, datetime, glob
import numpy as np
import pandas as pd
import torch
from pkg_resources import resource_filename
from pangolin.model import Pangolin, L, W, AR
from tqdm import tqdm

SCORE_TYPE = os.environ.get("SCORE_TYPE", "P(splice)")
N_PAD = int(os.environ.get("N_PAD", "5000"))

CITRINE_EXON1 = "ATGGTGTCCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTCAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAACTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGATGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAAGTGAAGTTCGAGGGCGACACCCTCGTGAACCGCATCGAGCTAAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAAGTGAACTTCAAGATCCGCCACAACATCGAG"
CITRINE_EXON2 = "GACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAG"
SMN2_INTRON6 = "GTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA"
SMN2_INTRON7 = "AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAAAAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCCAAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTGTGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTAGGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTCTGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAATTCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAG"
prefix = CITRINE_EXON1 + SMN2_INTRON6
suffix = SMN2_INTRON7 + CITRINE_EXON2

model_nums = [1, 3, 5, 7] if SCORE_TYPE == "Usage" else [0, 2, 4, 6]
IN_MAP = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
INDEX_MAP = {0: 1, 1: 2, 2: 4, 3: 5, 4: 7, 5: 8, 6: 10, 7: 11}


def get_input_sequence(seq):
    return re.sub('[^ACGTN]+', '', seq.upper().replace('U', 'T').replace(' ', ''))


def one_hot_encode(seq):
    seq = seq.replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('N', '0')
    return IN_MAP[np.array(list(map(int, list(seq))), dtype='int8')]


def load_models():
    models = []
    for i in model_nums:
        for j in range(1, 6):
            m = Pangolin(L, W, AR)
            w = torch.load(resource_filename("pangolin", "models/final.%d.%d.3" % (j, i)),
                           map_location="cuda" if torch.cuda.is_available() else "cpu")
            m.load_state_dict(w)
            m.eval()
            if torch.cuda.is_available():
                m.cuda()
            models.append(m)
    print("Models on GPU" if torch.cuda.is_available() else "Models on CPU", flush=True)
    return models


def run_models_batch(seqs, models):
    """seqs: list[str] (already N-padded). -> np.ndarray [B, n_model_nums, L_out]"""
    enc = np.stack([one_hot_encode(s).T for s in seqs])          # [B, 4, L]
    x = torch.from_numpy(enc).float()
    if torch.cuda.is_available():
        x = x.cuda()
    out = []
    for j, mn in enumerate(model_nums):
        preds = []
        for m in models[5 * j:5 * j + 5]:
            with torch.no_grad():
                preds.append(m(x)[:, INDEX_MAP[mn], :].cpu().numpy())   # [B, L_out]
        out.append(np.mean(preds, axis=0))
    return np.stack(out, axis=1)                                  # [B, n_mn, L_out]


ap = argparse.ArgumentParser()
ap.add_argument("--input_csv", default="/home/ubuntu/rerun2026/data.csv.gz")
ap.add_argument("--output_dir", required=True)
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shard_size", type=int, default=5000)
ap.add_argument("--no_npz", action="store_true")
args = ap.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
NPZ_DIR = os.path.join(args.output_dir, "fullpos_npz")
if not args.no_npz:
    os.makedirs(NPZ_DIR, exist_ok=True)
out_tsv = os.path.join(args.output_dir, "pangolin_junction_scores_MAY.tsv")

print("%s: reading %s" % (datetime.datetime.now(), args.input_csv), flush=True)
df = pd.read_csv(args.input_csv, low_memory=False)
df["snp"] = df["snp"].astype(str).str.strip()
for c in ("intron1", "exon", "intron2"):
    df[c] = df[c].astype(str).str.replace(" ", "", regex=False)
df = df[df["full_seq"].notnull() & df["event_id"].notnull()].copy()

# per-row coords, 1-based, in ref_seq (unpadded) coordinates - original convention
df["exon_start"] = len(prefix) + df["intron1"].str.len() + 1
df["exon_end"] = df["exon_start"] + df["exon"].str.len() - 1

if args.limit:
    df = df.head(args.limit).copy()

done = set()
if os.path.exists(out_tsv):
    try:
        done = set(pd.read_csv(out_tsv, sep="\t", usecols=["Reference"])["Reference"].astype(str))
    except Exception:
        done = set()
if done:
    before = len(df)
    df = df[~df["Reference"].astype(str).isin(done)].copy()
    print("Resuming: %d done, %d remaining" % (before - len(df), len(df)), flush=True)

print("Rows to score: %d" % len(df), flush=True)
if len(df) == 0:
    raise SystemExit(0)

models = load_models()

META = [c for c in ["Reference", "event_id", "event_id_161", "snp", "gene_exon",
                    "transcript_class", "seq_type", "exon_start", "exon_end"] if c in df.columns]
header_written = os.path.exists(out_tsv) and os.path.getsize(out_tsv) > 0
npz_buf, shard_idx = {}, 0
if not args.no_npz:
    ex = glob.glob(os.path.join(NPZ_DIR, "fullpos_*.npz"))
    if ex:
        shard_idx = max(int(os.path.basename(p).split("_")[1].split(".")[0]) for p in ex) + 1

checked_crop = False
for i in tqdm(range(0, len(df), args.batch_size)):
    batch = df.iloc[i:i + args.batch_size]
    seqs, metas = [], []
    for _, row in batch.iterrows():
        ref_seq = prefix + get_input_sequence(row["intron1"] + row["exon"] + row["intron2"]) + suffix
        seqs.append("N" * N_PAD + ref_seq + "N" * N_PAD)
        metas.append((row, len(ref_seq)))

    tracks = run_models_batch(seqs, models)     # [B, n_mn, L_out]

    if not checked_crop:
        # the original indexes s[exon_start-1] with exon_start in UNPADDED coords,
        # which is only correct if the model crops the N-pad. Verify, don't assume.
        L_out, L_ref = tracks.shape[2], metas[0][1]
        print("\nCROP CHECK: model out len=%d, unpadded ref_seq len=%d, padded=%d"
              % (L_out, L_ref, L_ref + 2 * N_PAD), flush=True)
        if L_out != L_ref:
            raise SystemExit("ABORT: output length %d != unpadded ref_seq length %d; "
                             "the s[exon_start-1] convention would be off by the pad." % (L_out, L_ref))
        checked_crop = True

    rows = []
    for (row, L_ref), tr in zip(metas, tracks):
        s, e = int(row["exon_start"]), int(row["exon_end"])
        rec = {k: row[k] for k in META}
        if 1 <= s <= tr.shape[1] and 1 <= e <= tr.shape[1]:
            for k, mn in enumerate(model_nums):
                rec["p3ss_m%d" % mn] = float(tr[k, s - 1])
                rec["p5ss_m%d" % mn] = float(tr[k, e - 1])
        else:
            for mn in model_nums:
                rec["p3ss_m%d" % mn] = np.nan
                rec["p5ss_m%d" % mn] = np.nan
        rec["track_len"] = int(tr.shape[1])
        rows.append(rec)
        if not args.no_npz:
            npz_buf[str(row["Reference"])] = tr.astype(np.float16)

    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False, mode="a", header=not header_written)
    header_written = True

    if not args.no_npz and len(npz_buf) >= args.shard_size:
        np.savez_compressed(os.path.join(NPZ_DIR, "fullpos_%04d.npz" % shard_idx), **npz_buf)
        npz_buf = {}
        shard_idx += 1

if not args.no_npz and npz_buf:
    np.savez_compressed(os.path.join(NPZ_DIR, "fullpos_%04d.npz" % shard_idx), **npz_buf)

print("%s: done -> %s" % (datetime.datetime.now(), out_tsv), flush=True)
