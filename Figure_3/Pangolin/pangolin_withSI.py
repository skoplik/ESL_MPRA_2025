"""
Pangolin scores for the *_with_SI PSI files.

The with_SI files add 1,893 rows (source_table == 'si') to the main file. They
are constructs already in the main file, quantified against an alternative 3'
splice site, so the same full_seq appears again with a different intron1/exon
split. Stage 1 (run_pangolin_MAY_batched.py) scores each row on its own split,
so only these rows need scoring; the MAY_v2 scores cover every main row as is.

  prep   write the SI rows, plus a random sample of already-scored main rows as
         a cross-check, as input for run_pangolin_MAY_batched.py
  merge  check the cross-check rows reproduce the MAY_v2 scores, append the SI
         scores to MAY_v2, and write delta logits for the SI variants

An SI variant's WT is the WT row on the same junction, keyed by
(event_id_161, exon_start, exon_end). event_id alone is not enough: an SI row
carries the same event_id as the MANE row, and up to three WTs share one.

Delta logit follows build_mega_MAY_v2.py: per model_num, logit(variant) -
logit(WT) at each splice site, take the most extreme across model_nums at each
site, then average the two sites.

Writes new files only.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

OUT_DIR = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output"
SI_INPUT = os.path.join(OUT_DIR, "1e-2_ALL_WTS_VARS_NO_DELTAS_with_SI.csv.gz")
V2_SCORES = "/ESL/ESL_MPRA/Figure_3/Pangolin/output_MAY_v2/pangolin_junction_scores_MAY.tsv"
MERGED_DIR = "/ESL/ESL_MPRA/Figure_3/Pangolin/output_MAY_v2_withSI"

MODEL_NUMS = [0, 2, 4, 6]
P3 = ["p3ss_m%d" % m for m in MODEL_NUMS]
P5 = ["p5ss_m%d" % m for m in MODEL_NUMS]
CELLS = ["HeLa", "K562", "MCF7", "HMC3", "HEK"]
DELTA_COLS = ["%s_delta_logit_pooled" % c for c in CELLS]
WTPSI_COLS = ["%s_wt_pooled_psi_raw" % c for c in CELLS]
JUNCTION = ["event_id_161", "exon_start", "exon_end"]
INPUT_COLS = ["Reference", "event_id", "event_id_161", "snp", "gene_exon", "transcript_class",
              "seq_type", "source_table", "intron1", "exon", "intron2", "full_seq"]


def logit_clip(p, eps=0.01):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def prep(args):
    df = pd.read_csv(args.input_csv, low_memory=False)
    si = df[df["source_table"] == "si"]
    check = df[df["source_table"] == "main"].sample(args.n_check, random_state=args.seed)
    out = pd.concat([si, check])[INPUT_COLS]
    if out["full_seq"].isna().any() or out["event_id"].isna().any():
        sys.exit("ABORT: null full_seq or event_id in the rows to score")
    out.to_csv(args.output_csv, index=False)
    print("wrote %s: %d SI rows + %d cross-check rows" % (args.output_csv, len(si), len(check)))


def merge(args):
    os.makedirs(args.out_dir, exist_ok=True)
    new = pd.read_csv(args.scores, sep="\t", low_memory=False)
    v2 = pd.read_csv(args.v2_scores, sep="\t", low_memory=False)
    data = pd.read_csv(args.input_csv, low_memory=False)
    for d in (new, v2, data):
        d["Reference"] = d["Reference"].astype(str)
        d["snp"] = d["snp"].astype(str).str.strip()
    si_refs = set(data.loc[data["source_table"] == "si", "Reference"])

    # 1. cross-check rows must reproduce MAY_v2
    chk = new[~new["Reference"].isin(si_refs)].merge(
        v2, on="Reference", suffixes=("_new", "_v2"))
    if chk.empty:
        sys.exit("ABORT: no cross-check rows found in the new scores")
    for c in ["event_id_161", "snp", "exon_start", "exon_end"]:
        bad = (chk[c + "_new"].astype(str) != chk[c + "_v2"].astype(str)).sum()
        if bad:
            sys.exit("ABORT: %d cross-check rows disagree with MAY_v2 on %s" % (bad, c))
    diff = max(float(np.abs(chk[c + "_new"] - chk[c + "_v2"]).max()) for c in P3 + P5)
    print("cross-check: %d rows, max |score diff| vs MAY_v2 = %.2e" % (len(chk), diff))
    if diff > args.tol:
        sys.exit("ABORT: cross-check scores differ from MAY_v2 by more than %g" % args.tol)

    # 2. append the SI rows to MAY_v2
    si = new[new["Reference"].isin(si_refs)]
    missing = si_refs - set(si["Reference"])
    if missing:
        sys.exit("ABORT: %d SI rows have no score" % len(missing))
    if si[P3 + P5].isna().any(axis=1).any():
        sys.exit("ABORT: NaN scores in SI rows")
    if set(si["Reference"]) & set(v2["Reference"]):
        sys.exit("ABORT: SI References already present in MAY_v2")
    merged = pd.concat([v2, si[v2.columns]], ignore_index=True)
    merged_path = os.path.join(args.out_dir, "pangolin_junction_scores_MAY_withSI.tsv")
    merged.to_csv(merged_path, sep="\t", index=False)
    print("wrote %s: %d MAY_v2 + %d SI = %d rows" % (merged_path, len(v2), len(si), len(merged)))

    # 3. delta logits for SI variants, WT on the same junction
    si = si.merge(data[["Reference", "source_table"] + DELTA_COLS + WTPSI_COLS],
                  on="Reference", how="left")
    wt = si[si["snp"] == "none"]
    if wt.duplicated(JUNCTION).any():
        sys.exit("ABORT: more than one SI WT on the same junction")
    var = si[si["snp"] != "none"].merge(
        wt[JUNCTION + P3 + P5], on=JUNCTION, how="left", suffixes=("", "_wt"))
    no_wt = var[P3[0] + "_wt"].isna()
    if no_wt.any():
        print("  %d SI variants have no WT on their junction (left NaN): %s"
              % (no_wt.sum(), var.loc[no_wt, "gene_exon"].value_counts().to_dict()))
    d3 = logit_clip(var[P3]) - logit_clip(var[[c + "_wt" for c in P3]])
    d5 = logit_clip(var[P5]) - logit_clip(var[[c + "_wt" for c in P5]])
    md3 = np.take_along_axis(d3, np.argmax(np.abs(d3), axis=1)[:, None], axis=1)[:, 0]
    md5 = np.take_along_axis(d5, np.argmax(np.abs(d5), axis=1)[:, None], axis=1)[:, 0]
    var["pangolin_delta_logit"] = np.nanmean(np.stack([md3, md5], axis=1), axis=1)
    var.loc[no_wt, "pangolin_delta_logit"] = np.nan

    for c in DELTA_COLS + WTPSI_COLS:
        var[c] = pd.to_numeric(var[c], errors="coerce")
    var["avg_delta_logit_pooled"] = var[DELTA_COLS].mean(axis=1)
    var["edge_filtered"] = ((var[WTPSI_COLS] == 0) | (var[WTPSI_COLS] == 1)).any(axis=1)
    keep = ["Reference", "event_id", "event_id_161", "gene_exon", "snp", "exon_start",
            "exon_end", "pangolin_delta_logit", "avg_delta_logit_pooled", "edge_filtered"]
    delta_path = os.path.join(args.out_dir, "pangolin_delta_logit_SI.csv")
    var[keep].to_csv(delta_path, index=False)
    print("wrote %s: %d SI variants" % (delta_path, len(var)))

    for label, sub in [("all", var), ("edge-filtered", var[~var["edge_filtered"]])]:
        s = sub[["pangolin_delta_logit", "avg_delta_logit_pooled"]].dropna()
        r = np.corrcoef(s.iloc[:, 0], s.iloc[:, 1])[0, 1] if len(s) > 2 else float("nan")
        print("  SI variants, %s: n=%d, r=%.4f" % (label, len(s), r))


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("prep")
p.add_argument("--input_csv", default=SI_INPUT)
p.add_argument("--output_csv", required=True)
p.add_argument("--n_check", type=int, default=500)
p.add_argument("--seed", type=int, default=7)
m = sub.add_parser("merge")
m.add_argument("--scores", required=True, help="stage-1 output for the prep rows")
m.add_argument("--input_csv", default=SI_INPUT)
m.add_argument("--v2_scores", default=V2_SCORES)
m.add_argument("--out_dir", default=MERGED_DIR)
m.add_argument("--tol", type=float, default=1e-4)
args = ap.parse_args()
prep(args) if args.cmd == "prep" else merge(args)
