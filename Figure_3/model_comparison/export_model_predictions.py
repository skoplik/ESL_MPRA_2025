#!/usr/bin/env python3
"""
Exports a shareable model-prediction table from the v2 mega file.

The mega files are 160-185 MB, over GitHub's limit. This writes ~5 MB: the
variant key, the WT PSI needed to reproduce the edge filter, the pooled
measurement, and one delta-logit column per model. The edge_filtered column
makes both rows of the benchmark table reproducible from this one file.

Verifies its output against the published n and r before exiting.
"""
import os
import numpy as np
import pandas as pd

OUTDIR = "/ESL/ESL_MPRA/Figure_3/model_comparison"
SRC = os.path.join(OUTDIR, "mega_pred_file_MAY_v2_NOEDGEFILTER.csv")
OUT = os.path.join(OUTDIR, "model_predictions_MAY_v2.csv.gz")

CELLS = ["HeLa", "K562", "MCF7", "HMC3", "HEK"]
WTPSI_COLS = ["%s_wt_pooled_psi_raw" % c for c in CELLS]

MODELS = [("spliceai_delta_logit", "SpliceAI"),
          ("alphagenome_delta_logit", "AlphaGenome"),
          ("pangolin_delta_logit", "Pangolin"),
          ("baseline_mmsplice_delta_logit", "MMSplice"),
          ("hal_delta_logit", "HAL")]

# what the published table says; the export is checked against this
EXPECTED = {
    "no_filter":   {"SpliceAI": (84021, 0.7473), "AlphaGenome": (84021, 0.7259),
                    "Pangolin": (84021, 0.7257), "MMSplice": (83573, 0.6163),
                    "HAL": (27031, 0.6681)},
    "edge_filter": {"SpliceAI": (71017, 0.7763), "AlphaGenome": (71017, 0.7625),
                    "Pangolin": (71017, 0.7612), "MMSplice": (70718, 0.6658),
                    "HAL": (22770, 0.6693)},
}


def main():
    print("Reading %s ..." % SRC)
    df = pd.read_csv(SRC, low_memory=False)
    print("  %d rows" % len(df))

    # intron1_len is part of the agreed unique key (event_id_161, snp, intron1_len);
    # it is what distinguishes the MANE junction from the alternative junction.
    df["intron1_len"] = df["intron1"].astype(str).str.len()

    # exact 0/1, matching build_mega_MAY_v2.py
    df["edge_filtered"] = df[WTPSI_COLS].apply(
        lambda r: (r == 0).any() or (r == 1).any(), axis=1)

    keep = (["Reference", "event_id", "event_id_161", "gene_exon", "snp",
             "intron1_len", "seq_type", "transcript_class", "source"]
            + WTPSI_COLS
            + ["edge_filtered", "avg_delta_logit_pooled"]
            + [c for c, _ in MODELS]
            + ["retrained_mmsplice_delta_logit", "mmsplice_is_test"])
    keep = [c for c in keep if c in df.columns]

    slim = df[keep].copy()
    slim.to_csv(OUT, index=False, compression="gzip")
    print("Wrote %s  (%.1f MB, %d rows, %d edge-filtered)"
          % (OUT, os.path.getsize(OUT) / 1e6, len(slim),
             int(slim["edge_filtered"].sum())))

    ok = True
    for tag, sub in [("no_filter", slim), ("edge_filter", slim[~slim["edge_filtered"]])]:
        print("\n[%s]" % tag)
        for col, label in MODELS:
            s = sub[[col, "avg_delta_logit_pooled"]].dropna()
            n = len(s)
            r = float(np.corrcoef(s[col], s["avg_delta_logit_pooled"])[0, 1])
            exp_n, exp_r = EXPECTED[tag][label]
            good = (n == exp_n) and (abs(r - exp_r) < 5e-4)
            ok = ok and good
            print("  %-14s n=%-7d r=%.4f   %s"
                  % (label, n, r, "OK" if good else
                     "MISMATCH expected n=%d r=%.4f" % (exp_n, exp_r)))

    if not ok:
        raise SystemExit("ABORT: export does not reproduce the published table")
    print("\nAll models reproduce the published benchmark table.")


if __name__ == "__main__":
    main()
