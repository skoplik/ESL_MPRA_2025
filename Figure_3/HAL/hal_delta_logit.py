"""
Turn a HAL web-server predictions file into per-variant delta logits.

HAL (http://splicing.cs.washington.edu/SE) returns WT_PSI and MUT_PSI as
percentages. The delta logit matches the convention in plot_hal_res.py and the
edge-filter convention in build_mega_MAY_v2.py:

    hal_delta_logit = logit(clip(MUT_PSI/100)) - logit(clip(WT_PSI/100))

with clip to [0.01, 0.99]. Reference is parsed from VARIANT_NAME = "Seq_<ref>".
This is a pure function of the predictions file; no supertable join is needed,
so it works the same for the main and with_SI inputs.

With --supertable, also prints the correlation against the measured mean delta
logit, with and without the exact 0/1 WT-PSI edge filter. Writes new files only.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.special import logit

CELLS = ["HeLa", "K562", "MCF7", "HMC3", "HEK"]
WTPSI_COLS = ["%s_wt_pooled_psi_raw" % c for c in CELLS]


def clip_psi(x, eps=1e-2):
    return x.clip(lower=eps, upper=1 - eps)


def hal_delta_logit(pred):
    wt = clip_psi(pd.to_numeric(pred["WT_PSI"], errors="coerce") / 100)
    mut = clip_psi(pd.to_numeric(pred["MUT_PSI"], errors="coerce") / 100)
    return logit(mut) - logit(wt)


ap = argparse.ArgumentParser()
ap.add_argument("--predictions", required=True, help="HAL web-server predictions .tsv")
ap.add_argument("--output", required=True, help="output Reference,hal_delta_logit csv")
ap.add_argument("--supertable", help="1e-2_ALL_WITH_WT[_with_SI].csv[.gz] to print r")
args = ap.parse_args()

pred = pd.read_csv(args.predictions, sep="\t")
pred["Reference"] = pred["VARIANT_NAME"].str.replace("Seq_", "", regex=False).astype(int)
pred["hal_delta_logit"] = hal_delta_logit(pred)
if pred["Reference"].duplicated().any():
    raise SystemExit("ABORT: duplicate Reference in predictions")
out = pred[["Reference", "hal_delta_logit"]].sort_values("Reference")
out.to_csv(args.output, index=False)
print("wrote %s: %d variants" % (args.output, len(out)))

if args.supertable:
    sup = pd.read_csv(args.supertable, low_memory=False)
    dl = [c for c in sup.columns if c.endswith("_delta_logit_pooled")]
    sup["exp_dlogit_avg"] = sup[dl].mean(axis=1, skipna=True)
    edge = ((sup[WTPSI_COLS] == 0) | (sup[WTPSI_COLS] == 1)).any(axis=1)
    m = out.merge(sup[["Reference", "exp_dlogit_avg", "source_table"]], on="Reference", how="left")
    m["edge"] = m["Reference"].isin(set(sup.loc[edge, "Reference"]))
    for label, sub in [("all", m), ("edge-filtered", m[~m["edge"]])]:
        s = sub[["hal_delta_logit", "exp_dlogit_avg"]].dropna()
        r = np.corrcoef(s.iloc[:, 0], s.iloc[:, 1])[0, 1] if len(s) > 2 else float("nan")
        print("  %-14s n=%d  r=%.4f" % (label, len(s), r))
    if "source_table" in m.columns:
        si = m[(m["source_table"] == "si") & (~m["edge"])][["hal_delta_logit", "exp_dlogit_avg"]].dropna()
        if len(si) > 2:
            r = np.corrcoef(si.iloc[:, 0], si.iloc[:, 1])[0, 1]
            print("  SI only (edge-filt) n=%d  r=%.4f" % (len(si), r))
