#!/usr/bin/env python3
"""Measurement noise of the double-variant additivity deviation (Figure 2E / S5E).

For each double variant with both constituent SNVs and the reference measured in the
same exon family, the deviation from additivity is
    dev = dlogit(double) - [dlogit(SNV1) + dlogit(SNV2)]
        = L_double - L_snv1 - L_snv2 + L_ref          (L = logit PSI)
The paper uses replicate-pooled dlogit averaged across cell lines and calls a pair
non-additive when |dev| > 1.

To see how large a deviation measurement noise alone produces, the deviation is
recomputed in two independent replicate halves of every cell line
(HeLa, K562, MCF7, HMC3: rep1 vs rep2; HEK: rep1+rep2 vs rep3+rep4), averaged across
the cell lines in which all four sequences are measured in both halves. Because both
halves measure the same pair, their difference is pure noise. The noise SD of the
full-data deviation is estimated as SD(dev_A - dev_B) / 2 (the full estimate averages
the two halves), with a robust MAD-based version alongside.

Writes a short summary to stdout; no figure.
"""
import re
import numpy as np
import pandas as pd

PSI = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz"
HALVES = {"HeLa": (["rep1"], ["rep2"]), "K562": (["rep1"], ["rep2"]),
          "MCF7": (["rep1"], ["rep2"]), "HMC3": (["rep1"], ["rep2"]),
          "HEK": (["rep1", "rep2"], ["rep3", "rep4"])}
THRESH = 1.0

cols = ["Reference", "event_id", "snp", "seq_type"] + \
       [f"{c}_{r}_logit" for c, (a, b) in HALVES.items() for r in a + b]
df = pd.read_csv(PSI, usecols=cols, low_memory=False)

def half_logit(frame, cell, reps):
    return frame[[f"{cell}_{r}_logit" for r in reps]].mean(axis=1, skipna=False)

for c, (a, b) in HALVES.items():
    df[f"{c}_A"] = half_logit(df, c, a)
    df[f"{c}_B"] = half_logit(df, c, b)
hcols = [f"{c}_{h}" for c in HALVES for h in "AB"]

def parse(s):
    return [(int(m.group(1)), m.group(2), m.group(3))
            for m in re.finditer(r"(\d+)\s*:\s*([ACGT])\s*>\s*([ACGT])", str(s))]

rows = []
for ev, g in df.groupby("event_id"):
    ref = g[g["snp"] == "none"]
    if ref.empty:
        continue
    ref = ref.iloc[0]
    singles = {}
    for _, r in g[g["seq_type"] == "single"].iterrows():
        v = parse(r["snp"])
        if len(v) == 1:
            singles[v[0]] = r
    for _, d in g[g["seq_type"] == "double"].iterrows():
        v = parse(d["snp"])
        if len(v) != 2 or v[0] not in singles or v[1] not in singles:
            continue
        s1, s2 = singles[v[0]], singles[v[1]]
        devA, devB = [], []
        for c in HALVES:
            vals = [x[[f"{c}_A", f"{c}_B"]].astype(float).values for x in (d, s1, s2, ref)]
            if np.isnan(np.concatenate(vals)).any():
                continue
            dd, a1, a2, rr = vals
            devA.append(dd[0] - a1[0] - a2[0] + rr[0])
            devB.append(dd[1] - a1[1] - a2[1] + rr[1])
        if devA:
            rows.append((ev, len(devA), np.mean(devA), np.mean(devB)))

res = pd.DataFrame(rows, columns=["event_id", "n_cells", "dev_A", "dev_B"])
diff = res["dev_A"] - res["dev_B"]
sd_full = diff.std(ddof=1) / 2
mad_full = 1.4826 * np.median(np.abs(diff - diff.median())) / 2
full = (res["dev_A"] + res["dev_B"]) / 2

from math import erf, sqrt
def p_exceed(t, s):
    return 1 - erf(t / (s * sqrt(2)))

print(f"pairs with both replicate halves measured: {len(res):,} "
      f"(median cell lines per pair {int(res['n_cells'].median())})")
print(f"noise SD of the full-data deviation: {sd_full:.3f} (robust MAD estimate {mad_full:.3f})")
print(f"cutoff of {THRESH} = {THRESH/sd_full:.1f} SD ({THRESH/mad_full:.1f} robust SD)")
print(f"expected fraction of truly additive pairs with |dev| > {THRESH} by noise alone "
      f"(normal): {100*p_exceed(THRESH, sd_full):.2f}% (robust: {100*p_exceed(THRESH, mad_full):.2f}%)")
print(f"empirical: fraction of pairs whose half-difference/2 exceeds {THRESH}: "
      f"{100*np.mean(np.abs(diff/2) > THRESH):.2f}%")
print(f"observed non-additive fraction in these pairs (|mean of halves| > {THRESH}): "
      f"{100*np.mean(np.abs(full) > THRESH):.2f}%")
for t in (0.5, 0.75, 1.0, 1.25, 1.5):
    print(f"  cutoff {t}: {t/sd_full:.1f} SD, expected noise exceedance {100*p_exceed(t, sd_full):.2f}%, "
          f"observed {100*np.mean(np.abs(full) > t):.2f}%")
