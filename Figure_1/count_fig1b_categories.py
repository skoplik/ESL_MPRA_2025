#!/usr/bin/env python3
"""
Counts the Figure 1B source categories from the measured data.

Counts against the assayed constructs (1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz), not the
244k design supertable, and **one row per unique full_seq** -- Figure 1B describes
assayed constructs, and its categories must partition the construct total (87,735).

The supertable `source` column is design provenance (which pool a construct came from),
NOT database membership. "ClinVar SNVs" means SNVs actually present in ClinVar, decided
by matching chrom:pos:ref>alt from variant_hg38 against the ClinVar VCF. Counting by
`source` instead gives 31,701 rather than 2,165.

Partition rule: references -> doubles -> singles matching ClinVar -> remaining singles
by `source` (exac, geuvadis) -> everything else is "additional". Reproduces the
published table to within one variant on ClinVar (2,165 vs 2,166).
"""
import re
import sys
import pandas as pd

DATA = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz"
CLINVAR_VCF = "/ESL/ESL_MPRA/Figure_1/inputs/clinvar_20240902.vcf"
EXPECTED_TOTAL = 87735

VAR_RE = re.compile(r"^(chr[^:]+):(\d+):([ACGTN]+)>([ACGTN]+)$")
PUBLISHED = {"references": 1323, "ClinVar SNVs": 2166, "ExAC SNVs": 5948,
             "Geuvadis SNVs": 61, "additional SNVs": 29758, "double variants": 48290}


def main():
    df = pd.read_csv(DATA, low_memory=False)
    df["snp"] = df["snp"].astype(str).str.strip()
    df["fs"] = df["full_seq"].astype(str).str.replace(" ", "", regex=False)

    one = df.drop_duplicates("fs").copy()
    one["n_snp"] = one["snp"].apply(lambda s: 0 if s == "none" else s.count(";") + 1)
    print("unique constructs: %d" % len(one))

    singles = one[one["n_snp"] == 1]
    keys = {}
    for idx, v in singles["variant_hg38"].items():
        if isinstance(v, str):
            m = VAR_RE.match(v.strip())
            if m:
                keys[idx] = (m.group(1), int(m.group(2)), m.group(3), m.group(4))

    wanted = {(c, p) for (c, p, _, _) in keys.values()}
    print("streaming ClinVar VCF for %d positions ..." % len(wanted))
    clinvar = set()
    with open(CLINVAR_VCF) as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t", 5)
            if len(f) < 5:
                continue
            chrom = f[0] if f[0].startswith("chr") else "chr" + f[0]
            try:
                pos = int(f[1])
            except ValueError:
                continue
            if (chrom, pos) in wanted:
                for alt in f[4].split(","):
                    clinvar.add((chrom, pos, f[3], alt))

    in_cv = {i for i, k in keys.items() if k in clinvar}
    src = one["source"].astype(str)
    rest = singles.index.difference(pd.Index(sorted(in_cv)))

    counts = {
        "references":      int((one["n_snp"] == 0).sum()),
        "ClinVar SNVs":    len(in_cv),
        "ExAC SNVs":       int((src.loc[rest] == "exac").sum()),
        "Geuvadis SNVs":   int((src.loc[rest] == "geuvadis").sum()),
        "additional SNVs": int(len(rest) - (src.loc[rest] == "exac").sum()
                               - (src.loc[rest] == "geuvadis").sum()),
        "double variants": int((one["n_snp"] > 1).sum()),
    }

    print("\n%-18s %>8s %>8s" .replace(">", "") % ("category", "published", "May"))
    for k, v in counts.items():
        print("%-18s %8d %8d" % (k, PUBLISHED[k], v))
    total = sum(counts.values())
    print("%-18s %8d %8d" % ("TOTAL", sum(PUBLISHED.values()), total))

    if total != EXPECTED_TOTAL:
        raise SystemExit("ABORT: partition sums to %d, expected %d" % (total, EXPECTED_TOTAL))
    print("\nPartition sums to %d unique constructs - OK" % total)


if __name__ == "__main__":
    main()
