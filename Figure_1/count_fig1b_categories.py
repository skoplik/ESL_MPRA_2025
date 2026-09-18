#!/usr/bin/env python3
"""
Counts the Figure 1B source categories from the measured data.

Counts against the assayed constructs (1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz), NOT the
244k design supertable -- Figure 1B describes what was assayed.

The supertable `source` column is design provenance (which pool a construct was drawn
from) and is NOT database membership. "ClinVar SNVs" in the figure means SNVs actually
present in ClinVar, so membership is decided by matching chrom:pos:ref>alt from
variant_hg38 against the ClinVar VCF. Counting by `source` instead overstates it by
more than 10x.

Reports categories both as overlapping sets and as a partition, because a variant can
be in more than one database.
"""
import gzip
import os
import re
import sys
import pandas as pd

DATA = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz"
CLINVAR_VCF = "/ESL/Data/Sequences/GRCh38_hg38/clinvar_20240902.vcf"

VAR_RE = re.compile(r"^(chr[^:]+):(\d+):([ACGTN]+)>([ACGTN]+)$")


def parse_variants(df):
    """variant_hg38 -> set of (chrom, pos, ref, alt); returns keys aligned to df.index."""
    keys = {}
    bad = 0
    for idx, v in df["variant_hg38"].items():
        if not isinstance(v, str):
            continue
        parts = [p for p in v.split(";") if p.strip()]
        ks = []
        for p in parts:
            m = VAR_RE.match(p.strip())
            if m:
                ks.append((m.group(1), int(m.group(2)), m.group(3), m.group(4)))
            else:
                bad += 1
        if ks:
            keys[idx] = ks
    return keys, bad


def load_clinvar(wanted_pos):
    """Stream the ClinVar VCF, keep only records at positions we care about."""
    found = set()
    opener = gzip.open if CLINVAR_VCF.endswith(".gz") else open
    with opener(CLINVAR_VCF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 5)
            if len(f) < 5:
                continue
            chrom = f[0] if f[0].startswith("chr") else "chr" + f[0]
            try:
                pos = int(f[1])
            except ValueError:
                continue
            if (chrom, pos) not in wanted_pos:
                continue
            for alt in f[4].split(","):
                found.add((chrom, pos, f[3], alt))
    return found


def main():
    print("Reading %s ..." % DATA)
    df = pd.read_csv(DATA, low_memory=False)
    df["snp"] = df["snp"].astype(str).str.strip()
    print("  %d rows" % len(df))

    wt = df[df["snp"] == "none"]
    var = df[df["snp"] != "none"].copy()
    var["n_snp"] = var["snp"].str.count(";") + 1
    singles = var[var["n_snp"] == 1]
    doubles = var[var["n_snp"] >= 2]

    print("\nreferences (snp == none)      : %7d rows, %7d unique full_seq"
          % (len(wt), wt["full_seq"].nunique()))
    print("single variants               : %7d rows" % len(singles))
    print("double (2+) variants          : %7d rows" % len(doubles))

    keys, bad = parse_variants(singles)
    print("\nsingles with a parseable variant_hg38: %d  (unparseable fields: %d)"
          % (len(keys), bad))

    wanted_pos = {(c, p) for ks in keys.values() for (c, p, _, _) in ks}
    print("distinct genomic positions to look up: %d" % len(wanted_pos))

    if not os.path.exists(CLINVAR_VCF):
        raise SystemExit("ABORT: ClinVar VCF not found at %s" % CLINVAR_VCF)
    print("Streaming ClinVar VCF (this takes a minute) ...")
    clinvar = load_clinvar(wanted_pos)
    print("  ClinVar records at our positions: %d" % len(clinvar))

    in_clinvar = {idx for idx, ks in keys.items() if any(k in clinvar for k in ks)}
    print("\n" + "=" * 62)
    print("SINGLE VARIANTS PRESENT IN CLINVAR : %d" % len(in_clinvar))
    print("  (counting by source == clinvar_single would give %d -- WRONG)"
          % int((singles["source"].astype(str) == "clinvar_single").sum()))
    print("=" * 62)

    print("\nsource column (design provenance, NOT database membership):")
    print(df["source"].value_counts(dropna=False).to_string())

    print("\nTODO: ExAC and Geuvadis membership need their own source files;")
    print("      /ESL/Figures/Resources/supertable_dbSNP155_Gencode_v26/ has dbSNP155")
    print("      ExAC/GnomAD match lists but they are keyed by the OLD supertable")
    print("      Reference, which renumbers -- match on coordinates instead.")


if __name__ == "__main__":
    main()
