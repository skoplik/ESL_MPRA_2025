#!/usr/bin/env python3
"""Reproduce the library-composition numbers quoted in the Results text.

Produces every number in the manuscript sentences:

  "COMPASS contains N references, N variants from Geuvadis; N from ExAC;
   N from ClinVar; and an additional N SNVs and N double variants ...
   In total, the library contains N sequences spanning N native human exons
   from N genes."

  "We found N sequences with measurements in at least one cell type. Of these,
   N lacked a matched reference ... In total, N variants remained with an
   associated reference, allowing calculation of dPSI."

Two conventions matter and are enforced here:

1. Counts come from NO_DELTAS, not ALL_WITH_WT. ALL_WITH_WT drops constructs
   that have no matched WT; those constructs were still assayed, so using it
   undercounts the library. (See section M3 of the paper-edits notes.)

2. ClinVar membership is NOT the `source` column. `source == "clinvar_single"`
   is the whole ClinVar-*targeted* pool and mixes variants reported in ClinVar
   with the randomly targeted SNVs. Actual ClinVar-database membership comes
   from the supertable ClinVar annotation (same file Figure 4C uses), joined on
   Reference at offset 0. ClinVar takes priority over the source label, which
   is what makes the published categories mutually exclusive and sum to the
   table size.

The offset-0 join is verified here rather than assumed: for every matched row
the construct-local SNP in the PSI table must equal the mutation recorded in
the ClinVar file. Offsets -1 and +1 are reported for contrast.

Run (pandas lives in the ec2-user site-packages on this instance):

  PYTHONPATH=/home/ec2-user/.local/lib/python3.7/site-packages \
      /usr/bin/python3.7 Figure_1/compute_library_composition.py
"""
import argparse
import ast
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(
    REPO, "Data_Pre-Processing", "Post-process_STAR_PSIs", "output",
    "1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz")
CLINVAR = os.path.join(
    REPO, "Figure_4", "inputs", "supertable_ClinVar_matched_in_supertable.txt")
CELLS = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]


def clinvar_refs(path):
    """Reference ids and construct-local mutation tags from the ClinVar file."""
    cv = pd.read_csv(path, sep="\t")
    cv["mut_ref"] = pd.to_numeric(cv["mut_ref"], errors="coerce")
    cv = cv.dropna(subset=["mut_ref"]).copy()
    cv["Reference"] = cv["mut_ref"].astype(int)

    def tag(cell):
        """'[(146, ("T", "C"))]' -> '146:T>C', matching the `snp` column."""
        try:
            muts = ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            return None
        if isinstance(muts, list) and len(muts) == 1:
            pos, (ref, alt) = muts[0]
            return "%d:%s>%s" % (pos, ref, alt)
        return None

    cv["snp_tag"] = cv["mut_ref_mut_positions"].map(tag)
    return cv[["Reference", "snp_tag"]]


def verify_offset(variants, cv):
    """Report allele agreement at offsets -1/0/+1; return the verified offset."""
    print("\n=== Reference join offset ===")
    best = None
    for off in (-1, 0, 1):
        shifted = cv.assign(Reference=cv["Reference"] + off)
        j = variants.merge(shifted, on="Reference", how="inner")
        tagged = j.dropna(subset=["snp_tag"])
        agree = (tagged["snp"] == tagged["snp_tag"]).mean() if len(tagged) else 0.0
        print("  offset %+d: %5d variant rows matched, allele agreement %.4f"
              % (off, len(j), agree))
        if agree == 1.0 and len(j):
            best = off
    if best is None:
        raise SystemExit("no offset gives full allele agreement - check inputs")
    print("  -> using offset %+d" % best)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", default=MAIN)
    ap.add_argument("--clinvar", default=CLINVAR)
    ap.add_argument("--out", default=os.path.join(
        REPO, "Figure_1", "outputs", "library_composition.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.main, low_memory=False)
    print("input: %s" % args.main)
    print("rows : %d" % len(df))

    is_ref = df["seq_type"].eq("none")
    variants = df.loc[~is_ref, ["Reference", "source", "snp"]]

    cv = clinvar_refs(args.clinvar)
    print("\nClinVar annotation: %s" % args.clinvar)
    print("  %d unique References" % cv["Reference"].nunique())
    offset = verify_offset(variants, cv)

    in_clinvar = df["Reference"].isin(set(cv["Reference"] + offset)) & ~is_ref
    print("\nClinVar-matched variant rows: %d" % int(in_clinvar.sum()))
    print("  by source label:")
    for src, n in df.loc[in_clinvar, "source"].value_counts().items():
        print("    %-15s %d" % (src, n))

    # Exclusive categories: ClinVar wins over the source label, so the
    # categories partition the table exactly as the published breakdown does.
    rows = [("references", int(is_ref.sum()))]
    for src, label in (("geuvadis", "Geuvadis"), ("exac", "ExAC")):
        rows.append((label, int((df["source"].eq(src) & ~is_ref & ~in_clinvar).sum())))
    rows.append(("ClinVar", int(in_clinvar.sum())))
    rows.append(("additional SNVs",
                 int((df["source"].eq("clinvar_single") & ~is_ref & ~in_clinvar).sum())))
    rows.append(("double variants",
                 int((df["source"].eq("clinvar_double") & ~is_ref & ~in_clinvar).sum())))

    print("\n=== library composition ===")
    for label, n in rows:
        print("  %-18s %7d" % (label, n))
    total = sum(n for _, n in rows)
    print("  %-18s %7d  (table rows %d)" % ("TOTAL", total, len(df)))
    if total != len(df):
        raise SystemExit("categories do not partition the table")

    print("\n=== spanned by ===")
    print("  native exons (event_id_161) %6d   <- the number quoted in the text"
          % df["event_id_161"].nunique())
    print("  exons (exon-coord event_id) %6d   <- different definition, not a data change"
          % df["event_id"].nunique())
    genes = df["gene_exon"].astype(str).str.split().str[0]
    print("  genes                       %6d" % genes.nunique())

    measured = df[[c + "_pooled_psi_raw" for c in CELLS]].notna().any(axis=1)
    has_wt = df["wt_reference"].notna()
    d_ok = df[[c + "_dpsi_pooled" for c in CELLS]].notna().any(axis=1)
    print("\n=== measurement / dPSI ===")
    print("  sequences measured in >=1 cell type    %6d" % int(measured.sum()))
    print("  lacking a matched reference            %6d" % int((~is_ref & ~has_wt).sum()))
    print("  variants with a matched reference      %6d" % int((~is_ref & has_wt).sum()))
    print("  ... and a computable dPSI              %6d   <- the number quoted in the text"
          % int((~is_ref & d_ok).sum()))
    print("\n  per cell line:")
    for c in CELLS:
        print("    %-6s %6d" % (c, int(df[c + "_pooled_psi_raw"].notna().sum())))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows, columns=["category", "n"]).to_csv(args.out, index=False)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
