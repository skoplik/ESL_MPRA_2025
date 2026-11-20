#!/usr/bin/env python3
"""Build Supplementary Table 1 - reference PSI and dlogit ranges for near-saturation exons.

Spec taken verbatim from the SI caption:

  "This table reports all exons with at least 300 assayed variants. For each
   exon we provide the gene and exon number, event identifier with the genomic
   coordinates (hg38) of the variable sequence, coordinates of the exon start
   and stop, the mean reference PSI calculated across all cell lines, and the
   minimum and maximum average dlogit(PSI) across all cell lines. The minimum
   and maximum is determined across all variants for that exon. We also report
   the total number of variants observed for that exon."

Counts come from NO_DELTAS per section M3 of the paper-edits notes: it is the
superset that includes constructs with no matched WT, so nothing assayed is
silently dropped.
"""
import os
import numpy as np
import pandas as pd

MAIN = ("/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/"
        "1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz")
OUT_DIR = "/ESL/ESL_MPRA/SI_figures/supp_tables"
CELLS = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]
MIN_VARIANTS = 300

# gene_exon labels that are wrong or ambiguous in the source data, fixed by exon
# coordinates (event_id). Numbering follows the MANE Select transcript
# (GENCODE v38). CACNA1C: the source labels two different loci "exon 31".
#   chr12:2648475-2648507 (33 nt) is MANE exon 31.
#   chr12:2633629-2633712 (84 nt) is not in MANE; it is the mutually exclusive
#   partner of MANE exon 30 (chr12:2634297-2634380), so "exon 30 (alt.)".
LABEL_FIXES = {
    "chr12:2633629-2633712:+": "CACNA1C exon 30 (alt.)",
    "chr12:2648475-2648507:+": "CACNA1C exon 31",
}


def main():
    df = pd.read_csv(MAIN, low_memory=False)
    print("rows: %d" % len(df))

    psi_cols = [c + "_pooled_psi_raw" for c in CELLS]
    dl_cols = [c + "_delta_logit_pooled" for c in CELLS]

    is_ref = df["seq_type"].eq("none")
    measured = df[psi_cols].notna().any(axis=1)

    # average dlogit across cell lines, per variant
    df["avg_dlogit"] = df[dl_cols].mean(axis=1, skipna=True)

    # the exon identity is the construct-level locus; exon coordinates are
    # carried separately so both appear in the table as the caption asks
    KEY = "event_id_161"

    variants = df[~is_ref & measured]
    n_by_exon = variants.groupby(KEY).size()
    keep = n_by_exon[n_by_exon >= MIN_VARIANTS].index
    print("exons with >= %d assayed variants: %d" % (MIN_VARIANTS, len(keep)))

    rows = []
    for key in keep:
        v = variants[variants[KEY] == key]
        r = df[(df[KEY] == key) & is_ref]

        ref_psi = np.nan
        if len(r):
            ref_psi = r.iloc[0][psi_cols].astype(float).mean(skipna=True)

        dl = v["avg_dlogit"].dropna()
        first = v.iloc[0]
        rows.append({
            "gene_exon": LABEL_FIXES.get(first["event_id"], first["gene_exon"]),
            "event_id_161_hg38": key,
            "event_id_exon_hg38": first["event_id"],
            "exon_start_hg38": first["exon_start_hg38"],
            "exon_end_hg38": first["exon_end_hg38"],
            "mean_reference_PSI": round(float(ref_psi), 4) if pd.notna(ref_psi) else np.nan,
            "min_avg_dlogit": round(float(dl.min()), 4) if len(dl) else np.nan,
            "max_avg_dlogit": round(float(dl.max()), 4) if len(dl) else np.nan,
            "n_variants": int(len(v)),
        })

    out = pd.DataFrame(rows).sort_values("gene_exon").reset_index(drop=True)
    dup = out[out["gene_exon"].duplicated(keep=False)]
    if len(dup):   # one label on two loci: add the locus to LABEL_FIXES
        raise SystemExit("duplicate gene_exon labels:\n" + dup.to_string(index=False))
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "Supplementary_Table_1.csv")
    out.to_csv(path, index=False)

    print("\nrows in table: %d" % len(out))
    print("n_variants: min %d  median %d  max %d"
          % (out["n_variants"].min(), out["n_variants"].median(), out["n_variants"].max()))
    print("mean reference PSI missing for %d exons" % out["mean_reference_PSI"].isna().sum())
    print("\nexons with reference PSI < 0.05 (only inclusion detectable):")
    low = out[out["mean_reference_PSI"] < 0.05]
    print(low[["gene_exon", "mean_reference_PSI", "n_variants"]].to_string(index=False)
          if len(low) else "  none")
    print("\nMEF2C rows (cited in the main text):")
    mef = out[out["gene_exon"].astype(str).str.startswith("MEF2C")]
    print(mef.to_string(index=False) if len(mef) else "  MEF2C not in the >=300 set")
    print("\nfirst 8 rows:")
    print(out.head(8).to_string(index=False))
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
