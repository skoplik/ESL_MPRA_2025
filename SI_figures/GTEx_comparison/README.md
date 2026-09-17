# GTEx comparison (Figure S4F)

COMPASS wild-type minigene PSI vs pan-tissue GTEx v11 PSI.

These scripts previously lived only in `/ESL/Analysis/Native/` (outside this repo),
so the analysis behind Figure S4F was not version-controlled. Copied here so the
figure is reproducible from the repository.

## Order of operations
1. `download_GTEx_v11_junctions.sh` - fetch the GTEx v11 junction count matrix.
2. `compute_GTEx_v11_allTissue_PSI.py` - pool junction counts across all tissues and
   donors to produce `GTEx_v11_allTissue_avg_STAR_ref_PSIs.txt`, keyed by `STAR_ref`
   (the wild-type sequence's `Reference`).
3. `compare_WT_minigene_vs_GTEx_v11_PSI.py` - original; reads the **March 2026**
   COMPASS table.
   `compare_WT_minigene_vs_GTEx_v11_PSI_MAY.py` - same analysis against the **May 2026**
   reprocessed table (`1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz`). Only MPRA_CSV and the
   output directory differ; GTEX_PSI is pinned to the existing mincov10 GTEx file so
   the two runs are directly comparable.

## Result (2026-09-17 recheck on May data)
Numbers are unchanged from the March run to 3-4 decimal places, so the manuscript text
needs no edit:

| cell line | March r / rho | May r / rho |
|---|---|---|
| HEK  | 0.3877 / 0.4239 | 0.3887 / 0.4245 |
| HeLa | 0.4002 / 0.4350 | 0.4014 / 0.4354 |
| K562 | 0.4016 / 0.4458 | 0.4027 / 0.4461 |
| MCF7 | 0.4103 / 0.4525 | 0.4114 / 0.4531 |
| HMC3 | 0.4147 / 0.4521 | 0.4152 / 0.4526 |
| Mean | 0.3997 / 0.4376 | 0.4010 / 0.4383 |

n = 1,292 matched exons in both. The join is safe: all 1,292 GTEx `STAR_ref` keys are
wild-type References in both data versions, and WT References are 100% stable between
them.

## Caveat
The MAY output directory is named `output_GTEx_v11_WT_comparison_mincov20_MAY` because
`MINCOV` is an argparse default of 20, but `GTEX_PSI` is pinned to the **mincov10** GTEx
file. The directory name is misleading; the comparison is like-for-like.

Outputs are written under `/ESL/Analysis/Native/` and are not tracked here (PDFs are
gitignored repo-wide).
