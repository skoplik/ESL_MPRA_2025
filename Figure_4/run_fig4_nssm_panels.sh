#!/usr/bin/env bash
# NSSM maps used in the main text and SI (May 2026 data):
#   Figure 4B  SPINT2 e3, DEPDC5 e6, SNCA e3, BRCA1 e13, PPT1 e8, BIN1 e16 (ALL)
#   Figure 4D-F MYH11 e42, SCN5A e24, LMNA e10 (ALL)
#   Figure 5B  BIN1 e16 (HEK)
#   Figure S9E COLQ e5 (ALL)
# Panels -> Figure_4/outputs/nssm_MAY/{ALL,HEK,HeLa,K562,HMC3,MCF7}/
# Figure S6 panels are made by SI_figures/S6_NSSM/run_figS6_panels.sh.
set -euo pipefail
REPO=/ESL/ESL_MPRA
OUT=$REPO/Figure_4/outputs/nssm_MAY
for ge in "spint2 exon 3" "depdc5 exon 6" "snca exon 3" "brca1 exon 13" "ppt1 exon 8" \
          "bin1 exon 16" "myh11 exon 42" "scn5a exon 24" "lmna exon 10" "colq exon 5"; do
  python3 $REPO/Figure_4/make_scn5a_ssm.py --gene-exon "$ge" --outdir "$OUT"
done

# BRCA1 exon 13 (Figure 4B) has 120 single mutants, below the 150 cutoff in the
# standard plotting script; the BRCA1 copy is identical except for a cutoff of 100.
python3 $REPO/Figure_4/make_scn5a_ssm.py --gene-exon "brca1 exon 13" --outdir "$OUT" \
  --script $REPO/Figure_4/ssm_single_muts_logits_BRCA1_exon13_fixed.py
