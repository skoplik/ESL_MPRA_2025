#!/usr/bin/env bash
# Figure S6: regenerate the 12 NSSM panels from the May 2026 data, then assemble.
# Panels  -> Figure_4/outputs/nssm_S6_MAY/{ALL,HEK,HeLa,K562,HMC3,MCF7}/
# Figure  -> SI_figures/SpliceAI_NSSM/assembled/figS6_nssm_maps_MAY.pdf
# Needs pandas, matplotlib, pymupdf.
set -euo pipefail
REPO=/ESL/ESL_MPRA
OUT=$REPO/Figure_4/outputs/nssm_S6_MAY
for ge in "lmna exon 10" "mutyh exon 6" "myh11 exon 42" "myh11 exon 6" "ryr1 exon 94" \
          "scn5a exon 24" "tnnt2 exon 4" "tnnt2 exon 5" "tsc2 exon 32" \
          "best1 exon 8" "hmbs exon 2" "spint2 exon 3"; do
  python3 $REPO/Figure_4/make_scn5a_ssm.py --gene-exon "$ge" --outdir "$OUT"
done
python3 $REPO/SI_figures/SpliceAI_NSSM/assemble_figS6_MAY.py --src "$OUT/ALL"
