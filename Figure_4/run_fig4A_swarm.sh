#!/usr/bin/env bash
# Figure 4A: average dlogit(PSI) by ClinVar class (B/LB, P/LP, CL/VUS), May 2026 data.
# Replaces run_clinvar_annots_071625.sh, which pointed at July 2025 inputs in /ESL/Figures_SK.
# ClinVar table is Reference-aligned to the May PSI file with no shift (verified 2166/2166 SNP matches).
set -euo pipefail
REPO=/ESL/ESL_MPRA
python3 $REPO/Figure_4/redo_clinvar_new_colors_t_tests.py \
  $REPO/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz \
  $REPO/Figure_4/inputs/supertable_ClinVar_matched_in_supertable.txt \
  $REPO/Figure_4/outputs/swarm_delta_logit_MAY.pdf
