#!/bin/bash
# Stage 2 — Merge per-replicate PSI files into the master CSV.
# Runs 02_merge_psi.py against the corrected supertable + patched per-rep PSI
# text files produced by 01_fix_sj_supertable.py.

BASE_DIR=/ESL  # change to your base directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

OUTPUT_DIR=/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output
PATCHED_PSI_DIR=$OUTPUT_DIR/recount_PSIs
SUPERTABLE_FILE=$OUTPUT_DIR/st_corrected.csv

HEK293_Rep1_PSI=$PATCHED_PSI_DIR/HEK293_Rep2_separate_recalc_PSIs_mincov10.txt
HEK293_Rep2_PSI=$PATCHED_PSI_DIR/HEK293_Rep3_separate_recalc_PSIs_mincov10.txt
HEK293_WT_Rep1_PSI=$PATCHED_PSI_DIR/HEK_WT_Rep1_20231101_separate_recalc_PSIs_mincov10.txt
HEK293_WT_Rep2_PSI=$PATCHED_PSI_DIR/HEK_WT_Rep2_20231101_separate_recalc_PSIs_mincov10.txt
HeLa_Rep1_PSI=$PATCHED_PSI_DIR/HeLa_Rep1_separate_recalc_PSIs_mincov10.txt
HeLa_Rep2_PSI=$PATCHED_PSI_DIR/HeLa_Rep3_20231031_separate_recalc_PSIs_mincov10.txt
K562_Rep1_PSI=$PATCHED_PSI_DIR/K562_Rep1_separate_recalc_PSIs_mincov10.txt
K562_Rep2_PSI=$PATCHED_PSI_DIR/K562_Rep3_20231031_separate_recalc_PSIs_mincov10.txt
MCF7_Rep1_PSI=$PATCHED_PSI_DIR/MCF7_Rep1_recalc_PSIs_mincov10.txt
MCF7_Rep2_PSI=$PATCHED_PSI_DIR/MCF7_Rep2_recalc_PSIs_mincov10.txt
HMC3_Rep1_PSI=$PATCHED_PSI_DIR/HMC3_Rep1_recalc_PSIs_mincov10.txt
HMC3_Rep2_PSI=$PATCHED_PSI_DIR/HMC3_Rep2_recalc_PSIs_mincov10.txt

python3 $SCRIPT_DIR/02_merge_psi.py \
--output_dir $OUTPUT_DIR \
--output_prefix 1e-2 \
--supertable_file $SUPERTABLE_FILE \
--HeLa_Rep1_PSI $HeLa_Rep1_PSI \
--HeLa_Rep2_PSI $HeLa_Rep2_PSI \
--K562_Rep1_PSI $K562_Rep1_PSI \
--K562_Rep2_PSI $K562_Rep2_PSI \
--MCF7_Rep1_PSI $MCF7_Rep1_PSI \
--MCF7_Rep2_PSI $MCF7_Rep2_PSI \
--HEK293_Rep1_PSI $HEK293_Rep1_PSI \
--HEK293_Rep2_PSI $HEK293_Rep2_PSI \
--HEK293_WT_Rep1_PSI $HEK293_WT_Rep1_PSI \
--HEK293_WT_Rep2_PSI $HEK293_WT_Rep2_PSI \
--HMC3_Rep1_PSI $HMC3_Rep1_PSI \
--HMC3_Rep2_PSI $HMC3_Rep2_PSI \
--clip=1e-2
