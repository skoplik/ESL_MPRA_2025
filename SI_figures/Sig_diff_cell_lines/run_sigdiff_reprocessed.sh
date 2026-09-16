#!/bin/bash
# Sig_diff cell lines - REPROCESSED (May 2026) recalc PSIs from STAR junctions
# Uses per-replicate recalc PSI files from the reprocessing pipeline output

RECOUNT=/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/recount_PSIs
SCRIPT=/ESL/ESL_MPRA/SI_figures/Sig_diff_cell_lines/plot_STAR_PSI_clustermap_4cell_lines_allReps_sigdiff_07_26_2024.py

HEK293_Rep1_PSI=$RECOUNT/HEK293_Rep2_separate_recalc_PSIs_mincov10.txt
HEK293_Rep2_PSI=$RECOUNT/HEK293_Rep3_separate_recalc_PSIs_mincov10.txt
HEK293_WT_Rep1=$RECOUNT/HEK_WT_Rep1_20231101_separate_recalc_PSIs_mincov10.txt
HEK293_WT_Rep2=$RECOUNT/HEK_WT_Rep2_20231101_separate_recalc_PSIs_mincov10.txt

HeLa_Rep1_PSI=$RECOUNT/HeLa_Rep1_separate_recalc_PSIs_mincov10.txt
HeLa_Rep2_PSI=$RECOUNT/HeLa_Rep3_20231031_separate_recalc_PSIs_mincov10.txt

K562_Rep1_PSI=$RECOUNT/K562_Rep1_separate_recalc_PSIs_mincov10.txt
K562_Rep2_PSI=$RECOUNT/K562_Rep3_20231031_separate_recalc_PSIs_mincov10.txt

MCF7_Rep1_PSI=$RECOUNT/MCF7_Rep1_recalc_PSIs_mincov10.txt
MCF7_Rep2_PSI=$RECOUNT/MCF7_Rep2_recalc_PSIs_mincov10.txt

HMC3_Rep1_PSI=$RECOUNT/HMC3_Rep1_recalc_PSIs_mincov10.txt
HMC3_Rep2_PSI=$RECOUNT/HMC3_Rep2_recalc_PSIs_mincov10.txt

output_dir=/ESL/ESL_MPRA/SI_figures/Sig_diff_cell_lines/outputs

/usr/bin/python3.7 $SCRIPT \
--HEK293_PSI_files $HEK293_Rep1_PSI,$HEK293_Rep2_PSI,$HEK293_WT_Rep1,$HEK293_WT_Rep2 \
--HeLa_PSI_files $HeLa_Rep1_PSI,$HeLa_Rep2_PSI \
--K562_PSI_files $K562_Rep1_PSI,$K562_Rep2_PSI \
--MCF7_PSI_files $MCF7_Rep1_PSI,$MCF7_Rep2_PSI \
--HMC3_PSI_files $HMC3_Rep1_PSI,$HMC3_Rep2_PSI \
--output_dir $output_dir --output_prefix "HEK293_HeLa_K562_MCF7_HMC3_compare_any_reps" \
--found_in_all_samples "False"
