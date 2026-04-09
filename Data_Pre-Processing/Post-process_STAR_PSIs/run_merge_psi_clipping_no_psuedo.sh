#!/bin/bash
# Merges per-replicate PSI files across all five cell lines into a single master CSV.
# Input PSI files are produced by get_STAR_separate_splicing_profiles.py (Step 2b).
# Processed PSI files are available on GEO (GSE307247) as supplementary files.
#
# Dependencies (Python packages): numpy, pandas, scipy
# Install Python packages: pip install numpy pandas scipy

BASE_DIR=/ESL  # change to your base directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

STAR_PSI_dir=$BASE_DIR/Analysis/STAR_alignment/separate_concat_2023_09_19_d1c_ms75_from_s3_2023_11_27/recount_SJs
PSI_dir_2024_07_26=$STAR_PSI_dir/2024_07_26
HEK293_Rep1_PSI=$STAR_PSI_dir/HEK293_Rep2_separate_splicing_profiles/HEK293_Rep2_separate_recalc_PSIs_mincov10.txt
HEK293_Rep2_PSI=$STAR_PSI_dir/HEK293_Rep3_separate_splicing_profiles/HEK293_Rep3_separate_recalc_PSIs_mincov10.txt
HEK293_WT_Rep1_PSI=$BASE_DIR/Analysis/WT_Library/separate/recount_SJs/HEK_WT_Rep1_20231101_separate_splicing_profiles/HEK_WT_Rep1_20231101_separate_recalc_PSIs_mincov10.txt
HEK293_WT_Rep2_PSI=$BASE_DIR/Analysis/WT_Library/separate/recount_SJs/HEK_WT_Rep2_20231101_separate_splicing_profiles/HEK_WT_Rep2_20231101_separate_recalc_PSIs_mincov10.txt
HeLa_Rep1_PSI=$STAR_PSI_dir/HeLa_Rep1_separate_splicing_profiles/HeLa_Rep1_separate_recalc_PSIs_mincov10.txt
HeLa_Rep2_PSI=$STAR_PSI_dir/HeLa_Rep3_20231031_separate_splicing_profiles/HeLa_Rep3_20231031_separate_recalc_PSIs_mincov10.txt
K562_Rep1_PSI=$STAR_PSI_dir/K562_Rep1_separate_splicing_profiles/K562_Rep1_separate_recalc_PSIs_mincov10.txt
K562_Rep2_PSI=$STAR_PSI_dir/K562_Rep3_20231031_separate_splicing_profiles/K562_Rep3_20231031_separate_recalc_PSIs_mincov10.txt
MCF7_Rep1_PSI=$PSI_dir_2024_07_26/MCF7_Rep1_separate_splicing_profiles/MCF7_Rep1_recalc_PSIs_mincov10.txt
MCF7_Rep2_PSI=$PSI_dir_2024_07_26/MCF7_Rep2_separate_splicing_profiles/MCF7_Rep2_recalc_PSIs_mincov10.txt
HMC3_Rep1_PSI=$PSI_dir_2024_07_26/HMC3_Rep1_separate_splicing_profiles/HMC3_Rep1_recalc_PSIs_mincov10.txt
HMC3_Rep2_PSI=$PSI_dir_2024_07_26/HMC3_Rep2_separate_splicing_profiles/HMC3_Rep2_recalc_PSIs_mincov10.txt

output_dir=$BASE_DIR/Figures_SK/General_preprocessing/output_03_16_2026
supertable_file=$BASE_DIR/Figures_SK/General_preprocessing/fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv

python3 $SCRIPT_DIR/merge_psi_07_18_2025_clipping_no_psuedo.py \
--output_dir $output_dir \
--output_prefix 03_16_2026_1e-2 \
--supertable_file $supertable_file \
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
