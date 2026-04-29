#!/bin/bash

BASE_DIR=/ESL  # change to your base directory

script=$BASE_DIR/Figures_SK/SSMs_single_muts/ssm_single_muts_logits_06_04_2025_with_clinvar.py
input_csv=$BASE_DIR/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WITH_WT.csv
output_dir=$BASE_DIR/Figures_SK/SSMs_single_muts/output_logit_clipped_08_27_2025_with_clinvar_more_ssms_coolwarm
clinvar_file=$BASE_DIR/Figures/Variant_analyses/heatmap_plots/supertable_ClinVar_matched_in_supertable.txt

python $script $input_csv $output_dir $clinvar_file
