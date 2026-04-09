#!/bin/bash

BASE_DIR=/ESL  # change to your base directory

script=$BASE_DIR/Figures_SK/redo_pop_variants/redo_clinvar_new_colors_t_tests.py
input_csv=$BASE_DIR/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WITH_WT.csv
clinvar_file=$BASE_DIR/Figures/Variant_analyses/heatmap_plots/supertable_ClinVar_matched_in_supertable.txt
output_pdf=$BASE_DIR/Figures_SK/redo_pop_variants/clinvar_updated_7_15_25/swarm_delta_logit.pdf

python3 $script $input_csv $clinvar_file $output_pdf
