#!/bin/bash

BASE_DIR=/ESL  # change to your base directory

script=$BASE_DIR/Figures_SK/Start_2_end_psi_fig3/scripts/start_2_end_clipped_updated_07_30_25.py
input_csv=$BASE_DIR/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WITH_WT.csv
output_dir=$BASE_DIR/Figures_SK/Start_2_end_psi_fig3/output_07_30_25

python3 $script \
--input_csv $input_csv \
--output_dir $output_dir \
--output_prefix start2end_pooled_density
