#!/bin/bash

BASE_DIR=/ESL  # change to your base directory

script=$BASE_DIR/Figures_SK/dpsi_position_fig1/scripts/plot_dpsi_by_pos_merged_logit.py
psi_merged_file=$BASE_DIR/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WITH_WT.csv
output_dir=$BASE_DIR/Figures_SK/dpsi_position_fig1/output_07_21_2025

python3 $script \
--output_dir $output_dir \
--cell_psi_merged_file $psi_merged_file \
--output_prefix all_cells_merged_06_09_2025_rasterized_redo_07_21_2025
