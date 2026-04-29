#!/bin/bash

BASE_DIR=/ESL  # change to your base directory

script=$BASE_DIR/Figures_SK/effect_size_final/plot_rbs_along_seq.py
input_csv=$BASE_DIR/Figures_SK/effect_size_final/out_06_09_2025/out_06_09_2025_psi_supetable_fimo_merged.csv
output_prefix=$BASE_DIR/Figures_SK/effect_size_final/out_06_09_2025/plots/rbps_along_seq_07_01_2025

python $script $input_csv $output_prefix
