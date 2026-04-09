#!/bin/bash

BASE_DIR=/ESL  # change to your base directory

script=$BASE_DIR/Figures_SK/General_preprocessing/scripts/comp_pooled_to_repl.py
input_dir=$BASE_DIR/Figures_SK/General_preprocessing/output_5_16_2025
output_dir=$input_dir/comp_pool_to_repl_plots

python $script $input_dir/05_16_2025_HeLa_WITH_WT.csv HeLa $output_dir
python $script $input_dir/05_16_2025_HEK_WITH_WT.csv  HEK  $output_dir
python $script $input_dir/05_16_2025_HMC3_WITH_WT.csv HMC3 $output_dir
python $script $input_dir/05_16_2025_K562_WITH_WT.csv K562 $output_dir
python $script $input_dir/05_16_2025_MCF7_WITH_WT.csv MCF7 $output_dir
