#!/bin/bash

BASE_DIR=/ESL  # change to your base directory

script=$BASE_DIR/Figures_SK/compare_snv_dnv_additive/analyze_dnv_snv.py
input_csv=$BASE_DIR/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WITH_WT.csv
output_dir=$BASE_DIR/Figures_SK/compare_snv_dnv_additive/7_13_2025/7_13_2025

python $script \
  --csv $input_csv \
  --threshold 3 \
  --out $output_dir
