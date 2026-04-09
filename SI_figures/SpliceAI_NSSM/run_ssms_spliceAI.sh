#!/bin/bash

BASE_DIR=/ESL  # change to your base directory

script=$BASE_DIR/Figures_SK/Spliceai/build_spliceai_ssms.py
spliceai_psi=$BASE_DIR/Figures_SK/Spliceai/all/spliceai_psi_logit.tsv
spliceai_missing=$BASE_DIR/Figures_SK/Spliceai/all/spliceai_psi_logit_missing_vars.tsv
supertable_file=$BASE_DIR/Figures_SK/General_preprocessing/fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv
coords_table=$BASE_DIR/Figures_SK/General_preprocessing/output_7_13_2025/table_gene_exon_coords_psi_dlogits.tsv
output_dir=$BASE_DIR/Figures_SK/Spliceai/heatmaps_ssms

python $script \
  $spliceai_psi \
  $spliceai_missing \
  $supertable_file \
  $coords_table \
  $output_dir
