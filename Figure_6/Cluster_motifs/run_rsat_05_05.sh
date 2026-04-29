#!/bin/bash

BASE_DIR=/ESL  # change to your base directory

export LD_LIBRARY_PATH=$BASE_DIR/src_download/libpng/lib:$LD_LIBRARY_PATH
$BASE_DIR/src_download/R-4.2.0/bin/Rscript -e 'library(ComplexHeatmap)'

Rscript $BASE_DIR/src_download/matrix-clustering_stand-alone/matrix-clustering.R \
-i $BASE_DIR/Figures_SK/Cluster_motifs/rsat_out_05_05_2025/input_motifs.txt \
-o $BASE_DIR/Figures_SK/Cluster_motifs/rsat_out_05_05_2025/motifs_05_05_2025 \
  --comparison_metric Ncor \
  --linkage_method complete \
  --cor_th 0.50 \
  --Ncor_th 0.40 \
  --w_th 3 \
  --no_rc TRUE
