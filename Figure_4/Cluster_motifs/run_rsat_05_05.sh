export LD_LIBRARY_PATH=/ESL/src_download/libpng/lib:$LD_LIBRARY_PATH
/ESL/src_download/R-4.2.0/bin/Rscript -e 'library(ComplexHeatmap)'

Rscript /ESL/src_download/matrix-clustering_stand-alone/matrix-clustering.R \
-i /ESL/Figures_SK/Cluster_motifs/rsat_out_05_05_2025/input_motifs.txt \
-o /ESL/Figures_SK/Cluster_motifs/rsat_out_05_05_2025/motifs_05_05_2025 \
  --comparison_metric Ncor \
  --linkage_method complete \
  --cor_th 0.50 \
  --Ncor_th 0.40 \
  --w_th 3 \
  --no_rc TRUE 
  
  