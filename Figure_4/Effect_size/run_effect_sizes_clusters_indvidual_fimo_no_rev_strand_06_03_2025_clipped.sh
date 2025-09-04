output_dir=/ESL/Figures_SK/effect_size_final/out_08_10_2025
fimo_out_file=/ESL/Figures_SK/effect_size_final/fimo_not_on_clusters_no_rc/merged_fimo_output_clusters.tsv
indv_cell_lines_file=/ESL/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WITH_WT.csv
cluster_file=/ESL/Figures_SK/Cluster_motifs/rsat_out_05_05_2025/motifs_05_05_2025_tables/clusters.tab

python3 /ESL/Figures_SK/effect_size_final/effect_size_scipt_delta_logit_on_clusters.py \
--output_dir $output_dir \
--output_prefix out_08_10_2025 \
--fimo_out_file $fimo_out_file \
--indv_cell_lines_file $indv_cell_lines_file \
--cluster_file $cluster_file \
--numboots 0