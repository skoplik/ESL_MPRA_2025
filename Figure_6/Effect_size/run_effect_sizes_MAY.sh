#!/bin/bash
# Effect sizes on the MAY 2026 data.
#
# Replaces run_effect_sizes_clusters_indvidual_fimo_no_rev_strand_06_03_2025_clipped.sh,
# which read PSIs from Figures_SK/General_preprocessing/output_7_13_2025 (July 2025) and
# called the Figures_SK copy of the script. This uses the May reprocessed PSIs and the
# in-repo script (identical apart from four lines setting editable vector text for
# Illustrator). FIMO scan and motif clustering are unchanged - neither depends on PSI.
#
# Writes to a NEW output dir; the 2025 outputs are left untouched.

set -e
REPO=/ESL/ESL_MPRA
SK=/ESL/Figures_SK

script=$REPO/Figure_6/Effect_size/effect_size_scipt_delta_logit_on_clusters.py
output_dir=$REPO/Figure_6/outputs/effect_size_MAY_v2
fimo_out_file=$SK/effect_size_final/fimo_not_on_clusters_no_rc/merged_fimo_output_clusters.tsv
indv_cell_lines_file=$REPO/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz
cluster_file=$SK/Cluster_motifs/rsat_out_05_05_2025/motifs_05_05_2025_tables/clusters.tab

mkdir -p $output_dir
python3 $script \
--output_dir $output_dir \
--output_prefix out_MAY_v2 \
--fimo_out_file $fimo_out_file \
--indv_cell_lines_file $indv_cell_lines_file \
--cluster_file $cluster_file \
--numboots 0
