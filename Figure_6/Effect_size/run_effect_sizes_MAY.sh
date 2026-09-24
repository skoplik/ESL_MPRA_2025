#!/bin/bash
# Effect sizes on the MAY 2026 data.
#
# Replaces run_effect_sizes_clusters_indvidual_fimo_no_rev_strand_06_03_2025_clipped.sh,
# which read PSIs from Figures_SK/General_preprocessing/output_7_13_2025 (July 2025) and
# called the Figures_SK copy of the script. This uses the May reprocessed PSIs and the
# in-repo script (identical apart from four lines setting editable vector text for
# Illustrator). FIMO scan and motif clustering are unchanged - neither depends on PSI.
#
# Writes to a NEW output dir; earlier outputs are left untouched.
#
# 2026-09-24: switched from 1e-2_ALL_WITH_WT.csv.gz to 1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz.
# Both are the same May 24 drop and partition intron1/exon/intron2 identically across all
# 87,001 shared References, so junctions are unchanged - but NO_DELTAS carries 2,130
# event_ids against ALL_WITH_WT's 1,162, so many more exon families reach the effect-size
# calculation. Everything downstream is computed from this file plus the FIMO hits.

set -e
REPO=/ESL/ESL_MPRA
SK=/ESL/Figures_SK

script=$REPO/Figure_6/Effect_size/01_compute_effect_sizes.py
output_dir=$REPO/Figure_6/outputs/effect_size_MAY_full
fimo_out_file=$REPO/Figure_6/inputs/fimo/merged_fimo_output_clusters.tsv
indv_cell_lines_file=$REPO/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz
cluster_file=$REPO/Figure_6/inputs/motif_clusters/clusters.tab

mkdir -p $output_dir
python3 $script \
--output_dir $output_dir \
--output_prefix out_MAY_full \
--fimo_out_file $fimo_out_file \
--indv_cell_lines_file $indv_cell_lines_file \
--cluster_file $cluster_file \
--numboots 0
