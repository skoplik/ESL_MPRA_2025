#!/bin/bash

GitHub_path=/ESL/GitHub/ESL

supertable_file=/ESL/Data/Sequences/supertable.tsv
gtf_file=/ESL/Analysis/clustering_barcodes_DNA/concat_2023_09_19_d1c_ms75/shorter3p/assembled_GTF_supertable_ESL_concat_2023_09_19_subsampleparams_d1c_ms75_shorter3p_iterate_mincov5.gtf

STAR_pass1_dir=/ESL/Analysis/STAR_alignment/separate_concat_2023_09_19_d1c_ms75/MCF7_Rep2_20231031_separate_STAR_pass1
mincov=10

MCF7_Rep2_20231031_prefix=MCF7_Rep2_20231031_separate
MCF7_Rep2_20231031_output_dir=/ESL/Analysis/STAR_alignment/separate_concat_2023_09_19_d1c_ms75/${MCF7_Rep2_20231031_prefix}_splicing_profiles

python3 $GitHub_path/get_STAR_separate_splicing_profiles.py \
--GTF_all_refs_file $gtf_file --STAR_pass1_dir $STAR_pass1_dir --supertable_file $supertable_file \
--mincov $mincov --output_dir $MCF7_Rep2_20231031_output_dir --output_prefix $MCF7_Rep2_20231031_prefix
