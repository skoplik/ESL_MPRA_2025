#!/bin/bash
#FASTAs derived from DNA clustering are availble on GEO (GSE307247) as suppplementary files 

which python3


# determining DNA barcode clusters
GitHub_path=/ESL/GitHub/ESL
output_dir=/ESL/Analysis/clustering_barcodes_DNA/concat_2023_09_19_d1c_ms75/shorter3p
output_prefix=ESL_concat_2023_09_19_subsampleparams_d1c_ms75_shorter3p_iterate

# Taking clusters and make FASTA reference
sequence_clusters=/ESL/Analysis/clustering_barcodes_DNA/concat_2023_09_19_d1c_ms75/shorter3p/ESL_concat_2023_09_19_subsampleparams_d1c_ms75_shorter3p_iterate_trimmedShared_barcode_clusters_consensus_seq.txt
supertable=/ESL/Data/Sequences/supertable.tsv

# Threshold from mini library
min_coverage=5

python3 $GitHub_path/make_GFF3_STAR_supertable.py --supertable $supertable \
--STAR_fasta_file $output_dir/${output_prefix}_mincov${min_coverage}_reference.fasta \
--output_dir $output_dir --output_prefix assembled_GFF3_supertable_${output_prefix}_mincov${min_coverage}

python3 $GitHub_path/make_GTF_STAR_supertable.py --supertable $supertable \
--STAR_fasta_file $output_dir/${output_prefix}_mincov${min_coverage}_reference.fasta \
--output_dir $output_dir --output_prefix assembled_GTF_supertable_${output_prefix}_mincov${min_coverage}
