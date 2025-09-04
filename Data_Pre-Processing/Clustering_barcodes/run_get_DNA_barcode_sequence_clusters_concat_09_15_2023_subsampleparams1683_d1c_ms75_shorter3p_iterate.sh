#!/bin/bash
#update code to specify your file paths. DNA-Seq is availavle on GEO (GSE307247) as GSM9219929
#FASTAs derived from DNA clustering are availble on GEO (GSE307247) as suppplementary files 

which python3

# trim plasmid DNA reads
processes=3
shared_region=TGTGAAACAAAATGCTTTTTAACATCCA
concat_data_dir=/ESL/Data/MiSeq/concat_2023_09_19

mkdir $concat_data_dir

ESL_DNA_R1_fastq=$concat_data_dir/concat_2023_09_19_R1.fastq.gz
ESL_DNA_R2_fastq=$concat_data_dir/concat_2023_09_19_R2.fastq.gz
trimmed_data_dir=$concat_data_dir/cutadapt_output
mkdir $trimmed_data_dir


cutadapt -j $processes -m 1 \
-g AMY_26=${shared_region} \
-g AMY_36=NNNNNNNNNNNNNNNNNVMVCYG${shared_region} \
-g AMY_37=NNNNNNNNNNNNNNNNNVMV${shared_region} \
-g AMY_38=NNNNNNNNNNNNNNNNN${shared_region} \
-o $trimmed_data_dir/concat_2023_09_19_R1_trimmedShared.fastq.gz -p $trimmed_data_dir/concat_2023_09_19_R2_trimmedShared.fastq.gz \
 $ESL_DNA_R1_fastq $ESL_DNA_R2_fastq


# determining DNA barcode clusters
GitHub_path=/ESL/GitHub/ESL
output_dir=/ESL/Analysis/clustering_barcodes_DNA/concat_2023_09_19_d1c_ms75/shorter3p
output_prefix=ESL_concat_2023_09_19_subsampleparams_d1c_ms75_shorter3p_iterate
processes=3


python3 $GitHub_path/get_DNA_barcode_sequence_clusters_relaxed_parallelized_subsample_1683_d1c_ms75_shorter3p_iterate.py \
--read1_fastq $trimmed_data_dir/concat_2023_09_19_R1_trimmedShared.fastq.gz \
--read2_fastq $trimmed_data_dir/concat_2023_09_19_R2_trimmedShared.fastq.gz --output_dir $output_dir --p $processes \
--output_prefix ${output_prefix}_trimmedShared


# Taking clusters and make FASTA reference
sequence_clusters=$output_dir/${output_prefix}_trimmedShared_barcode_clusters_consensus_seq.txt
supertable=/home/ec2-user/environment/ESL/Data/Sequences/supertable.tsv


# Previous min coverage.
# This is still valid cutoff based on higher depth subsampling of mini library (7 (true) vs. 2 (false)), and allows any plasmid drop between preps.
min_coverage=5

time python3 $GitHub_path/assemble_FASTA_from_clusters_relaxed_with_close_match_bowtie.py --supertable $supertable \
--sequence_clusters $sequence_clusters --output_dir $output_dir --min_coverage $min_coverage \
--output_prefix ${output_prefix}_mincov${min_coverage} --shared_mismatch_allowed "-1" --p $processes
