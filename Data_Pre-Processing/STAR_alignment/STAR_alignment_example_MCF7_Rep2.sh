# Example pipeline for MCF7 (applies similarly to all cell lines).
# Raw and processed data files are available on GEO under accession GSE307247.

GitHub_path=/ESL/GitHub/ESL
supertable_file=/ESL/Data/Sequences/supertable.tsv
processors=62
picard_jar=/ESL/GitHub/ESL/picard/build/libs/picard-2.26.4-2-g843f2db-SNAPSHOT-all.jar
fgbio_jar=/ESL/GitHub/fgbio/target/scala-2.13/fgbio-1.5.0-b01fc04-SNAPSHOT.jar

DNA_barcode_fasta_file=/ESL/Analysis/clustering_barcodes_DNA/concat_2023_09_19_d1c_ms75/shorter3p/ESL_concat_2023_09_19_subsampleparams_d1c_ms75_shorter3p_iterate_mincov5_reference_with_close_matches.fasta
gtf_file=/ESL/Analysis/clustering_barcodes_DNA/concat_2023_09_19_d1c_ms75/shorter3p/assembled_GTF_supertable_ESL_concat_2023_09_19_subsampleparams_d1c_ms75_shorter3p_iterate_mincov5.gtf
filter_output_dir=/ESL/Analysis/Filter_RNA-seq_barcodes/filtered_2023_10_31_concat_2023_09_19_d1c_ms75
STAR_output_dir=/ESL/Analysis/STAR_alignment/separate_concat_2023_09_19_d1c_ms75
STAR_FASTA_file=$DNA_barcode_fasta_file
data_dir=/ESL/Data/NextSeq/2023_10_31/231031_NB501203_0669_AHV5TCAFX5/2023_10_31_bcl2fastq_output/merge_technical_reps

# MCF7 Rep2 example
MCF7_Rep2_name=MCF7_Rep2_20231031

# Filter RNA-seq reads by DNA barcode clusters

python3 $GitHub_path/filter_RNA_from_DNA_barcode_clusters_1MM_separate.py \
	--read1_fastq $data_dir/trimmed_reads/${MCF7_Rep2_name}_R1_trimmed.fastq.gz \
	--read2_fastq $data_dir/trimmed_reads/${MCF7_Rep2_name}_R3_trimmed.fastq.gz \
	--UMI_fastq  $data_dir/${MCF7_Rep2_name}_R2.fastq.gz \
	--DNA_barcode_fasta_file $DNA_barcode_fasta_file \
	--output_dir $filter_output_dir/${MCF7_Rep2_name}_filtered_separate \
	--output_prefix ${MCF7_Rep2_name}_filtered --p $processors


cd $filter_output_dir
tar zcf ${MCF7_Rep2_name}_filtered_separate.tar.gz ${MCF7_Rep2_name}_filtered_separate &

# Run STAR alignment and PSI quantification
separate_FASTQ_dir=$filter_output_dir/${MCF7_Rep2_name}_filtered_separate
separate_FASTQ_dir_prefix=$separate_FASTQ_dir/${MCF7_Rep2_name}_filtered
barcode_ref_file=$separate_FASTQ_dir/${MCF7_Rep2_name}_filtered_reference_barcode_counts.txt
output_path_prefix=${MCF7_Rep2_name}_separate

python3 $GitHub_path/run_STAR_separate_PSI_parallelized.py \
	--supertable_file $supertable_file \
	--GTF_all_refs_file $gtf_file \
	--output_dir $STAR_output_dir \
	--all_refs_FASTA_file $STAR_FASTA_file \
	--separate_FASTQ_dir_prefix $separate_FASTQ_dir_prefix \
	--barcode_ref_file $barcode_ref_file \
	--output_prefix $output_path_prefix \
	--picard_jar $picard_jar \
	--fgbio_jar $fgbio_jar \
	--p $processors
