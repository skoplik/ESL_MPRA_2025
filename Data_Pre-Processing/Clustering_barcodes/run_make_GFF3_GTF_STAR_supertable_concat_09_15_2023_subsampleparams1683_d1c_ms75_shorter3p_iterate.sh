#!/bin/bash
# FASTAs derived from DNA clustering are available on GEO (GSE307247) as supplementary files.
# Reads cluster output produced by run_get_DNA_barcode_sequence_clusters_*.sh (Step 1a).
#
# Dependencies (Python packages): biopython
# Install Python packages: pip install biopython

BASE_DIR=/ESL  # change to your base directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

which python3


# determining DNA barcode clusters
output_dir=$BASE_DIR/Analysis/clustering_barcodes_DNA/concat_2023_09_19_d1c_ms75/shorter3p
output_prefix=ESL_concat_2023_09_19_subsampleparams_d1c_ms75_shorter3p_iterate

# Taking clusters and make FASTA reference
sequence_clusters=$output_dir/${output_prefix}_trimmedShared_barcode_clusters_consensus_seq.txt
supertable=$BASE_DIR/Data/Sequences/supertable.tsv

# Threshold from mini library
min_coverage=5

python3 $SCRIPT_DIR/make_GFF3_STAR_supertable.py --supertable $supertable \
--STAR_fasta_file $output_dir/${output_prefix}_mincov${min_coverage}_reference.fasta \
--output_dir $output_dir --output_prefix assembled_GFF3_supertable_${output_prefix}_mincov${min_coverage}

python3 $SCRIPT_DIR/make_GTF_STAR_supertable.py --supertable $supertable \
--STAR_fasta_file $output_dir/${output_prefix}_mincov${min_coverage}_reference.fasta \
--output_dir $output_dir --output_prefix assembled_GTF_supertable_${output_prefix}_mincov${min_coverage}
