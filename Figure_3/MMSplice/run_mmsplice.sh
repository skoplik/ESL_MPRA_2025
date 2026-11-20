#!/bin/bash
set -e

# ============================================================
# MMSplice prediction pipeline
#
# WORKFLOW:
#   This script has two stages. Run them on different machines:
#
#   STAGE 1 (this machine — CPU only):
#     Generates the three input files needed by MMSplice:
#       - synthetic_reference.fa    (FASTA of synthetic exon constructs)
#       - synthetic_reference.gtf   (GTF annotation for the FASTA)
#       - synthetic_variants.vcf.gz (bgzipped + tabix-indexed VCF of all variants)
#     After Stage 1, copy the entire INPUT_DIR to your GPU machine.
#
#   STAGE 2 (GPU machine — run run_mmsplice_dataloader.py directly):
#     Runs kipoi MMSplice predictions. This is the slow step (~hours on CPU,
#     much faster on GPU). On the GPU machine, run:
#
#       python3 run_mmsplice_dataloader.py \
#         --vcf_path   outputs/input_files/synthetic_variants.vcf.gz \
#         --gtf_path   outputs/input_files/synthetic_reference.gtf \
#         --fasta_path outputs/input_files/synthetic_reference.fa \
#         --output_path outputs/mmsplice/mmsplice_predictions.csv
#
#     Copy the resulting mmsplice_predictions.csv back here when done.
# ============================================================

DATA=/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz
INPUT_DIR=/ESL/ESL_MPRA/Figure_3/MMSplice/outputs/input_files
SCRIPTS=/ESL/ESL_MPRA/Figure_3/MMSplice

mkdir -p $INPUT_DIR

echo "=== Step 1: Building MMSplice input files (FASTA, GTF, VCF) ==="
python3 $SCRIPTS/make_mmsplice_inputs.py \
  --main_data $DATA \
  --output_dir $INPUT_DIR

echo ""
echo "=== Step 2: Sort, compress and index VCF ==="
grep "^#" $INPUT_DIR/synthetic_variants.vcf > $INPUT_DIR/synthetic_variants_sorted.vcf
grep -v "^#" $INPUT_DIR/synthetic_variants.vcf | sort -k1,1n -k2,2n >> $INPUT_DIR/synthetic_variants_sorted.vcf
mv $INPUT_DIR/synthetic_variants_sorted.vcf $INPUT_DIR/synthetic_variants.vcf
bgzip -f $INPUT_DIR/synthetic_variants.vcf
tabix -p vcf $INPUT_DIR/synthetic_variants.vcf.gz

echo ""
echo "=== Stage 1 complete ==="
echo "Input files written to: $INPUT_DIR"
echo "  synthetic_reference.fa"
echo "  synthetic_reference.gtf"
echo "  synthetic_variants.vcf.gz  (+ .tbi index)"
echo ""
echo "Copy these files to a GPU machine and run Stage 2 (see header comment)."
