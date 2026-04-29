#!/bin/bash
set -e

DATA=/ESL/Figures_SK/General_preprocessing/output_04_26_2026/04_26_2026_1e-2_ALL_WITH_WT.csv
INPUT_DIR=/ESL/Figures_SK_Updated_Apr_2026/Figure_3/model_processing/outputs/mmsplice/input_files
SCRIPTS=/ESL/Figures_SK_Updated_Apr_2026/Figure_3/model_processing

mkdir -p $INPUT_DIR

echo "=== Step 1: Building MMSplice input files (FASTA, GTF, VCF) ==="
python3 $SCRIPTS/make_mmsplice_inputs.py \
  --main_data $DATA \
  --output_dir $INPUT_DIR

echo ""
echo "=== Step 2: Sort, compress and index VCF ==="
# Sort by chrom (numeric) then position, keeping header
grep "^#" $INPUT_DIR/synthetic_variants.vcf > $INPUT_DIR/synthetic_variants_sorted.vcf
grep -v "^#" $INPUT_DIR/synthetic_variants.vcf | sort -k1,1n -k2,2n >> $INPUT_DIR/synthetic_variants_sorted.vcf
mv $INPUT_DIR/synthetic_variants_sorted.vcf $INPUT_DIR/synthetic_variants.vcf
bgzip -f $INPUT_DIR/synthetic_variants.vcf
tabix -p vcf $INPUT_DIR/synthetic_variants.vcf.gz

echo ""
echo "=== Step 3: Running MMSplice kipoi dataloader ==="
python3 $SCRIPTS/run_mmsplice_dataloader.py \
  --vcf_path   $INPUT_DIR/synthetic_variants.vcf.gz \
  --gtf_path   $INPUT_DIR/synthetic_reference.gtf \
  --fasta_path $INPUT_DIR/synthetic_reference.fa \
  --output_path $SCRIPTS/outputs/mmsplice/mmsplice_predictions.csv

echo ""
echo "=== Done ==="
echo "Predictions saved to: $SCRIPTS/outputs/mmsplice/mmsplice_predictions.csv"
