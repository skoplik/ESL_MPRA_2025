#!/bin/bash
set -e
DATA=/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz
INPUT_DIR=/ESL/ESL_MPRA/Figure_3/MMSplice/outputs/input_files_MAY_20260911
SCRIPTS=/ESL/ESL_MPRA/Figure_3/MMSplice
echo "=== Step 1: Building MMSplice inputs from MAY data ==="
/usr/bin/python3.7 $SCRIPTS/make_mmsplice_inputs.py --main_data $DATA --output_dir $INPUT_DIR
echo "=== Step 2: sort + bgzip + tabix VCF ==="
grep "^#" $INPUT_DIR/synthetic_variants.vcf > $INPUT_DIR/synthetic_variants_sorted.vcf
grep -v "^#" $INPUT_DIR/synthetic_variants.vcf | sort -k1,1n -k2,2n >> $INPUT_DIR/synthetic_variants_sorted.vcf
mv $INPUT_DIR/synthetic_variants_sorted.vcf $INPUT_DIR/synthetic_variants.vcf
bgzip -f $INPUT_DIR/synthetic_variants.vcf
tabix -p vcf $INPUT_DIR/synthetic_variants.vcf.gz
echo "=== DONE MMSplice MAY inputs ==="
ls -la $INPUT_DIR
