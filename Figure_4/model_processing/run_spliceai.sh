#!/bin/bash
set -e

DATA=/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz
OUTDIR=/ESL/ESL_MPRA/Figure_4/model_processing/outputs/spliceai
SCRIPTS=/ESL/ESL_MPRA/Figure_4/model_processing

mkdir -p $OUTDIR

echo "=== Running SpliceAI on all sequences (GPU required) ==="
python3 $SCRIPTS/run_spliceai_all.py \
  --input_csv $DATA \
  --output_dir $OUTDIR

echo ""
echo "=== Done ==="
echo "Raw scores saved to: $OUTDIR/spliceai_raw_preds_all.tsv"
echo "Next: run process_spliceai.py to compute delta logit from raw scores."
