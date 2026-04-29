#!/bin/bash
set -e

DATA=/ESL/Figures_SK/General_preprocessing/output_04_26_2026/04_26_2026_1e-2_ALL_WITH_WT.csv
OUTDIR=/ESL/Figures_SK_Updated_Apr_2026/Figure_3/model_processing/outputs/spliceai
SCRIPTS=/ESL/Figures_SK_Updated_Apr_2026/Figure_3/model_processing

mkdir -p $OUTDIR

echo "=== Running SpliceAI on all sequences (GPU required) ==="
python3 $SCRIPTS/run_spliceai_all.py \
  --input_csv $DATA \
  --output_dir $OUTDIR

echo ""
echo "=== Done ==="
echo "Raw scores saved to: $OUTDIR/spliceai_raw_preds_all.tsv"
echo "Next: run process_spliceai.py to compute delta logit from raw scores."
