#!/bin/bash
set -e

SUPERTABLE=/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/st_corrected.csv
DATA=/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv
OUTDIR=/ESL/ESL_MPRA/Figure_3/SpliceAI/output
SCRIPTS=/ESL/ESL_MPRA/Figure_3/SpliceAI

mkdir -p $OUTDIR

echo "=== Running SpliceAI on supertable sequences (GPU required) ==="
python3 $SCRIPTS/run_spliceai_supertable.py \
  --input_csv $SUPERTABLE \
  --output_dir $OUTDIR

echo ""
echo "=== Processing SpliceAI predictions ==="
python3 $SCRIPTS/process_spliceai_supertable.py \
  --raw_preds $OUTDIR/spliceai_raw_preds_supertable.tsv \
  --supertable $SUPERTABLE \
  --data $DATA \
  --output_dir $OUTDIR

echo ""
echo "=== Done ==="
echo "Full supertable predictions: $OUTDIR/spliceai_supertable_predictions.csv"
echo "COMPASS data predictions:    $OUTDIR/spliceai_data_predictions.csv"
echo "Correlation summary:         $OUTDIR/spliceai_vs_exp_correlation_summary.json"
