#!/bin/bash
set -e

DATA=/ESL/Figures_SK/General_preprocessing/output_04_26_2026/04_26_2026_1e-2_ALL_WITH_WT.csv
OUTDIR=/ESL/ESL_MPRA/Figure_3/model_processing/outputs/hal
SCRIPTS=/ESL/ESL_MPRA/Figure_3/model_processing

mkdir -p $OUTDIR

echo "=== Building HAL input files ==="
python3 $SCRIPTS/make_input_hal.py \
  --input_csv $DATA \
  --output_zip $OUTDIR/hal_input_variants_only_avgwtpsi_exon6nt.tsv.zip \
  --output_plot_csv $OUTDIR/hal_plotting_input.csv

echo ""
echo "=== Done ==="
echo "HAL input zip: $OUTDIR/hal_input_variants_only_avgwtpsi_exon6nt.tsv.zip"
echo ""
echo "Next: submit zip to HAL at http://splicing.cs.washington.edu/SE and save predictions to:"
echo "  $OUTDIR/hal_predictions.tsv"
