#!/bin/bash
set -e

DATA=/ESL/Figures_SK/General_preprocessing/output_04_26_2026/04_26_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv
OUTDIR=/ESL/Figures_SK_Updated_Apr_2026/Figure_1/outputs/plots
SCRIPTS=/ESL/Figures_SK_Updated_Apr_2026/Figure_1

mkdir -p $OUTDIR

echo "=== PSI Distribution ==="
python3 $SCRIPTS/plot_psi_dist.py \
  --input_csv $DATA \
  --output_dir $OUTDIR

echo "=== Replicate Correlations ==="
python3 $SCRIPTS/plot_corrs.py \
  --input_csv $DATA \
  --output_dir $OUTDIR

echo "=== Done ==="
