#!/bin/bash
set -e

DATA=/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz
OUTDIR=/ESL/ESL_MPRA/Figure_1/outputs/plots
SCRIPTS=/ESL/ESL_MPRA/Figure_1

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
