#!/bin/bash
set -e

DATA=/ESL/Figures_SK/General_preprocessing/output_04_26_2026/04_26_2026_1e-2_ALL_WITH_WT.csv
OUTDIR=/ESL/Figures_SK_Updated_Apr_2026/Figure_2/outputs/plots
SCRIPTS=/ESL/Figures_SK_Updated_Apr_2026/Figure_2

mkdir -p $OUTDIR

echo "=== Start-to-end PSI ==="
python3 $SCRIPTS/start_2_end_clipped.py \
  --input_csv $DATA \
  --output_dir $OUTDIR \
  --output_prefix start2end

echo "=== Position effects ==="
python3 $SCRIPTS/plot_dpsi_by_pos.py \
  --cell_psi_merged_file $DATA \
  --output_dir $OUTDIR \
  --output_prefix d
  psi_by_pos

echo "=== Double variant additivity ==="
python3 $SCRIPTS/analyze_dnv_snv.py \
  --csv $DATA \
  --out $OUTDIR/dnv_snv

echo "=== Done ==="
