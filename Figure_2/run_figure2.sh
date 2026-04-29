#!/bin/bash
set -e

DATA=/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/ALL_WITH_WT.csv
OUTDIR=/ESL/ESL_MPRA/Figure_2/outputs/plots
SCRIPTS=/ESL/ESL_MPRA/Figure_2/other_analyses

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
