#!/bin/bash
# Rebuild COMPASS PSI outputs with Type B (var/ref boundary mismatch) corrections
# applied at source. PSI for affected variants is recomputed at the reference
# junction before pooling, rather than patching a previously built CSV.
#
# Outputs go to /ESL/Figures_SK/General_preprocessing/output_04_17_2026/
# After the merge, run add_annotations_04_17_2026.py to add transcript_id /
# exon coords, then 06_build_with_wt_corrected.py for the WITH_WT file.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAR_PSI_DIR=/ESL/Analysis/STAR_alignment/separate_concat_2023_09_19_d1c_ms75_from_s3_2023_11_27/recount_SJs
PSI_2024=/ESL/Analysis/STAR_alignment/separate_concat_2023_09_19_d1c_ms75_from_s3_2023_11_27/recount_SJs/2024_07_26
WT_DIR=/ESL/Analysis/WT_Library/separate/recount_SJs

HEK_REP1=$STAR_PSI_DIR/HEK293_Rep2_separate_splicing_profiles/HEK293_Rep2_separate_recalc_PSIs_mincov10.txt
HEK_REP2=$STAR_PSI_DIR/HEK293_Rep3_separate_splicing_profiles/HEK293_Rep3_separate_recalc_PSIs_mincov10.txt
HEK_WT_REP1=$WT_DIR/HEK_WT_Rep1_20231101_separate_splicing_profiles/HEK_WT_Rep1_20231101_separate_recalc_PSIs_mincov10.txt
HEK_WT_REP2=$WT_DIR/HEK_WT_Rep2_20231101_separate_splicing_profiles/HEK_WT_Rep2_20231101_separate_recalc_PSIs_mincov10.txt
HELA_REP1=$STAR_PSI_DIR/HeLa_Rep1_separate_splicing_profiles/HeLa_Rep1_separate_recalc_PSIs_mincov10.txt
HELA_REP2=$STAR_PSI_DIR/HeLa_Rep3_20231031_separate_splicing_profiles/HeLa_Rep3_20231031_separate_recalc_PSIs_mincov10.txt
K562_REP1=$STAR_PSI_DIR/K562_Rep1_separate_splicing_profiles/K562_Rep1_separate_recalc_PSIs_mincov10.txt
K562_REP2=$STAR_PSI_DIR/K562_Rep3_20231031_separate_splicing_profiles/K562_Rep3_20231031_separate_recalc_PSIs_mincov10.txt
MCF7_REP1=$PSI_2024/MCF7_Rep1_separate_splicing_profiles/MCF7_Rep1_recalc_PSIs_mincov10.txt
MCF7_REP2=$PSI_2024/MCF7_Rep2_separate_splicing_profiles/MCF7_Rep2_recalc_PSIs_mincov10.txt
HMC3_REP1=$PSI_2024/HMC3_Rep1_separate_splicing_profiles/HMC3_Rep1_recalc_PSIs_mincov10.txt
HMC3_REP2=$PSI_2024/HMC3_Rep2_separate_splicing_profiles/HMC3_Rep2_recalc_PSIs_mincov10.txt

PKL_BASE=$STAR_PSI_DIR
PKL_2024=$PSI_2024

PKL_HELA_REP1=$PKL_BASE/HeLa_Rep1_separate_splicing_profiles/HeLa_Rep1_separate_all_splicing_counts.p
PKL_HELA_REP2=$PKL_BASE/HeLa_Rep3_20231031_separate_splicing_profiles/HeLa_Rep3_20231031_separate_all_splicing_counts.p
PKL_K562_REP1=$PKL_BASE/K562_Rep1_separate_splicing_profiles/K562_Rep1_separate_all_splicing_counts.p
PKL_K562_REP2=$PKL_BASE/K562_Rep3_20231031_separate_splicing_profiles/K562_Rep3_20231031_separate_all_splicing_counts.p
PKL_MCF7_REP1=$PKL_2024/MCF7_Rep1_separate_splicing_profiles/MCF7_Rep1_all_splicing_counts.p
PKL_MCF7_REP2=$PKL_2024/MCF7_Rep2_separate_splicing_profiles/MCF7_Rep2_all_splicing_counts.p
PKL_HMC3_REP1=$PKL_2024/HMC3_Rep1_separate_splicing_profiles/HMC3_Rep1_all_splicing_counts.p
PKL_HMC3_REP2=$PKL_2024/HMC3_Rep2_separate_splicing_profiles/HMC3_Rep2_all_splicing_counts.p
PKL_HEK_REP1=$PKL_BASE/HEK293_Rep2_separate_splicing_profiles/HEK293_Rep2_separate_all_splicing_counts.p
PKL_HEK_REP2=$PKL_BASE/HEK293_Rep3_separate_splicing_profiles/HEK293_Rep3_separate_all_splicing_counts.p
PKL_HEK_WT_REP1=$WT_DIR/HEK_WT_Rep1_20231101_separate_splicing_profiles/HEK_WT_Rep1_20231101_separate_all_splicing_counts.p
PKL_HEK_WT_REP2=$WT_DIR/HEK_WT_Rep2_20231101_separate_splicing_profiles/HEK_WT_Rep2_20231101_separate_all_splicing_counts.p

BOUNDARY_MISMATCHES=/ESL/Figures_SK/ambiguous_sjs/boundary_mismatches.csv
SUPERTABLE=/ESL/Figures_SK/General_preprocessing/fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv
OUTPUT_DIR=/ESL/Figures_SK/General_preprocessing/output_04_17_2026

echo "=== Step 1: Merge PSI with Type B junction corrections ==="
python3 "$SCRIPT_DIR/merge_psi_04_17_2026_type_b_corrected.py" \
    --output_dir "$OUTPUT_DIR" \
    --output_prefix "04_17_2026_1e-2" \
    --supertable_file "$SUPERTABLE" \
    --boundary_mismatches "$BOUNDARY_MISMATCHES" \
    --HeLa_Rep1_PSI "$HELA_REP1" \
    --HeLa_Rep2_PSI "$HELA_REP2" \
    --K562_Rep1_PSI "$K562_REP1" \
    --K562_Rep2_PSI "$K562_REP2" \
    --MCF7_Rep1_PSI "$MCF7_REP1" \
    --MCF7_Rep2_PSI "$MCF7_REP2" \
    --HMC3_Rep1_PSI "$HMC3_REP1" \
    --HMC3_Rep2_PSI "$HMC3_REP2" \
    --HEK293_Rep1_PSI "$HEK_REP1" \
    --HEK293_Rep2_PSI "$HEK_REP2" \
    --HEK293_WT_Rep1_PSI "$HEK_WT_REP1" \
    --HEK293_WT_Rep2_PSI "$HEK_WT_REP2" \
    --pkl_HeLa_rep1 "$PKL_HELA_REP1" \
    --pkl_HeLa_rep2 "$PKL_HELA_REP2" \
    --pkl_K562_rep1 "$PKL_K562_REP1" \
    --pkl_K562_rep2 "$PKL_K562_REP2" \
    --pkl_MCF7_rep1 "$PKL_MCF7_REP1" \
    --pkl_MCF7_rep2 "$PKL_MCF7_REP2" \
    --pkl_HMC3_rep1 "$PKL_HMC3_REP1" \
    --pkl_HMC3_rep2 "$PKL_HMC3_REP2" \
    --pkl_HEK_rep1  "$PKL_HEK_REP1" \
    --pkl_HEK_rep2  "$PKL_HEK_REP2" \
    --pkl_HEK_wt_rep1 "$PKL_HEK_WT_REP1" \
    --pkl_HEK_wt_rep2 "$PKL_HEK_WT_REP2" \
    --clip=1e-2

echo "=== Step 2: Add annotations (transcript_id, exon coords, variant_hg38) ==="
python3 "$SCRIPT_DIR/add_annotations_04_17_2026.py"

echo "=== Step 3: Build WITH_WT file ==="
python3 "$SCRIPT_DIR/06_build_with_wt_corrected.py"

echo "=== Step 4: Build Type B SI table (boundary mismatch corrected PSI) ==="
python3 "$SCRIPT_DIR/04_build_type_b_si.py"

echo "=== Step 5: Build alt transcript SI table (Type A ambiguous events) ==="
python3 "$SCRIPT_DIR/05_build_alt_transcript_si_table.py"

echo ""
echo "Done. Outputs in $OUTPUT_DIR"
echo "SI files:"
echo "  /ESL/Figures_SK/ambiguous_sjs/SI_boundary_mismatch.csv"
echo "  /ESL/Figures_SK/ambiguous_sjs/SI_alt_transcript_psi.csv"
