# Ambiguous Splice Junction Correction

Fixes PSI values for events where multiple Gencode transcripts annotate the same
exon with different intron1/exon lengths (different splice junctions).

## Problem

Some MPRA exons map to multiple Gencode transcripts with distinct 5' intronic
flanks, producing different splice junction coordinates. The original pipeline
assigned PSI at whichever junction the construct was annotated to, which could
be wrong (e.g. KCTD10 exon 4: annotated junction had 0 reads, MANE junction
had PSI ≈ 0.93).

## Pipeline (run in order)

All output paths are under `/ESL/Figures_SK/General_preprocessing/`.

### 00_prepare_working_supertable.py
- Input: `fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv.gz` (source, never modified)
- Adds `mane_status` column from Gencode v48 GTF
- Output: `output_04_24_2026/st_working_04_24_2026.csv`

### 07_apply_best_junction_ambiguous_events.py
- Detects ambiguous events by intron1_len diversity in the supertable (33 events)
- For each ambiguous event, tests the primary reference construct's read counts
  at every transcript's junction across all replicates
- Picks the junction with ≥2 valid reps, preferring MANE Select > MANE Plus
  Clinical > lowest Reference
- Recomputes all PSI/logit/delta columns for every row in ambiguous events
  at the chosen junction
- Supertable output adds `chosen_junction_transcript_id`,
  `chosen_junction_mane_status`, `chosen_junction_exon_start_hg38`,
  `chosen_junction_exon_end_hg38` columns (per-row annotations preserved)
- Inputs:
  - `output_04_24_2026/st_working_04_24_2026.csv`
  - `output_03_16_2026/03_16_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv`
- Outputs:
  - `output_04_24_2026/st_04_24_2026.csv`
  - `output_04_24_2026/04_24_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv`
  - `output_04_24_2026/04_24_2026_1e-2_ALL_WITH_WT.csv`

### 08_update_sequences_for_chosen_junction.py
- Updates `transcript_id`, `exon_start_hg38`, `exon_end_hg38` in the main table
  to reflect the chosen junction for rows where the annotated intron1 length
  doesn't match the chosen junction's intron1 length
- Case 1: if the same SNP exists as a row with the chosen intron1_len, copies
  full sequences + metadata from that row
- Case 2: if no matching row exists (variant designed only for the non-chosen
  transcript), updates metadata only — sequences (intron1/exon/intron2/full_seq)
  are left as-is since they reflect the actual synthesized construct
- Does NOT modify the supertable
- Inputs:
  - `output_04_24_2026/04_24_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv`
  - `output_04_24_2026/st_04_24_2026.csv`
- Outputs:
  - `output_04_24_2026/04_24_2026_1e-2_ALL_WTS_VARS_NO_DELTAS_seqfix.csv`
  - `output_04_24_2026/04_24_2026_1e-2_ALL_WITH_WT_seqfix.csv`

### 05_build_alt_transcript_si_table.py
- For each ambiguous event, computes PSI at ALL non-primary transcript junctions
  for every row (WT and variants)
- Produces one row per (Reference, alt_transcript_id)
- Computes dPSI and delta logit relative to the WT at each alt junction
- Inputs:
  - `output_04_24_2026/st_working_04_24_2026.csv`
  - `output_04_24_2026/04_24_2026_1e-2_ALL_WTS_VARS_NO_DELTAS_seqfix.csv`
- Output: `/ESL/Figures_SK/ambiguous_sjs/SI_alt_transcript_psi.csv`
  (3,156 rows, 31 events, 32 unique alt transcripts)

## Final outputs

All files under `/ESL/Figures_SK/General_preprocessing/output_04_24_2026/` except the SI table.

| File | Description |
|------|-------------|
| `04_24_2026_1e-2_ALL_WTS_VARS_NO_DELTAS_seqfix.csv` | Main table (87,546 rows) |
| `04_24_2026_1e-2_ALL_WITH_WT_seqfix.csv` | With-WT subset (84,998 rows) |
| `st_04_24_2026.csv` | Supertable with chosen_junction_* cols |
| `/ESL/Figures_SK/ambiguous_sjs/SI_alt_transcript_psi.csv` | SI alt transcript table |

## Notes

- 6 of 33 ambiguous events had no reads at any junction and were skipped
- 11 events had a non-primary transcript chosen (all MANE Select preferred)
- Sequences in the main table reflect actual synthesized constructs; transcript_id
  and exon coordinates reflect the junction used for PSI computation
- Source supertable gz is never modified; all outputs go to new files
