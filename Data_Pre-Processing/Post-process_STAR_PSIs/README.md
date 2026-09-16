# Post-processing — Merge PSI Across Cell Lines

Pipeline that converts per-replicate PSI text files into the COMPASS master
CSV. Stage 1 enforces a canonical splice junction per ambiguous event (MANE
Select preferred, strand-aware); Stage 2 merges PSI across replicates and
cell lines and adds a `wt_reference` column; Stage 3 produces an SI table of
PSI at alternative-isoform splice junctions; Stage 6 concatenates the main
and SI tables into convenience files for downstream analysis. Stages 4-5 are
diagnostic post-pipeline analyses.

## Scripts

| Script | Role |
|---|---|
| `01_fix_sj_supertable.py` | Per-event MANE-canonical SJ rewrite. For each ambiguous event (multiple WT splits in `event_id_161`), pick MANE Select if it fits + has reads at WT pkl, else fall back to alt with most WT reads. For each unique `full_seq` in the event, if no row has the canonical SJ, rewrite the lowest-Reference row of that full_seq to the canonical SJ. Writes corrected supertable + per-rep PSI text files. Alt-junction discovery (writes `st_alt_junctions.csv`) runs **after** the rewrite so displaced annotations get rescued. |
| `02_run_merge_psi.sh` → `02_merge_psi.py` | Clip PSI, compute logit, pool reps, compute SDs, merge across cell lines, **add `wt_reference` column** (= Reference of the WT row at the same event + junction). **Drops barcode-cluster duplicates** (`transcript_class='duplicate'`) from the merged all-cell files. |
| `03_alt_transcript_si_table.py` | For each ambiguous event, compute PSI at alt-isoform junctions for every WT and variant row. Adds `low_coverage_rescue` flag (rows that fail the WT+variant pooled-PSI pair filter), re-keys `Reference` to synthetic 244,001+ with `original_reference` pointing back, adds `wt_reference`. |
| `04_list_alt_events.py` | Per-event alt-transcript summary (supertable + gencode-only). Diagnostic. |
| `05_find_skipped_events.py` | Identify ambiguous events that didn't get a MANE row in the corrected supertable. Diagnostic. |
| `06_merge_si_into_data.py` | Concatenate main NO_DELTAS / ALL_WITH_WT with the SI table into single tables. Drops `low_coverage_rescue=True` SI rows. Adds `source_table` column (`main` or `si`). |

Each script has a `BASE_DIR=/ESL` (shell) or hardcoded path constants (Python)
near the top — change these to your local base directory before running.

## Inputs

```
$BASE_DIR/Analysis/STAR_alignment/.../recount_SJs/<CELL>_<rep>_separate_splicing_profiles/
    *_recalc_PSIs_mincov10.txt    # per-rep PSI text file
    *_all_splicing_counts.p       # per-rep junction-count pickle
$BASE_DIR/Analysis/WT_Library/separate/recount_SJs/HEK_WT_Rep{1,2}_*_separate_splicing_profiles/...
$BASE_DIR/ESL_MPRA/Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25_strandfix.csv
$BASE_DIR/Figures_SK/General_preprocessing/fix_supertable_2/gencode.v48.annotation.gtf
```

The strandfix supertable corrects an exon-coordinate mirroring bug for - strand
events from the original `st_final_with_snp_and_coords_05_30_25.csv.gz`. Source
files are read-only.

## Run

```bash
python 01_fix_sj_supertable.py     # ~13 min (pkl loading dominates)
bash   02_run_merge_psi.sh         # ~3 min
python 03_alt_transcript_si_table.py
python 06_merge_si_into_data.py    # generates the *_with_SI.csv merged files
python 04_list_alt_events.py       # post-pipeline summary (diagnostic)
python 05_find_skipped_events.py   # post-pipeline diagnostic
```

## Outputs (in `output/`)

| File | Description |
|---|---|
| `st_corrected.csv` | Full corrected supertable (244,000 rows). New columns: `event_id_161` (chr:start-end:strand window), `event_id` (chr:exon_start-exon_end:strand for the canonical SJ — used for WT/variant pairing in Stage 2), `transcript_class` (`MANE` / `alt` / `duplicate`), `n_rows_per_full_seq`, `alt_transcripts_in_supertable`, `alt_transcripts_gencode_only`. |
| `1e-2_ALL_WTS_VARS_NO_DELTAS.csv` | Primary master CSV (all cell lines, every WT and variant; barcode duplicates dropped). Includes `wt_reference` column. |
| `1e-2_ALL_WITH_WT.csv` | Subset filtered to event_ids that contain both a WT row and ≥1 variant row. ~87k rows. Includes `wt_reference`. |
| `1e-2_<CELL>_{WITH_WT,VARIANTS_ONLY,WTS_VARS_NO_DELTAS}.csv` | Per-cell-line trios. Includes `wt_reference`. |
| `1e-2_ALL_WTS_VARS_NO_DELTAS_with_SI.csv` | NO_DELTAS + SI rows concatenated. `source_table` ∈ {`main`, `si`}; SI rows have synthetic Reference 244,001+ with `original_reference` pointing to the main-file MANE-row Reference. Excludes `low_coverage_rescue` SI rows. |
| `1e-2_ALL_WITH_WT_with_SI.csv` | ALL_WITH_WT + SI rows, same convention. |
| `st_alt_junctions.csv` | One row per (event_id_161, gencode-discovered alt_transcript_id). `canonical_reference` anchors to the WT MANE row. |
| `ambiguous_sjs/SI_alt_transcript_psi.csv` | Per-row PSI at each alternative-isoform junction for WT + variants in ambiguous events. Synthetic Reference (244,001+), `original_reference` → MANE-row Reference, `si_row_id` as stable cross-run key, per-cell + per-rep alt-junction counts, `low_coverage_rescue` flag, `wt_reference`. |
| `alt_events.csv` | Per-event summary of all alt transcripts (supertable + gencode-only). |
| `ambig_events_status.csv` | Per-event diagnostic: which ambiguous events have a MANE row, which don't. |
| `recount_PSIs/` | Per-replicate PSI text files with PSI recomputed at the chosen canonical SJ. |

## Stage 1 detail — canonical SJ per event

An **ambiguous event** = `event_id_161` with >1 unique `(intron1_len, exon_len)`
among WT rows.

For each ambiguous event:

1. **Pick canonical SJ.** Try MANE Select via `find_mane_split_for_gene()` — fit
   the gene's MANE exon into the construct window (`junction_in_window` is
   strand-aware: for - strand events `intron1_len = ev_end - exon_end_hg38`).
   Accept MANE if it has reads in ≥2 reps at the WT canonical pkl key. Else
   fall back to alt with most WT reads via `gather_alt_candidates()`. Else skip.
2. **Apply per full_seq.** For each unique `full_seq` in the event, check if
   any row already has the canonical SJ. If not, rewrite the lowest-Reference
   row of that full_seq: re-slice `intron1`/`exon`/`intron2` from `full_seq`
   and overwrite `transcript_id` / `mane_status` / `exon_start_hg38` /
   `exon_end_hg38` / `intron1_len` / `exon_len`.

Pre-existing rows at non-canonical SJs are preserved as `transcript_class='alt'`
(so KCTD10's 3 isoforms stay as 3 rows per variant). True barcode-cluster
duplicates (same `full_seq` + same SJ) get `transcript_class='duplicate'` and
are dropped from the merged file by Stage 2.

## Stage 2 detail — merge

- PSI clipping to `[0.01, 0.99]` (configurable via `--clip`)
- `logit_PSI = log(clipped / (1 - clipped))`
- Per-rep `_included` / `_excluded` / `_psi_raw` / `_psi_clipped` / `_logit`
- Pooled `_pooled_*` weighted by Coverage
- WT pool grouped by `event_id` (chr:exon_start-exon_end:strand); variant rows
  paired against the WT sharing their SJ → `dpsi_pooled` and
  `delta_logit_pooled` per event
- HEK rep handling: variant reps 1+2 (`HEK293_Rep{1,2}_PSI`), WT reps 3+4
  (`HEK293_WT_Rep{1,2}_PSI`); WT mean per `event_id` maps onto variant rows
- **Drops `transcript_class == 'duplicate'` rows** (identical PSI to the
  canonical row of the same `(full_seq, SJ)` group)

## Stage 3 detail — alt-transcript SI table

For each ambiguous event with multiple candidate junctions (supertable
transcripts + gencode-discovered alts), compute PSI at the alt junction for
every WT and variant row. Output schema mirrors the main CSV with
`_alt`-suffixed PSI columns and an `alt_transcript_id` key. Filtered to alts
with ≥1 WT + ≥1 variant passing pooled-PSI coverage in any cell line.

**SI is complementary to `ALL_WITH_WT.csv`, not a duplicate.** For variants,
the SI script skips any alt junction that already has its own Reference row
in NO_DELTAS for the same `full_seq` (`03_alt_transcript_si_table.py:315`).
So when looking up all isoforms of one variant:

- Supertable-design alts (the ambiguous-event sibling rows) live in
  `ALL_WITH_WT.csv` under their own References.
- Alt junctions *not* represented as their own Reference row appear only in
  `SI_alt_transcript_psi.csv`.

Example — KCTD10 exon 4 variant 62:T>G (3 transcript annotations):
the (54, 87) MANE row is `ALL_WITH_WT.csv` Ref 99223; the (51, 90) alt-51
row is `ALL_WITH_WT.csv` Ref 99222; the (84, 57) alt-84 has no supertable
variant row, so its PSI lives in the SI file under the MANE Ref 99223. WT
baselines for SI dPSI come from the SI file's own WT rows
(stage 03 always emits WTs to keep that baseline self-contained).

Reference indices are **1-indexed across all data files** (NO_DELTAS,
ALL_WITH_WT, per-rep PSI, SI). The supertable (`st_corrected.csv`) is
**0-indexed**: `data_file_Reference = supertable_Reference + 1`.

## Construct geometry constants

```
SHARED_5P = 286            # length of the shared 5' adapter before variable region
MINCOV    = 10             # min reads to assign PSI
E_JXN     = (26, 871)      # construct-coordinate exclusion junction

# inclusion junctions for transcript with intron1_len i1 and exon_len ex:
i1_jxn = (26, SHARED_5P + i1)
i2_jxn = (SHARED_5P + i1 + ex + 1, 871)
```

## Dependencies

`numpy`, `pandas`, `scipy`
