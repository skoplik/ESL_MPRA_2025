# Post-processing — Merge PSI Across Cell Lines

Five-stage pipeline that converts per-replicate PSI text files into the COMPASS
master CSV. Stage 1 enforces a canonical splice junction per ambiguous event
(MANE Select preferred, strand-aware); Stage 2 merges PSI across replicates and
cell lines; Stage 3 produces an SI table of PSI at gencode-discovered alt
junctions; Stages 4-5 are post-pipeline analyses.

## Scripts

| Script | Role |
|---|---|
| `01_fix_sj_supertable.py` | Per-event MANE-canonical SJ rewrite. For each ambiguous event (multiple WT splits in `event_id_161`), pick MANE Select if it fits + has reads at WT pkl, else fall back to alt with most WT reads. For each unique `full_seq` in the event, if no row has the canonical SJ, rewrite the lowest-Reference row of that full_seq to the canonical SJ. Writes corrected supertable + per-rep PSI text files. |
| `02_run_merge_psi.sh` → `02_merge_psi.py` | Clip PSI, compute logit, pool reps, compute SDs, merge across cell lines. **Drops barcode-cluster duplicates** (rows with `transcript_class='duplicate'`) — these have identical PSI to their canonical row. |
| `03_alt_transcript_si_table.py` | For each ambiguous event with gencode-discovered alt junctions, compute PSI at each alt junction for every WT and variant row. Filtered to alts with WT + variant coverage. |
| `04_list_alt_events.py` | Per-event alt-transcript summary (supertable + gencode-only). |
| `05_find_skipped_events.py` | Identify ambiguous events that didn't get a MANE row in the corrected supertable (e.g. PAAF1 / SLCO4A1 / AFDN — intronic-variant constructs with no MANE annotation). |

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
python 04_list_alt_events.py       # post-pipeline summary
python 05_find_skipped_events.py   # post-pipeline diagnostic
```

## Outputs (in `output/`)

| File | Description |
|---|---|
| `st_corrected.csv` | Full corrected supertable (244,000 rows). New columns: `event_id_161` (chr:start-end:strand window), `event_id` (chr:exon_start-exon_end:strand for the canonical SJ — used for WT/variant pairing in Stage 2), `transcript_class` (`MANE` / `alt` / `duplicate`), `n_rows_per_full_seq`, `alt_transcripts_in_supertable`, `alt_transcripts_gencode_only`. |
| `1e-2_ALL_WTS_VARS_NO_DELTAS.csv` | Primary master CSV (all cell lines, every WT and variant; barcode duplicates dropped). |
| `1e-2_ALL_WITH_WT.csv` | Subset filtered to event_ids that contain both a WT row and ≥1 variant row. ~87k rows. |
| `1e-2_<CELL>_{WITH_WT,VARIANTS_ONLY,WTS_VARS_NO_DELTAS}.csv` | Per-cell-line trios. |
| `st_alt_junctions.csv` | One row per (event_id_161, gencode-discovered alt_transcript_id). `canonical_reference` anchors to the WT MANE row. |
| `ambiguous_sjs/SI_alt_transcript_psi.csv` | Per-row PSI at each gencode-only alt junction for WT + variants in the 11 events with such alts. |
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

For each ambiguous event with gencode-discovered alt junctions (= alts present
in gencode v48 but not in the supertable design), compute PSI at the alt
junction for every WT and variant row. Output schema mirrors the main CSV with
`_alt`-suffixed PSI columns and an `alt_transcript_id` key. Filtered to alts
with ≥1 WT + ≥1 variant passing pooled-PSI coverage in any cell line.

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
