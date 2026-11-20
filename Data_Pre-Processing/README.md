# COMPASS Data Pre-Processing Pipeline

Code for processing raw sequencing data into PSI measurements for the COMPASS MPRA
(Koplik et al., bioRxiv 2025). Raw and processed data are available on GEO under
accession **GSE307247**.

---

## Directory Layout

Scripts are in `/ESL/ESL_MPRA/Data_Pre-Processing/`. Each shell script has a single
`BASE_DIR=/ESL` variable at the top — **change this to your local base directory**
and all paths in that script will update accordingly.

| `$BASE_DIR/` subdirectory | Contents |
|---------------------------|----------|
| `Data/` | Raw MiSeq (DNA) and NextSeq (RNA) FASTQ files; reference supertable |
| `Analysis/` | Intermediate outputs: barcode clusters, STAR alignments, per-replicate PSI files |
| `Figures_SK/General_preprocessing/` | Final merged PSI CSV (primary data file for all downstream analyses) |

---

## Pipeline Overview

```
Raw MiSeq (DNA-Seq)          Raw NextSeq (RNA-Seq)
       |                              |
  [Step 1]                      [Step 2]
  DNA barcode                   Filter RNA reads
  clustering                    by barcode, then
  → FASTA reference             STAR alignment
  → GTF annotation              → per-barcode PSI files
                                      |
                                 [Step 3]
                                 Merge PSI across
                                 cell lines & replicates
                                 → master CSV
```

---

## Dependencies

Install Python packages:

```bash
pip install -r requirements.txt
```

External tools (see `requirements.txt` for versions and links):
`STAR`, `samtools`, `bowtie2`, `cutadapt`, `starcode`, `Picard`, `fgbio`

---

## Step 1 — DNA Barcode Clustering

**Input:** Raw MiSeq paired-end FASTQ files (DNA-Seq library).
Available on GEO as GSM9219929.

**Script:** `Clustering_barcodes/run_get_DNA_barcode_sequence_clusters_concat_09_15_2023_subsampleparams1683_d1c_ms75_shorter3p_iterate.sh`

What it does:
1. Trims plasmid adapter sequences with `cutadapt`
2. Clusters 20 nt barcodes with `starcode` (distance=1)
3. Calls consensus sequences per cluster (majority rule, ≥60% agreement)
4. Assembles a FASTA reference and identifies close-match sequences via `bowtie2`

**Key outputs** (in `/ESL/Analysis/clustering_barcodes_DNA/concat_2023_09_19_d1c_ms75/shorter3p/`):
- `*_mincov5_reference_with_close_matches.fasta` — reference sequences for STAR alignment
- `*_barcode_clusters_consensus_seq.txt` — consensus sequences per cluster

Then generate GTF/GFF3 annotations for STAR:

**Script:** `Clustering_barcodes/run_make_GFF3_GTF_STAR_supertable_concat_09_15_2023_subsampleparams1683_d1c_ms75_shorter3p_iterate.sh`

**Key outputs:**
- `assembled_GTF_supertable_*.gtf` — used as STAR genome annotation
- `assembled_GFF3_supertable_*.gff3`

---

## Step 2 — RNA-Seq Filtering and STAR Alignment

**Input:** Trimmed RNA-Seq paired-end FASTQ files (NextSeq). Available on GEO (GSE307247).

**Script:** `STAR_alignment/STAR_alignment_example_MCF7_Rep2.sh`
(example for MCF7 Rep2; run analogously for all cell lines and replicates)

What it does:
1. Filters RNA-Seq reads against the DNA barcode reference (1 mismatch allowed)
2. Runs two-pass STAR alignment per barcode with UMI-based deduplication (fgbio)
3. Calculates PSI from STAR splice junction counts (min coverage 10 reads)

**Key outputs** (per cell line/replicate, in `/ESL/Analysis/STAR_alignment/`):
- `*_separate_STAR_pass1/` — STAR alignment outputs

Then recalculate splicing profiles:

**Script:** `STAR_alignment/run_get_STAR_separate_splicing_profiles_MCF7_Rep2_20231031.sh`

**Key outputs:**
- `*_separate_recalc_PSIs_mincov10.txt` — PSI values per barcode (≥10 reads)

---

## Step 3 — Merge PSI Across Cell Lines (with ambiguous-SJ correction)

**Input:** Per-replicate `*_recalc_PSIs_mincov10.txt` files from Step 2 for all five
cell lines (HeLa, K562, MCF7, HMC3, HEK293) and all replicates (2 per cell line for
HeLa/K562/MCF7/HMC3; 4 for HEK293, split into variant reps 1–2 and WT reps 3–4),
plus the source supertable and Gencode v48 GTF.

**Scripts:** `Post-process_STAR_PSIs/` — run in order

| Stage | Script | What it does |
|------:|--------|--------------|
| 1 | `01_fix_sj_supertable.py` | Some MPRA exons map to multiple Gencode transcripts with different intron1/exon lengths, producing different splice-junction coordinates. For each such ambiguous event, this script picks one canonical Gencode-annotated junction (MANE Select preferred; variant-pool fallback when the primary reference is a synthesis dropout), rewrites every row in the event to use that junction (re-deriving `intron1`/`exon`/`intron2` from `full_seq`), and patches the per-replicate PSI text files at the chosen junction. Without this step, e.g. KCTD10 exon 4 ref 99222 reports PSI = 0 because reads land at the MANE junction rather than the originally-annotated one. |
| 2 | `02_run_merge_psi.sh` → `02_merge_psi.py` | Clips PSI to [0.01, 0.99], computes logit(PSI), pools replicates, computes per-replicate SDs, computes `dpsi_pooled` and `delta_logit_pooled` per event, and merges all five cell lines into the master table. Carries `transcript_id`, `exon_start_hg38`, `exon_end_hg38`, `variant_hg38` through from the corrected supertable; emits both `ALL_WTS_VARS_NO_DELTAS.csv` and a merged `ALL_WITH_WT.csv`. |
| 3 | `03_alt_transcript_si_table.py` | For each ambiguous event, computes PSI at every non-chosen Gencode transcript's junction for all rows; emits an SI table with `_alt`-suffixed columns. |

**Key outputs** (under `/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/`):

| File | Description |
|------|-------------|
| `ALL_WTS_VARS_NO_DELTAS.csv` | Full COMPASS dataset, all cell lines, every WT and variant — primary data file used for all downstream analyses |
| `ALL_WITH_WT.csv` | Subset filtered to events containing both a WT and at least one variant |
| `st_corrected.csv` | Corrected supertable: one transcript / intron1_len / exon_start_hg38 / exon_end_hg38 per ambiguous event |
| `ambiguous_sjs/SI_alt_transcript_psi.csv` | PSI at every non-chosen alt transcript's junction for ambiguous events |

The legacy single-stage merge script (`merge_psi_07_18_2025_clipping_no_psuedo.py`)
and its predecessor outputs are archived under `Post-process_STAR_PSIs/old/` for
reproducibility of older outputs and are not run in the current pipeline.

See `Post-process_STAR_PSIs/README_04_26_2026.md` for full per-stage detail
(junction-selection logic, fallback rules, schema, construct geometry).

---

## Reference Files

| File | Path | Status |
|------|------|--------|
| Supertable (source, with SNP/coord annotations) | `/ESL/ESL_MPRA/Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25.csv.gz` | **Current** — pipeline input for Step 3 |
| Corrected supertable (ambiguous SJ fix applied) | `/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/st_corrected.csv` | **Current** — generated by running Stage 1; used by Stage 2 |
| Gencode v48 GTF | `/ESL/Figures_SK/General_preprocessing/fix_supertable_2/gencode.v48.annotation.gtf` | Large file, not in repo |
| Barcode FASTA reference | `/ESL/Analysis/clustering_barcodes_DNA/concat_2023_09_19_d1c_ms75/shorter3p/*_reference_with_close_matches.fasta` | Output of Step 1 |
| Construct GTF annotation | `/ESL/Analysis/clustering_barcodes_DNA/concat_2023_09_19_d1c_ms75/shorter3p/assembled_GTF_supertable_*.gtf` | Output of Step 1 |
