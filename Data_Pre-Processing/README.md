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

## Step 3 — Merge PSI Across Cell Lines

**Input:** Per-replicate `*_recalc_PSIs_mincov10.txt` files from Step 2 for all five
cell lines (HeLa, K562, MCF7, HMC3, HEK293) and all replicates (2 per cell line for
HeLa/K562/MCF7/HMC3; 4 for HEK293, split into variant reps 1–2 and WT reps 3–4).

**Script:** `Post-process_STAR_PSIs/run_merge_psi_clipping_no_psuedo.sh`

What it does:
- Clips PSI to [0.01, 0.99], computes logit(PSI)
- Computes delta logit(PSI) = logit(PSI_variant) − logit(PSI_WT) per event
- Pools replicates; computes per-replicate standard deviations
- Merges all cell lines into a single master table

**Key output:**
- `/ESL/Figures_SK/General_preprocessing/output_03_16_2026/03_16_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv`
  — primary data file used for all downstream analyses

---

## Reference Files

| File | Path |
|------|------|
| Supertable (sequence metadata) | `/ESL/Data/Sequences/supertable.tsv` |
| Final supertable with coordinates | `/ESL/Figures_SK/General_preprocessing/fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv` |
| Barcode FASTA reference | `/ESL/Analysis/clustering_barcodes_DNA/concat_2023_09_19_d1c_ms75/shorter3p/*_reference_with_close_matches.fasta` |
| GTF annotation | `/ESL/Analysis/clustering_barcodes_DNA/concat_2023_09_19_d1c_ms75/shorter3p/assembled_GTF_supertable_*.gtf` |
