# Splicing MPRA Analysis

This repository contains scripts and resources for the splicing MPRA analysis.  
Raw and processed data are available on GEO under accession **GSE307247**.  

The following bioinformatic pipeline is demonstrated with **MCF7 Rep2**, but the same workflow applies to all cell lines and replicates.  
Update file paths in the scripts (`GitHub_path`, `supertable_file`, `gtf_file`, `data_dir`, etc.) to match your environment.

`Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25.csv.gz` is the most up-to-date supertable file with additional sequence annotations.

The older supertable file (with the same sequence content but fewer metadata annotations) is available at `Data_Pre-Processing/supertable.tsv.gz`.

## Bioinformatic Pipeline Order

1. **DNA Clustering (Optional)**  
   Cluster DNA barcodes and generate the reference FASTA and GTF/GFF annotation files.  
   - The reference FASTA is already provided on GEO, so this step can be skipped if you want to use the pre-generated files.

2. **RNA-seq Filtering**  
   Filter RNA-seq reads by DNA barcode clusters to remove mismatched or low-quality barcodes.  
   - This prepares the input FASTQ files for alignment.  

3. **STAR Alignment**  
   Map filtered RNA-seq reads to the reference sequences using STAR.  
   Example script: `bash STAR_alignment_example_MCF7_Rep2.sh`  
   - Most of the provided scripts for STAR alignment use **MCF7 Rep2** as an example.  
   - Replace the prefixes to process other cell lines.

4. **Splicing Profiles**  
   Extract exon inclusion and splicing profiles from the STAR alignment outputs.  
   Example script: `bash run_get_STAR_separate_splicing_profiles_MCF7_Rep2_20231031.sh`

5. **Post-processing of PSIs**  
   Merge and post-process STAR outputs to generate PSI, ΔPSI, and logit values across all cell lines.  
   Example script: `bash /ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/run_merge_psi_clipping_no_psuedo.sh`

---

All other analysis scripts in this repository are run on these **post-processed data files**.
# ESL_MPRA_2025
