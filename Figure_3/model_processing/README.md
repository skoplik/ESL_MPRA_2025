# Figure 3 Model Processing — April 2026 Data Update

Scripts to rebuild model prediction inputs and rerun predictions using the April 26 2026 COMPASS data. Required because exon/intron sequences and splice site positions changed in the new pipeline run, making old prediction files stale.

All inputs use:
- `/ESL/Figures_SK/General_preprocessing/output_04_26_2026/04_26_2026_1e-2_ALL_WITH_WT.csv`

Outputs go to `outputs/` subdirectories.

---

## SpliceAI

**Run on GPU.**

```bash
bash run_spliceai.sh
```

Runs `run_spliceai_all.py` — scores all sequences in one batched pass using the 5 SpliceAI models. Saves raw per-nucleotide scores to:
- `outputs/spliceai/spliceai_raw_preds_all.tsv`

After GPU run completes, run `process_spliceai.py` (TODO) to compute Δlogit from raw scores using updated exon coordinates.

---

## MMSplice

**Step 1 (local):** Rebuild synthetic FASTA, GTF, and VCF input files from new sequences.

```bash
bash run_mmsplice.sh
```

This also runs `bgzip` + `tabix` on the VCF. Output files:
- `outputs/mmsplice/input_files/synthetic_reference.fa`
- `outputs/mmsplice/input_files/synthetic_reference.gtf`
- `outputs/mmsplice/input_files/synthetic_variants.vcf.gz` (+ `.tbi`)

**Step 2 (manual):** Run the kipoi MMSplice dataloader with the new input files:

```bash
python3 run_mmsplice_dataloader.py \
  --vcf_path   outputs/mmsplice/input_files/synthetic_variants.vcf.gz \
  --gtf_path   outputs/mmsplice/input_files/synthetic_reference.gtf \
  --fasta_path outputs/mmsplice/input_files/synthetic_reference.fa \
  --output_path outputs/mmsplice/mmsplice_predictions.csv
```

---

## HAL

**Step 1 (local):** Build HAL input zip from new sequences and exon sizes.

```bash
bash run_hal.sh
```

Output:
- `outputs/hal/hal_input_variants_only_avgwtpsi_exon6nt.tsv.zip`
- `outputs/hal/hal_plotting_input.csv`

**Step 2 (manual):** Submit the zip to HAL at http://splicing.cs.washington.edu/SE. Save predictions to `outputs/hal/hal_predictions.tsv`.
