# Figure 3 — Model Comparisons

Scripts to rebuild model prediction inputs and rerun predictions using the current COMPASS data.

All inputs use:
- `/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz`

Outputs go to each model's `outputs/` subdirectory.

---

## SpliceAI — **Run on GPU**

**Step 1 (local):** Build input files.

```bash
bash SpliceAI/run_spliceai.sh
```

**Step 2 (GPU machine):** Run predictions. Copy `SpliceAI/outputs/` to a GPU machine and run:

```bash
python3 run_spliceai_all.py
```

Saves raw per-nucleotide scores to `outputs/spliceai/spliceai_raw_preds_all.tsv`. Copy back when done, then run `process_spliceai.py` to compute Δlogit from raw scores.

---

## MMSplice — **Step 2 must run on GPU**

The kipoi MMSplice dataloader uses TensorFlow and is very slow on CPU (~hours). Run Step 2 on a machine with a GPU.

**Step 1 (local):** Build synthetic FASTA, GTF, and VCF input files.

```bash
bash MMSplice/run_mmsplice.sh
```

Output files written to `MMSplice/outputs/input_files/`:
- `synthetic_reference.fa`   — FASTA of all synthetic exon constructs
- `synthetic_reference.gtf`  — GTF annotation for the FASTA
- `synthetic_variants.vcf.gz` + `.tbi` — all variants in VCF format

**Step 2 (GPU machine):** Copy the `input_files/` directory to a GPU machine, then run:

```bash
python3 run_mmsplice_dataloader.py \
  --vcf_path   input_files/synthetic_variants.vcf.gz \
  --gtf_path   input_files/synthetic_reference.gtf \
  --fasta_path input_files/synthetic_reference.fa \
  --output_path outputs/mmsplice/mmsplice_predictions.csv
```

Copy the resulting `mmsplice_predictions.csv.gz` back to `MMSplice/outputs/mmsplice/` when done.

---

## HAL

HAL runs via the web server — no local GPU needed.

**Step 1 (local):** Build HAL input zip.

```bash
bash HAL/run_hal.sh
```

Output:
- `HAL/outputs/hal_input_variants_only_avgwtpsi_exon6nt.tsv.zip`
- `HAL/outputs/hal_plotting_input.csv`

**Step 2 (web):** Upload the zip to http://splicing.cs.washington.edu/SE. Download results and save to `HAL/outputs/hal_predictions.tsv`.

**Step 3 (local):** Plot results.

```bash
python3 HAL/plot_hal_res.py
```

Output plots saved to `HAL/outputs/plots/`.
