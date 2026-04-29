import argparse
import os
import threading
import time
import pandas as pd
import numpy as np
from mmsplice import MMSplice, predict_all_table
from mmsplice.vcf_dataloader import SplicingVCFDataloader
import warnings
import tensorflow as tf
from pyfaidx import Fasta
import pysam

parser = argparse.ArgumentParser()
parser.add_argument("--vcf_path",    required=True, help="bgzipped + tabix-indexed VCF")
parser.add_argument("--gtf_path",    required=True)
parser.add_argument("--fasta_path",  required=True)
parser.add_argument("--output_path", required=True)
args = parser.parse_args()

assert os.path.exists(args.vcf_path),   f"VCF not found: {args.vcf_path}"
assert os.path.exists(args.gtf_path),   f"GTF not found: {args.gtf_path}"
assert os.path.exists(args.fasta_path), f"FASTA not found: {args.fasta_path}"

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

print("\n=== Counting variants in VCF ===")
vcf_count = pysam.VariantFile(args.vcf_path)
all_ids = {rec.id for rec in vcf_count.fetch()}
total_variants = len(all_ids)
print(f"Total variants in VCF: {total_variants:,}")

print("\n=== Initializing SplicingVCFDataloader ===")
dl = SplicingVCFDataloader(args.gtf_path, args.fasta_path, args.vcf_path, tissue_specific=False)
print("DataLoader initialized successfully")

print("\n=== Running MMSplice predictions ===")
model = MMSplice()

result_container = {}
error_container = {}

def run_predictions():
    try:
        result_container["df"] = predict_all_table(
            model, dl, batch_size=256, pathogenicity=True, splicing_efficiency=True
        )
    except Exception as e:
        error_container["err"] = e

t = threading.Thread(target=run_predictions, daemon=True)
t.start()

start = time.time()
spinner = ["|", "/", "-", "\\"]
i = 0
while t.is_alive():
    elapsed = int(time.time() - start)
    mins, secs = divmod(elapsed, 60)
    print(f"\r  {spinner[i % 4]}  Running... {mins:02d}:{secs:02d} elapsed", end="", flush=True)
    i += 1
    time.sleep(1)

print()  # newline after spinner

if "err" in error_container:
    raise error_container["err"]

pred_df = result_container["df"]
elapsed = int(time.time() - start)
mins, secs = divmod(elapsed, 60)
print(f"Prediction complete in {mins:02d}:{secs:02d}. {len(pred_df):,} variants processed.")

print("\n=== Checking for missing variants ===")
vcf = pysam.VariantFile(args.vcf_path)
predicted_ids = set(pred_df["ID"].astype(str).tolist())
missing_ids = sorted(all_ids - predicted_ids)
print(f"Total variants in VCF:     {total_variants:,}")
print(f"Variants with predictions: {len(predicted_ids):,}")
print(f"Missing:                   {len(missing_ids):,}")

pred_df.to_csv(args.output_path, index=False)
print(f"\nSaved MMSplice predictions to: {args.output_path}")
