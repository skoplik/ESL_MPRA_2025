import argparse
import os
import threading
import time
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import pandas as pd
import numpy as np
from mmsplice import MMSplice, predict_all_table
from mmsplice.vcf_dataloader import SplicingVCFDataloader
from pyfaidx import Fasta
import pysam

# Use GPU if available; fall back to CPU with a warning
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU detected: {[g.name for g in gpus]}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("WARNING: No GPU detected. Predictions will run on CPU and may take several hours.")

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

print("\n=== Initializing SplicingVCFDataloader (this may take several minutes) ===")
dl_result = {}
dl_error = {}

def _init_dl():
    try:
        dl_result["dl"] = SplicingVCFDataloader(args.gtf_path, args.fasta_path, args.vcf_path, tissue_specific=False)
    except Exception as e:
        dl_error["err"] = e

t = threading.Thread(target=_init_dl, daemon=True)
t.start()
spinner = ["|", "/", "-", "\\"]
i = 0
init_start = time.time()
while t.is_alive():
    elapsed = int(time.time() - init_start)
    mins, secs = divmod(elapsed, 60)
    print(f"\r  {spinner[i % 4]}  Initializing dataloader... {mins:02d}:{secs:02d}", end="", flush=True)
    i += 1
    time.sleep(1)
print()
if "err" in dl_error:
    raise dl_error["err"]
dl = dl_result["dl"]
print("DataLoader initialized successfully")

print("\n=== Running MMSplice predictions ===")
model = MMSplice()
start = time.time()
pred_df = predict_all_table(model, dl, batch_size=1024, pathogenicity=True, splicing_efficiency=True, progress=True)
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
