import pandas as pd
import numpy as np
from mmsplice import MMSplice, predict_all_table
from mmsplice.vcf_dataloader import SplicingVCFDataloader
import warnings
import tensorflow as tf
import os
from pyfaidx import Fasta
import pysam


# === Input files ===
vcf_path = "/ESL/ESL_MPRA/Figure_3/model_processing/outputs/mmsplice/input_files/synthetic_all_variants.vcf.gz"
gtf_path = "/ESL/ESL_MPRA/Figure_3/model_processing/outputs/mmsplice/input_files/synthetic_reference.gtf"
fasta_path = "/ESL/ESL_MPRA/Figure_3/model_processing/outputs/mmsplice/input_files/synthetic_reference.fa"
output_path = "/ESL/ESL_MPRA/Figure_3/model_processing/outputs/mmsplice/mmsplice_predictions.csv"


assert os.path.exists(vcf_path), f"VCF not found: {vcf_path}"
assert os.path.exists(gtf_path), f"GTF not found: {gtf_path}"
assert os.path.exists(fasta_path), f"FASTA not found: {fasta_path}"

# === Initialize SplicingVCFDataloader ===
print("\n=== Initializing SplicingVCFDataloader ===")
try:
    dl =SplicingVCFDataloader(
        gtf_path,
        fasta_path,
        vcf_path,
        tissue_specific=False
    )
    print("DataLoader initialized successfully")
except Exception as e:
    print("\n=== ERROR: Failed to initialize SplicingVCFDataloader ===")
    raise e

# === Run MMSplice model ===
print("\n=== Running MMSplice predictions ===")
model = MMSplice()

try:
    pred_df = predict_all_table(model, dl, batch_size=256, pathogenicity=True, splicing_efficiency=True)
    print(f"Prediction complete. {len(pred_df)} variants processed.")
except Exception as e:
    print("\n=== ERROR: Failed during prediction ===")
    raise e

# === Extract all variant IDs from VCF ===
print("\n=== Checking for missing variants ===")
vcf = pysam.VariantFile(vcf_path)
all_ids = set()
for rec in vcf.fetch():
    all_ids.add(rec.id)

predicted_ids = set(pred_df["ID"].astype(str).tolist())
missing_ids = sorted(all_ids - predicted_ids)

print(f"\nTotal variants in VCF: {len(all_ids)}")
print(f"Variants with predictions: {len(predicted_ids)}")

# === Save predictions ===
pred_df.to_csv(output_path, index=False)
print("\nSaved MMSplice predictions to:", output_path)


