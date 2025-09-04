import subprocess
import shutil
import os

# === Input/output paths ===
vcf_in_path = "/ESL/Figures_SK/mmsplice2/input_files_doubles/synthetic_variants.vcf"
vcf_out_path = "/ESL/Figures_SK/mmsplice2/input_files_doubles/synthetic_all_variants.vcf"
vcf_bgz_path = vcf_out_path + ".gz"

# === Read full VCF
with open(vcf_in_path) as f:
    vcf_lines = f.readlines()

vcf_header = [line for line in vcf_lines if line.startswith('#')]
vcf_body = [line for line in vcf_lines if not line.startswith('#')]

# === Parse all records with CHROM and POS for full sort
records = []
for line in vcf_body:
    fields = line.strip().split('\t')
    chrom = fields[0]
    pos = int(fields[1])
    records.append(((chrom, pos), line))

# === Sort by (CHROM, POS)
records.sort(key=lambda x: (x[0][0], x[0][1]))
sorted_body = [r[1] for r in records]

# === Write uncompressed sorted VCF
with open(vcf_out_path, 'w') as out:
    out.writelines(vcf_header)
    out.writelines(sorted_body)

print(f"✓ Wrote VCF (uncompressed): {vcf_out_path}")

# === Compress with bgzip
tmp_for_bgzip = vcf_out_path + ".tmp"
shutil.copyfile(vcf_out_path, tmp_for_bgzip)

with open(vcf_bgz_path, "wb") as out_gz:
    subprocess.run(["bgzip", "-c", tmp_for_bgzip], stdout=out_gz, check=True)

# === Index with tabix
subprocess.run(["tabix", "-f", "-p", "vcf", vcf_bgz_path], check=True)
print(f"✓ BGZIP + Tabix complete: {vcf_bgz_path}")

# === Cleanup temp file
os.remove(tmp_for_bgzip)
