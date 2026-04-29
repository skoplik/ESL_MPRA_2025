import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import os
import re

def parse_transfac(filepath):
    motifs = []
    with open(filepath) as f:
        lines = f.readlines()
    current_pwm = []
    motif_name = None
    reading_pwm = False

    for line in lines:
        line = line.strip()
        if line.startswith("ID"):
            motif_name = line.split()[1]
        elif line.startswith("P0"):
            reading_pwm = True
        elif line.startswith("XX") or line == "//":
            if current_pwm:
                pwm_df = pd.DataFrame(current_pwm, columns=["A", "C", "G", "T"])
                pwm_df.columns = ['A', 'C', 'G', 'U']  # Convert to RNA
                motifs.append((motif_name, pwm_df))
                current_pwm = []
                reading_pwm = False
        elif reading_pwm:
            tokens = line.split()
            if len(tokens) == 5 and tokens[0].isdigit():
                current_pwm.append([float(x) for x in tokens[1:]])

    return motifs

# === Load and parse TRANSFAC motif file ===
transfac_file = "/ESL/ESL_MPRA/Figure_4/Cluster_motifs/output_cleaned_pwms/filtered_motifs_final.transfac"
output_dir = "/ESL/ESL_MPRA/Figure_4/Cluster_motifs/Logos_Indv"
os.makedirs(output_dir, exist_ok=True)

motifs = parse_transfac(transfac_file)

# === Plot and save logos ===
for i, (name, pwm_df) in enumerate(motifs):
    trimmed_name = '_'.join(name.split('_')[:10])  # Keep first 10 tokens
    safe_name = re.sub(r"[^\w\-_.]", "_", trimmed_name)  # Sanitize
    print(safe_name)
    plt.figure(figsize=(8, 2))
    logomaker.Logo(pwm_df, color_scheme='classic')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"logo_{safe_name}.pdf"))
    plt.close()
