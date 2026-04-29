import pandas as pd
from collections import defaultdict
import re

# === File paths
cluster_tab = "/ESL/ESL_MPRA/Figure_6/Cluster_motifs/clusters.tab"
rbp_mapping_file = "/ESL/ESL_MPRA/Figure_6/Effect_size/outputs/plots/rncmpt_to_rbpname_full_mapping.csv"
cluster_output_csv = "/ESL/ESL_MPRA/Figure_6/Effect_size/outputs/plots/cluster_to_rbps_parsed.csv"
motif_output_csv = "/ESL/ESL_MPRA/Figure_6/Effect_size/outputs/plots/motif_to_rbps_parsed.csv"

# === Load RNCMPT mapping
motif_map_df = pd.read_csv(rbp_mapping_file)
motif_map_df['Motif_ID'] = motif_map_df['Motif_ID'].astype(str).str.strip()
motif_map_df['RBP_Name'] = motif_map_df['RBP_Name'].astype(str).str.strip().str.upper()
motif_renamer = dict(zip(motif_map_df['Motif_ID'], motif_map_df['RBP_Name']))

# === Initialize mappings
cluster_to_rbps = defaultdict(set)
motif_to_rbps = defaultdict(set)

with open(cluster_tab, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) != 3:
            continue

        cluster_id = parts[0].strip()
        motif_field = parts[1].strip()
        annotation_field = parts[2].strip()

        # Extract motif IDs from motif_field
        motif_ids = []
        for raw in motif_field.split(","):
            raw = raw.strip()
            m = re.search(r"motif(\d+)", raw)
            if m:
                motif_ids.append(f"motif{m.group(1)}")

        # Split annotation_field by both commas and pipes
        annotations = re.split(r"[,\|]", annotation_field)

        for entry in annotations:
            entry = entry.strip()
            if ":" not in entry:
                continue
            rhs = entry.split(":", 1)[1].strip()
            rbp_id = rhs.split("_")[0].strip()

            if rbp_id.startswith("RNCMPT"):
                cleaned = motif_renamer.get(rbp_id, rbp_id).upper()
            else:
                cleaned = rbp_id.upper()

            if cleaned.endswith("CONSTRUCT"):
                cleaned = cleaned.replace("CONSTRUCT", "").strip()

            if cleaned:
                cluster_to_rbps[cluster_id].add(cleaned)
                for motif in motif_ids:
                    motif_to_rbps[motif].add(cleaned)

# === Export cluster → RBP
cluster_rows = []
for cluster in sorted(cluster_to_rbps.keys()):
    rbps = sorted(cluster_to_rbps[cluster])
    cluster_rows.append({"cluster": cluster, "NAME": ", ".join(rbps)})
pd.DataFrame(cluster_rows).to_csv(cluster_output_csv, index=False)

# === Export motif → RBP
motif_rows = []
for motif in sorted(motif_to_rbps.keys()):
    rbps = sorted(motif_to_rbps[motif])
    motif_rows.append({"motif": motif, "NAME": ", ".join(rbps)})
pd.DataFrame(motif_rows).to_csv(motif_output_csv, index=False)

print(f"Exported {len(cluster_rows)} clusters to: {cluster_output_csv}")
print(f"Exported {len(motif_rows)} motifs to: {motif_output_csv}")

# === Input file (use either cluster or motif output)
input_csv = cluster_output_csv
# input_csv = motif_output_csv

# === Output file for unique RBPs
output_txt = "/ESL/ESL_MPRA/Figure_6/Effect_size/outputs/plots/unique_rbp_names.txt"

# === Load and parse
df = pd.read_csv(input_csv)
all_rbps = set()

for rbp_list in df["NAME"]:
    for rbp in str(rbp_list).split(","):
        rbp_clean = rbp.strip()
        if rbp_clean:
            all_rbps.add(rbp_clean)

# === Write sorted list
with open(output_txt, "w") as f:
    for rbp in sorted(all_rbps):
        f.write(rbp + "\n")

print(f"Exported {len(all_rbps)} unique RBP names to: {output_txt}")
