"""
Prepare a clean working copy of the supertable from the gz backup.
Adds mane_status column from Gencode GTF.
Writes to output directory — never modifies the source gz.

Input:  st_final_with_snp_and_coords_05_30_25.csv.gz  (original, never modified)
Output: output_04_24_2026/st_working_04_24_2026.csv
"""

import re
import gzip
import pandas as pd

GZ_IN  = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv.gz"
GTF    = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/gencode.v48.annotation.gtf"
OUT    = "/ESL/Figures_SK/General_preprocessing/output_04_24_2026/st_working_04_24_2026.csv"

print("Loading supertable from gz...")
with gzip.open(GZ_IN, "rt") as f:
    st = pd.read_csv(f, low_memory=False)
print(f"  Rows: {len(st):,}")

st["_i1"] = st["intron1"].str.len()
ic = st.groupby("event_id")["_i1"].nunique()
print(f"  Ambiguous events (i1_len diversity): {(ic>1).sum()}")
st = st.drop(columns=["_i1"])

print("Parsing MANE status from GTF...")
mane_select = set()
mane_plus   = set()
with open(GTF) as f:
    for line in f:
        if '\ttranscript\t' not in line:
            continue
        m = re.search(r'transcript_id "([^"]+)"', line)
        if not m:
            continue
        tx = m.group(1).split('.')[0]
        if 'MANE_Select' in line:
            mane_select.add(tx)
        if 'MANE_Plus_Clinical' in line:
            mane_plus.add(tx)
print(f"  MANE Select: {len(mane_select):,}  MANE Plus Clinical: {len(mane_plus):,}")

def mane_label(tx_id):
    tx = str(tx_id).split('.')[0]
    if tx in mane_select:
        return 'MANE Select'
    if tx in mane_plus:
        return 'MANE Plus Clinical'
    return ''

st["mane_status"] = st["transcript_id"].map(mane_label)
print("mane_status distribution:")
print(st["mane_status"].value_counts(dropna=False).to_string())

st.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}")
