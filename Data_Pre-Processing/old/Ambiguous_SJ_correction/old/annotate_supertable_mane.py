"""
Add mane_status column to the supertable based on transcript_id.
Reads MANE Select / MANE Plus Clinical from the Gencode GTF and annotates in-place.
"""

import re
import pandas as pd

GTF        = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/gencode.v48.annotation.gtf"
SUPERTABLE = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv"

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

print(f"  MANE Select:        {len(mane_select):,}")
print(f"  MANE Plus Clinical: {len(mane_plus):,}")

def mane_label(transcript_id):
    tx = str(transcript_id).split('.')[0]
    if tx in mane_select:
        return 'MANE Select'
    if tx in mane_plus:
        return 'MANE Plus Clinical'
    return ''

print("Loading supertable...")
st = pd.read_csv(SUPERTABLE, low_memory=False)
print(f"  Rows: {len(st):,}")

st["mane_status"] = st["transcript_id"].map(mane_label)

counts = st["mane_status"].value_counts(dropna=False)
print("mane_status distribution:")
print(counts.to_string())

st.to_csv(SUPERTABLE, index=False)
print(f"\nSaved: {SUPERTABLE}")
