"""
Build WITH_WT style file from the Type B corrected NO_DELTAS file.

Keeps only events where both WT (snp=='none') and at least one variant exist.
All 144 columns are preserved (no column dropping).

Input:  04_17_2026_1e-2_ALL_WTS_VARS_NO_DELTAS_corrected.csv
Output: 04_17_2026_1e-2_ALL_WITH_WT_corrected.csv
"""

import pandas as pd

INPUT  = "/ESL/Figures_SK/General_preprocessing/output_04_17_2026/04_17_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"
OUTPUT = "/ESL/Figures_SK/General_preprocessing/output_04_17_2026/04_17_2026_1e-2_ALL_WITH_WT.csv"

print("Loading corrected NO_DELTAS...")
df = pd.read_csv(INPUT, low_memory=False)
print(f"  Total rows: {len(df):,}")

wt_eids  = set(df.loc[df["snp"] == "none",  "event_id"])
var_eids = set(df.loc[df["snp"] != "none",  "event_id"])
both_eids = wt_eids & var_eids

with_wt = df[df["event_id"].isin(both_eids)].copy()
print(f"  Events with both WT and variant: {len(both_eids):,}")
print(f"  Rows in WITH_WT: {len(with_wt):,}")
print(f"    WT rows:      {(with_wt['snp']=='none').sum():,}")
print(f"    Variant rows: {(with_wt['snp']!='none').sum():,}")

with_wt.to_csv(OUTPUT, index=False)
print(f"\nSaved: {OUTPUT}")
