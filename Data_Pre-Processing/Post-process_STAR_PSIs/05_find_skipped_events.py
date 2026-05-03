"""Identify ambiguous events in the new st_corrected.csv that were SKIPPED by
Stage 1's MANE-canonical SJ assignment (no canonical SJ chosen) and explain
why — typically because no MANE row exists in the supertable for that event,
or no alt with reads as fallback.

Run after Stages 1-3.
"""
import os
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
ST = os.path.join(OUT_DIR, "st_corrected.csv")

print(f"Loading {ST}...")
st = pd.read_csv(ST, low_memory=False)


def ambig_events(st):
    """Return event_id_161 → list of WT splits, for events with >1 WT split."""
    wt = st[st["snp"] == "none"]
    splits_per_event = (
        wt.groupby("event_id_161")
          .apply(lambda g: sorted({(int(r["intron1_len"]), int(r["exon_len"]))
                                    for _, r in g.iterrows()}))
    )
    return splits_per_event[splits_per_event.apply(len) > 1]


amb = ambig_events(st)
print(f"\nAmbiguous events (multiple WT splits): {len(amb)}")

print("\n" + "="*80)
print("Per-ambiguous-event check: does any row have transcript_class == 'MANE'?")
print("="*80)

n_ok = 0
n_no_mane = 0
records = []
for eid, splits in amb.items():
    grp = st[st["event_id_161"] == eid]
    gene_exon = grp["gene_exon"].iloc[0]
    has_mane_row    = (grp["transcript_class"] == "MANE").any()
    has_mane_status = (grp["mane_status"] == "MANE Select").any()
    n_unique_full_seqs = grp["full_seq"].nunique()
    splits_str = ", ".join(f"{i}/{e}" for i, e in splits)
    if has_mane_row:
        n_ok += 1
        flag = ""
    else:
        n_no_mane += 1
        flag = "  ★ NO MANE row — Stage 1 likely fell back to alt or skipped"
    records.append({
        "event_id_161": eid,
        "gene_exon": gene_exon,
        "n_wt_splits": len(splits),
        "wt_splits": splits_str,
        "n_unique_full_seqs": n_unique_full_seqs,
        "has_mane_row": has_mane_row,
        "has_mane_status_in_supertable": has_mane_status,
    })
    if flag:
        print(f"  {eid}  {gene_exon}")
        print(f"    WT splits: {splits_str}    has_mane_in_st: {has_mane_status}{flag}")

df = pd.DataFrame(records)
out_csv = os.path.join(OUT_DIR, "ambig_events_status.csv")
df.to_csv(out_csv, index=False)

print(f"\nSummary:")
print(f"  Ambiguous events with a MANE row:    {n_ok}")
print(f"  Ambiguous events with no MANE row:   {n_no_mane}")
print(f"\nSaved per-event status: {out_csv}")
