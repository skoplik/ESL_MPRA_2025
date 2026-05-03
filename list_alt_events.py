"""Build a unified list of exon families (event_id_161) that have any alt
transcript annotations — either in the supertable or discovered in gencode v48.
Run after the post-processing pipeline (Stages 1-3) has produced st_corrected.csv
and st_alt_junctions.csv.
"""
import os
import pandas as pd

PIPE_OUT = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output"
ST  = os.path.join(PIPE_OUT, "st_corrected.csv")
SIA = os.path.join(PIPE_OUT, "st_alt_junctions.csv")
OUT = "/ESL/ESL_MPRA/alt_events.csv"

print("Loading st_corrected.csv...")
st = pd.read_csv(ST, low_memory=False)
print("Loading st_alt_junctions.csv...")
sia = pd.read_csv(SIA, low_memory=False)


def supertable_alts(g):
    txs = set(g["transcript_id"].dropna().astype(str))
    return ";".join(sorted(txs))


print("Building per-event supertable + gencode summary...")
ev_st = (st.groupby("event_id_161")
           .apply(lambda g: pd.Series({
               "gene_exon": g["gene_exon"].iloc[0],
               "n_transcripts_in_supertable": g["transcript_id"].nunique(),
               "supertable_transcripts":      supertable_alts(g),
               "n_unique_full_seqs":          g["full_seq"].nunique(),
           })).reset_index())

ev_gen = (sia.groupby("event_id_161")
            .agg(n_gencode_only_alts=("alt_transcript_id", "nunique"),
                 gencode_only_transcripts=("alt_transcript_id",
                                           lambda s: ";".join(sorted(s.dropna().astype(str).unique())))
            ).reset_index())

merged = ev_st.merge(ev_gen, on="event_id_161", how="left")
merged["n_gencode_only_alts"]      = merged["n_gencode_only_alts"].fillna(0).astype(int)
merged["gencode_only_transcripts"] = merged["gencode_only_transcripts"].fillna("")

alts = merged[(merged["n_transcripts_in_supertable"] > 1) | (merged["n_gencode_only_alts"] > 0)].copy()
alts["total_alts"] = (alts["n_transcripts_in_supertable"] - 1) + alts["n_gencode_only_alts"]
alts = alts.sort_values("total_alts", ascending=False)

alts.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}")
print(f"Total events with any alts: {len(alts):,}")
print(f"  Events with supertable alts (>1 tx): {(alts['n_transcripts_in_supertable']>1).sum():,}")
print(f"  Events with gencode-only alts:       {(alts['n_gencode_only_alts']>0).sum():,}")
print(f"  Events with both:                    {((alts['n_transcripts_in_supertable']>1) & (alts['n_gencode_only_alts']>0)).sum():,}")
print()
print("Top 20 events by total alts:")
print(alts[["event_id_161","gene_exon","n_transcripts_in_supertable","n_gencode_only_alts"]].head(20).to_string(index=False))
