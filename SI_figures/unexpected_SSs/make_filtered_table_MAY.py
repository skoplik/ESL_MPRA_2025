#!/usr/bin/env python3
"""Build cryptic_junctions_per_sequence_filtered.csv for the May rebuild.

No script in the repo produced the Sep 2025 `_filtered` table; it was made ad
hoc. The filter it encodes is the one cryptic_ss_figures_figS2.py applies at
load time: drop sequences whose Parent_ref is in the ambiguous-splice-junction
list. Reproduced here so panel A has a generator.
"""
import pandas as pd
D = "/ESL/ESL_MPRA/SI_figures/unexpected_SSs/may_rebuild/"
AMB = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/ambiguous_sjs/SI_alt_transcript_psi.csv"
seq = pd.read_csv(D + "cryptic_junctions_per_sequence.csv")
amb = set(pd.read_csv(AMB, usecols=["Reference"])["Reference"].dropna().astype(int))
before = len(seq)
seq = seq[~seq["Parent_ref"].isin(amb)]
print("rows %d -> %d (excluded %d ambiguous)" % (before, len(seq), before - len(seq)))
seq.to_csv(D + "cryptic_junctions_per_sequence_filtered.csv", index=False)
print("wrote", D + "cryptic_junctions_per_sequence_filtered.csv")
