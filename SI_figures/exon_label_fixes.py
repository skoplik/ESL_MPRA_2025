"""gene_exon labels that are wrong or ambiguous in the source data, fixed by exon
coordinates (May-convention event_id). Shared by every SI figure/table that labels
exons, so they all agree. Numbering follows the MANE Select transcript (GENCODE v38).

CACNA1C: the source labels two different loci "exon 31".
  chr12:2648475-2648507 (33 nt) is MANE exon 31 (cassette).
  chr12:2633629-2633712 (84 nt) is not in MANE; it is the mutually exclusive
  partner of MANE exon 30 (chr12:2634297-2634380), so "exon 30 (alt.)".
"""
LABEL_FIXES = {
    "chr12:2633629-2633712:+": "CACNA1C exon 30 (alt.)",
    "chr12:2648475-2648507:+": "CACNA1C exon 31",
}


def label(event_id, gene_exon):
    return LABEL_FIXES.get(event_id, gene_exon)
