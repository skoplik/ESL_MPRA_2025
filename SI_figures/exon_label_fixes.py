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


def primary_loci(variants):
    """event_ids to keep when one library construct (event_id_161) was quantified
    against more than one version of its exon (alternative splice sites a few nt
    apart; 32 constructs in the May data, e.g. ADAM15 exon 20, PIGH exon 3).
    The loci share most of their sequences, so counting all of them double-counts.
    Keep, per construct, the locus with the most measured variants (ties: lowest
    event_id). `variants` = measured, non-reference rows with event_id and event_id_161.
    """
    n = variants.groupby(["event_id_161", "event_id"]).size().reset_index(name="n")
    n = n.sort_values(["event_id_161", "n", "event_id"], ascending=[True, False, True])
    return set(n.drop_duplicates("event_id_161")["event_id"])


def label(event_id, gene_exon):
    return LABEL_FIXES.get(event_id, gene_exon)
