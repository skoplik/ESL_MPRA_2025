"""
MMSplice input builder for the May data.

Keys each synthetic chromosome by event_id_161 rather than gene_exon, because a
gene_exon label can cover two loci (CACNA1C exon 31 is the 31a/31b pair, 15 kb
apart) -- keying by label dropped one locus and its 518 variants entirely.
Variants are restricted to the same junction annotation whose WT defines their
chromosome, so a locus with two annotations contributes each variant once.

The VCF ID convention (Reference+1) is unchanged, so the verified downstream
mapping still holds.
"""
import pandas as pd, os

CITRINE_EXON1="ATGGTGTCCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTCAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAACTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGATGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAAGTGAAGTTCGAGGGCGACACCCTCGTGAACCGCATCGAGCTAAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAAGTGAACTTCAAGATCCGCCACAACATCGAG"
CITRINE_EXON2="GACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAG"
SMN2_INTRON6="GTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA"
SMN2_INTRON7="AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAAAAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCCAAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTGTGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTAGGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTCTGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAATTCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAG"

LEFT = CITRINE_EXON1 + SMN2_INTRON6
RIGHT = SMN2_INTRON7 + CITRINE_EXON2
LEFT_LEN = len(LEFT)
KEY = "event_id_161"

OUT = "/ESL/ESL_MPRA/Figure_3/MMSplice/outputs/input_files_MAY_v3"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz",
                 low_memory=False, dtype=str)
df["is_ref"] = df["snp"] == "none"
df = df[df["full_seq"].notnull()].copy()
df["tc"] = df["transcript_class"].astype(str)

# primary-junction rule, per LOCUS (not per gene_exon label)
ehm = df.groupby(KEY)["tc"].apply(lambda s: (s == "MANE").any())
df["ehm"] = df[KEY].map(ehm)
df["keep"] = (df["tc"] == "MANE") | (~df["ehm"])
df = df[df["keep"] | df["is_ref"]].copy()

# one WT per LOCUS: prefer MANE, then lowest Reference (deterministic)
wt = df[df["is_ref"]].copy()
wt["m"] = (wt["tc"] == "MANE").astype(int)
wt["ri"] = wt["Reference"].astype(int)
wt = wt.sort_values([KEY, "m", "ri"], ascending=[True, False, True]).drop_duplicates(KEY)

k2chr = {g: int(r) + 1 for g, r in zip(wt[KEY], wt["Reference"].astype(int))}
k2seq = wt.set_index(KEY)["full_seq"].to_dict()
k2i1 = {g: len(s) for g, s in zip(wt[KEY], wt["intron1"])}
k2ex = {g: len(s) for g, s in zip(wt[KEY], wt["exon"])}
k2gene = wt.set_index(KEY)["gene_exon"].to_dict()
# the annotation whose WT defines this chromosome - variants must match it
k2event = wt.set_index(KEY)["event_id"].to_dict()


def gname(ge):
    return str(ge).split(" exon ")[0].replace(" ", "")


V = df[(~df["is_ref"]) & (df[KEY].isin(k2chr))].copy()
before = len(V)
V = V[[e == k2event.get(k) for k, e in zip(V[KEY], V["event_id"])]].copy()
print("variants: %d -> %d after restricting to the chromosome's own annotation" % (before, len(V)))

with open(OUT + "/synthetic_reference.fa", "w") as fa, open(OUT + "/synthetic_reference.gtf", "w") as gtf:
    for k, chrom in sorted(k2chr.items(), key=lambda kv: kv[1]):
        seq = LEFT + k2seq[k] + RIGHT
        sl = len(seq); c = str(chrom)
        es = LEFT_LEN + k2i1[k] + 1; ee = es + k2ex[k] - 1
        fa.write(">%s\n%s\n" % (c, seq))
        a = ('gene_id "%s"; transcript_id "%s"; gene_name "%s"; gene_type "protein_coding"; '
             'gene_biotype "protein_coding";' % (c, c, gname(k2gene[k])))
        gtf.write("%s\tsup\tgene\t1\t%d\t.\t+\t.\t%s\n" % (c, sl, a))
        gtf.write('%s\tsup\ttranscript\t1\t%d\t.\t+\t.\t%s exon_id "%s";\n' % (c, sl, a, c))
        gtf.write('%s\tsup\texon\t1\t%d\t.\t+\t.\t%s exon_id "c1";\n' % (c, LEFT_LEN, a))
        gtf.write('%s\tsup\texon\t%d\t%d\t.\t+\t.\t%s exon_id "core";\n' % (c, es, ee, a))
        gtf.write('%s\tsup\texon\t%d\t%d\t.\t+\t.\t%s exon_id "c2";\n' % (c, ee + 1, sl, a))

rows = []
for _, r in V.iterrows():
    ws = k2seq.get(r[KEY]); c = k2chr.get(r[KEY])
    if ws is None:
        continue
    P = []; RB = []; AB = []
    for snv in str(r["snp"]).split(";"):
        if ":" not in snv or ">" not in snv:
            continue
        ps, ch = snv.split(":")
        rr, aa = ch.split(">")
        rr = rr.strip(); aa = aa.strip(); p = int(ps.strip())
        P.append(p); RB.append((p, rr)); AB.append((p, aa))
    if not P:
        continue
    mn, mx = min(P), max(P)
    if mx >= len(ws):
        continue
    rs = list(ws[mn:mx + 1]); al = rs[:]
    for (p, x), (_, y) in zip(RB, AB):
        o = p - mn
        if 0 <= o < len(rs) and rs[o].upper() == x.upper():
            al[o] = y
    rstr = "".join(rs).replace(" ", ""); astr = "".join(al).replace(" ", "")
    if rstr == astr:
        continue
    rows.append((int(c), LEFT_LEN + mn + 1, int(r["Reference"]) + 1, rstr, astr))

rows.sort(key=lambda t: (t[0], t[1]))
hdr = ['##fileformat=VCFv4.2',
       '##INFO=<ID=S,Number=1,Type=String,Description="s">',
       '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO']
with open(OUT + "/synthetic_variants.vcf", "w") as f:
    f.write("\n".join(hdr) + "\n")
    for c, p, i, rr, aa in rows:
        f.write("%s\t%s\t%s\t%s\t%s\t.\tPASS\tS=.\n" % (c, p, i, rr, aa))

print("WT loci (chromosomes): %d   VCF variants: %d" % (len(k2chr), len(rows)))
ge = pd.Series([k2gene[k] for k in k2chr])
print("distinct gene_exon labels covered: %d" % ge.nunique())
print("gene_exons split across >1 chromosome: %d" % int((ge.value_counts() > 1).sum()))
