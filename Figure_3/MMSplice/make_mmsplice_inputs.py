import argparse
import os
import pandas as pd

CITRINE_EXON1 = "ATGGTGTCCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTCAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAACTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGATGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAAGTGAAGTTCGAGGGCGACACCCTCGTGAACCGCATCGAGCTAAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAAGTGAACTTCAAGATCCGCCACAACATCGAG"
CITRINE_EXON2 = "GACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAG"
SMN2_INTRON6 = "GTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA"
SMN2_INTRON7 = "AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAAAAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCCAAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTGTGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTAGGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTCTGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAATTCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAG"

LEFT_PADDING = CITRINE_EXON1 + SMN2_INTRON6
RIGHT_PADDING = SMN2_INTRON7 + CITRINE_EXON2
LEFT_LEN = len(LEFT_PADDING)

parser = argparse.ArgumentParser()
parser.add_argument("--main_data", required=True, help="ALL_WITH_WT.csv")
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
fasta_out = os.path.join(args.output_dir, "synthetic_reference.fa")
gtf_out   = os.path.join(args.output_dir, "synthetic_reference.gtf")
vcf_out   = os.path.join(args.output_dir, "synthetic_variants.vcf")

print("Loading data...")
df = pd.read_csv(args.main_data, low_memory=False, dtype=str)
df.columns = df.columns.str.strip()
df = df[df["full_seq"].notnull()].copy()
df["is_ref"] = df["snp"] == "none"
df["Reference"] = df["Reference"].astype(int)
df["Reference"] = df["Reference"] + 1

# Deduplicate on full_seq keeping lowest Reference
df = df.sort_values("Reference").drop_duplicates("full_seq", keep="first").copy()

# Only WT + single + double
df = df[(df["seq_type"].isin(["single", "double"])) | (df["is_ref"])].copy()

# Map each event_id to its WT Reference (used as synthetic chromosome name)
ref_map = df[df["is_ref"]].groupby("event_id")["Reference"].min().to_dict()
df["synthetic_chr"] = df["event_id"].map(ref_map)
df = df[df["synthetic_chr"].notnull()].copy()

# Build synthetic sequences and exon coords
df["synthetic_seq"] = LEFT_PADDING + df["full_seq"] + RIGHT_PADDING
df["exon_start_synthetic"] = LEFT_LEN + df["intron1"].str.len().astype(int) + 1  # 1-based
df["exon_end_synthetic"]   = df["exon_start_synthetic"] + df["exon"].str.len().astype(int) - 1

print(f"Total sequences: {len(df):,}  |  WT refs: {df['is_ref'].sum():,}  |  Variants: {(~df['is_ref']).sum():,}")

# === Write FASTA ===
fasta_df = df[df["is_ref"]].drop_duplicates("synthetic_chr")
with open(fasta_out, "w") as f:
    for _, row in fasta_df.iterrows():
        f.write(f">{row['synthetic_chr']}\n{row['synthetic_seq']}\n")
print(f"FASTA written: {fasta_out}  ({len(fasta_df):,} sequences)")

# === Write GTF ===
with open(gtf_out, "w") as gtf:
    for _, row in fasta_df.iterrows():
        chrom = str(row["synthetic_chr"])
        gene = row.get("gene_name", chrom)
        base_attrs = f'gene_id "{gene}"; gene_name "{gene}"; transcript_id "{chrom}";'
        seq_len = len(row["synthetic_seq"])

        gtf.write(f"{chrom}\tsupertable\ttranscript\t1\t{seq_len}\t.\t+\t.\t{base_attrs} exon_id \"{chrom}\";\n")

        c1_end    = len(CITRINE_EXON1)
        core_s    = int(row["exon_start_synthetic"])
        core_e    = int(row["exon_end_synthetic"])
        c2_start  = seq_len - len(CITRINE_EXON2) + 1

        gtf.write(f"{chrom}\tsupertable\texon\t1\t{c1_end}\t.\t+\t.\t{base_attrs} exon_id \"citrine1\";\n")
        gtf.write(f"{chrom}\tsupertable\texon\t{core_s}\t{core_e}\t.\t+\t.\t{base_attrs} exon_id \"core_exon\";\n")
        gtf.write(f"{chrom}\tsupertable\texon\t{c2_start}\t{seq_len}\t.\t+\t.\t{base_attrs} exon_id \"citrine2\";\n")
print(f"GTF written: {gtf_out}")

# === Write VCF ===
vcf_header = [
    "##fileformat=VCFv4.2",
    "##INFO=<ID=SOURCE,Number=1,Type=String,Description=\"Original source\">",
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
]

eventid_to_refseq = df[df["is_ref"]].set_index("event_id")["full_seq"].to_dict()
vcf_rows = []
for _, row in df[~df["is_ref"]].iterrows():
    snp_field = row["snp"]
    if not isinstance(snp_field, str) or ":" not in snp_field:
        continue
    try:
        chrom   = str(ref_map.get(row["event_id"]))
        wt_seq  = eventid_to_refseq.get(row["event_id"])
        if wt_seq is None:
            continue

        positions, ref_bases, alt_bases = [], [], []
        for snv in snp_field.split(";"):
            if ":" not in snv or ">" not in snv:
                continue
            pos_str, change = snv.split(":")
            ref_b, alt_b = change.split(">")
            pos = int(pos_str)
            positions.append(pos)
            ref_bases.append((pos, ref_b))
            alt_bases.append((pos, alt_b))

        if not positions:
            continue

        min_pos  = min(positions)
        max_pos  = max(positions)
        ref_seq  = list(wt_seq[min_pos: max_pos + 1])
        alt_seq  = ref_seq.copy()

        for (pos, rb), (_, ab) in zip(ref_bases, alt_bases):
            offset = pos - min_pos
            if 0 <= offset < len(ref_seq) and ref_seq[offset].upper() == rb.upper():
                alt_seq[offset] = ab

        ref_str = "".join(ref_seq).replace(" ", "")
        alt_str = "".join(alt_seq).replace(" ", "")
        if ref_str == alt_str:
            continue

        vcf_rows.append([chrom, LEFT_LEN + min_pos + 1, int(row["Reference"]),
                         ref_str, alt_str, ".", "PASS", f"SOURCE={row['source']}"])
    except Exception:
        continue

with open(vcf_out, "w") as f:
    f.write("\n".join(vcf_header) + "\n")
    for r in vcf_rows:
        f.write("\t".join(map(str, r)) + "\n")
print(f"VCF written: {vcf_out}  ({len(vcf_rows):,} variants)")
print("\nDone. Next step: bgzip + tabix the VCF, then run MMSplice kipoi dataloader.")
