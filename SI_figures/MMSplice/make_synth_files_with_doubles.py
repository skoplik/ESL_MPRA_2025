import pandas as pd
import os

# === Paths ===
supertable_path = "/ESL/ESL_MPRA/Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25.csv.gz"
fasta_out = "/ESL/ESL_MPRA/Figure_3/model_processing/outputs/mmsplice/input_files/synthetic_reference.fa"
gtf_out = "/ESL/ESL_MPRA/Figure_3/model_processing/outputs/mmsplice/input_files/synthetic_reference.gtf"
vcf_out = "/ESL/ESL_MPRA/Figure_3/model_processing/outputs/mmsplice/input_files/synthetic_variants.vcf"

# === Padding sequences ===
CITRINE_EXON1 = ("ATGGTGTCCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTCAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAACTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGATGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAAGTGAAGTTCGAGGGCGACACCCTCGTGAACCGCATCGAGCTAAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAAGTGAACTTCAAGATCCGCCACAACATCGAG")
CITRINE_EXON2 = ("GACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAG")
SMN2_INTRON6 = ("GTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA")
SMN2_INTRON7 = ("AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAAAAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCCAAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTGTGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTAGGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTCTGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAATTCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAG")

LEFT_PADDING = CITRINE_EXON1 + SMN2_INTRON6
RIGHT_PADDING = SMN2_INTRON7 + CITRINE_EXON2
LEFT_LEN = len(LEFT_PADDING)

# === Load data ===
df = pd.read_csv(supertable_path, dtype=str)
df.columns = df.columns.str.strip()
df = df[df["full_seq"].notnull()].copy()
df["is_ref"] = df["snp"] == "none"
df["Reference"] = df["Reference"].astype(int)
df["Reference"] = df["Reference"] + 1

# === Deduplicate on full_seq keeping lowest Reference
df = df.sort_values("Reference").drop_duplicates("full_seq", keep="first").copy()

# === Identify reference sequences
ref_map = df[df["is_ref"]].groupby("event_id")["Reference"].min().to_dict()
df["synthetic_chr"] = df["event_id"].map(ref_map)
df = df[df["synthetic_chr"].notnull()].copy()

# === Only retain WT, single and double mutants
df = df[(df["seq_type"].isin(["single", "double"])) | (df["is_ref"])].copy()

# === Build synthetic sequences
df["synthetic_seq"] = LEFT_PADDING + df["full_seq"] + RIGHT_PADDING
df["exon_start_synthetic"] = LEFT_LEN + df["intron1"].str.len().astype(int)
df["exon_end_synthetic"] = df["exon_start_synthetic"] + df["exon"].str.len().astype(int) - 1
df["exon_start_synthetic"] = df["exon_start_synthetic"] + 1
df["exon_end_synthetic"] = df["exon_end_synthetic"] + 1

# === Write FASTA
fasta_df = df[df["is_ref"]].drop_duplicates("synthetic_chr")
with open(fasta_out, "w") as f:
    for _, row in fasta_df.iterrows():
        f.write(f">{row['synthetic_chr']}\n{row['synthetic_seq']}\n")

# === Write GTF
with open(gtf_out, "w") as gtf:
    for _, row in fasta_df.iterrows():
        chrom = str(row["synthetic_chr"])
        strand = "+"
        gene = row["gene_name"]
        transcript_id = exon_id = chrom
        base_attrs = f'gene_id "{gene}"; gene_name "{gene}"; transcript_id "{transcript_id}";'

        seq_len = len(row["synthetic_seq"])
        gtf.write(f"{chrom}\tsupertable\ttranscript\t1\t{seq_len}\t.\t{strand}\t.\t{base_attrs} exon_id \"{exon_id}\";\n")

        c1_end = len(CITRINE_EXON1)
        core_start = row["exon_start_synthetic"]
        core_end = row["exon_end_synthetic"]
        c2_start = seq_len - len(CITRINE_EXON2) + 1
        c2_end = seq_len

        gtf.write(f"{chrom}\tsupertable\texon\t1\t{c1_end}\t.\t{strand}\t.\t{base_attrs} exon_id \"citrine1\";\n")
        gtf.write(f"{chrom}\tsupertable\texon\t{core_start}\t{core_end}\t.\t{strand}\t.\t{base_attrs} exon_id \"core_exon\";\n")
        gtf.write(f"{chrom}\tsupertable\texon\t{c2_start}\t{c2_end}\t.\t{strand}\t.\t{base_attrs} exon_id \"citrine2\";\n")

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
        chrom = str(ref_map.get(row["event_id"]))
        source = row["source"]
        ref_idx = int(row["Reference"])
        wt_seq = eventid_to_refseq.get(row["event_id"])
        if wt_seq is None:
            continue

        ref_bases = []
        alt_bases = []
        positions = []

        for snv in snp_field.split(";"):
            if ":" not in snv or ">" not in snv:
                continue
            pos_str, change = snv.split(":")
            ref_base, alt_base = change.split(">")
            pos = int(pos_str)
            positions.append(pos)
            ref_bases.append((pos, ref_base))
            alt_bases.append((pos, alt_base))

        if not positions:
            continue

        min_pos = min(positions)
        max_pos = max(positions)
        span_len = max_pos - min_pos + 1

        ref_seq = list(wt_seq[min_pos : min_pos + span_len])
        alt_seq = ref_seq.copy()

        for (pos, ref_base), (_, alt_base) in zip(ref_bases, alt_bases):
            offset = pos - min_pos
            if offset < 0 or offset >= len(ref_seq):
                continue
            if ref_seq[offset].upper() != ref_base.upper():
                continue
            alt_seq[offset] = alt_base

        ref_str = "".join(ref_seq).replace(" ", "")
        alt_str = "".join(alt_seq).replace(" ", "")

        if ref_str == alt_str:
            continue

        vcf_rows.append([
            chrom,
            LEFT_LEN + min_pos + 1,
            ref_idx,
            ref_str,
            alt_str,
            ".",
            "PASS",
            f"SOURCE={source}"
        ])
    except Exception:
        continue

with open(vcf_out, "w") as f:
    f.write("\n".join(vcf_header) + "\n")
    for row in vcf_rows:
        f.write("\t".join(map(str, row)) + "\n")

print("Synthetic FASTA, GTF, and corrected VCF written using WT reference sequences.")
