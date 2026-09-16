
import os, sys
from Bio import SeqIO
from Bio import pairwise2
rna_dir="/ESL/ESL_MPRA/Figure_5/BIN1RNA_fastq"
exon_included="CACCACGACAGCAGGAAGAGAGCTGGAGGCTGCTTCACTTGCCGCCGTCTCCCCTGGCTCCTGGGCTCCAGCCGCAGGTTGGGTCCCACCCGCCACCTCCGAGGCCTCTGCTGGCTGAGATGGGGACTTGGGGAGGGTGGCCC"
exon_excluded="CACCACGACAGCAGGAAGAGAGCTCTGAGATGGGGACTTGGGGAGGGTGGCCC"
max_mismatches=3
def match(read,target):
    for a in pairwise2.align.localms(read,target,2,-1,-2,-2,one_alignment_only=True):
        s1,s2,score,start,end=a
        mm=sum(1 for x,y in zip(s1[start:end],s2[start:end]) if x!=y)
        if mm<=max_mismatches: return True
    return False
out=open("/ESL/ESL_MPRA/Figure_5/regenerated_panels/PE_PSI_values.tsv","w")
out.write("sample\tincluded\texcluded\tPSI\n")
os.makedirs("/ESL/ESL_MPRA/Figure_5/regenerated_panels",exist_ok=True)
for f in sorted(os.listdir(rna_dir)):
    if not f.endswith(".fastq"): continue
    inc=exc=0
    for rec in SeqIO.parse(os.path.join(rna_dir,f),"fastq"):
        rs=str(rec.seq)
        if match(rs,exon_included): inc+=1
        elif match(rs,exon_excluded): exc+=1
    psi=inc/(inc+exc) if (inc+exc)>0 else float("nan")
    line=f"{f}\t{inc}\t{exc}\t{psi:.4f}\n"
    out.write(line); out.flush()
    print(line.strip())
out.close()
print("PE_PSI_DONE")
