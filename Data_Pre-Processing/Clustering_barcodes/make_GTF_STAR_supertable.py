"""
Making GTF file for STAR alignment of ESL MPRA.

Version: 0.0.1
Author: Angela M Yu, Samantha E. Koplik, 2020-2022

Copyright (C) 2020-2022 Angela M Yu, Samantha E. Koplik.
All rights reserved.
Distributed under the terms of the GNU General Public License, see 'LICENSE'.
"""

import os
from collections import defaultdict
import sys, getopt
from Bio import SeqIO

if __name__== "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"", ["supertable=","STAR_fasta_file=", "output_dir=", "output_prefix="])
        opts = dict(opts)
        print(opts)
        supertable_file = opts["--supertable"]
        output_dir = opts["--output_dir"]
        STAR_fasta_file=opts["--STAR_fasta_file"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path_prefix = "%s/%s" % (output_dir, opts["--output_prefix"])
    except:
        print("Failed to get opts")
        sys.exit(2)
    
    # define expected shared sequences in RNA-seq amplicons
    shared_5p_RNA_seq_trimmed = "CTTCAAGATCCGCCACAACATCGAGGTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA"
    shared_3p_RNA_seq = "AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAAAAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCCAAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTGTGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTAGGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTCTGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAATTCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGGACTGATAGTAAGGCCCATTACCTGC"
    
    # store parsed sequences
    agilent_barcode_seqs_dict = {}  # agilent seq -> barcode from DNA-seq
    agilent_seqID_desc_dict = {} # agilent seq -> [index_1_count, seqID, desc]
    output_order = []
    agilent_WT_seqs = set()  # for looking up close matches
    agilent_WT_seqs_ref = {}
    agilent_ref_to_seq = {}
    agilent_seq_lens = {}  # index 1 -> (len(AgilentIntron1), len(AgilentExon), len(AgilentIntron2))
    
    print("Loading supertable sequences")
    with open(supertable_file, "r") as f:
        f.readline()  # throw away header
        for line in f:
            row = line.strip().split("\t")
            index_0_count = row[0]
            index_1_count = str(int(index_0_count) + 1)
            eventID=row[3]
            exonID=row[5]
            geneName=row[6]
            snp=row[11]
            AgilentIntron1=row[7]
            AgilentIntron2=row[8]
            AgilentExon=row[4]
            desc_source=row[13]
            AgilentSequence = AgilentIntron1 + AgilentExon + AgilentIntron2
            SequenceID='Event ID:'+ eventID + '|' + 'Exon ID:' + exonID + '|' +\
            'Gene Name:'+ geneName + '|' + 'SNP:' + snp
            Description=desc_source
            # Load variables and account for duplicate sequences in supertable
            if AgilentSequence not in agilent_seqID_desc_dict:
                agilent_barcode_seqs_dict[AgilentSequence] = []
                agilent_seqID_desc_dict[AgilentSequence] = [(index_1_count, SequenceID, Description)]
                output_order.append(AgilentSequence)
                agilent_seq_lens[index_1_count] = (len(AgilentIntron1), len(AgilentExon), len(AgilentIntron2))
            else:
                agilent_seqID_desc_dict[AgilentSequence].append((index_1_count, SequenceID, Description))
            if snp == "none":
                if AgilentSequence not in agilent_WT_seqs_ref:
                    agilent_WT_seqs.add(AgilentSequence)
                    agilent_WT_seqs_ref[AgilentSequence] = index_1_count
            agilent_ref_to_seq[index_1_count] = AgilentSequence
    
    max_index_1_count = index_1_count
    
    print("agilent_barcode_seqs_dict length: ", len(agilent_barcode_seqs_dict))
    print("Few keys of agilent_barcode_seqs_dict: ", list(agilent_barcode_seqs_dict.keys())[:10])
    
    # GTF needs exon boundary locations, needs to be in this order based on legacy code downstream
    # 1. Agilent exon
    # 2. Citrine exon 1 based on sequencing read = 1 - 25
    # 3. Citrine exon 2 based on sequencing read = 872 - 1115
    citrine_exon1_GTF = ["ESL", "exon", "1", "25", ".", "+", ".", "."]
    citrine_exon2_GTF = ["ESL", "exon", "872", "1115", ".", "+", ".", "."]
    agilent_GTF_list1 = ["ESL", "exon"]
    agilent_GTF_list2 = [".", "+", ".", "."]
    with open(output_path_prefix+".gtf", "w") as f:
        with open(STAR_fasta_file, "r") as fasta:
            recs = SeqIO.parse(fasta,"fasta")
            for rec in recs:
                sequence = str(rec.seq)
                ID=rec.id
                ID_name=ID.split('_')
                if ID_name[0] in agilent_seq_lens.keys():
                    a_exon_5_pos = len(shared_5p_RNA_seq_trimmed) + agilent_seq_lens[ID_name[0]][0] + 1
                    a_exon_3_pos = len(shared_5p_RNA_seq_trimmed) + agilent_seq_lens[ID_name[0]][0] + agilent_seq_lens[ID_name[0]][1]
                    f.write("\t".join([ID] + agilent_GTF_list1 + [str(a_exon_5_pos), str(a_exon_3_pos)] + agilent_GTF_list2) + "\n")
                    f.write("\t".join([ID] + citrine_exon1_GTF) + "\n")
                    f.write("\t".join([ID] + citrine_exon2_GTF) + "\n")
    
    