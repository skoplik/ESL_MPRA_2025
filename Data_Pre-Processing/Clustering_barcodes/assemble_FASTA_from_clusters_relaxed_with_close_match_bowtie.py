"""
assemble_FASTA_from_clusters.py

Parse barcode clusters' consensus sequences
"""

import os
import sys
import getopt
from Bio import pairwise2
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
import multiprocessing as mp
import Levenshtein


"""
reverse complement, taken from R2D2:
https://github.com/LucksLab/R2D2/blob/master/NAU.py
"""
def rev(s):
    #This section was taken from Cole's code
    nuc_table = { 'A' : 'T',
                'T' : 'A',
                'C' : 'G',
                'G' : 'C',
                'U' : 'A',
                'a' : 't',
                't' : 'a',
                'c' : 'g',
                'g' : 'c',
                'u' : 'a',  }
    sl = list(s)

    try:
        rsl = [nuc_table[x] for x in sl]
    except:
        print >> sys.stderr, "Error: adapter sequences must contain only A,C,G,T,U"
        exit(1)
    rsl.reverse()

    return ''.join(rsl)


def supertable_and_close_match_first_filter(line):
    var = line.strip().split("\t")
    # ignore barcodes with insufficient coverage
    if int(var[2]) < min_coverage:
        return None
    # ignore sequences that are too long or short
    if abs(len(var[1]) - 161) > 8:
        return None
    # change for only counting perfect matches
    if var[1] in agilent_barcode_seqs_dict:
        #agilent_barcode_seqs_dict[var[1]].append(var[0])  # should be only in master
        return ("supertable", var[0], var[1])
    else:
        # give back sequences that need to be aligned with bowtie
        return ("to_align", var[0], var[1])
    return None


if __name__== "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"", ["sequence_clusters=", "supertable=", "min_coverage=", "output_dir=", "output_prefix=", "shared_mismatch_allowed=", "p="])
        opts = dict(opts)
        print(opts)
        sequence_clusters_file = opts["--sequence_clusters"]
        supertable_file = opts["--supertable"]
        min_coverage = int(opts["--min_coverage"])
        output_dir = opts["--output_dir"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path_prefix = "%s/%s" % (output_dir, opts["--output_prefix"])
        if "--shared_mismatch_allowed" in opts:
            if opts["--shared_mismatch_allowed"] == "-1":
                alignment_score_limit = -1  # turn off shared region search
            else:
                alignment_score_limit = 258 - int(opts["--shared_mismatch_allowed"])
        else:
            alignment_score_limit = 251  # default of 7 mismatches allowed in shared region
        if "--p" in opts:
            processes = int(opts["--p"])
        else:
            processes = 1
    except getopt.GetoptError as err:
        print("Failed to get opts")
        print(str(err))
        sys.exit(2)

    # define expected shared sequences in plasmid DNA
    #R1_5p_shared_seq = "TGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA"
    R1_5p_shared_seq = "TATAAAGCTATCTATATATAGCTATCTATATCTA"
    R1_3p_shared_seq = "AAAGTGAATCTTACTTTTGT"
    R2_shared_seq = "ACTGATAGTAAGGCCCATTACCTGC"
    
    # define expected share sequences in RNA-seq amplicons
    shared_5p_RNA_seq_trimmed = "CTTCAAGATCCGCCACAACATCGAGGTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA"
    shared_3p_RNA_seq = "AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAAAAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCCAAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTGTGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTAGGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTCTGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAATTCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGGACTGATAGTAAGGCCCATTACCTGC"
    
    # store parsed sequences
    agilent_barcode_seqs_dict = {}  # agilent seq -> barcode from DNA-seq
    agilent_seqID_desc_dict = {} # agilent seq -> [index_1_count, seqID, desc]
    output_order = []
    agilent_WT_seqs = set()  # for looking up close matches
    agilent_WT_seqs_ref = {}
    agilent_ref_to_seq = {}
    
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
    
    # write out agilent WT seqs into FASTA
    print("len(agilent_WT_seqs): ", len(agilent_WT_seqs))
    agilent_WT_records = []
    for AgilentSequence in agilent_WT_seqs:
        agilent_WT_records.append(SeqRecord(Seq(AgilentSequence), id=agilent_WT_seqs_ref[AgilentSequence]))
    SeqIO.write(agilent_WT_records, output_path_prefix + '_agilent_WT_seqs.fasta', 'fasta')
    
    # build bt2 index from agilent WT seqs
    os.system("bowtie2-build %s %s/agilent_WT_bt2" % (output_path_prefix + '_agilent_WT_seqs.fasta', output_dir))
    
    found_count = 0
    not_found_count = 0
    p = mp.Pool(processes)
    pool_count = 0
    print("Reading in sequence clusters = ", sequence_clusters_file)
    with open(sequence_clusters_file, "r") as f:
        f.readline()  # throw away header
        
        # multiprocess search
        with open(output_path_prefix + "_to_align.fasta", "w") as f_out:
            for result in p.imap(supertable_and_close_match_first_filter, f):  #, chunksize=100):  # iterate results as available
                if result is not None:
                    if result[0] == "supertable":
                        agilent_barcode_seqs_dict[result[2]].append(result[1])
                    elif result[0] == "to_align":
                        f_out.write(">%s\n%s\n" % (result[1], result[2]))
                pool_count += 1
                if pool_count % 100000 == 0:
                    print("pool_count = ", pool_count)
    p.close()
    p.join()
    
    print("Assembling output and writing out duplicates")
    record_list = []
    with open(output_path_prefix + "_duplicates.txt", "w") as f:
        for curr_seq in output_order:
            if len(agilent_seqID_desc_dict[curr_seq]) > 1:
                # duplicate sequence in supertable
                # will only use first match in FASTA, but keeping record of all here
                f.write("\t".join(agilent_barcode_seqs_dict[curr_seq]) + "\n")
                f.write("\n".join(["\t".join(items) for items in agilent_seqID_desc_dict[curr_seq]]) + "\n")
            
            # more than one barcode for this variant sequence
            if len(agilent_barcode_seqs_dict[curr_seq]) > 1:
                index_1_count, SequenceID, Description = agilent_seqID_desc_dict[curr_seq][0]
                for curr_barcode_count, curr_barcode in enumerate(agilent_barcode_seqs_dict[curr_seq]):
                    ReferenceSequence = shared_5p_RNA_seq_trimmed + curr_seq + shared_3p_RNA_seq + curr_barcode
                    record = SeqRecord(Seq(ReferenceSequence), id="%s_%s" % (index_1_count, curr_barcode_count+1), description=SequenceID+";"+Description)
                    record_list.append(record)
            elif len(agilent_barcode_seqs_dict[curr_seq]) == 1:  # only one barcode for this variant
                index_1_count, SequenceID, Description = agilent_seqID_desc_dict[curr_seq][0]
                curr_barcode = agilent_barcode_seqs_dict[curr_seq][0]
                ReferenceSequence = shared_5p_RNA_seq_trimmed + curr_seq + shared_3p_RNA_seq + curr_barcode
                record = SeqRecord(Seq(ReferenceSequence), id=index_1_count, description=SequenceID+";"+Description)
                record_list.append(record)
    
    # align with bowtie2 to determine close matches
    os.system("bowtie2 -f -p %s -x %s/agilent_WT_bt2 -U %s -S %s/output.sam" % (processes, output_dir, output_path_prefix + "_to_align.fasta", output_dir))
    os.system("samtools view -F 4 -F 16 -S %s/output.sam > %s/output_forward_mapped.sam" % (output_dir, output_dir))  # only keep mappings on forward strand
    
    # parse NM tag and only keep up to 8 edit distance away (~95% of 161 nt)
    close_matches = defaultdict(list)
    close_matches_partner = {}
    with open("%s/output_forward_mapped.sam" % (output_dir), "r") as f:
        for line in f:
            var = line.strip().split("\t")
            supertable_mapped = var[2]
            cluster_seq = var[9]
            # NM in the SAM file doesn't capture indels, so need to calculate
            NM = int([v.split("NM:i:")[1] for v in var if "NM:i:" in v][0])
            L_dist = Levenshtein.distance(agilent_ref_to_seq[supertable_mapped], cluster_seq)
            if NM <= 8 and L_dist <= 8:
                barcode = var[0]
                close_matches[cluster_seq].append(barcode)
                close_matches_partner[cluster_seq] = (agilent_ref_to_seq[supertable_mapped], str(NM), str(L_dist))

    # output close matches
    close_match_record_list = []
    index_1_count = max_index_1_count
    for close_match_seq, barcodes_list in close_matches.items():
        index_1_count = str(int(index_1_count) + 1)
        if len(barcodes_list) > 1:
            for curr_barcode_count, curr_barcode in enumerate(barcodes_list):
                ReferenceSequence = shared_5p_RNA_seq_trimmed + close_match_seq + shared_3p_RNA_seq + curr_barcode
                record = SeqRecord(Seq(ReferenceSequence), id="%s_%s" % (index_1_count, curr_barcode_count+1), description=close_match_seq)
                close_match_record_list.append(record)
        else:
            ReferenceSequence = shared_5p_RNA_seq_trimmed + close_match_seq + shared_3p_RNA_seq + barcodes_list[0]
            record = SeqRecord(Seq(ReferenceSequence), id=index_1_count, description=close_match_seq)
            close_match_record_list.append(record)
    
    print("Writing output")
    SeqIO.write(record_list, output_path_prefix + '_reference.fasta', 'fasta')
    SeqIO.write(close_match_record_list, output_path_prefix + '_close_matches_reference.fasta', 'fasta')
    SeqIO.write(record_list + close_match_record_list, output_path_prefix + '_reference_with_close_matches.fasta', 'fasta')
    
    with open(output_path_prefix + "_close_matches_partner.txt", "w") as f:
        f.write("close_match\tsupertable_WT\tsupertable_WT_ref_index1\tNM_edit_distance\tLevenshtein_distance\n")
        for k, v in close_matches_partner.items():
            close_match_seq, NM, L_dist = v
            supertable_wt_index = ",".join([ele[0] for ele in agilent_seqID_desc_dict[close_match_seq]])
            f.write("\t".join([k, close_match_seq, supertable_wt_index, NM, L_dist]) + "\n")
    