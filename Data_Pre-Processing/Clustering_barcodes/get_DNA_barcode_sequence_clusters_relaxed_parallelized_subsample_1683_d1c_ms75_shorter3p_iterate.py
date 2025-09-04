# -*- coding: utf-8 -*-
"""
get_DNA_barcode_sequence_clusters.py

Parse DNA-seq for expected R2 structure, pull out barcodes + sequences, cluster with starcode.
"""

import os
from collections import defaultdict
import sys, getopt
import gzip
from Bio import SeqIO, bgzf, pairwise2
# Used to convert the fastq stream into a file handle
from io import StringIO
import multiprocessing as mp
import numpy as np
import pickle


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


def rev_withN(s):
    #This section was taken from Cole's code
    nuc_table = { 'A' : 'T',
                'T' : 'A',
                'C' : 'G',
                'G' : 'C',
                'U' : 'A',
                'N' : 'N',
                'a' : 't',
                't' : 'a',
                'c' : 'g',
                'g' : 'c',
                'u' : 'a',
                'n' : 'n' }
    sl = list(s)

    try:
        rsl = [nuc_table[x] for x in sl]
    except:
        print >> sys.stderr, "Error: adapter sequences must contain only A,C,G,T,U,N"
        exit(1)
    rsl.reverse()

    return ''.join(rsl)


def findBarcodes_R2_shared(read1_data, read2_data):    
    seq_R1 = read1_data.seq
    seq_R2 = read2_data.seq

    #R2_5_prime_shared = "GCAGGTAATGGGCCTTACTAT"
    #R2_5_prime_shared_forward = "ACTGATAGTAAGGCCCATTACCTGC"
    R2_5_prime_shared_forward = "TAGTAAGGCCCATTACCTGC"
    
    result = None
    
    # Case: Allow 3 mismatches in 21 nt shared region in Read 2

    if "N" in seq_R2:
        return result

    seq_R2 = rev(seq_R2)  # change to forward direction
    #seq_R2 = rev_withN(seq_R2)  # change to forward direction
    
    # Printing out seq_R2 to debug output
    """
    if "N" not in seq_R2:
        result = (seq_R2)
    """
    
    score = []
    start_pos = []
    end_pos = []
    len_R2_5_prime_shared_forward = len(R2_5_prime_shared_forward)
    for alignment in pairwise2.align.localms(R2_5_prime_shared_forward, seq_R2, 1, -1, -1, -1):
        score.append(alignment[2])
        # save end positions in the context of seq_R2 and not of the alignment which can have gaps
        aligned_R2_part = alignment[1][:alignment[4]].replace("-", "")
        end_pos.append(len(aligned_R2_part))
        aligned_R2_part = alignment[1][:alignment[3]].replace("-", "")
        start_pos.append(len(aligned_R2_part))
    max_score = max(score) if len(score) >= 1 else -1
    #if max_score >= round(len_R2_5_prime_shared_forward * 0.9): 
    if max_score >= round(len_R2_5_prime_shared_forward * 0.75): 
        #if all sequences correct with allowable mismatches, save barcode in dict with seq id
        max_indices = [i for i, j in enumerate(score) if j == max_score]
        max_end_pos = [j for i, j in enumerate(end_pos) if i in max_indices]
        max_start_pos = [j for i, j in enumerate(start_pos) if i in max_indices]
        
        # pick position closest to expected between best matching alignment
        closest_end_position_range = min([abs(x - len_R2_5_prime_shared_forward) for x in max_end_pos])
        closest_end_indices, closest_end_positions = zip(*[(i,x) for i, x in enumerate(max_end_pos) if abs(x-len_R2_5_prime_shared_forward) == closest_end_position_range])
        closest_start_positions = [(i,x) for i, x in enumerate(max_start_pos) if i in closest_end_indices]
            
        # further pick position closest to expected start position 0
        closest_start_position_range = min([x for i, x in closest_start_positions])
        closest_start_indices, closest_start_positions = zip(*[(i,x) for i, x in enumerate(max_start_pos) if x == closest_start_position_range])
        closest_end_positions = [x for i, x in zip(closest_end_indices, closest_end_positions) if i in closest_start_indices]
            
        # Potentially can still have multiple equally good alignments after this filtering, using first
        barcode = seq_R2[closest_end_positions[0]:]
        if len(barcode) == 20:
            result = (barcode, seq_R1, seq_R2[:closest_end_positions[0]], read2_data.id)
        else:
            result = (seq_R1, seq_R2)
    else:
        result = (seq_R1, )
        
    return result


def findBarcodes_R2_shared_helper(args):
    """
    Helper for findBarcodes_R2_shared() to unpack arguments.
    """
    return findBarcodes_R2_shared(*args)


"""
Majority rule consensus calling from a list of sequences.
Tweaked from Johannes's APARENT code:
https://github.com/johli/aparent/blob/2156b2826e0afcc21a8c8dc041c38364d1008cd3/data/random_mpra/individual_library/doubledope/unprocessed_data/doubledope_dna_processing.ipynb
"""
def call_consensus_sequence(fullseq_count_dict):
    # Assuming all sequences are the same length
    len_sequence = max([len(s) for s in fullseq_count_dict.keys()])
    bp_map = np.zeros((len_sequence, 4))
    total_count = 0
    for sequence, count in fullseq_count_dict.items():
        total_count += count
        for j in range(0, len(sequence)) :
            if sequence[j] == 'A' :
                bp_map[j, 0] += count
            elif sequence[j] == 'C' :
                bp_map[j, 1] += count
            elif sequence[j] == 'G' :
                bp_map[j, 2] += count
            elif sequence[j] == 'T' :
                bp_map[j, 3] += count

    consensus_sequence = ''
    for j in range(0, len_sequence) :
        max_i = int(np.argmax(bp_map[j, :]))
        # exclude positions that don't have at least 25% of the counts
        if bp_map[j, max_i] < total_count/4.:
            continue
        # check for any ambiguous positions with enough coverage and throw out current consensus if not >= 60%
        if bp_map[j, max_i] < sum(bp_map[j,:]) * 0.6:
            return None
        if max_i == 0 :
            consensus_sequence += 'A'
        elif max_i == 1 :
            consensus_sequence += 'C'
        elif max_i == 2 :
            consensus_sequence += 'G'
        elif max_i == 3 :
            consensus_sequence += 'T'
    
    return consensus_sequence
    

def local_alignment_handling(search_seq, fullseq):
    score = []
    start_pos = []
    end_pos = []
    for alignment in pairwise2.align.localms(search_seq, fullseq, 1, -1, -1, -1):
        score.append(alignment[2])
        # save end positions in the context of seq_R2 and not of the alignment which can have gaps
        aligned_R2_part = alignment[1][:alignment[4]].replace("-", "")
        end_pos.append(len(aligned_R2_part))
        aligned_R2_part = alignment[1][:alignment[3]].replace("-", "")
        start_pos.append(len(aligned_R2_part))
    max_score = max(score) if len(score) > 0 else -1
    #if max_score >= round(len(search_seq) * 0.9): 
    if max_score >= round(len(search_seq) * 0.75): 
        #if all sequences correct with allowable mismatches, save barcode in dict with seq id
        max_indices = [i for i, j in enumerate(score) if j == max_score]
        max_end_pos = [j for i, j in enumerate(end_pos) if i in max_indices]
        max_start_pos = [j for i, j in enumerate(start_pos) if i in max_indices]
            
        # pick position closest to expected between best matching alignment
        closest_end_position_range = min([abs(x - len(search_seq)) for x in max_end_pos])
        closest_end_indices, closest_end_positions = zip(*[(i,x) for i, x in enumerate(max_end_pos) if abs(x-len(search_seq)) == closest_end_position_range])
        closest_start_positions = [(i,x) for i, x in enumerate(max_start_pos) if i in closest_end_indices]
        
        # further pick position closest to expected start position 0
        closest_start_position_range = min([x for i, x in closest_start_positions])
        closest_start_indices, closest_start_positions = zip(*[(i,x) for i, x in enumerate(max_start_pos) if x == closest_start_position_range])
        closest_end_positions = [x for i, x in zip(closest_end_indices, closest_end_positions) if i in closest_start_indices]
        return closest_start_positions, closest_end_positions
    return None, None
    
    
def remove_shared_regions(fullseq):
    """
    shared_5p_region = "TATAAAGCTATCTATATATAGCTATCTATATCTA"
    shared_3p_region = "AAAGTGAATCTTACTTTTGT"
    shared_region_R2 = "ACTGATAGTAAGGCCCATTACCTGC"
    """
    shared_5p_region = "TAAAGCTATCTATATATAGCTATCTATATCTA"
    #shared_3p_region = "AAAGTGAATCTTACTTTT"
    shared_3p_region = "AAAGTGAATCT"
    #shared_3p_region = "AAAGTGAA"
    shared_region_R2 = "TGATAGTAAGGCCCATTACCTGC"
    
    closest_start_positions, closest_end_positions = local_alignment_handling(shared_5p_region, fullseq)
    if closest_end_positions is None:
        #return
        trimmed_5p = fullseq
    else:
        trimmed_5p = fullseq[(closest_end_positions[0]):]
    
    closest_start_positions, closest_end_positions = local_alignment_handling(shared_3p_region, trimmed_5p)
    if closest_start_positions is None:
        #return
        #trimmed_5p_3p = trimmed_5p[:161]  # works in the mini library scenario
        trimmed_5p_3p = trimmed_5p  # works in the mini library scenario
        if len(trimmed_5p_3p) > 161:
            # soft trim because overhang may be too short for 18-mer search
            extra_3p = trimmed_5p[161:]
            closest_start_positions, closest_end_positions = local_alignment_handling(shared_3p_region[:len(extra_3p)], extra_3p)
            if closest_start_positions is not None:
                trimmed_5p_3p = trimmed_5p[:161] + extra_3p[:(closest_start_positions[0])]
            
            elif len(extra_3p) > 0:
                closest_start_positions, closest_end_positions = local_alignment_handling(shared_3p_region[:len(extra_3p)], extra_3p)
                if closest_start_positions is not None:
                    trimmed_5p_3p = trimmed_5p[:161] + extra_3p[:(closest_start_positions[0])]
                else:
                    extra_3p = trimmed_5p[(161-8):]
                    closest_start_positions, closest_end_positions = local_alignment_handling(shared_3p_region[:len(extra_3p)], extra_3p)
                    if closest_start_positions is not None:
                        trimmed_5p_3p = trimmed_5p[:(161-8)] + extra_3p[:(closest_start_positions[0])]
    else:
        trimmed_5p_3p = trimmed_5p[:(closest_start_positions[0])]
    
    return trimmed_5p_3p
    
    
if __name__== "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"", ["read1_fastq=", "read2_fastq=", "output_dir=", "output_prefix=", "p="])
        opts = dict(opts)
        print(opts)
        read1_fastq_file = opts["--read1_fastq"]
        read2_fastq_file = opts["--read2_fastq"]
        output_dir = opts["--output_dir"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path_prefix = "%s/%s" % (output_dir, opts["--output_prefix"])
        if "--p" in opts:
            processes = int(opts["--p"])
        else:
            processes = 1
    except:
        print("Failed to get opts")
        sys.exit(2)

    print("Reading in FASTQ's and parsing for barcodes")
    p = mp.Pool(processes)
    barcodes = defaultdict(str)
    barcode_count = defaultdict(int)
    barcode_fullsequence_dict = defaultdict(dict)  # unfortunately can't nest defaultdicts
    
    # For testing output
    with open(output_path_prefix + "_debugging.txt", "w") as f:
        f.write("Debug_output\n")
    
    unmapped_count = 0
    wrong_barcode_len = 0
    no_match_count = 0
    with gzip.open(read1_fastq_file, "rt") as f_R1_input:
        with gzip.open(read2_fastq_file, "rt") as f_R2_input:
            R1_input_iterator = SeqIO.parse(f_R1_input, "fastq")
            R2_input_iterator = SeqIO.parse(f_R2_input, "fastq")
            for result in p.imap(findBarcodes_R2_shared_helper, zip(R1_input_iterator, R2_input_iterator), chunksize=100):  # iterate results as available
                if result is not None:
                    if len(result) == 4:
                        barcode, seq_R1, seq_R2_part, key_R2 = result
                        fullseq = seq_R1
                        barcodes[key_R2] = barcode
                        barcode_count[barcode] += 1
                        if fullseq in barcode_fullsequence_dict[barcode]:
                            barcode_fullsequence_dict[barcode][fullseq] += 1
                        else:
                            barcode_fullsequence_dict[barcode][fullseq] = 1
                    elif len(result) == 2:
                        wrong_barcode_len += 1
                    elif len(result) == 1:
                        no_match_count += 1
                    else:
                        seq_R2 = result
                        # For testing output
                        with open(output_path_prefix + "_debugging.txt", "a") as f:
                            f.write(seq_R2 + "\n")
                else:
                    unmapped_count += 1
    p.close()
    p.join()
    print("unmapped_count = ", unmapped_count)
    print("wrong_barcode_len = ", wrong_barcode_len)
    print("no_match_count = ", no_match_count)
    
    with open(output_path_prefix+'_barcode_counts.txt', 'w') as f:
        for k, v in barcode_count.items():
            s=str(k)+"\t"+str(v)+"\n" 
            b=f.write(s)
            
    with open(output_path_prefix+'_barcodes_ids.txt', 'w') as f:
        for k, v in barcodes.items():
            s=str(k)+"\t"+str(v)+"\n" 
            b=f.write(s)
    
    # call starcode
    print("starcode -i %s -o %s -t %s --print-clusters -d 1 -c" % (output_path_prefix+'_barcode_counts.txt', output_path_prefix+"_barcode_clusters.txt", processes))
    os.system("starcode -i %s -o %s -t %s --print-clusters -d 1 -c" % (output_path_prefix+'_barcode_counts.txt', output_path_prefix+"_barcode_clusters.txt", processes))
    #os.system("starcode -i %s -o %s -t %s --print-clusters -r 1" % (output_path_prefix+'_barcode_counts.txt', output_path_prefix+"_barcode_clusters.txt", processes))

    # parse starcode barcode clusters
    print("Parsing starcode clusters")
    starcode_cluster_sequences = defaultdict(dict)  # unfortunately can't nest defaultdicts
    with open(output_path_prefix+"_barcode_clusters.txt", "r") as f:
        for line in f:
            barcode_consensus, coverage, barcodes_list = line.strip().split("\t")
            barcodes_cluster = barcodes_list.split(",")
            # parse through all reads matching this barcode cluster
            for barcode in barcodes_cluster:
                for fullseq, fullseq_count in barcode_fullsequence_dict[barcode].items():
                    if fullseq in starcode_cluster_sequences[barcode_consensus]:
                        starcode_cluster_sequences[barcode_consensus][fullseq] += fullseq_count
                    else:
                        starcode_cluster_sequences[barcode_consensus][fullseq] = fullseq_count
    
    # Save starcode_cluster_sequences
    with open(output_path_prefix + '_starcode_cluster_sequences.p', 'wb') as handle:
        pickle.dump(starcode_cluster_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Write out sequence consensus calls for each barcode cluster
    print("Writing out starcode consensus sequences")
    with open(output_path_prefix+"_barcode_clusters_consensus_seq.txt", "w") as f:
        f.write("barcode_consensus\tsequence_consensus\tcoverage\n")
        for barcode_consensus, fullseq_count_dict in starcode_cluster_sequences.items():
            fullseq_consensus = call_consensus_sequence(fullseq_count_dict)
            if fullseq_consensus is not None:
                trimmed_consensus = remove_shared_regions(fullseq_consensus)
                if trimmed_consensus is not None:
                    #if len(barcode_consensus) == 20 and len(trimmed_consensus) == 161:
                    if len(barcode_consensus) == 20:
                        f.write("%s\t%s\t%s\n" % (barcode_consensus, trimmed_consensus, sum(fullseq_count_dict.values())))
