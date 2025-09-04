# -*- coding: utf-8 -*-
"""
filter_RNA_from_DNA_barcode_clusters.py

Filter RNA-seq FASTQ's against known DNA barcode clusters. Should use adapter trimmed reads.
DNA reference barcodes will be assumed as last 20 nt in FASTA file "--DNA_barcode_fasta_file".
The filtered FASTQ's will be later mapped with START.
"""

import sys, getopt, os
import gzip
from Bio import SeqIO, pairwise2
from collections import defaultdict
import multiprocess as mp  # multiprocessing has issues with SeqIO
import time
import gc

CONST_NT_MAP = ['A', 'C', 'G', 'T']


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


def get_hamming_neighbor_1_all(seq, seq_map, start_r, end_r) :
    matches = []
    for i in range(start_r, end_r) :
        for base1 in CONST_NT_MAP :
            mut_seq = seq[:i] + base1 + seq[i+1:]
            if mut_seq in seq_map :
                matches.append(mut_seq)  # will only return all found
    return matches


def find_barcode_match(R1_input_data, R2_input_data, UMI_input_data):
    # Get barcode from current R2
    seq_R2 = R2_input_data[1]
    result = None
    
    if "N" in seq_R2:
        return result
        
    seq_R2 = rev(seq_R2)  # change to forward direction
    score = []
    start_pos = []
    end_pos = []
    
    for alignment in pairwise2.align.localms(R2_5_prime_shared_forward, seq_R2, 1, -1, -1, -1):
        score.append(alignment[2])
        # save end positions in the context of seq_R2 and not of the alignment which can have gaps
        aligned_R2_part = alignment[1][:alignment[4]].replace("-", "")
        end_pos.append(len(aligned_R2_part))
        aligned_R2_part = alignment[1][:alignment[3]].replace("-", "")
        start_pos.append(len(aligned_R2_part))
    del alignment

    max_score = max(score)
    if max_score >= round(len_R2_5_prime_shared_forward * 0.9): 
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
        barcode=seq_R2[closest_end_positions[0]:]
        if len(barcode) < 20:
            return result
        else:
            barcode = barcode[:20]
            # Check current R2 barcode against reference
            if barcode in DNA_barcode_dict:
                result = (barcode, R1_input_data, R2_input_data, UMI_input_data)
            else:
                mismatch_matches = get_hamming_neighbor_1_all(barcode, DNA_barcode_dict, 0, len(barcode))
                if len(mismatch_matches) == 1:
                    result = (mismatch_matches[0], R1_input_data, R2_input_data, UMI_input_data)
        
        del seq_R2, barcode
        del score, start_pos, end_pos, max_score
        del max_indices, max_end_pos, max_start_pos
        del closest_end_position_range, closest_end_indices, closest_end_positions
        del closest_start_position_range, closest_start_indices, closest_start_positions

        # wait 1e-4 second in case the parent is backlogged
        #time.sleep(1e-4)
        
        return result
   
    # wait 1e-4 second in case the parent is backlogged
    #time.sleep(1e-4)
    
    del seq_R2
    del score, start_pos, end_pos
    return result


def find_barcode_match_helper(args):
    """
    Helper for find_barcode_match() to unpack arguments.
    """
    return find_barcode_match(*args)


def write_append_FASTQ(output_fastq_file, seq_record, out_type = "a"):
    with open(output_fastq_file, out_type) as f_output:
        f_output.write("\n".join([seq_record[0], seq_record[1], "+", seq_record[2]]) + "\n")
    del f_output
    return


def FASTQ_iterator(file_handle):
    while True:
        read_name = file_handle.readline().rstrip()
        if not read_name:
            break
        seq = file_handle.readline().rstrip()
        _ = file_handle.readline()
        qual = file_handle.readline().rstrip()
        yield (read_name, seq, qual)


def sizeof_fmt(num, suffix='B'):
    """ by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def gzip_and_rm_barcodes(barcode):
    for file_name in output_barcode_keys[barcode]:
        with open(file_name, 'rb') as f_in:
            with gzip.open(file_name + ".gz", 'wb') as f_out:
                f_out.writelines(f_in)
        os.remove(file_name)
    return

            
if __name__== "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"", ["read1_fastq=", "read2_fastq=", "UMI_fastq=", "DNA_barcode_fasta_file=", "output_dir=", "output_prefix=", "p="])
        opts = dict(opts)
        print(opts)
        read1_fastq_file = opts["--read1_fastq"]
        read2_fastq_file = opts["--read2_fastq"]
        UMI_fastq_file = opts["--UMI_fastq"]
        DNA_barcode_fasta_file = opts["--DNA_barcode_fasta_file"]
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
    
    gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
    
    # load DNA barcodes
    print("Loading DNA barcodes")
    DNA_barcode_dict = {}  # barcode --> reference number
    DNA_barcode_fasta_iterator = SeqIO.parse(DNA_barcode_fasta_file, "fasta")
    for fasta_data in DNA_barcode_fasta_iterator:
        barcode = str(fasta_data.seq)[-20:]
        reference_name = str(fasta_data.id)
        DNA_barcode_dict[barcode] = reference_name
    del DNA_barcode_fasta_iterator, DNA_barcode_fasta_file
    del reference_name

    # Parse through RNA-seq fastq.gz files for R1, R2, and index
    # Need to open all three as iterators at the same time to avoid memory issues + parse in order
    # Likewise, need to output all 3 in order
    # Stacking these with statements is ugly, but gets the job done...
    
    # name all file output handles
    output_barcode_keys = {}
    for key in DNA_barcode_dict.keys():
        f1_barcode_out = "%s_%s_R1.fastq" % (output_path_prefix, key)
        f2_barcode_out = "%s_%s_R2.fastq" % (output_path_prefix, key)
        fUMI_barcode_out = "%s_%s_UMI.fastq" % (output_path_prefix, key)
        output_barcode_keys[key] = (f1_barcode_out, f2_barcode_out, fUMI_barcode_out)
    del f1_barcode_out, f2_barcode_out, fUMI_barcode_out
    del key
    
    R2_5_prime_shared_forward = "ACTGATAGTAAGGCCCATTACCTGC"
    len_R2_5_prime_shared_forward = len(R2_5_prime_shared_forward)
    
    barcode_counts = defaultdict(int)  # barcode --> # found in RNA-seq
    
    p = mp.Pool(processes)
    
    count = 0
    with gzip.open(read1_fastq_file, "rt") as f_R1_input:
        with gzip.open(read2_fastq_file, "rt") as f_R2_input:
            with gzip.open(UMI_fastq_file, "rt") as f_UMI_input:
                R1_input_iterator = FASTQ_iterator(f_R1_input)
                R2_input_iterator = FASTQ_iterator(f_R2_input)
                UMI_input_iterator = FASTQ_iterator(f_UMI_input)
                
                for result in p.imap(find_barcode_match_helper, zip(R1_input_iterator, R2_input_iterator, UMI_input_iterator), chunksize=1):  # iterate results as available
                    if result is not None:
                        barcode, R1_input_data, R2_input_data, UMI_input_data = result
                        output_R1_fastq_file, output_R2_fastq_file, output_UMI_fastq_file = output_barcode_keys[barcode]
                        barcode_counts[barcode] += 1
                        count += 1
                        
                        if barcode_counts[barcode] == 1:
                            write_append_FASTQ(output_R1_fastq_file, R1_input_data, out_type = "w")
                            write_append_FASTQ(output_R2_fastq_file, R2_input_data, out_type = "w")
                            write_append_FASTQ(output_UMI_fastq_file, UMI_input_data, out_type = "w")
                        else:
                            write_append_FASTQ(output_R1_fastq_file, R1_input_data, out_type = "a")
                            write_append_FASTQ(output_R2_fastq_file, R2_input_data, out_type = "a")
                            write_append_FASTQ(output_UMI_fastq_file, UMI_input_data, out_type = "a")
                        del output_R1_fastq_file, output_R2_fastq_file, output_UMI_fastq_file
                        
                        if count % 2000000 == 0 or count == 1:
                            print("\ncount = ", count)
                            for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                                key= lambda x: -x[1])[:10]:
                                print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
                            del name, size
                            #gc.collect()
    
    
    # gzip FASTQ's
    """
    for barcode in barcode_counts.keys():
        for file_name in output_barcode_keys[barcode]:
            with open(file_name, 'rb') as f_in:
                with gzip.open(file_name + ".gz", 'wb') as f_out:
                    f_out.writelines(f_in)
            os.remove(file_name)
    """
    _ = p.imap(gzip_and_rm_barcodes, barcode_counts.keys(), chunksize=1)
    p.close()
    p.join()
    
    # write out reference counts
    with open(output_path_prefix + "_reference_barcode_counts.txt", "w") as f:
        reference_counts = {DNA_barcode_dict[k]: (k,v) for k, v in barcode_counts.items()}
        f.write("Reference\tbarcode\tcount\n")
        f.write("\n".join(["\t".join([k, v[0], str(v[1])]) for k,v in sorted(reference_counts.items())]) + "\n")