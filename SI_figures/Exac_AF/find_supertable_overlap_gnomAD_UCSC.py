"""
find_supertable_overlap_coordinates.py 


"""

import sys, getopt
import os
from collections import defaultdict
import requests
import json
import gc


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


def make_FASTA(seqnames, sequences, output_file_name):
    with open(output_file_name, "w") as f:
        for seqname, sequence in zip(seqnames, sequences):
            f.write(">%s\n" % (seqname))
            f.write(sequence + "\n")
    return output_file_name


def curl_REST_gnomAD_UCSC(chromosome, start, end):
    get_chr_start_end_str = "chrom=%s;start=%s;end=%s" % (chromosome, start, end)  # chromosome starts with 'chr'
    response = requests.get('https://api.genome.ucsc.edu/getData/track?genome=hg38;track=gnomadGenomesVariantsV3_1_1;' + get_chr_start_end_str,)
    found_gnomAD_variants = json.loads(response.__dict__['_content'])['gnomadGenomesVariantsV3_1_1']
    response = requests.get('https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=%s;start=%s;end=%s' % (chromosome, start, end),)
    sequence = json.loads(response.__dict__['_content'])['dna'].upper()
    return found_gnomAD_variants, sequence


if __name__== "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"", ["gtf_file=", "supertable_file=", "BLAT_reference=", "BLAT_exe_path=", "output_dir=", "output_prefix="])
        opts = dict(opts)
        print(opts)
        
        gtf_file = opts["--gtf_file"]
        supertable_file = opts["--supertable_file"]
        output_dir = opts["--output_dir"]
        BLAT_reference = opts["--BLAT_reference"]
        BLAT_exe_path = opts["--BLAT_exe_path"]
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path_prefix = "%s/%s" % (output_dir, opts["--output_prefix"])
        """
        # small instance
        gtf_file = "/home/ec2-user/environment/Data/Sequences/Gencode_v26/gencode.v26.annotation.gff3"
        supertable_file = "/home/ec2-user/environment/Data/Sequences/supertable.tsv"
        output_dir = "/home/ec2-user/environment/Analysis/gnomAD"
        output_prefix = "gnomAD_Gencode_v26"
        output_path_prefix = "%s/%s" % (output_dir, output_prefix)
        BLAT_reference = "/home/ec2-user/environment/Data/Sequences/Gencode_v26/GRCh38.p10.genome.2bit"
        BLAT_exe_path = "/home/ec2-user/environment/src_download/UCSC_utilities/blat/"
        """
    except:
        print("Failed to get opts")
        sys.exit(2)
        
    # Load exons in GTF file
    exon_dict = defaultdict(list)
    exon_start_dict = defaultdict(list)
    exon_end_dict = defaultdict(list)
    transcript_dict = defaultdict(list)
    with open(gtf_file, "r") as f:
        for line in f:
            var = line.strip().split("\t")
            # only looking at exons
            if line[0] == "#":
                continue
            if var[2] != "exon":
                continue
            transcript_info = dict([ele.split("=") for ele in var[8].split(";")])
            transcript_id = transcript_info['transcript_id']
            gene_name = transcript_info['gene_name']
            exon_number = transcript_info['exon_number']
            
            # load lines that match exons
            exon_dict[",".join([var[0], var[3], var[4], var[6]])].append(var[:-1] + [transcript_info])
            exon_start_dict[",".join([var[0], var[3], var[6]])].append(var[:-1] + [transcript_info])
            exon_end_dict[",".join([var[0], var[4], var[6]])].append(var[:-1] + [transcript_info])
            
            # keep track of transcripts
            transcript_dict[transcript_id].append((gene_name, exon_number, var[0], var[3], var[4], var[6]))
    
    # Load in supertable and use BLAT find correct location in genome
    wt_supertable_exons = {}
    sequence_name_list = []
    sequence_list = []
    wt_eventID_supertableRef_dict = {}  # supertable eventID -> supertable reference
    wt_fullseq_supertable_dict = {}  # fullseq -> first supertable reference
    #wt_strand_dict = {}  # supertable ref -> strand
    
    with open(supertable_file, "r") as f:
        f.readline()  # throw out header
        for line in f:
            var = line.strip().split("\t")
            # skip mutants
            if var[11] != "none":
                continue
            #curr_supertable_ref = var[0]
            curr_supertable_ref = str(int(var[0]) + 1)
            event_id = var[3]
            exon = var[4]
            intron1 = var[7]
            intron2 = var[8]
            fullseq_supertable = intron1 + exon + intron2
            strand = event_id.split(":")[-1]
            #wt_strand_dict[curr_supertable_ref] = strand
            if event_id not in wt_eventID_supertableRef_dict:
                if fullseq_supertable not in wt_fullseq_supertable_dict.keys():
                    wt_fullseq_supertable_dict[fullseq_supertable] = curr_supertable_ref
                    wt_eventID_supertableRef_dict[event_id] = curr_supertable_ref
                else:
                    wt_eventID_supertableRef_dict[event_id] = wt_fullseq_supertable_dict[fullseq_supertable]
            if fullseq_supertable not in sequence_list:
                sequence_name_list.append(curr_supertable_ref)
                sequence_list.append(fullseq_supertable)
                wt_supertable_exons[curr_supertable_ref] = (exon, intron1, intron2)
    
    # Second pass read through of supertable for all variants
    HEK_found_wt = set()
    HEK_found_wt_variant_matches = {}
    HEK_found_variant_wt_matches = {}
    mutation_positions = {}  # supertable ref -> [mut_index]
    HEK_found_ClinVar = set()
    mut_supertable_dict = defaultdict(list)  # supertable wt reference -> list of mutant reference
    mut_fullseq_dict = {}  # fullseq of var -> supertable ref
    all_fullseq_supertable_refs_dict = defaultdict(list)  # full seq -> [supertable references]
    with open(supertable_file, "r") as f:
        f.readline()  # throw out header
        for line in f:
            var = line.split("\t")
            curr_supertable_ref = str(int(var[0]) + 1)
            event_id = var[3]
            gene_name = var[6]
            snp = var[11]
            exon = var[4]
            intron1 = var[7]
            intron2 = var[8]
            fullseq_supertable = intron1 + exon + intron2
            seq_tuple = (exon, intron1, intron2)
            all_fullseq_supertable_refs_dict[fullseq_supertable].append(curr_supertable_ref)
            # parse WT
            if snp != "none":
                # Ignore if variant fullseq already found
                if fullseq_supertable in mut_fullseq_dict.keys():
                    continue
                
                curr_supertable_wt_ref = wt_eventID_supertableRef_dict[event_id]
                wt_exon, wt_5p_intron, wt_3p_intron = wt_supertable_exons[curr_supertable_wt_ref]
                
                # Ignore annotated mutants but subsequence is the same as WT
                if fullseq_supertable == wt_5p_intron + wt_exon + wt_3p_intron:
                    continue
                mut_fullseq_dict[fullseq_supertable] = curr_supertable_ref
                
                # get mut positions
                mut_positions = [i for i, chr_tup in enumerate(zip(fullseq_supertable, wt_5p_intron + wt_exon + wt_3p_intron)) if chr_tup[0] != chr_tup[1]]
                mutation_positions[curr_supertable_ref] = mut_positions
                
                # group mutants by their WT reference number
                mut_supertable_dict[curr_supertable_wt_ref].append(curr_supertable_ref)
    
    # output mut_fullseq_dict
    #with open("%s_mut_fullseq_dict.txt" % (output_path_prefix), "w") as f:
    #    f.write("\n".join(["\t".join(item) for item in mut_fullseq_dict.items()]) + "\n")
        
    # make FASTA files for BLAT input
    make_FASTA(sequence_name_list, sequence_list, output_path_prefix + "_BLAT_fullseq_temp.fa")
    
    # call BLAT on supertable intron1 + exon + intron2
    print("%s/blat %s %s %s_BLAT_fullseq_temp.psl" % (BLAT_exe_path, BLAT_reference, output_path_prefix + "_BLAT_fullseq_temp.fa", output_path_prefix))
    # Commenting out the system call and ran separately because of memory issues on this AWS instance
    #os.system("%s/blat %s %s %s_BLAT_fullseq_temp.psl" % (BLAT_exe_path, BLAT_reference, output_path_prefix + "_BLAT_fullseq_temp.fa", output_path_prefix))
    
    # read in BLAT fullseq output to make sure all of supertable sequence is correct
    matches = []
    overlap_found = []
    overlap_range_found = []
    significant_overlap_found = []
    significant_overlap_range_found =[]
    total_supertable_line_matches = set()
    total_supertable_line_range_matches = set()
    break_flag = False
    gnomAD_output_keys = ['chrom', 'chromStart', 'chromEnd', 'name', 'ref', 'alt', 'FILTER', 'AC', 'AN', 'AF', 'rsId', 'genes', 'variation_type']
    
    # make header for output file
    with open("%s_gnomAD_matches.txt" % (output_path_prefix), "w") as f:
        f.write("\t".join(gnomAD_output_keys + ['full_seq', 'supertable_1_num']) + "\n")
    
    with open("%s_BLAT_fullseq_temp.psl" % (output_path_prefix), "r") as f_BLAT:
        for line_psl in f_BLAT:
            var = line_psl.split()
            
            if len(var) != 21:
                continue
            if var[0] == "161":
                supertable_matched_ref = var[9]
                chromosome = var[13]
                strand = var[8]
                start_BLAT = int(var[15])
                end_BLAT = int(var[16])
                len_intron1 = len(wt_supertable_exons[supertable_matched_ref][1])
                len_intron2 = len(wt_supertable_exons[supertable_matched_ref][2])
                # negative strand
                if strand == "-":
                    exon_start_BLAT = start_BLAT + len_intron2 + 1
                    exon_end_BLAT = end_BLAT - len_intron1
                else:  # positive strand
                    exon_start_BLAT = start_BLAT + len_intron1 + 1
                    exon_end_BLAT = end_BLAT - len_intron2
                
                # ignore non-"chr" chromosomes
                if chromosome[:3] != "chr" or "_" in chromosome:
                    continue
                # get gnomAD variants at supertable_ref location
                print("supertable_matched_ref = ", supertable_matched_ref)
                print("chromosome = ", chromosome)
                print("start_BLAT = ", start_BLAT)
                print("end_BLAT = ", end_BLAT)
                found_gnomAD_variants, gnomAD_sequence = curl_REST_gnomAD_UCSC(chromosome, start_BLAT, end_BLAT)
                print("gnomAD_sequence original = ", gnomAD_sequence)
                
                # compare sequences from supertable and REST API to make sure there is no discrepancy
                supertable_sequence = wt_supertable_exons[supertable_matched_ref][1] + wt_supertable_exons[supertable_matched_ref][0] + wt_supertable_exons[supertable_matched_ref][2]
                # revcomp to keep gnomAD_sequence in same orientation as supertable if negative strand
                gnomAD_sequence_corrected = gnomAD_sequence
                if strand == "-":
                    gnomAD_sequence_corrected = rev(gnomAD_sequence)
                # double check supertable seq and UCSC's sequence are the same
                if supertable_sequence != gnomAD_sequence_corrected:
                    print("supertable_matched_ref = ", supertable_matched_ref)
                    print("supertable_sequence = ", supertable_sequence)
                    print("gnomAD_sequence_corrected = ", gnomAD_sequence_corrected)
                    print("found_gnomAD_variants[0] = ", found_gnomAD_variants[0])
                    print("var = ", var)
                    print("chromosome = ", chromosome)
                    print("start_BLAT = ", start_BLAT)
                    print("end_BLAT = ", end_BLAT)
                    raise AssertionError("supertable sequence does not match UCSC's sequence at same genomic location")
                
                # go through all gnomAD variants
                for curr_gnomAD_var in found_gnomAD_variants:
                    
                    # ignore in case of insertions running over variable region
                    #if curr_gnomAD_var['chromEnd'] > end_BLAT or curr_gnomAD_var['chromStart'] < start_BLAT:
                    if curr_gnomAD_var['chromEnd'] > end_BLAT or curr_gnomAD_var['chromStart'] < start_BLAT + 1:
                        continue
                    
                    gnomAD_var_seq = list(gnomAD_sequence_corrected)  # initialize with gnomAD ref seq
                    print("curr_gnomAD_var = ", curr_gnomAD_var)
                    print("start_BLAT = ", start_BLAT)
                    print("end_BLAT = ", end_BLAT)
                    print("gnomAD_sequence_corrected = ", gnomAD_sequence_corrected)
                    
                    # double check gnomAD ref matches the sequence
                    if strand == "-":
                        if curr_gnomAD_var['chromEnd'] - start_BLAT - len(curr_gnomAD_var['ref']) > 0:
                            get_var_region = "".join(gnomAD_var_seq[-(curr_gnomAD_var['chromEnd'] - start_BLAT):-(curr_gnomAD_var['chromEnd'] - start_BLAT - len(curr_gnomAD_var['ref']))])
                        #elif curr_gnomAD_var['chromEnd'] - start_BLAT - len(curr_gnomAD_var['ref']) == 0:
                        #    get_var_region = "".join(gnomAD_var_seq[-(curr_gnomAD_var['chromEnd'] - start_BLAT):])
                        else:
                            get_var_region = "".join(gnomAD_var_seq[-(len(curr_gnomAD_var['ref'])):])
                    else:
                        if curr_gnomAD_var['chromStart'] == start_BLAT:
                            get_var_region = "".join(gnomAD_var_seq[:len(curr_gnomAD_var['ref'])])
                        else:
                            get_var_region = "".join(gnomAD_var_seq[(curr_gnomAD_var['chromEnd'] - start_BLAT - len(curr_gnomAD_var['ref'])):(curr_gnomAD_var['chromEnd'] - start_BLAT)])
                    print("get_var_region = ", get_var_region)
                    
                    if rev(curr_gnomAD_var['ref']) != get_var_region and strand == "-":
                        print("supertable_matched_ref = ", supertable_matched_ref)
                        print("curr_gnomAD_var = ", curr_gnomAD_var)
                        print("gnomAD_sequence = ", gnomAD_sequence)
                        print("strand = ", strand)
                        print("gnomAD_var_seq = ", gnomAD_var_seq)
                        print("start_BLAT = ", start_BLAT)
                        print("end_BLAT = ", end_BLAT)
                        print("curr_gnomAD_var['ref'] = ", curr_gnomAD_var['ref'])
                        print("rev(curr_gnomAD_var['ref']) = ", rev(curr_gnomAD_var['ref']))
                        print("get_var_region = ", get_var_region)
                        print("curr_gnomAD_var['chromEnd'] - start_BLAT = ", curr_gnomAD_var['chromEnd'] - start_BLAT)
                        raise AssertionError("gnomAD ref nt not matching")
                    elif curr_gnomAD_var['ref'] != get_var_region and strand == "+":
                        print("start_BLAT = ", start_BLAT)
                        print("end_BLAT = ", end_BLAT)
                        print("get_var_region = ", get_var_region)
                        raise AssertionError("gnomAD ref nt not matching")
                    # assign alt identity
                    if strand == "-":
                        gnomAD_var_seq = gnomAD_var_seq[:-(curr_gnomAD_var['chromEnd'] - start_BLAT - len(curr_gnomAD_var['ref']) + 1)] + [','] + list(rev(curr_gnomAD_var['alt'])) + [','] + gnomAD_var_seq[-(curr_gnomAD_var['chromEnd'] - start_BLAT - len(curr_gnomAD_var['ref'])):]
                    else:
                        gnomAD_var_seq = gnomAD_var_seq[:curr_gnomAD_var['chromEnd'] - start_BLAT - len(curr_gnomAD_var['ref'])] + [','] + list(curr_gnomAD_var['alt']) + [','] + gnomAD_var_seq[(curr_gnomAD_var['chromEnd'] - start_BLAT):]
                    gnomAD_var_seq = "".join(gnomAD_var_seq)  # recast to str
                    print("gnomAD_var_seq edited = ", gnomAD_var_seq)
                    gnomAD_var_seq = gnomAD_var_seq.replace(",", "")  # remove debug commas
                    print("gnomAD_var_seq final = ", gnomAD_var_seq)
                    
                    # lookup if gnomAD variant is in the supertable
                    if gnomAD_var_seq in mut_fullseq_dict.keys():
                        # save matched gnomAD variant for output 
                        """
                        >>> found_gnomAD_variants[1]
{'chrom': 'chr10', 'chromStart': 101775119, 'chromEnd': 101775120, 'name': '9c47360257c32ce300142fe826029352', 'score': 0, 'strand': '.', 'thickStart': 101775119, 'thickEnd': 101775120, 'reserved': '95,95,95', 'ref': 'C', 'alt': 'T', 'FILTER': 'PASS', 'AC': '1', 'AN': '152138', 'AF': '6.57298e-06', 'faf95': '0.00000', 'nhomalt': '0', 'rsId': 'rs577352323', 'genes': 'FGF8, LOC105378457, N/A', 'annot': 'other', 'variation_type': 'intron_variant,non_coding_transcript_variant,NMD_transcript_variant,downstream_gene_variant,TF_binding_site_variant', '_startPos': 101775120, '_displayName': 'chr10-101775120-C-T', '_dataOffset': '805141033881', '_dataLen': 1533}
                        """
                        # matched supertable ref
                        curr_gnomAD_supertable_ref = mut_fullseq_dict[gnomAD_var_seq]
                        
                        # append match to output tsv
                        with open("%s_gnomAD_matches.txt" % (output_path_prefix), "a") as f:
                            f.write("\t".join([str(curr_gnomAD_var[k]) for k in gnomAD_output_keys] + [gnomAD_var_seq, str(curr_gnomAD_supertable_ref)]) + "\n")
                
                
    