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


if __name__== "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"", ["gtf_file=", "supertable_file=", "ExAC_vcf=", "BLAT_reference=", "BLAT_exe_path=", "output_dir=", "output_prefix="])
        opts = dict(opts)
        print(opts)
        
        gtf_file = opts["--gtf_file"]
        supertable_file = opts["--supertable_file"]
        ExAC_vcf = opts["--ExAC_vcf"]
        output_dir = opts["--output_dir"]
        BLAT_reference = opts["--BLAT_reference"]
        BLAT_exe_path = opts["--BLAT_exe_path"]
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path_prefix = "%s/%s" % (output_dir, opts["--output_prefix"])
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
            transcript_info = dict([ele.split(" ") for ele in var[8].split("; ")])
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
    with open("%s_mut_fullseq_dict.txt" % (output_path_prefix), "w") as f:
        f.write("\n".join(["\t".join(item) for item in mut_fullseq_dict.items()]) + "\n")
        
    # make FASTA files for BLAT input
    make_FASTA(sequence_name_list, sequence_list, output_path_prefix + "_BLAT_fullseq_temp.fa")
    
    # call BLAT on supertable intron1 + exon + intron2
    print("%s/blat %s %s %s_BLAT_fullseq_temp.psl" % (BLAT_exe_path, BLAT_reference, output_path_prefix + "_BLAT_fullseq_temp.fa", output_path_prefix))
    # Commenting out the system call and ran separately because of memory issues on this AWS instance
    os.system("%s/blat %s %s %s_BLAT_fullseq_temp.psl" % (BLAT_exe_path, BLAT_reference, output_path_prefix + "_BLAT_fullseq_temp.fa", output_path_prefix))
    
    # load ExAC processed file
    #ExAC_dict = defaultdict(list)  # chr -> [[all ExAC_dict info]]
    ExAC_dict = defaultdict(dict)  # chr -> pos -> [all ExAC_dict info]
    ExAC_output_keys = ['chrom', 'pos', 'ID', 'ref', 'alt', 'QUAL', "FILTER", 'AC', 'AN', 'AF']
    with open(ExAC_vcf, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            var = line.strip().split("\t")
            ExAC_chrom = "chr" + var[0]
            curr_pos = int(var[1])
            info_dict = dict([item.split("=") if len(item.split("=")) == 2 else ['ID', item] for item in var[7].split(";")])
            curr_ExAC_info = dict(zip(ExAC_output_keys, var[:7] + [info_dict["AC"], info_dict["AN"], info_dict["AF"]]))
            # handle duplicated lines
            if curr_pos not in ExAC_dict[ExAC_chrom]:
                ExAC_dict[ExAC_chrom][curr_pos] = [curr_ExAC_info]
            elif var not in ExAC_dict[ExAC_chrom][curr_pos]:
                ExAC_dict[ExAC_chrom][curr_pos].append(curr_ExAC_info)
    print("ExAC_dict.keys() = ", ExAC_dict.keys())
    
    # read in BLAT fullseq output to make sure all of supertable sequence is correct
    matches = []
    overlap_found = []
    overlap_range_found = []
    significant_overlap_found = []
    significant_overlap_range_found =[]
    total_supertable_line_matches = set()
    total_supertable_line_range_matches = set()
    break_flag = False
    
    
    # make header for output file
    with open("%s_ExAC_matches.txt" % (output_path_prefix), "w") as f:
        f.write("\t".join(ExAC_output_keys + ['full_seq', 'supertable_1_num', 'study_alt', 'study_AF']) + "\n")
    
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
                # get ExAC variants at supertable_ref location
                print("supertable_matched_ref = ", supertable_matched_ref)
                print("chromosome = ", chromosome)
                print("start_BLAT = ", start_BLAT)
                print("end_BLAT = ", end_BLAT)
                found_chr_variants = ExAC_dict[chromosome]
                found_chr_variants = [ExAC_dict[chromosome][pos] for pos in ExAC_dict[chromosome] if pos >= start_BLAT and pos <= end_BLAT]
                found_chr_variants = [item for sublist in found_chr_variants for item in sublist]  # flatten list of lists
                print("found_chr_variants = ", found_chr_variants)
                
                # compare sequences from supertable and REST API to make sure there is no discrepancy
                supertable_sequence = wt_supertable_exons[supertable_matched_ref][1] + wt_supertable_exons[supertable_matched_ref][0] + wt_supertable_exons[supertable_matched_ref][2]
                
                # go through all ExAC variants
                for curr_ExAC_var in found_chr_variants:
                    # recast chromEnd and chromStart
                    print("curr_ExAC_var = ", curr_ExAC_var)
                    curr_ExAC_var['pos'] = int(curr_ExAC_var['pos'])
                    chromEnd = curr_ExAC_var['pos'] + len(curr_ExAC_var['ref'])
                    
                    # ignore in case of insertions running over variable region
                    if curr_ExAC_var['pos'] > end_BLAT + 1 or curr_ExAC_var['pos'] < start_BLAT + 1 or chromEnd - start_BLAT > 161 + 1:
                        continue
                    
                    ExAC_var_seq = list(supertable_sequence)  # initialize with supertable ref seq
                    print("curr_ExAC_var = ", curr_ExAC_var)
                    print("init ExAC_var_seq = ", ExAC_var_seq)
                    
                    # double check ExAC ref matches the sequence
                    if strand == "-":
                        if chromEnd - start_BLAT - 1 - len(curr_ExAC_var['ref']) > 0:
                            get_var_region = "".join(ExAC_var_seq[-(chromEnd - start_BLAT - 1):-(chromEnd - start_BLAT - 1 - len(curr_ExAC_var['ref']))])
                        else:
                            get_var_region = "".join(ExAC_var_seq[-(len(curr_ExAC_var['ref'])):])
                    else:
                        if curr_ExAC_var['pos'] == start_BLAT:
                            get_var_region = "".join(ExAC_var_seq[:len(curr_ExAC_var['ref'])])
                        else:
                            get_var_region = "".join(ExAC_var_seq[(curr_ExAC_var['pos'] - start_BLAT - 1):(curr_ExAC_var['pos'] - start_BLAT - 1 + len(curr_ExAC_var['ref']))])
                    print("get_var_region = ", get_var_region)
                    
                    if rev(curr_ExAC_var['ref']) != get_var_region and strand == "-":
                        print("supertable_matched_ref = ", supertable_matched_ref)
                        print("curr_ExAC_var = ", curr_ExAC_var)
                        print("strand = ", strand)
                        print("ExAC_var_seq = ", ExAC_var_seq)
                        print("start_BLAT = ", start_BLAT)
                        print("end_BLAT = ", end_BLAT)
                        print("curr_ExAC_var['ref'] = ", curr_ExAC_var['ref'])
                        print("rev(curr_ExAC_var['ref']) = ", rev(curr_ExAC_var['ref']))
                        print("get_var_region = ", get_var_region)
                        print("chromEnd - start_BLAT = ", chromEnd - start_BLAT)
                        raise AssertionError("ExAC ref nt not matching")
                    elif curr_ExAC_var['ref'] != get_var_region and strand == "+":
                        print("start_BLAT = ", start_BLAT)
                        print("end_BLAT = ", end_BLAT)
                        print("get_var_region = ", get_var_region)
                        print("chromEnd - start_BLAT = ", chromEnd - start_BLAT)
                        print("curr_ExAC_var['pos'] = ", curr_ExAC_var['pos'])
                        raise AssertionError("ExAC ref nt not matching")
                    
                    for curr_alt, curr_AF in zip(curr_ExAC_var['alt'].split(","), curr_ExAC_var['AF'].split(",")):
                        # assign alt identity
                        if strand == "-":
                            #ExAC_var_seq_modified = ExAC_var_seq[:-(chromEnd - start_BLAT - len(curr_ExAC_var['ref']))] + [','] + list(rev(curr_alt)) + [','] + ExAC_var_seq[-(chromEnd - start_BLAT - 1 - len(curr_ExAC_var['ref'])):]
                            #-(chromEnd - start_BLAT - 1):-(chromEnd - start_BLAT - 1 - len(curr_ExAC_var['ref']))
                            ExAC_var_seq_modified = ExAC_var_seq[:-(chromEnd - start_BLAT - 1)] + [','] + list(rev(curr_alt)) + [','] + ExAC_var_seq[-(chromEnd - start_BLAT - 1 - len(curr_ExAC_var['ref'])):]
                        else:
                            ExAC_var_seq_modified = ExAC_var_seq[:(curr_ExAC_var['pos'] - start_BLAT - 1)] + [','] + list(curr_alt) + [','] + ExAC_var_seq[(chromEnd - start_BLAT - 1):]
                        ExAC_var_seq_modified = "".join(ExAC_var_seq_modified)  # recast to str
                        print("ExAC_var_seq edited = ", ExAC_var_seq_modified)
                        ExAC_var_seq_final = ExAC_var_seq_modified.replace(",", "")  # remove debug commas
                        print("ExAC_var_seq final = ", ExAC_var_seq_final)
                        
                        # lookup if ExAC variant is in the supertable
                        if ExAC_var_seq_final in mut_fullseq_dict.keys():
                            # save matched ExAC variant for output 
                            
                            # matched supertable ref
                            curr_ExAC_supertable_ref = mut_fullseq_dict[ExAC_var_seq_final]
                            if curr_AF[:3] != "AF=":
                                curr_AF = "AF=" + curr_AF
                            
                            # append match to output tsv
                            with open("%s_ExAC_matches.txt" % (output_path_prefix), "a") as f:
                                f.write("\t".join([str(curr_ExAC_var[k]) for k in ExAC_output_keys] + [ExAC_var_seq_final, str(curr_ExAC_supertable_ref), curr_alt, curr_AF]) + "\n")
                    
                
    