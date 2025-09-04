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


if __name__== "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"", ["gtf_file=", "supertable_file=", "dbSNP_table_file=", "BLAT_psl_file=", "output_dir=", "output_prefix="])
        opts = dict(opts)
        print(opts)
        
        gtf_file = opts["--gtf_file"]
        supertable_file = opts["--supertable_file"]
        dbSNP_table_file = opts["--dbSNP_table_file"]
        BLAT_psl_file = opts["--BLAT_psl_file"]
        output_dir = opts["--output_dir"]
        
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
    with open("%s_mut_fullseq_dict.txt" % (output_path_prefix), "w") as f:
        f.write("\n".join(["\t".join(item) for item in mut_fullseq_dict.items()]) + "\n")
    
    studies_order = ["1000_genomes", "dbGaP_PopFreq", "TOPMED", "KOREAN", "SGDP_PRJ", "Qatari", "NorthernSweden", "Siberian", "TWINSUK", "TOMMO", \
    "ALSPAC", "GENOME_DK", "GnomAD", "GoNL", "Estonian", "Vietnamese", "Korea1K", "HapMap", "PRJEB36033", "HGDP_Stanford", "Daghestan", \
    "PAGE_STUDY", "Chileans", "MGP", "PRJEB37584", "GoESP", "ExAC", "GnomAD_exomes", "FINRISK", "PharmGKB", "PRJEB37766"]
    # There is some duplication of lines and names within the table file, so I am splitting on chromosome to make the search faster without dropping unique rows
    dbSNP_dict = defaultdict(list)  # chr -> [[all dbSNP info]]
    # Load dbSNP from table output file
    with open(dbSNP_table_file, "r") as f:
        header = f.readline()[1:].strip().split("\t")
        for line in f:
            # 2 additional header lines from concat, so skipping
            if line[0] == "#":
                continue
            var = line.strip().split("\t")
            dbSNP_chrom = var[0]
            # handle duplicated lines
            if var not in dbSNP_dict[dbSNP_chrom]:
                dbSNP_dict[dbSNP_chrom].append({h: v for h,v in zip(header, var)})
    
    # read in BLAT fullseq output to make sure all of supertable sequence is correct
    matches = []
    overlap_found = []
    overlap_range_found = []
    significant_overlap_found = []
    significant_overlap_range_found =[]
    total_supertable_line_matches = set()
    total_supertable_line_range_matches = set()
    break_flag = False
    dbSNP_output_keys = ['chrom', 'chromStart', 'chromEnd', 'name', 'ref', 'altCount', 'alts', 'shiftBases', 'freqSourceCount', \
    'minorAlleleFreq', 'majorAllele', 'minorAllele', 'maxFuncImpact', 'class', 'ucscNotes', '_dataOffset', '_dataLen', \
    'study_refAllele', 'study_altAllele', 'study_altAlleleFreq']
    
    # make header for output files
    for study in studies_order:
        with open("%s_dbSNP155_matches_%s.txt" % (output_path_prefix, study), "w") as f:
            f.write("\t".join(dbSNP_output_keys + ['full_seq', 'supertable_1_num']) + "\n")
    
    with open(BLAT_psl_file, "r") as f_BLAT:
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
                # get dbSNP variants at supertable_ref location
                print("supertable_matched_ref = ", supertable_matched_ref)
                print("chromosome = ", chromosome)
                print("start_BLAT = ", start_BLAT)
                print("end_BLAT = ", end_BLAT)
                found_chr_variants = dbSNP_dict[chromosome]
                
                supertable_sequence = wt_supertable_exons[supertable_matched_ref][1] + wt_supertable_exons[supertable_matched_ref][0] + wt_supertable_exons[supertable_matched_ref][2]
                
                # go through all dbSNP variants
                for curr_dbSNP_var in found_chr_variants:
                    # recast chromEnd and chromStart
                    curr_dbSNP_var['chromStart'] = int(curr_dbSNP_var['chromStart'])
                    curr_dbSNP_var['chromEnd'] = int(curr_dbSNP_var['chromEnd'])
                    
                    # ignore in case of insertions running over variable region
                    if curr_dbSNP_var['chromEnd'] > end_BLAT or curr_dbSNP_var['chromStart'] < start_BLAT + 1:
                        continue
                    
                    dbSNP_var_seq = list(supertable_sequence)  # initialize with supertable ref seq
                    print("curr_dbSNP_var = ", curr_dbSNP_var)
                    print("init dbSNP_var_seq = ", dbSNP_var_seq)
                    
                    # double check dbSNP ref matches the sequence
                    if strand == "-":
                        if curr_dbSNP_var['chromEnd'] - start_BLAT - len(curr_dbSNP_var['ref']) > 0:
                            get_var_region = "".join(dbSNP_var_seq[-(curr_dbSNP_var['chromEnd'] - start_BLAT):-(curr_dbSNP_var['chromEnd'] - start_BLAT - len(curr_dbSNP_var['ref']))])
                        else:
                            get_var_region = "".join(dbSNP_var_seq[-(len(curr_dbSNP_var['ref'])):])
                    else:
                        if curr_dbSNP_var['chromStart'] == start_BLAT:
                            get_var_region = "".join(dbSNP_var_seq[:len(curr_dbSNP_var['ref'])])
                        else:
                            get_var_region = "".join(dbSNP_var_seq[(curr_dbSNP_var['chromEnd'] - start_BLAT - len(curr_dbSNP_var['ref'])):(curr_dbSNP_var['chromEnd'] - start_BLAT)])
                    print("get_var_region = ", get_var_region)
                    
                    if rev(curr_dbSNP_var['ref']) != get_var_region and strand == "-":
                        print("supertable_matched_ref = ", supertable_matched_ref)
                        print("curr_dbSNP_var = ", curr_dbSNP_var)
                        print("supertable_sequence = ", supertable_sequence)
                        print("strand = ", strand)
                        print("dbSNP_var_seq = ", dbSNP_var_seq)
                        print("start_BLAT = ", start_BLAT)
                        print("end_BLAT = ", end_BLAT)
                        print("curr_dbSNP_var['ref'] = ", curr_dbSNP_var['ref'])
                        print("rev(curr_dbSNP_var['ref']) = ", rev(curr_dbSNP_var['ref']))
                        print("get_var_region = ", get_var_region)
                        print("curr_dbSNP_var['chromEnd'] - start_BLAT = ", curr_dbSNP_var['chromEnd'] - start_BLAT)
                        raise AssertionError("dbSNP ref nt not matching")
                    elif curr_dbSNP_var['ref'] != get_var_region and strand == "+":
                        print("start_BLAT = ", start_BLAT)
                        print("end_BLAT = ", end_BLAT)
                        print("get_var_region = ", get_var_region)
                        raise AssertionError("dbSNP ref nt not matching")
                    
                    """
                    # go through all alts in this line
                    # This is assuming char in col "ref". There are lines and studies where the minor and major alleles are reversed which is not considered here.
                    for alt in curr_dbSNP_var['alts'].split(",")[:-1]:
                        # go through each study
                        for study_num, curr_minor_allele_info in enumerate(zip(curr_dbSNP_var["minorAllele"].split(",")[:-1], curr_dbSNP_var["minorAlleleFreq"].split(",")[:-1])):
                            curr_minor_allele, curr_minor_allele_freq = curr_minor_allele_info
                            curr_study = studies_order[study_num]
                            if curr_minor_allele == alt:
                                # assign alt identity
                                if strand == "-":
                                    dbSNP_var_seq_modified = dbSNP_var_seq[:-(curr_dbSNP_var['chromEnd'] - start_BLAT - len(curr_dbSNP_var['ref']) + 1)] + [','] + list(rev(curr_minor_allele)) + [','] + dbSNP_var_seq[-(curr_dbSNP_var['chromEnd'] - start_BLAT - len(curr_dbSNP_var['ref'])):]
                                else:
                                    dbSNP_var_seq_modified = dbSNP_var_seq[:curr_dbSNP_var['chromEnd'] - start_BLAT - len(curr_dbSNP_var['ref'])] + [','] + list(curr_minor_allele) + [','] + dbSNP_var_seq[(curr_dbSNP_var['chromEnd'] - start_BLAT):]
                                dbSNP_var_seq_modified = "".join(dbSNP_var_seq_modified)  # recast to str
                                dbSNP_var_seq_final = dbSNP_var_seq_modified.replace(",", "")  # remove debug commas

                                # lookup if dbSNP variant is in the supertable
                                if dbSNP_var_seq_final in mut_fullseq_dict.keys():
                                    # save matched dbSNP variant for output
                                    # matched supertable ref
                                    curr_dbSNP_supertable_ref = mut_fullseq_dict[dbSNP_var_seq_final]
                                    
                                    # append match to output tsv
                                    with open("%s_dbSNP155_matches_%s.txt" % (output_path_prefix, curr_study), "a") as f:
                                        f.write("\t".join([str(curr_dbSNP_var[k]) for k in dbSNP_output_keys] + [dbSNP_var_seq_final, str(curr_dbSNP_supertable_ref)]) + "\n")
                    """
                            
                    for alt in curr_dbSNP_var['alts'].split(",")[:-1]:
                        # go through each study
                        for study_num, curr_allele_info in enumerate(zip(curr_dbSNP_var["majorAllele"].split(",")[:-1], curr_dbSNP_var["minorAllele"].split(",")[:-1], curr_dbSNP_var["minorAlleleFreq"].split(",")[:-1])):
                            curr_major_allele, curr_minor_allele, curr_minor_allele_freq = curr_allele_info
                            curr_study = studies_order[study_num]
                            print("curr_study = ", curr_study)
                            print("curr_allele_info = ", curr_allele_info)
                            print("curr_minor_allele == alt = ", curr_minor_allele == alt)
                            print("curr_major_allele == alt = ", curr_major_allele == alt)
                            print("curr_dbSNP_var['ref'] == curr_minor_allele = ", curr_dbSNP_var['ref'] == curr_minor_allele)
                            if curr_minor_allele == alt or (curr_major_allele == alt and curr_dbSNP_var['ref'] == curr_minor_allele):
                                print("passed")
                                # assign alt identity
                                if strand == "-" and len(curr_dbSNP_var['ref']) > 0:
                                    dbSNP_var_seq_modified = dbSNP_var_seq[:-(curr_dbSNP_var['chromEnd'] - start_BLAT - len(curr_dbSNP_var['ref']) + 1)] + [','] + list(rev(alt)) + [','] + dbSNP_var_seq[-(curr_dbSNP_var['chromEnd'] - start_BLAT - len(curr_dbSNP_var['ref'])):]
                                elif strand == "+" and len(curr_dbSNP_var['ref']) > 0:
                                    dbSNP_var_seq_modified = dbSNP_var_seq[:curr_dbSNP_var['chromEnd'] - start_BLAT - len(curr_dbSNP_var['ref'])] + [','] + list(alt) + [','] + dbSNP_var_seq[(curr_dbSNP_var['chromEnd'] - start_BLAT):]
                                elif strand == "-" and len(curr_dbSNP_var['ref']) == 0:
                                    print("HIT")
                                    dbSNP_var_seq_modified = dbSNP_var_seq[:-(curr_dbSNP_var['chromEnd'] - start_BLAT)] + [','] + list(rev(alt)) + [','] + dbSNP_var_seq[-(curr_dbSNP_var['chromEnd'] - start_BLAT - len(curr_dbSNP_var['ref'])):]
                                else:
                                    dbSNP_var_seq_modified = dbSNP_var_seq[:(curr_dbSNP_var['chromEnd'] - start_BLAT + 1)] + [','] + list(alt) + [','] + dbSNP_var_seq[(curr_dbSNP_var['chromEnd'] - start_BLAT):]
                                dbSNP_var_seq_modified = "".join(dbSNP_var_seq_modified)  # recast to str
                                dbSNP_var_seq_final = dbSNP_var_seq_modified.replace(",", "")  # remove debug commas
                                print("dbSNP_var_seq_final = ", dbSNP_var_seq_final)

                                # lookup if dbSNP variant is in the supertable
                                if dbSNP_var_seq_final in mut_fullseq_dict.keys():
                                    # save matched dbSNP variant for output
                                    # matched supertable ref
                                    curr_dbSNP_supertable_ref = mut_fullseq_dict[dbSNP_var_seq_final]
                                    
                                    if (curr_major_allele == alt and curr_dbSNP_var['ref'] == curr_minor_allele):
                                        curr_dbSNP_var["study_altAlleleFreq"] = str(1 - float(curr_minor_allele_freq))
                                        curr_dbSNP_var["study_refAllele"] = alt
                                        curr_dbSNP_var["study_altAllele"] = curr_dbSNP_var['ref']
                                    else:
                                        curr_dbSNP_var["study_altAlleleFreq"] = curr_minor_allele_freq
                                        curr_dbSNP_var["study_refAllele"] = curr_dbSNP_var['ref']
                                        curr_dbSNP_var["study_altAllele"] = alt
                                    # append match to output tsv
                                    with open("%s_dbSNP155_matches_%s.txt" % (output_path_prefix, curr_study), "a") as f:
                                        f.write("\t".join([str(curr_dbSNP_var[k]) for k in dbSNP_output_keys] + [dbSNP_var_seq_final, str(curr_dbSNP_supertable_ref)]) + "\n")
                            
    