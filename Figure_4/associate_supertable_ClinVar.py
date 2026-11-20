"""
Label ClinVar variants in supertable
"""

import getopt, sys, os
from collections import defaultdict
import pickle


def make_FASTA(seqnames, sequences, output_file_name):
    with open(output_file_name, "w") as f:
        for seqname, sequence in zip(seqnames, sequences):
            f.write(">%s\n" % (seqname))
            f.write(sequence + "\n")
    return output_file_name


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
        opts, args = getopt.getopt(sys.argv[1:],"", ["supertable_file=", "output_dir=", "output_prefix=", "ClinVar_VCF=", "BLAT_reference=", "BLAT_exe_path="])
        opts = dict(opts)
        print(opts)
        supertable_file = opts["--supertable_file"]
        output_dir = opts["--output_dir"]
        ClinVar_vcf = opts["--ClinVar_VCF"]
        BLAT_reference = opts["--BLAT_reference"]
        BLAT_exe_path = opts["--BLAT_exe_path"]
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        grouped_plots_dir = output_dir + "/grouped_plots/"
        if not os.path.exists(grouped_plots_dir):
            os.makedirs(grouped_plots_dir)
        output_prefix = opts["--output_prefix"]
        output_path_prefix = "%s/%s" % (output_dir, output_prefix)
    except:
        print("Failed to get opts")
        sys.exit(2)
    
    
    # Load ClinVar
    clinvar_dict = {}  # chr -> {pos : (ref, var, ID, INFO)}
    with open(ClinVar_vcf, "r") as f:
        for line in f:
            # skip header
            if line[0] == "#":
                continue
            var = line.strip().split("\t")
            curr_chr = "chr%s" % (var[0])
            if var[4] == ".":
                var[4] = ""
            elif "N" in var[4]:
                # skip variants with N's
                continue
            var_info_tuple = (var[3], var[4], var[2], var[7])
            if curr_chr not in clinvar_dict:
                clinvar_dict[curr_chr] = { int(var[1]) - 1 : [var_info_tuple] }
            elif int(var[1]) - 1 in clinvar_dict[curr_chr]:
                clinvar_dict[curr_chr][int(var[1]) - 1].append(var_info_tuple)
            else:
                clinvar_dict[curr_chr][int(var[1]) - 1] = [var_info_tuple]
    
    # Load in supertable and separate WT and variant sequences
    wt_supertable_dict = {}  # supertable reference -> seq_tuple
    wt_eventID_supertableRef_dict = {}  # supertable eventID -> supertable reference
    mut_supertable_dict = defaultdict(list)  # supertable wt reference -> list of mutant reference
    supertable_gene_snp = {}  # supertable reference -> gene + SNP + event_id
    gene_name_dict = {}
    exon_only_mut_supertable_ref_list = []
    intron_only_mut_supertable_ref_list = []
    both_exon_intron_mut_supertable_ref_list = []
    exac_geuvadis_refs = []
    sequence_name_list = []
    sequence_list = []
    all_wt_fullseq_dict = {}  # wt ref -> fullseq
    wt_event_equivalency_dict = {}  # wt event -> used wt ref
    used_fullseq_wt_dict = {}  # fullseq -> wt ref
    
    # First pass read through of supertable for all WT's
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
            source = var[13]
            # parse WT
            if var[11] == "none":
                wt_supertable_dict[curr_supertable_ref] = seq_tuple
                all_wt_fullseq_dict[curr_supertable_ref] = fullseq_supertable
                if event_id not in wt_eventID_supertableRef_dict:
                    wt_eventID_supertableRef_dict[event_id] = curr_supertable_ref
                supertable_gene_snp[curr_supertable_ref] = (gene_name, snp, event_id)
                # store sequences for getting correct coordinate
                if fullseq_supertable not in sequence_list:
                    sequence_name_list.append(var[0])
                    sequence_list.append(fullseq_supertable)
                    wt_event_equivalency_dict[event_id] = curr_supertable_ref
                    used_fullseq_wt_dict[fullseq_supertable] = curr_supertable_ref
                else:
                    wt_event_equivalency_dict[event_id] = used_fullseq_wt_dict[fullseq_supertable]
            gene_name_dict[curr_supertable_ref] = gene_name
            # collect Exac and Geuvadis source references
            if source == "exac" or source == "geuvadis":
                exac_geuvadis_refs.append(curr_supertable_ref)
    
    # make FASTA files for BLAT input
    make_FASTA(sequence_name_list, sequence_list, output_path_prefix + "_BLAT_fullseq_temp.fa")
    
    # call BLAT on supertable intron1 + exon + intron2
    print("%s/blat %s %s %s_BLAT_fullseq_temp.psl" % (BLAT_exe_path, BLAT_reference, output_path_prefix + "_BLAT_fullseq_temp.fa", output_path_prefix))
    # Commenting out the system call and ran separately because of memory issues on this AWS instance
    os.system("%s/blat %s %s %s_BLAT_fullseq_temp.psl" % (BLAT_exe_path, BLAT_reference, output_path_prefix + "_BLAT_fullseq_temp.fa", output_path_prefix))
    
    # read in BLAT fullseq output to make sure all of supertable sequence is correct
    ClinVar_parent_ref = defaultdict(list)  # WT_supertable_ref -> [clinvar_dict[chromosome][location]]
    with open("%s_BLAT_fullseq_temp.psl" % (output_path_prefix), "r") as f_BLAT:
        for line_psl in f_BLAT:
            var = line_psl.split()
            if len(var) != 21:
                continue
            if var[0] == "161":
                supertable_matched_ref = str(int(var[9]) + 1)
                chromosome = var[13]
                if "_" in chromosome or "K" in chromosome or "G" in chromosome:  # ignore contigs
                    continue
                strand = var[8]
                start_BLAT = int(var[15])
                end_BLAT = int(var[16])
                ClinVar_overlaps = [{'ClinVar_info': clinvar_dict[chromosome][location], "ClinVar_location": location} for location in clinvar_dict[chromosome].keys() if location >= start_BLAT and location <= end_BLAT]
                for ClinVar_overlap in ClinVar_overlaps:
                    ClinVar_overlap["supertable_overlap"] = (chromosome, strand, start_BLAT, end_BLAT)
                if len(ClinVar_overlaps) > 0:
                    ClinVar_parent_ref[supertable_matched_ref].append(ClinVar_overlaps)
    
    # Flatten the ClinVar matches which is a list of sublists for every key
    for key in ClinVar_parent_ref:
        ClinVar_parent_ref[key] = [item for sublist in ClinVar_parent_ref[key] for item in sublist]
    
    # Second pass read through of supertable for all variants
    mutation_positions = {}  # supertable ref -> [mut_index]
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
            # parse WT
            if snp != "none":
                curr_supertable_wt_ref = wt_event_equivalency_dict[event_id]
                wt_exon, wt_5p_intron, wt_3p_intron = wt_supertable_dict[curr_supertable_wt_ref]
                # Ignore annotated mutants but subsequence is the same as WT
                if fullseq_supertable == wt_5p_intron + wt_exon + wt_3p_intron:
                    continue
                # Count number of exon-only, intron-only, or both exon+intron mutations in supertable
                if exon != wt_exon and intron1 == wt_5p_intron and intron2 == wt_3p_intron:
                    exon_only_mut_supertable_ref_list.append(curr_supertable_ref)
                elif exon == wt_exon and (intron1 != wt_5p_intron or intron2 != wt_3p_intron):
                    intron_only_mut_supertable_ref_list.append(curr_supertable_ref)
                else:
                    both_exon_intron_mut_supertable_ref_list.append(curr_supertable_ref)
                # get mut positions
                mut_positions = [(i, chr_tup) for i, chr_tup in enumerate(zip(wt_5p_intron + wt_exon + wt_3p_intron, fullseq_supertable)) if chr_tup[0] != chr_tup[1]]
                mutation_positions[curr_supertable_ref] = mut_positions
                
                # group mutants by their WT reference number
                mut_supertable_dict[curr_supertable_wt_ref].append(curr_supertable_ref)
                supertable_gene_snp[curr_supertable_ref] = (gene_name, snp, event_id)
    
    print("Supertable's len(exon_only_mut_supertable_ref_list): ", len(exon_only_mut_supertable_ref_list))
    print("Supertable's len(intron_only_mut_supertable_ref_list): ", len(intron_only_mut_supertable_ref_list))
    print("Supertable's len(both_exon_intron_mut_supertable_ref_list): ", len(both_exon_intron_mut_supertable_ref_list))
    
    print("len(mut_supertable_dict) = ", len(mut_supertable_dict))
    print("sum([len(v) for v in mut_supertable_dict.values()]) = ", sum([len(v) for v in mut_supertable_dict.values()]))
    
    ClinVar_matched_dict = {}  # mut_ref -> ClinVar info
    clinvar_found = 0
    
    #break_flag = False
    for wt_ref, mut_refs in mut_supertable_dict.items():
        # check if exact mutation is a ClinVar mutation
        if wt_ref in ClinVar_parent_ref:
            #if break_flag:
            #    break
            for ClinVar_item in ClinVar_parent_ref[wt_ref]:
                for curr_SNP in ClinVar_item['ClinVar_info']:
                    supertable_chr, supertable_strand, supertable_start, supertable_end = ClinVar_item['supertable_overlap']
                    ClinVar_location = ClinVar_item['ClinVar_location']
                    REF, VAR, ID, INFO = curr_SNP
                    len_var_ref_diff = len(VAR) - len(REF)
                    if supertable_strand == "-":
                        #print("negative_strand")
                        expected_mut_pos = [(supertable_end - ClinVar_location - 1, (rev(REF), rev(VAR)))]
                        #break_flag = True
                        #break
                    else:
                        #print("positive_strand")
                        expected_mut_pos = [(ClinVar_location - supertable_start, (REF, VAR))]
                        #break_flag = True
                        #break
                    for mut_ref in mut_refs:
                        #print("ClinVar_item = ", ClinVar_item)
                        #print("mut_ref = ", mut_ref)
                        mut_ref_mut_positions = mutation_positions[mut_ref]
                        if mut_ref_mut_positions == expected_mut_pos:
                            #print("FOUND mut_ref = ", mut_ref)
                            clinvar_found += 1
                            ClinVar_matched_dict[mut_ref] = {"ClinVar_item": ClinVar_item, "wt_ref": wt_ref, "mut_ref_mut_positions": mut_ref_mut_positions, "gene": gene_name_dict[wt_ref]}
                            #break_flag = True
                            break
                    #break
    
    print("clinvar_found = ", clinvar_found)
    
    # output ClinVar matched mutants
    with open(output_path_prefix + "_ClinVar_matched_dict.p", "wb") as f:
        pickle.dump(ClinVar_matched_dict, f)
    
    with open(output_path_prefix + "_ClinVar_matched_in_supertable.txt", "w") as f:
        f.write("mut_ref\twt_ref\tgene\tmut_ref_mut_positions\tClinVar_item\n")
        for mut_ref, info_dict in ClinVar_matched_dict.items():
            f.write("\t".join([mut_ref, info_dict["wt_ref"], info_dict["gene"], str(info_dict["mut_ref_mut_positions"]), str(info_dict["ClinVar_item"])]) + "\n")
