"""
find_supertable_overlap_MFASS.py

"""

import sys, getopt
import os
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt


def make_FASTA(seqnames, sequences, output_file_name):
    with open(output_file_name, "w") as f:
        for seqname, sequence in zip(seqnames, sequences):
            f.write(">%s\n" % (seqname))
            f.write(sequence + "\n")
    return output_file_name


def load_PSI_file(PSI_file):
    PSI_dict = {}
    with open(PSI_file, "r") as f:
        f.readline()
        for line in f:
            var = line.split()
            supertable_ref = str(int(var[0]) - 1)  # STAR reference is 1-index and supertable is 0-index
            PSI_dict[supertable_ref] = float(var[1])
    return PSI_dict


if __name__== "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"", ["supertable_file=", "to_overlap_file=", "Rep1_PSI_file=", "Rep2_PSI_file=", "BLAT_reference=", "BLAT_exe_path=", "output_dir=", "output_prefix=", "filter_MFASS_high_replicability_smn1"])
        opts = dict(opts)
        print(opts)
        
        supertable_file = opts["--supertable_file"]
        to_overlap_file = opts["--to_overlap_file"]
        output_dir = opts["--output_dir"]
        BLAT_reference = opts["--BLAT_reference"]
        BLAT_exe_path = opts["--BLAT_exe_path"]
        Rep1_PSI_file = opts["--Rep1_PSI_file"]
        Rep2_PSI_file = opts["--Rep2_PSI_file"]
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path_prefix = "%s/%s" % (output_dir, opts["--output_prefix"])
        
        filter_MFASS_high_replicability_smn1 = False
        if "--filter_MFASS_high_replicability_smn1" in opts:
            filter_MFASS_high_replicability_smn1 = True
    except:
        print("Failed to get opts")
        sys.exit(2)
    
    # Load in file to find overlap with
    # Writing this specifically for `/ESL/GitHub/MFASS/processed_data/sre/sre_data_clean.txt`
    to_overlap_nat_dict = {}
    to_overlap_mut_dict = defaultdict(list)
    MFASS_locations = defaultdict(list)
    with open(to_overlap_file, "r") as f:
        f.readline()  # header
        for line in f:
            var = line.split()
            # find exon coordinates of skipped exon
            seq_type = var[12]
            replicability_smn1 = var[79]  #or replicability_smn1 != "\"high\""
            # ignore control constructs
            if seq_type == "control":
                continue
            # only consider high replicability_smn1
            if replicability_smn1 != "\"high\"" and filter_MFASS_high_replicability_smn1:
                continue
            curr_chr, curr_start, curr_end = var[3:6]  # in hg19
            curr_strand = "+" if var[6] == "1" else "-"
            smn1_dPSI = var[88]
            # only consider variants with SMN1 backbone data
            if smn1_dPSI == "NA":
                continue
            smn1_dPSI = float(smn1_dPSI)
            smn1_PSI = float(var[75])
            smn1_nat_PSI = float(var[90])
            category = var[13]
            seq = var[11].strip("\"")
            location_key = ",".join([curr_chr, curr_start, curr_end, curr_strand])
            if seq_type == "\"nat\"":
                to_overlap_nat_dict[location_key] = (seq, smn1_PSI, smn1_nat_PSI, smn1_dPSI)
                MFASS_locations[",".join([curr_chr, curr_strand])].append((curr_start, curr_end))
            elif seq_type == "\"mut\"":
                to_overlap_mut_dict[location_key].append((curr_strand, category, seq, smn1_PSI, smn1_nat_PSI, smn1_dPSI))
    
    # Load in supertable and use BLAT find correct location in genome
    wt_supertable_dict = {}
    mut_supertable_dict = defaultdict(list)
    sequence_name_list = []
    sequence_list = []
    exon_only_mut_supertable_ref_list = []
    with open(supertable_file, "r") as f:
        f.readline()  # throw out header
        curr_supertable_wt_seq = ""
        curr_supertable_wt_ref = ""
        for line in f:
            var = line.split("\t")
            curr_supertable_ref = var[0]
            exon = var[4]
            intron1 = var[7]
            intron2 = var[8]
            fullseq_supertable = intron1 + exon + intron2
            seq_tuple = (exon, intron1, intron2)
            # parse WT
            if var[11] == "none":
                sequence_name_list.append(curr_supertable_ref)
                sequence_list.append(fullseq_supertable)
                wt_supertable_dict[curr_supertable_ref] = seq_tuple
                curr_supertable_wt_seq = seq_tuple
                curr_supertable_wt_ref = curr_supertable_ref
            # parse a potential mutation
            else:
                # Remove annotated mutants but subsequence is the same as WT
                if fullseq_supertable == curr_supertable_wt_seq:
                    continue
                if exon != curr_supertable_wt_seq[0] and intron1 == curr_supertable_wt_seq[1] and intron2 == curr_supertable_wt_seq[2]:
                    exon_only_mut_supertable_ref_list.append(curr_supertable_ref)
                # group mutants by their WT reference number
                mut_supertable_dict[curr_supertable_wt_ref].append([curr_supertable_ref, seq_tuple])
    
    #print("exon_only_mut_supertable_ref_list:\n", exon_only_mut_supertable_ref_list)
    
    # make FASTA files for BLAT input
    make_FASTA(sequence_name_list, sequence_list, output_path_prefix + "_BLAT_fullseq_temp.fa")
    
    # call BLAT on supertable intron1 + exon + intron2
    os.system("%s/blat %s %s %s_BLAT_fullseq_temp.psl" % (BLAT_exe_path, BLAT_reference, output_path_prefix + "_BLAT_fullseq_temp.fa", output_path_prefix))
    
    # load PSI's from RNA-seq
    Rep1_PSI_dict = load_PSI_file(Rep1_PSI_file)
    Rep2_PSI_dict = load_PSI_file(Rep2_PSI_file)
    
    # read in BLAT fullseq output to make sure all of supertable sequence is correct
    overlap_wt_matches = []
    overlap_mut_matches = []
    overlap_matches_PSI_Rep1 = []
    overlap_matches_PSI_Rep2 = []
    overlap_mut_matches_PSI_Rep1 = []
    overlap_mut_matches_PSI_Rep2 = []
    overlap_matches_dPSI_Rep1 = []
    overlap_matches_dPSI_Rep2 = []
    overlap_matches_PSI_avg_Reps = []
    overlap_mut_matches_PSI_avg_Reps = []
    hit = 0
    rnd_exon_1nt_count = 0
    rnd_exon_2nt_count = 0
    with open("%s_BLAT_fullseq_temp.psl" % (output_path_prefix), "r") as f_BLAT:
        for line_psl in f_BLAT:
            var = line_psl.split()
            if len(var) != 21:
                continue
            # Found a perfect match so it's WT
            if var[0] == "161":
                supertable_matched_ref = var[9]
                chromosome = var[13]
                strand = var[8]
                start_BLAT = int(var[15])
                end_BLAT = int(var[16])
                len_intron1 = len(wt_supertable_dict[supertable_matched_ref][1])
                len_intron2 = len(wt_supertable_dict[supertable_matched_ref][2])
                # negative strand
                if strand == "-":
                    exon_start_BLAT = start_BLAT + len_intron2 + 1
                    exon_end_BLAT = end_BLAT - len_intron1
                else:  # positive strand
                    exon_start_BLAT = start_BLAT + len_intron1 + 1
                    exon_end_BLAT = end_BLAT - len_intron2
                # lookup potential overlaps with MFASS data
                for curr_MFASS_start, curr_MFASS_end in MFASS_locations[",".join(["\"%s\"" % (chromosome[3:]), strand])]:
                    curr_MFASS_start = int(curr_MFASS_start)
                    curr_MFASS_end = int(curr_MFASS_end)
                    if curr_MFASS_start <= exon_start_BLAT - 20 and curr_MFASS_end >= exon_end_BLAT + 20:
                        # overlap found in WT/natural
                        MFASS_location_key = ",".join(["\""+chromosome[3:]+"\"", str(curr_MFASS_start), str(curr_MFASS_end), strand])
                        MFASS_seq, smn1_PSI, smn1_nat_PSI, smn1_dPSI = to_overlap_nat_dict[MFASS_location_key]
                        overlap_wt_matches.append((supertable_matched_ref, MFASS_location_key))
                        wt_supertable_seq_tuple = wt_supertable_dict[supertable_matched_ref]
                        wt_supertable_seq = "".join([wt_supertable_seq_tuple[1], wt_supertable_seq_tuple[0], wt_supertable_seq_tuple[2]])
                        # compare measured PSI's
                        if supertable_matched_ref in Rep1_PSI_dict:
                            overlap_matches_PSI_Rep1.append((supertable_matched_ref, MFASS_location_key, Rep1_PSI_dict[supertable_matched_ref], float(smn1_nat_PSI), wt_supertable_seq, MFASS_seq))
                            if supertable_matched_ref in Rep2_PSI_dict:
                                overlap_matches_PSI_avg_Reps.append((supertable_matched_ref, MFASS_location_key, (Rep1_PSI_dict[supertable_matched_ref]+Rep2_PSI_dict[supertable_matched_ref])/2, float(smn1_nat_PSI)))
                        if supertable_matched_ref in Rep2_PSI_dict:
                            overlap_matches_PSI_Rep2.append((supertable_matched_ref, MFASS_location_key, Rep2_PSI_dict[supertable_matched_ref], float(smn1_nat_PSI), wt_supertable_seq, MFASS_seq))
                        # check mutants in both studies for overlap
                        if MFASS_location_key in to_overlap_mut_dict:
                            for MFASS_strand, MFASS_category, MFASS_seq, smn1_PSI, smn1_nat_PSI, smn1_dPSI in to_overlap_mut_dict[MFASS_location_key]:
                                if MFASS_category == "\"rnd_exon_1nt\"":
                                    rnd_exon_1nt_count += 1
                                if MFASS_category == "\"rnd_exon_2nt\"":
                                    rnd_exon_2nt_count += 1
                                if MFASS_strand == "-":
                                        hit += 1
                                for supertable_mut_ref, supertable_mut_seq_tuple in mut_supertable_dict[supertable_matched_ref]:
                                    supertable_mut_seq = "".join([supertable_mut_seq_tuple[1], supertable_mut_seq_tuple[0], supertable_mut_seq_tuple[2]])
                                    # find overlapping subsequences
                                    MFASS_overlap_start = 0 if max(curr_MFASS_start, start_BLAT) - curr_MFASS_start == 0 else max(curr_MFASS_start, start_BLAT) - curr_MFASS_start -1
                                    MFASS_overlap_end = len(MFASS_seq) + min(curr_MFASS_end, end_BLAT) - curr_MFASS_end
                                    supertable_overlap_start = 0 if max(curr_MFASS_start, start_BLAT) - start_BLAT == 0 else max(curr_MFASS_start, start_BLAT) - start_BLAT - 1
                                    supertable_overlap_end = len(supertable_mut_seq) + min(curr_MFASS_end, end_BLAT) - end_BLAT
                                    MFASS_subseq = MFASS_seq[MFASS_overlap_start:MFASS_overlap_end]
                                    supertable_subseq = supertable_mut_seq[supertable_overlap_start:supertable_overlap_end]
                                    # check if any MFASS mutations are outside of subsequence location
                                    if any([c.islower() for c in MFASS_seq[:MFASS_overlap_start] + MFASS_seq[MFASS_overlap_end:]]):
                                        continue
                                    # check if any supertable mutations are outside of subsequence location
                                    wt_seq_portions = wt_supertable_seq[:supertable_overlap_start] + wt_supertable_seq[supertable_overlap_end:]
                                    if supertable_mut_seq[:supertable_overlap_start] + supertable_mut_seq[supertable_overlap_end:] != wt_seq_portions:
                                        continue
                                    # check if MFASS and supertable sequence are the same
                                    if supertable_subseq == MFASS_subseq.upper():
                                        overlap_mut_matches.append((supertable_mut_ref, MFASS_location_key))
                                        if supertable_mut_ref in Rep1_PSI_dict:
                                            overlap_mut_matches_PSI_Rep1.append((supertable_mut_ref, MFASS_location_key, Rep1_PSI_dict[supertable_mut_ref], float(smn1_PSI), supertable_mut_seq, MFASS_seq))
                                            if supertable_matched_ref in Rep1_PSI_dict:
                                                print("dPSI Rep1")
                                                overlap_matches_dPSI_Rep1.append((Rep1_PSI_dict[supertable_mut_ref] - Rep1_PSI_dict[supertable_matched_ref], float(smn1_dPSI)))
                                            if supertable_mut_ref in Rep2_PSI_dict:
                                                overlap_mut_matches_PSI_avg_Reps.append((supertable_mut_ref, (Rep1_PSI_dict[supertable_mut_ref]+Rep2_PSI_dict[supertable_mut_ref])/2, float(smn1_PSI)))
                                        if supertable_mut_ref in Rep2_PSI_dict:
                                            overlap_mut_matches_PSI_Rep2.append((supertable_mut_ref, MFASS_location_key, Rep2_PSI_dict[supertable_mut_ref], float(smn1_PSI), supertable_mut_seq, MFASS_seq))
                                            if supertable_matched_ref in Rep2_PSI_dict:
                                                print("dPSI Rep2")
                                                overlap_matches_dPSI_Rep2.append((Rep2_PSI_dict[supertable_mut_ref] - Rep2_PSI_dict[supertable_matched_ref], float(smn1_dPSI)))
                                        #print("FOUND")
                                        #print("HIT")
                                        #hit += 1
                                        #break
                                #if hit > 0:
                                #    break
                    #if hit > 0:
                    #    break
                #if hit > 0:
                #    break
    
    MFASS_overlap_PSI_Rep1 = overlap_matches_PSI_Rep1 + overlap_mut_matches_PSI_Rep1
    MFASS_overlap_PSI_Rep2 = overlap_matches_PSI_Rep2 + overlap_mut_matches_PSI_Rep2
    
    with open("%s_MFASS_overlap_PSI_Rep1.txt" % (output_path_prefix), "w") as f:
        f.write("supertable_ref\tMFASS_loc\tsupertable_PSI\tMFASS_index\tsupertable_seq\tMFASS_seq\n")  # write out header
        f.write("\n".join(["\t".join([str(ele) for ele in tup]) for tup in MFASS_overlap_PSI_Rep1]) + "\n")
    
    with open("%s_MFASS_overlap_PSI_Rep2.txt" % (output_path_prefix), "w") as f:
        f.write("supertable_ref\tMFASS_loc\tsupertable_PSI\tMFASS_index\tsupertable_seq\tMFASS_seq\n")  # write out header
        f.write("\n".join(["\t".join([str(ele) for ele in tup]) for tup in MFASS_overlap_PSI_Rep2]) + "\n")
    
    _, _, Rep1_PSI, MFASS_PSI_1, _, _ = zip(*overlap_matches_PSI_Rep1)
    _, _, Rep2_PSI, MFASS_PSI_2, _, _  = zip(*overlap_matches_PSI_Rep2)
    
    print("number of wt matches, Rep1:\n", len(Rep1_PSI))
    print("number of wt matches, Rep2:\n", len(Rep2_PSI))
    
    correlation1_spearman = stats.spearmanr(Rep1_PSI, MFASS_PSI_1)
    correlation2_spearman = stats.spearmanr(Rep2_PSI, MFASS_PSI_2)
    print("wt correlation1_spearman:\n", correlation1_spearman)
    print("wt correlation2_spearman:\n", correlation2_spearman)
    
    correlation1_pearson = stats.pearsonr(Rep1_PSI, MFASS_PSI_1)
    correlation2_pearson = stats.pearsonr(Rep2_PSI, MFASS_PSI_2)
    print("wt correlation1_pearson:\n", correlation1_pearson)
    print("wt correlation2_pearson:\n", correlation2_pearson)
    
    Rep1_mut_PSI, Rep2_mut_PSI, MFASS_mut_PSI_1, MFASS_mut_PSI_2 = ((), (), (), ())
    if len(overlap_mut_matches_PSI_Rep1) > 0:
        _, _, Rep1_mut_PSI, MFASS_mut_PSI_1, _, _ = zip(*(overlap_mut_matches_PSI_Rep1))
    if len(overlap_mut_matches_PSI_Rep2) > 0:
        _, _, Rep2_mut_PSI, MFASS_mut_PSI_2, _, _ = zip(*(overlap_mut_matches_PSI_Rep2))
    
    print("number of mut matches, Rep1:\n", len(Rep1_mut_PSI))
    print("number of mut matches, Rep2:\n", len(Rep2_mut_PSI))
    
    correlation1_spearman = stats.spearmanr(Rep1_mut_PSI, MFASS_mut_PSI_1)
    correlation2_spearman = stats.spearmanr(Rep2_mut_PSI, MFASS_mut_PSI_2)
    print("mut correlation1_spearman:\n", correlation1_spearman)
    print("mut correlation2_spearman:\n", correlation2_spearman)
    
    if len(overlap_mut_matches_PSI_Rep1) >= 2:
        correlation1_pearson = stats.pearsonr(Rep1_mut_PSI, MFASS_mut_PSI_1)
        print("mut correlation1_pearson:\n", correlation1_pearson)
    if len(overlap_mut_matches_PSI_Rep2) >= 2:
        correlation2_pearson = stats.pearsonr(Rep2_mut_PSI, MFASS_mut_PSI_2)
        print("mut correlation2_pearson:\n", correlation2_pearson)
    
    print("number of all matches, Rep1:\n", len(Rep1_PSI + Rep1_mut_PSI))
    print("number of all matches, Rep2:\n", len(Rep2_PSI + Rep2_mut_PSI))
    
    correlation1_spearman = stats.spearmanr(Rep1_PSI+Rep1_mut_PSI, MFASS_PSI_1+MFASS_mut_PSI_1)
    correlation2_spearman = stats.spearmanr(Rep2_PSI+Rep2_mut_PSI, MFASS_PSI_2+MFASS_mut_PSI_2)
    print("all correlation1_spearman:\n", correlation1_spearman)
    print("all correlation2_spearman:\n", correlation2_spearman)
    
    correlation1_pearson = stats.pearsonr(Rep1_PSI+Rep1_mut_PSI, MFASS_PSI_1+MFASS_mut_PSI_1)
    correlation2_pearson = stats.pearsonr(Rep2_PSI+Rep2_mut_PSI, MFASS_PSI_2+MFASS_mut_PSI_2)
    print("all correlation1_pearson:\n", correlation1_pearson)
    print("all correlation2_pearson:\n", correlation2_pearson)
    
    print("all correlation1_pearson^2:\n", pow(correlation1_pearson[0], 2))
    print("all correlation2_pearson^2:\n", pow(correlation2_pearson[0], 2))
    
    plt.scatter(Rep1_PSI+Rep1_mut_PSI, MFASS_PSI_1+MFASS_mut_PSI_1, alpha=0.5)
    plt.xlabel("Rep1 PSI")
    plt.ylabel("MFASS avg index")
    plt.savefig(output_path_prefix+"_Rep1.png")
    plt.clf()
    
    plt.scatter(Rep2_PSI+Rep2_mut_PSI, MFASS_PSI_2+MFASS_mut_PSI_2, alpha=0.5)
    plt.xlabel("Rep2 PSI")
    plt.ylabel("MFASS avg index")
    plt.savefig(output_path_prefix+"_Rep2.png")
    plt.clf()
    
    # Average between reps
    _, _, Rep_avg_PSI, MFASS_PSI_avg = zip(*overlap_matches_PSI_avg_Reps)
    correlation_spearman = stats.spearmanr(Rep_avg_PSI, MFASS_PSI_avg)
    correlation_pearson = stats.pearsonr(Rep_avg_PSI, MFASS_PSI_avg)
    print("Rep avg Spearman:\n", correlation_spearman)
    print("Rep avg Pearson:\n", correlation_pearson)
    
    print("rnd_exon_1nt_count:\n", rnd_exon_1nt_count)
    print("rnd_exon_2nt_count:\n", rnd_exon_2nt_count)
    
    plt.scatter(Rep_avg_PSI, MFASS_PSI_avg, alpha=0.5)
    plt.xlabel("Rep avg PSI")
    plt.ylabel("MFASS avg index")
    plt.savefig(output_path_prefix+"_Rep_avg.png")
    plt.clf()
    
    # Write out correlations for all overlaps
    with open(output_path_prefix + "_correlation.txt", "w") as f:
        f.write("Pearson correlation of WT and mutant sequences: Rep1 vs. MFASS avg: %s, %s\n" % (correlation1_pearson[0], correlation1_pearson[1]))
        f.write("Pearson correlation of WT and mutant sequences: Rep2 vs. MFASS avg: %s, %s\n" % (correlation2_pearson[0], correlation2_pearson[1]))
        f.write("Pearson correlation^2 of WT and mutant sequences: Rep1 vs. MFASS avg: %s\n" % (pow(correlation1_pearson[0], 2)))
        f.write("Pearson correlation^2 of WT and mutant sequences: Rep2 vs. MFASS avg: %s\n" % (pow(correlation2_pearson[0], 2)))
        f.write("Pearson correlation of WT and mutant sequences: Rep1 Rep2 avg vs. MFASS avg: %s, %s\n" % (correlation_pearson[0], correlation_pearson[1]))
        f.write("Pearson correlation^2 of WT and mutant sequences: Rep1 Rep2 avg vs. MFASS avg: %s\n" % (pow(correlation_pearson[0], 2)))
    
    # Look at dPSI overlaps
    print("overlap_matches_dPSI_Rep1: ", overlap_matches_dPSI_Rep1)
    print("overlap_matches_dPSI_Rep2: ", overlap_matches_dPSI_Rep2)
    
    print("Length overlap_matches_dPSI_Rep1: ", len(overlap_matches_dPSI_Rep1))
    print("Length overlap_matches_dPSI_Rep2: ", len(overlap_matches_dPSI_Rep2))
    
    if len(overlap_matches_dPSI_Rep1) > 0 and len(overlap_matches_dPSI_Rep2) > 0:
        # Correlation in dPSI
        Rep1_dPSI, MFASS_dPSI_avg_1 = zip(*overlap_matches_dPSI_Rep1)
        Rep2_dPSI, MFASS_dPSI_avg_2 = zip(*overlap_matches_dPSI_Rep2)
        
        correlation_all_dPSI_pearson = stats.pearsonr(Rep1_dPSI+Rep2_dPSI, MFASS_dPSI_avg_1+MFASS_dPSI_avg_2)
        print("all correlation dPSI pearson:\n", correlation_all_dPSI_pearson)
    
