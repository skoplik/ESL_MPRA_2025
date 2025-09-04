"""
Copyright 
GPL license
"""

import os
from collections import defaultdict
import sys, getopt
from Bio import SeqIO
import shutil
import glob
import pathlib
import pickle


def create_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir
    

def remove_file(f):
    """ Removes a single file f """
    try:
        os.remove(f)
    except OSError:
        pass


"""
https://github.com/LucksLab/R2D2/blob/master/OSU.py
"""
def remove_files(files):
    """ Remove files in the list files """
    for f in files:
        remove_file(f)
    
    
if __name__== "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"", ["GTF_all_refs_file=","STAR_pass1_dir=", "supertable_file=", "mincov=", "output_dir=", "output_prefix="])
        opts = dict(opts)
        print(opts)
        gtf_file = opts["--GTF_all_refs_file"]
        output_dir = opts["--output_dir"]
        STAR_pass1_dir = opts["--STAR_pass1_dir"]
        supertable_file = opts["--supertable_file"]
        mincov = int(opts["--mincov"]) if "--mincov" in opts else 10
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path_prefix = "%s/%s" % (output_dir, opts["--output_prefix"])
    except:
        print("Failed to get opts")
        sys.exit(2)
    
    # collect WT reference numbers, has extras here
    wt_supertable_dict = {}  # supertable reference -> seq_tuple
    wt_eventID_supertableRef_dict = {}  # supertable eventID -> supertable reference
    mut_supertable_dict = defaultdict(list)  # supertable wt reference -> list of mutant reference
    supertable_gene_snp = {}  # supertable reference -> gene + SNP + event_id
    gene_name_dict = {}
    
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
            if var[11] == "none":
                wt_supertable_dict[curr_supertable_ref] = seq_tuple
                if event_id not in wt_eventID_supertableRef_dict:
                    wt_eventID_supertableRef_dict[event_id] = curr_supertable_ref
                supertable_gene_snp[curr_supertable_ref] = (gene_name, snp, event_id)
            gene_name_dict[curr_supertable_ref] = gene_name
    print("len(wt_supertable_dict.keys()) = ", len(wt_supertable_dict.keys()))
    
    # load GTF for expected exon junctions
    exon_dict = defaultdict(list)
    exon_start_dict = defaultdict(list)
    exon_end_dict = defaultdict(list)
    transcript_dict = defaultdict(list)
    all_expected_jxns = {}  # parent_ref -> {'i1': [start, end], 'i2': [start, end], 'e': [start, end]}
    with open(gtf_file, "r") as f:
        curr_expected_junctions = {'i1': [0,0], 'i2': [0,0], 'e': [0,0]}
        prev_parent_ref = -1
        junction_counts = {'i1': 0, 'i2': 0, 'e': 0}
        junction_multi_counts = {'i1': 0, 'i2': 0, 'e': 0}
        for line in f:
            var = line.strip().split("\t")
            # only looking at exons
            if line[0] == "#":
                continue
            if var[2] != "exon":
                continue
            
            curr_parent_ref = int(var[0].split("_")[0])
            
            if prev_parent_ref == -1:
                prev_parent_ref = curr_parent_ref
                # repeat code, just the variable exon part as the pipeline stands
                if var[4] != "1" and var[4] != "872":
                    curr_expected_junctions['i1'][1] = str(int(var[3]) - 1)
                    curr_expected_junctions['i2'][0] = str(int(var[4]) + 1)
            elif curr_parent_ref == prev_parent_ref:
                if var[3] == "1":
                    curr_expected_junctions['i1'][0] = str(int(var[4]) + 1)
                    curr_expected_junctions['e'][0] = str(int(var[4]) + 1)
                elif var[3] == "872":
                    curr_expected_junctions['i2'][1] = str(int(var[3]) - 1)
                    curr_expected_junctions['e'][1] = str(int(var[3]) - 1)
                else:
                    curr_expected_junctions['i1'][1] = str(int(var[3]) - 1)
                    curr_expected_junctions['i2'][0] = str(int(var[4]) + 1)
            else:  # moved to new parent ref
                # save expected junctions to parent ref
                all_expected_jxns[prev_parent_ref] = curr_expected_junctions
                
                # reset variables
                prev_parent_ref = curr_parent_ref
                curr_expected_junctions = {'i1': [0,0], 'i2': [0,0], 'e': [0,0]}
                junction_counts = {'i1': 0, 'i2': 0, 'e': 0}
                junction_multi_counts = {'i1': 0, 'i2': 0, 'e': 0}
                
                # repeat code, just the variable exon part as the pipeline stands
                if var[4] != "1" and var[4] != "872":
                    curr_expected_junctions['i1'][1] = str(int(var[3]) - 1)
                    curr_expected_junctions['i2'][0] = str(int(var[4]) + 1)
        # save last reference in GTF
        all_expected_jxns[curr_parent_ref] = curr_expected_junctions
    
    # Load all splicing counts
    all_splicing_counts = {}  # parent_ref -> (SJ 5', SJ 3') -> read count
    
    for curr_reporter_dir in glob.glob(STAR_pass1_dir + "/*/"):
        reporter_num = pathlib.PurePath(curr_reporter_dir).name
        curr_parent_ref = int(reporter_num.split("_")[0])
        if curr_parent_ref > 244000:
            continue
        
        # collect reads
        SJ_file = "%s/parsed_alignment/%s_unique_UMIconsensusSJ.out.tab" % (curr_reporter_dir, reporter_num)
        with open(SJ_file, "r") as sj_f:
            for line_sj in sj_f:
                var_sj = line_sj.strip().split()
                SJ_5p = int(var_sj[1])
                SJ_3p = int(var_sj[2])
                SJ_tuple = (SJ_5p, SJ_3p)
                SJ_count = int(var_sj[6])
                
                # load junction and counts into all_splicing_counts
                if curr_parent_ref not in all_splicing_counts:
                    all_splicing_counts[curr_parent_ref] = { SJ_tuple : SJ_count }
                elif SJ_tuple in all_splicing_counts[curr_parent_ref]:
                    all_splicing_counts[curr_parent_ref][SJ_tuple] += SJ_count
                elif SJ_tuple not in all_splicing_counts[curr_parent_ref]:
                    all_splicing_counts[curr_parent_ref][SJ_tuple] = SJ_count
                else:
                    print("EDGE CASE = ", SJ_tuple, SJ_count)
        
    # Calculate overall and conditional frequencies for 26 and 871
    all_freqs = {}
    all_26_freqs = {}
    all_871_freqs = {}
    all_i1_freqs = {}
    all_i2_freqs = {}
    all_e_freqs = {}
    all_PSIs_cov = {}
    for curr_parent_ref, SJ_read_count_dict in sorted(all_splicing_counts.items()):
        print("curr_parent_ref = ", curr_parent_ref)
        total_counts = sum(SJ_read_count_dict.values())
        if total_counts == 0:
            continue
        # min coverage filter, defaults 10
        #if total_counts < mincov:
        #    continue
        
        curr_all_freqs = { jxn_tuple : count / float(total_counts) for jxn_tuple, count in SJ_read_count_dict.items() }
        all_freqs[curr_parent_ref] = curr_all_freqs
        
        total_26_counts = sum([count for jxn_tuple, count in SJ_read_count_dict.items() if jxn_tuple[0] == 26])
        if total_26_counts > 0:
            curr_26_freqs = { jxn_tuple : count / float(total_26_counts) for jxn_tuple, count in SJ_read_count_dict.items() if jxn_tuple[0] == 26 }
            all_26_freqs[curr_parent_ref] = curr_26_freqs
        
        total_871_counts = sum([count for jxn_tuple, count in SJ_read_count_dict.items() if jxn_tuple[1] == 871])
        if total_871_counts > 0:
            curr_871_freqs = { jxn_tuple : count / float(total_871_counts) for jxn_tuple, count in SJ_read_count_dict.items() if jxn_tuple[1] == 871 }
            all_871_freqs[curr_parent_ref] = curr_871_freqs
        
        # calculate PSI at expected junctions
        curr_all_junction_counts = all_splicing_counts[curr_parent_ref]
        expected_junctions = all_expected_jxns[curr_parent_ref]
        print("curr_all_junction_counts = ", curr_all_junction_counts)
        print("expected_junctions = ", expected_junctions)
        junction_counts = {'i1': 0, 'i2': 0, 'e': 0}
        if tuple([int(z) for z in expected_junctions['i1']]) in curr_all_junction_counts:
            junction_counts['i1'] = curr_all_junction_counts[tuple([int(z) for z in expected_junctions['i1']])]
        if tuple([int(z) for z in expected_junctions['i2']]) in curr_all_junction_counts:
            junction_counts['i2'] = curr_all_junction_counts[tuple([int(z) for z in expected_junctions['i2']])]
        if tuple([int(z) for z in expected_junctions['e']]) in curr_all_junction_counts:
            junction_counts['e'] = curr_all_junction_counts[tuple([int(z) for z in expected_junctions['e']])]
        
        included = min(junction_counts['i1'], junction_counts['i2'])
        total_count = (included + junction_counts['e'])
        PSI = included / total_count if total_count > 0 else -1
        
        all_PSIs_cov[int(curr_parent_ref)] = (PSI, total_count)
        
    # Pickle all_splicing_counts
    with open(output_path_prefix + "_all_splicing_counts.p", "wb") as f:
        pickle.dump(all_splicing_counts, f)
    
    with open(output_path_prefix + "_all_freqs.p", "wb") as f:
        pickle.dump(all_freqs, f)
    
    with open(output_path_prefix + "_all_26_freqs.p", "wb") as f:
        pickle.dump(all_26_freqs, f)
    
    with open(output_path_prefix + "_all_871_freqs.p", "wb") as f:
        pickle.dump(all_871_freqs, f)
    
    with open(output_path_prefix + "_all_expected_jxns.p", "wb") as f:
        pickle.dump(all_expected_jxns, f)
    
    # Write out to file
    with open(output_path_prefix + "_all_splicing_counts.txt", "w") as f:
        f.write("Parent_ref\tSJ_5\tSJ_3\tCount\tAll_freqs\tConditional_26_freqs\tConditional_871_freqs\ti1_jxn\ti2_jxn\te_jxn\n")
        for parent_ref, SJ_read_count_dict in sorted(all_splicing_counts.items()):
            total_counts = sum(SJ_read_count_dict.values())
            # min coverage filter, defaults 10
            if total_counts < mincov:
                continue
            f.write("\n".join(["\t".join( \
            [str(parent_ref), str(k[0]), str(k[1]), str(v), str(all_freqs[parent_ref][k]), \
            "N/A" if parent_ref not in all_26_freqs else str(all_26_freqs[parent_ref][k]) if k in all_26_freqs[parent_ref] else "N/A", \
            "N/A" if parent_ref not in all_871_freqs else str(all_871_freqs[parent_ref][k]) if k in all_871_freqs[parent_ref] else "N/A", \
            str([str(k[0]), str(k[1])] == all_expected_jxns[parent_ref]["i1"]), \
            str([str(k[0]), str(k[1])] == all_expected_jxns[parent_ref]["i2"]), \
            str([str(k[0]), str(k[1])] == all_expected_jxns[parent_ref]["e"]) ]) \
            for k, v in SJ_read_count_dict.items() if parent_ref in all_freqs ]) + "\n")
    
    # write out PSI's
    with open(output_path_prefix + "_recalc_all_PSIs.txt", "w") as f:
        f.write("Reference\tPSI\tCoverage\n")
        for ref in sorted(all_PSIs_cov.keys()):
            f.write("%s\t%s\t%s\n" % (ref, all_PSIs_cov[ref][0], all_PSIs_cov[ref][1]))
    
    with open("%s_recalc_PSIs_mincov%s.txt" % (output_path_prefix, mincov), "w") as f:
        f.write("Reference\tPSI\tCoverage\n")
        for ref in sorted(all_PSIs_cov.keys()):
            if all_PSIs_cov[ref][1] >= mincov:
                f.write("%s\t%s\t%s\n" % (ref, all_PSIs_cov[ref][0], all_PSIs_cov[ref][1]))
    