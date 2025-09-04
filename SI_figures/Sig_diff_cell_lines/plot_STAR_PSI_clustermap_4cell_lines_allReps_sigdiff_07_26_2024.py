"""
Plot heatmap of STAR_PSI output files. Only significantly different sequences in at least 1 cell line vs. cell line. 
"""

from scipy.stats import linregress, fisher_exact
import getopt, sys, os, resource
resource.setrlimit(resource.RLIMIT_STACK, [0x10000000, resource.RLIM_INFINITY])
sys.setrecursionlimit(0x100000)
import re
import seaborn as sns; sns.set_theme(color_codes=True)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest, test_proportions_2indep
from statsmodels.stats.multitest import fdrcorrection
from itertools import combinations
import pickle
from collections import defaultdict


def load_STAR_PSI_file(filename):
    PSI_dict = {}  # reference --> PSI
    coverage_dict = {}
    with open(filename, "r") as f:
        f.readline()  # throw away header
        for line in f:
            var = line.split("\t")
            PSI_dict[var[0]] = float(var[1])
            coverage_dict[var[0]] = float(var[2])
    return PSI_dict, coverage_dict
    
    
def calculate_average_PSI(STAR_PSI_file_list):
    # Load replicates individually and get overlapping references
    PSI_rep_dicts = []
    PSI_rep_coverage_dicts = []
    intersection_PSI_references = set()
    for PSI_file in STAR_PSI_file_list:
        PSI_dict, coverage_dict = load_STAR_PSI_file(PSI_file)
        PSI_rep_dicts.append(PSI_dict)
        PSI_rep_coverage_dicts.append(coverage_dict)
        if len(intersection_PSI_references) == 0:
            intersection_PSI_references = set(PSI_dict.keys())
        else:
            intersection_PSI_references = intersection_PSI_references.intersection(set(PSI_dict.keys()))
    
    # Calculate average PSI's and prune out references with replicates not within 1 PSI of each other
    num_reps = len(PSI_rep_dicts)
    PSI_avg_dict = {}
    PSI_avg_coverage_dict = {}
    for intersection_PSI_reference in intersection_PSI_references:
        curr_PSIs = [d[intersection_PSI_reference] for d in PSI_rep_dicts]
        curr_coverages = [d[intersection_PSI_reference] for d in PSI_rep_coverage_dicts]
        if abs(max(curr_PSIs) - min(curr_PSIs)) <= 1:
            PSI_avg_dict[intersection_PSI_reference] = sum(curr_PSIs) / num_reps
            PSI_avg_coverage_dict[intersection_PSI_reference] = sum(curr_coverages) / num_reps
    print("Number of references in PSI_avg_dict: ", len(PSI_avg_dict))
    
    return PSI_avg_dict, PSI_avg_coverage_dict


def parse_PSI_reps_to_list_dicts(STAR_PSI_file_list, found_in_all_samples=True):
    # Load replicates individually and count spliced in & out
    PSI_rep_dicts = []
    PSI_rep_coverage_dicts = []
    intersection_PSI_references = None
    for PSI_file in STAR_PSI_file_list:
        PSI_dict, coverage_dict = load_STAR_PSI_file(PSI_file)
        PSI_rep_dicts.append(PSI_dict)
        PSI_rep_coverage_dicts.append(coverage_dict)
        if intersection_PSI_references is None:
            intersection_PSI_references = set(PSI_dict.keys())
        elif found_in_all_samples:
            intersection_PSI_references = intersection_PSI_references.intersection(set(PSI_dict.keys()))
        else:
            intersection_PSI_references = intersection_PSI_references.union(set(PSI_dict.keys()))
    
    return PSI_rep_dicts, PSI_rep_coverage_dicts, intersection_PSI_references


if __name__== "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"", ["HEK293_PSI_files=", "HeLa_PSI_files=", "K562_PSI_files=", "MCF7_PSI_files=", "HMC3_PSI_files=", "found_in_all_samples=", "output_dir=", "output_prefix="])
        opts = dict(opts)
        print(opts)
        HEK293_PSI_files_list = opts["--HEK293_PSI_files"].split(",")
        HeLa_PSI_files_list = opts["--HeLa_PSI_files"].split(",")
        K562_PSI_files_list = opts["--K562_PSI_files"].split(",")
        MCF7_PSI_files_list = opts["--MCF7_PSI_files"].split(",")
        HMC3_PSI_files_list = opts["--HMC3_PSI_files"].split(",")
        found_in_all_samples = "True" == opts["--found_in_all_samples"] if "--found_in_all_samples" in opts else True
        print("found_in_all_samples = ", found_in_all_samples, flush=True)
        output_dir = opts["--output_dir"]
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path_prefix = "%s/%s" % (output_dir, opts["--output_prefix"])
        
    except:
        print("Failed to get opts")
        sys.exit(2)
    
    # Load HEK293 data and take avg PSI and counts
    HEK293_PSI_dicts, HEK293_coverage_dicts, HEK293_intersection_PSI_references = parse_PSI_reps_to_list_dicts(HEK293_PSI_files_list, found_in_all_samples)
    HeLa_PSI_dicts, HeLa_coverage_dicts, HeLa_intersection_PSI_references = parse_PSI_reps_to_list_dicts(HeLa_PSI_files_list, found_in_all_samples)
    K562_PSI_dicts, K562_coverage_dicts, K562_intersection_PSI_references = parse_PSI_reps_to_list_dicts(K562_PSI_files_list, found_in_all_samples)
    MCF7_PSI_dicts, MCF7_coverage_dicts, MCF7_intersection_PSI_references = parse_PSI_reps_to_list_dicts(MCF7_PSI_files_list, found_in_all_samples)
    HMC3_PSI_dicts, HMC3_coverage_dicts, HMC3_intersection_PSI_references = parse_PSI_reps_to_list_dicts(HMC3_PSI_files_list, found_in_all_samples)
    
    all_cell_PSI_dicts = [HEK293_PSI_dicts, HeLa_PSI_dicts, K562_PSI_dicts, MCF7_PSI_dicts, HMC3_PSI_dicts]
    max_num_rep_comparisons = pow(max([len(rs) for rs in all_cell_PSI_dicts]), 2)  # upper bound estimate
    all_cell_coverage_dicts = [HEK293_coverage_dicts, HeLa_coverage_dicts, K562_coverage_dicts, MCF7_coverage_dicts, HMC3_coverage_dicts]
    if found_in_all_samples:
        all_intersection_PSI_references = HEK293_intersection_PSI_references.intersection(HeLa_intersection_PSI_references, K562_intersection_PSI_references, MCF7_intersection_PSI_references, HMC3_intersection_PSI_references)
    else:
        all_intersection_PSI_references = HEK293_intersection_PSI_references.union(HeLa_intersection_PSI_references, K562_intersection_PSI_references, MCF7_intersection_PSI_references, HMC3_intersection_PSI_references)
    
    with open(output_path_prefix + "_all_intersection_PSI_references.p", "wb") as f:
        pickle.dump(all_intersection_PSI_references, f, pickle.HIGHEST_PROTOCOL)
    
    
    # Make STAR PSI dataframe
    # Probably can do more elegantly, but need this sooner
    STAR_PSI_merged = pd.DataFrame(list(HEK293_PSI_dicts[0].items()), \
        columns = ['Reference', 'HEK293_PSI_Rep1'])
    print("STAR_PSI_merged initial = ", STAR_PSI_merged)
    HEK293_Rep2_df = pd.DataFrame(list(HEK293_PSI_dicts[1].items()), \
        columns = ['Reference', 'HEK293_PSI_Rep2'])
    
    if len(HEK293_PSI_dicts) > 2:
        HEK293_WT_Rep1_df = pd.DataFrame(list(HEK293_PSI_dicts[2].items()), \
            columns = ['Reference', 'HEK293_WT_PSI_Rep1'])
        HEK293_WT_Rep2_df = pd.DataFrame(list(HEK293_PSI_dicts[3].items()), \
            columns = ['Reference', 'HEK293_WT_PSI_Rep2'])
    
    HeLa_Rep1_df = pd.DataFrame(list(HeLa_PSI_dicts[0].items()), \
        columns = ['Reference', 'HeLa_PSI_Rep1'])
    HeLa_Rep2_df = pd.DataFrame(list(HeLa_PSI_dicts[1].items()), \
        columns = ['Reference', 'HeLa_PSI_Rep2'])
    
    K562_Rep1_df = pd.DataFrame(list(K562_PSI_dicts[0].items()), \
        columns = ['Reference', 'K562_PSI_Rep1'])
    K562_Rep2_df = pd.DataFrame(list(K562_PSI_dicts[1].items()), \
        columns = ['Reference', 'K562_PSI_Rep2'])
    
    MCF7_Rep1_df = pd.DataFrame(list(MCF7_PSI_dicts[0].items()), \
        columns = ['Reference', 'MCF7_PSI_Rep1'])
    MCF7_Rep2_df = pd.DataFrame(list(MCF7_PSI_dicts[1].items()), \
        columns = ['Reference', 'MCF7_PSI_Rep2'])
    
    HMC3_Rep1_df = pd.DataFrame(list(HMC3_PSI_dicts[0].items()), \
        columns = ['Reference', 'HMC3_PSI_Rep1'])
    HMC3_Rep2_df = pd.DataFrame(list(HMC3_PSI_dicts[1].items()), \
        columns = ['Reference', 'HMC3_PSI_Rep2'])
    
    if len(HEK293_PSI_dicts) > 3:
        dfs_to_merge_in = [HEK293_Rep2_df, HEK293_WT_Rep1_df, HEK293_WT_Rep2_df, HeLa_Rep1_df, HeLa_Rep2_df, K562_Rep1_df, K562_Rep2_df, MCF7_Rep1_df, MCF7_Rep2_df, HMC3_Rep1_df, HMC3_Rep2_df]
    else:
        dfs_to_merge_in = [HEK293_Rep2_df, HeLa_Rep1_df, HeLa_Rep2_df, K562_Rep1_df, K562_Rep2_df, MCF7_Rep1_df, MCF7_Rep2_df, HMC3_Rep1_df, HMC3_Rep2_df]
    
    for curr_df in dfs_to_merge_in:
        STAR_PSI_merged = STAR_PSI_merged.merge(curr_df, how='outer', on="Reference")
    STAR_PSI_merged.set_index("Reference", inplace=True)
    print("STAR_PSI_merged = ", STAR_PSI_merged)
    STAR_PSI_merged.to_csv(output_path_prefix + '_STAR_PSI_merged.tsv', sep="\t")
    
    # calculate p-values
    references_p = {}
    all_references_p_list = []
    num_tests = 0
    compare_combinations = list(combinations(range(len(all_cell_PSI_dicts)), 2))
    # references_p_by_cell_comparisons == cc_string -> {ref : [p-values] }
    references_p_by_cell_comparisons = {",".join([str(c) for c in cc]): defaultdict(list) for cc in compare_combinations}
    # For each sequence
    for ref in sorted(all_intersection_PSI_references):
        # For each cell type vs. cell type
        for i, j in compare_combinations:
            curr_cell_type_1_PSI_dicts = all_cell_PSI_dicts[i]
            curr_cell_type_2_PSI_dicts = all_cell_PSI_dicts[j]
            ref_in_cell_type_1 = [di for di, d in enumerate(curr_cell_type_1_PSI_dicts) if ref in d]
            ref_in_cell_type_2 = [di for di, d in enumerate(curr_cell_type_2_PSI_dicts) if ref in d]
            # need at least 1 replicates per cell type
            if len(ref_in_cell_type_1) >= 1 and len(ref_in_cell_type_2) >= 1:
                all_rep_comparisons_p_values = []
                # For each replicate vs. each replicate
                for ct1_rep in ref_in_cell_type_1:
                    for ct2_rep in ref_in_cell_type_2:
                        num_tests += 1
                        cc_string = ",".join([str(i), str(j)])
                        
                        curr_PSI_1 = all_cell_PSI_dicts[i][ct1_rep][ref]
                        coverage_1 = all_cell_coverage_dicts[i][ct1_rep][ref]
                        curr_PSI_2 = all_cell_PSI_dicts[j][ct2_rep][ref]
                        coverage_2 = all_cell_coverage_dicts[j][ct2_rep][ref]
                        
                        # two-sided test
                        #_, p_value = proportions_ztest([round(curr_PSI_1*coverage_1), round(curr_PSI_2*coverage_2)], [coverage_1, coverage_2])
                        # Fischer's exact two-sided test
                        ref_included = round(curr_PSI_1 * coverage_1)
                        ref_excluded = round((1.0 - curr_PSI_1) * coverage_1)
                        var_included = round(curr_PSI_2 * coverage_2)
                        var_excluded = round((1.0 - curr_PSI_2) * coverage_2)
                        statistic, p_value = fisher_exact([[ref_included, var_included], [ref_excluded, var_excluded]], "two-sided")
                        
                        # source code for this test doesn't add 0.5 for logit-adjusted before hitting a divide by zero, so pre-adding 0.5
                        """
                        # ‘diff’: ‘agresti-caffo’
                        _, p_value = test_proportions_2indep(curr_PSI_1*coverage_1 + 0.5, coverage_1 + 1, \
                            curr_PSI_2*coverage_2 + 0.5, coverage_2 + 1, compare="diff")
                        """
                        #_, p_value = test_proportions_2indep(curr_PSI_1*coverage_1 + 0.5, coverage_1 + 1, \
                        #    curr_PSI_2*coverage_2 + 0.5, coverage_2 + 1, compare="odds-ratio", method="logit")
                        # add to running list of all p-values
                        all_references_p_list.append((ref, p_value))
                        references_p_by_cell_comparisons[cc_string][ref].append(p_value)
                        
                        if p_value < 0.05:
                            if ref not in references_p: 
                                references_p[ref] = [p_value]
                            else:
                                references_p[ref].append(p_value)
    print("number of tests performed = ", num_tests)
    print("Number of significant references found before correction = ", len(references_p.keys()))
    
    
    # Benjamini/Hochberg independent FDR correction FWER 0.05
    BH_significant_dict = {}
    # Keys here are from compare_combinations
    for k, cc_dict in references_p_by_cell_comparisons.items():
        # TODO: assuming every reference has the same number of replicates. i.e. cc_dict has values of the same length.
        #num_rep_comparisons = len(list(cc_dict.values())[0])
        all_adjp_references = None
        consensus_not_significant_adjp_references = None
        for i in range(max_num_rep_comparisons):
            comparison_items = [(pi, p[i]) for pi,p in cc_dict.items() if len(p) > i]
            if len(comparison_items) == 0:
                continue
            reference_order, reference_p_values = zip(*comparison_items)
            rejected_bool, adjusted_p = fdrcorrection(reference_p_values)
            significant_adjp_references = [k for k, b in zip(reference_order, rejected_bool) if b == True]
            not_significant_adjp_references = [k for k, b in zip(reference_order, rejected_bool) if b == False]
            print("len(significant_adjp_references) = ", len(significant_adjp_references))
            if all_adjp_references is None:
                all_adjp_references = set(reference_order)
                consensus_not_significant_adjp_references = set(not_significant_adjp_references)
            else:
                all_adjp_references = all_adjp_references.union(set(reference_order))
                consensus_not_significant_adjp_references = consensus_not_significant_adjp_references.union(set(not_significant_adjp_references))
        consensus_significant_adjp_references = all_adjp_references - consensus_not_significant_adjp_references
        BH_significant_dict[k] = list(sorted(consensus_significant_adjp_references))
    
    all_BH_significant_refs = set().union(*list(BH_significant_dict.values()))
    print("Number of significant from Benjamini/Hochberg = ", [(k, len(v)) for k,v in BH_significant_dict.items()])
    print("Total significant from Benjamini/Hochberg = ", sum([len(k) for k in BH_significant_dict.values()]))
    
    # Save the significant hits
    with open(output_path_prefix + '_all_cell_PSI_dicts.p', 'wb') as f:
        pickle.dump(all_cell_PSI_dicts, f, pickle.HIGHEST_PROTOCOL)
    
    with open(output_path_prefix + '_references_p_by_cell_comparisons.p', 'wb') as f:
        pickle.dump(references_p_by_cell_comparisons, f, pickle.HIGHEST_PROTOCOL)
    
    with open(output_path_prefix + '_BH_significant_dict.p', 'wb') as f:
        pickle.dump(BH_significant_dict, f, pickle.HIGHEST_PROTOCOL)
    
    with open(output_path_prefix + '_BH_significant.txt', 'w') as f:
        f.write("Cell_type_1\tCell_type_2\tSignificant_reference\n")
        celltype_lookup = {"0": "HEK293", "1": "HeLa", "2": "K562", "3": "MCF7", "4": "HMC3"}
        for k, curr_sig_list in BH_significant_dict.items():
            cell_type_1, cell_type_2 = [celltype_lookup[i] for i in k.split(",")]
            for curr_sig_num in curr_sig_list:
                f.write("\t".join([cell_type_1, cell_type_2, curr_sig_num]) + "\n")
    
    
    #BH FDR correction
    # filter down df to only significant references
    STAR_PSI_merged_BH = STAR_PSI_merged.loc[STAR_PSI_merged.index.isin(all_BH_significant_refs)]
    STAR_PSI_merged_BH_orig = STAR_PSI_merged_BH.copy()
    STAR_PSI_merged_BH_avg = STAR_PSI_merged_BH.copy()
    
    # handle references with NaN
    STAR_PSI_merged_BH = STAR_PSI_merged_BH.fillna(-1)
    STAR_PSI_merged_BH.to_csv(output_path_prefix + '_Fisher_exact_BH.tsv', sep="\t")

    mask = np.zeros_like(STAR_PSI_merged_BH)
    mask[STAR_PSI_merged_BH == -1] = True 
    
    fig, ax = plt.subplots(figsize=(6, 125))
    g = sns.clustermap(STAR_PSI_merged_BH, yticklabels=True, cmap="viridis", mask=mask, col_cluster=False)
    g.ax_row_dendrogram.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path_prefix+"_Fisher_exact_BH.pdf")
    plt.clf()
    
    
    # make an averaged by cell line version of a heatmap
    if len(HEK293_PSI_dicts) > 3:
        STAR_PSI_merged_BH_avg["HEK293_avg"] = STAR_PSI_merged_BH_orig[['HEK293_PSI_Rep1', 'HEK293_PSI_Rep2', 'HEK293_WT_PSI_Rep1', 'HEK293_WT_PSI_Rep2']].mean(axis=1)
    else:
        STAR_PSI_merged_BH_avg["HEK293_avg"] = STAR_PSI_merged_BH_orig[['HEK293_PSI_Rep1', 'HEK293_PSI_Rep2']].mean(axis=1)
    STAR_PSI_merged_BH_avg["HeLa_avg"] = STAR_PSI_merged_BH_orig[['HeLa_PSI_Rep1', 'HeLa_PSI_Rep2']].mean(axis=1)
    STAR_PSI_merged_BH_avg["K562_avg"] = STAR_PSI_merged_BH_orig[['K562_PSI_Rep1', 'K562_PSI_Rep2']].mean(axis=1)
    STAR_PSI_merged_BH_avg["MCF7_avg"] = STAR_PSI_merged_BH_orig[['MCF7_PSI_Rep1', 'MCF7_PSI_Rep2']].mean(axis=1)
    STAR_PSI_merged_BH_avg["HMC3_avg"] = STAR_PSI_merged_BH_orig[['HMC3_PSI_Rep1', 'HMC3_PSI_Rep2']].mean(axis=1)
    
    # handle references with NaN
    STAR_PSI_merged_BH_avg = STAR_PSI_merged_BH_avg.fillna(-1)
    STAR_PSI_merged_BH_avg.to_csv(output_path_prefix + '_Fisher_exact_BH_avg.tsv', sep="\t")
    STAR_PSI_merged_BH_avg = STAR_PSI_merged_BH_avg[["HEK293_avg", "HeLa_avg", "K562_avg", "MCF7_avg", "HMC3_avg"]]
    
    mask = np.zeros_like(STAR_PSI_merged_BH_avg)
    mask[STAR_PSI_merged_BH_avg == -1] = True 
    
    fig, ax = plt.subplots(figsize=(6, 125))
    g = sns.clustermap(STAR_PSI_merged_BH_avg, yticklabels=True, cmap="viridis", mask=mask, col_cluster=False)
    g.ax_row_dendrogram.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path_prefix+"_Fisher_exact_BH_avg.pdf")
    plt.clf()
    
    