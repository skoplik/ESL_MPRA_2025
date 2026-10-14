import numpy as np
import pandas as pd


def get_average_coverage(cov_df, cell_types):
    cell_types = [i + "_total_pooled" for i in cell_types]
    total = cov_df[cell_types].mean(axis=1)
    return total


def get_average_psi(cov_df, cell_types):
    cell_types = [i + "_pooled_psi_clipped" for i in cell_types]
    total = cov_df[cell_types].mean(axis=1)
    return total


# Gets the index of the pool. Since each item in family_list
# is a family with some size >= 1, the total size of the samples
# in that split (test, val, train) is the sum of those sizes.
# Hence, the index of the pool is also calculated from that sum.
def find_index_by_size(fams, target_size, start_index = 0):
    # size of each fam is given at index 0
    end_index = start_index
    curr_size = 0
    while end_index < len(fams) and curr_size + fams[end_index][0] <= target_size:
        curr_size += fams[end_index][0]
        end_index += 1
    return end_index


def get_split_families(family_list, proportions = None, pool_sizes = None):
    """
    heuristic to find good splits
    families list:
    a list of tuples (size_of_family, total_coverage, event_id, avg_psi)
    
    Concatenates every third family together so the higher coverage
    samples will be at the beginning of each third
    
    From a pool of possible families, assign families to test with the 
    highest coverage while mantaining the test size (5%)
    
    Do the same for the validation set, while including families
    not used by test. Optimize for highest total coverage while keeping
    val size below (5%)
    
    For the validation and test sets, split them in half, sorted by the 
    average psi across cells. This allows for a better balance of PSIs 
    across the different splits. 
    
    The samples from each pool are chosen to maximize the total coverage
    using knapsack. This allows for the maximum coverage to be reached given
    the size constrains of each split. 
    
    The leftover families are given to the train set. Due to concatenation
    at the beginning, it will contain the last third of families, guaranteeing
    at least some of the high coverage families are included. 
    
    """
    if proportions is None:
        # default split is 90 | 5 | 5 for train | val | test
        proportions = {}
        proportions["test"]  = 0.05
        proportions["val"]   = 0.05
        proportions["train"] = 0.9
    if pool_sizes is None:
        # a higher pool size increases the chance good families
        # are included. Too high of a pool size means high coverage
        # families are excluded from the train set
        pool_sizes = {}
        pool_sizes["test"] = 0.33
        pool_sizes["val"] = 0.33
    
    # sort by average coverage. Sorting by total coverage for the concatenation
    # means the test set will get very large families, lowering the chance the test
    # set will have a high total coverage given its size constraints
    sorted_by_coverage = sorted(family_list, key = lambda x : -(x[1]/(x[0] + 1)))
    
    # the actual size of each split is given by the sum of the family sizes
    n = sum([i[0] for i in family_list])
    n1 = int(n * pool_sizes["test"])
    n2 = int(n * pool_sizes["val"])
    
    # concatenate every third family together and 
    # create pools to optimize splits from
    list_1 = sorted_by_coverage[::3]
    list_2 = sorted_by_coverage[1::3]
    list_3 = sorted_by_coverage[2::3]
    concat_list = list_1 + list_2 + list_3
    
    test_indx = find_index_by_size(concat_list, n1)
    val_indx = find_index_by_size(concat_list, n2, start_index = test_indx)
    
    test_pool = concat_list[0:test_indx]
    val_pool = concat_list[test_indx:val_indx]
    train_pool = concat_list[val_indx:]
    
    # sort pool by psi, to allow an event amount of high psi and low psi
    # need somewhat of a balance of labels in the dataset
    test_pool = sorted(test_pool, key = lambda x : x[3])
    val_pool = sorted(val_pool, key = lambda x : x[3])
    
    test_indxL = find_index_by_size(test_pool, n1 // 2)
    test_poolL = test_pool[0 : test_indxL]
    test_poolR = test_pool[test_indxL :]
    
    val_indxL = find_index_by_size(val_pool, n2 // 2)
    val_poolL = val_pool[0 : val_indxL]
    val_poolR = val_pool[val_indxL :]
    
    # Split test and validation into two subsets to balance PSI across splits.
    proportions = [int(proportions["test"] * n / 2), int(proportions["test"] * n / 2), 
                   int(proportions["val"] * n / 2), int(proportions["val"] * n / 2)]
    pools = [test_poolL,test_poolR,val_poolL,val_poolR,train_pool]    
    splits_out = []

    # Run knapsack on each pool and give unchosen families to the next split
    for i in range(len(proportions)):
        m = knapsack(pools[i], max_size = proportions[i])
        chosen = traceback(m, pools[i])
        splits_out.append([pools[i][j] for j in chosen])
        for j in range(len(pools[i])):
            if j not in chosen:
                pools[i + 1].append(pools[i][j])
    
    # The leftover famlilies are given to the training dataset
    test_ = splits_out[0] + splits_out[1]
    val_ = splits_out[2] + splits_out[3]
    train_ = pools[4]
    
    return train_, val_, test_


def knapsack(pool, max_size):
    n = len(pool)
    m = np.zeros((n+1, max_size+1))
    for i in range(1,n+1):
        for j in range(1,max_size+1):
            if pool[i-1][0] > j:
                m[i][j] = m[i-1][j]
            else:
                m[i][j] = max(m[i-1][j], m[i-1][j-pool[i-1][0]] + pool[i-1][1])
    return m


def traceback(m, pool):
    chosen = set()
    i, j = len(m) - 1, len(m[0]) - 1
    while i > 0:
        initial = (i, j)
        if m[i - 1][j] == m[i][j]:
            i -= 1
        else:
            chosen.add(i-1)
            j = j - pool[i-1][0]
            i = i - 1
    return chosen


# len(x) - 1 so that the reference is not included in the 
# calculation of the split size.
# Returns both the average coverage and mean psi. Still returns the legth
def get_both(x):
    return (len(x) - 1, np.sum(x["Avg_coverage"]), np.mean(x["Avg_PSI"]))
    

# Returns the families that are present in the dataset (splice_vs_super_full)
def filter_families_with_ref(family_list, family_set):
    family_filtered = []
    for i in family_list:
        if i[2] in family_set:
            family_filtered.append(i)
    return family_filtered


def get_family_list(coverage_df_filtered, splice_vs_super_full):
	
	# set up variables to run knapsack on
	# average the coverage across cells, sum over the variants
	# in the family to get the total coverage for each family
	fams = coverage_df_filtered.groupby("event_id")[["Avg_coverage", "Avg_PSI"]].apply(get_both)
	family_list = [(i[1][0], i[1][1], i[0], i[1][2]) for i in fams.items()]

	# remove nans
	family_list_no_nan = []
	for i in family_list:
		if not np.isnan(i[1]):
			family_list_no_nan.append(i)
	family_list = family_list_no_nan

	family_set = set(splice_vs_super_full["event_id"])
	family_list = filter_families_with_ref(family_list, family_set)
	return family_list


names = [
('MMSplice_Ref_ID', 'MMsplice Ref ID'),
('MMSplice_VCF_Variant', 'MMSplice VCF Var'),
('region', 'region'),
('MMSplice_Delta_Logit_PSI', 'delta_logit_psi'),
('Supertable_Mean_Delta_Logit_PSI', 'mean_delta_logit_pooled'),
('WT_Avg_PSI', 'wt_avg_psi'),
('WT_Avg_Logit', 'wt_avg_logit'),
('MMSplice_Predicted_PSI', 'predicted_variant_psi'),
('Supertable_Observed_PSI', 'avg_variant_psi'),
('MMSplice_Predicted_Delta_PSI', 'predicted_delta_psi'),
('Supertable_Observed_Delta_PSI', 'experimental_delta_psi'),
('HEK_Supertable_Delta_Logit', 'HEK_delta_logit_pooled'),
('HeLa_Supertable_Delta_Logit', 'HeLa_delta_logit_pooled'),
('K562_Supertable_Delta_Logit', 'K562_delta_logit_pooled'),
('MCF7_Supertable_Delta_Logit', 'MCF7_delta_logit_pooled'),
('HMC3_Supertable_Delta_Logit', 'HMC3_delta_logit_pooled'),
('Reference', 'Reference'),
('event_id', 'event_id'),
('gene_exon', 'gene_exon'),
('snp', 'snp'),
('source', 'source'),
('seq_type', 'seq_type'),
('intron1', 'intron1'),
('exon', 'exon'),
('intron2', 'intron2'),
('full_seq', 'full_seq')
]


def rename_splice_vs_super(splice_vs_super_double_variants):
	splice_vs_super_double_variants=splice_vs_super_double_variants.rename(columns = {names[i][1] : names[i][0] for i in range(len(names))})
	return splice_vs_super_double_variants


def get_split_families_mm(coverage_df, splice_vs_super, proportions = None, pool_sizes = {"test":0.33, "val":0.32}):
	cell_types = ["HeLa", "MCF7", "HMC3", "HEK", "K562"]
	avg_coverage = get_average_coverage(coverage_df, cell_types)
	avg_psi = get_average_psi(coverage_df, cell_types)
	coverage_df["Avg_coverage"] = avg_coverage
	coverage_df["Avg_PSI"] = avg_psi
	splice_vs_super = rename_splice_vs_super(splice_vs_super)
	coverage_df_filtered = coverage_df[(coverage_df["Reference"].isin(splice_vs_super["Reference"])) | (coverage_df["snp"] == "none")]
	family_list = get_family_list(coverage_df_filtered, splice_vs_super)	
	return get_split_families(family_list, proportions, pool_sizes)


def visualize_splits(train, val, test):
	# Visualizing the splits
	# Note that the train set is almost guaranteed to 
	# have a higher total coverage than val and test since
	# the total family size is 18 times greater

	sizes = []
	n_fams = []
	tot_cov = []
	mean_cov = []
	mean_fam_cov = []
	total_size = 0
	for i in [train, val, test]:
	    size = sum([j[0] for j in i])
	    sizes.append(size)
	    total_size += size
	    n_fams.append(len(i))
	    tot_cov.append(np.sum([j[1] for j in i]))
	    mean_cov.append(np.sum([j[1] for j in i]) / np.sum([j[0] for j in i]))
	    mean_fam_cov.append(np.mean([j[1] for j in i]))

	percent_ = [round(size*100/total_size, 3) for size in sizes]


	train_size = sum([i[0] for i in train])
	val_size = sum([i[0] for i in val])
	test_size = sum([i[0] for i in test])
	total = train_size + val_size + test_size

	splits_df = pd.DataFrame({"Size": sizes, "Percent":percent_, "Number of Families": n_fams, "Total Coverage":tot_cov,\
				 "Mean Sample Coverage":mean_cov, "Mean Family Coverage":mean_fam_cov})

	splits_df.index = ["Train", "Val", "Test"]
	for i in splits_df:
		print(i)
		print(splits_df[i])
		print()
	return splits_df



if __name__ == "__main__":
	merged_PSI_cov_logit_df = pd.read_csv("/storage/yu/Projects/P000--ESL_continuation/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WTS_VARS_NO_DELTAS.csv")

	splice_vs_super = pd.read_csv("/storage/yu/Projects/P000--ESL_continuation/Data/MMSplice_2025_07_23/double_variants/mmsplice_vs_supertable_vcfid_filtered.csv")
	result = get_split_families_mm(merged_PSI_cov_logit_df, splice_vs_super)
	print(len(result[0]))
	print(len(result[1]))
	print(len(result[2]))

	visualize_splits(*result)

