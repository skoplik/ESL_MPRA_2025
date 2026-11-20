import numpy as np
from src.data_helpers.knapsack.knapsack2d import knapsack2D, traceback2D

def find_index_by_value(pools_psi_sorted, mean):
    # run binary search
    l = 0
    r = len(pools_psi_sorted) - 1
    while l < r:
        m = (l + r) // 2
        if pools_psi_sorted[m][3] < mean:
            l = m + 1
        else:
            r = m - 1
    for i in (l, m, r):
        if pools_psi_sorted[i][3] > mean:
            return i
    return max(l, m, r) + 1


def find_index_by_size(fams, target_size, start_index = 0):
    # size of each fam is given at index 0
    end_index = start_index
    curr_size = 0
    while end_index < len(fams) and curr_size + fams[end_index][0] <= target_size:
        curr_size += fams[end_index][0]
        end_index += 1
    return end_index


def get_split_families_2D_knapsack(family_list, proportions = None, pool_sizes = None, num_hits = 859, A = 0, B = 4, mean_psi = None):
    
    if proportions is None:
        # default split is 90 | 5 | 5 for train | val | test
        proportions = {}
        proportions["test"]  = 0.05
        proportions["val"]   = 0.05
        proportions["train"] = 0.9
    if pool_sizes is None:
        # a higher pool size increases the chance good families
        # are included. Too high of a pool size means high coverage
        # families are excluded form the train set
        pool_sizes = {}
        pool_sizes["test"] = 0.33
        pool_sizes["val"] = 0.33
    
    # sort by average coverage. Sorting by total coverage for the concatenation
    # means the test set will get very large families, lowering the chance the test
    # set will have a high total coverage given its size constraints
    sorted_by_coverage = sorted(family_list, key = lambda x : -(x[0]/(x[1])))
    
    # the actual size of each split is given by the sum of the family sizes
    n = sum([i[0] for i in family_list])
    n1 = int(n * pool_sizes["test"])
    n2 = int(n * pool_sizes["val"])
    
    # concatenate every third family together and 
    # create pools to optimize splits form
    list_1 = sorted_by_coverage[::3]
    list_2 = sorted_by_coverage[1::3]
    list_3 = sorted_by_coverage[2::3]
    concat_list = list_1 + list_2 + list_3
    
    test_indx = find_index_by_size(concat_list, n1)
    val_indx = find_index_by_size(concat_list, n2, start_index = test_indx)
    
    
    test_pool = concat_list[0:test_indx]
    val_pool = concat_list[test_indx:val_indx]
    train_pool = concat_list[val_indx:]
    
    hit_proportion = num_hits / n
    
    pools = [test_pool, val_pool, train_pool]
    
    proportions = [n * proportions["test"], n * proportions["val"]]
    
    splits_out = []
    for i in range(len(proportions)):
        if mean_psi is not None and i < len(mean_psi):
            num_below_mean, num_above_mean, mean = mean_psi[i]
            pools_psi_sorted = sorted(pools[i], key = lambda x : x[3])
            total_pool_limit = int(proportions[i])
            diff = num_below_mean - num_above_mean
            if diff > 0:
                psi_limit_high = (total_pool_limit // 2) + (diff // 2) 
                psi_limit_low = (total_pool_limit // 2) - (diff // 2) 
            else:
                psi_limit_high = (total_pool_limit // 2) - (diff // 2) 
                psi_limit_low = (total_pool_limit // 2) + (diff // 2)
            if psi_limit_high + psi_limit_low < total_pool_limit:
                if psi_limit_low < psi_limit_high:
                    psi_limit_low += 1
                else:
                    psi_limit_high += 1
            mean_point = find_index_by_value(pools_psi_sorted, mean)
            pools_low_psi = pools_psi_sorted[:mean_point]
            pools_high_psi = pools_psi_sorted[mean_point:]
            size_pools_low_psi = int(sum(j[0] for j in pools_low_psi))
            size_pools_high_psi = int(sum(j[0] for j in pools_high_psi))
            p_low = np.array([j[3] for j in pools_low_psi])
            m1 = knapsack2D(pools_high_psi, max_sizeA = psi_limit_high, max_sizeB = num_hits, score_index = 1, A = 0, B = 4)
            m2 = knapsack2D(pools_low_psi, max_sizeA = psi_limit_low, max_sizeB = num_hits, score_index = 1, A = 0, B = 4)
            chosen1 = traceback2D(m1, pools_high_psi, A = A, B = B)
            chosen2 = traceback2D(m2, pools_low_psi, A = A, B = B)
            high_psi_samples = [pools_high_psi[j] for j in chosen1]
            low_psi_samples = [pools_low_psi[j] for j in chosen2]
            p_high_n = sum(j[0] for j in high_psi_samples)
            p_low_n = sum(j[0] for j in low_psi_samples)
            unchosen1 = [pools_high_psi[j] for j in range(len(pools_high_psi)) if j not in chosen1]
            unchosen2 = [pools_low_psi[j]  for j in range(len(pools_low_psi))  if j not in chosen2] 
            chosen1_families = set(j[2] for j in high_psi_samples)
            chosen2_families = set(j[2] for j in low_psi_samples)
            unchosen2_families = set(j[2] for j in unchosen2)
            if p_high_n < psi_limit_high:
                pools_extra_psi = unchosen1 + unchosen2
                m3 = knapsack2D(pools_extra_psi, max_sizeA = psi_limit_high - p_high_n, max_sizeB = num_hits)
                chosen3 = traceback2D(m3, pools_extra_psi, A = A, B = B)
                high_psi_samples += [pools_extra_psi[j] for j in chosen3]
                unchosen3 = [pools_extra_psi[j] for j in range(len(pools_extra_psi)) if j not in chosen3]
            elif p_low_n < psi_limit_low:
                pools_extra_psi = unchosen1 + unchosen2
                m3 = knapsack2D(pools_extra_psi, max_sizeA = psi_limit_low - p_low_n, max_sizeB = num_hits)
                chosen3 = traceback2D(m3, pools_extra_psi, A = A, B = B)
                low_psi_samples += [pools_extra_psi[j] for j in chosen3]
                unchosen3 = [pools_extra_psi[j] for j in range(len(pools_extra_psi)) if j not in chosen3]
            else:
                unchosen3 = unchosen1 + unchosen2
            splits_out.append(high_psi_samples + low_psi_samples)
            pools[i + 1] += unchosen3
        else:
            size_of_curr_pool = sum(k[0] for k in pools[i])
            m = knapsack2D(pools[i], max_sizeA = int(proportions[i]), max_sizeB = num_hits, score_index = 1, A = 0, B = 4)
            chosen = traceback2D(m, pools[i], A = A, B = B)
            families_chosen = [pools[i][j] for j in chosen]
            families_not_chosen = [pools[i][j] for j in range(len(pools[i])) if j not in chosen]
            splits_out.append([pools[i][j] for j in chosen])
            pools[i + 1] += [pools[i][j] for j in range(len(pools[i])) if j not in chosen]
    splits_out.append(pools[2])
    return splits_out


def get_split_families_balanced_psi(family_list, proportions = None, pool_sizes = None, size_index = 0, cov_index = 1, score_index = 1, num_hits = 859, A = 0, B = 4, mean_psi = None, balance_psi = True, fam_size_has_ref = False):
    """
    Gets split families using 2d knapsack to 
    bound the number of hits in each split (keeping them equal)
    and also with PSI balancing. If num_hits = 1, ignore the hits.
    if mean_psi is passed, assume some samples from a previous split
    exists (corresponding to the train val and test proportions).
    mean_psi is of the form [num_below_mean, num_above_mean, mean] for 
    each split. If balance_psi is False, ignore PSI balancing and make
    splits purely based on the coverage.
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
        # families are excluded form the train set
        pool_sizes = {}
        pool_sizes["test"] = 0.33
        pool_sizes["val"] = 0.33
    
    # sort by average coverage. Sorting by total coverage for the concatenation
    # means the test set will get very large families, lowering the chance the test
    # set will have a high total coverage given its size constraints
    if fam_size_has_ref:
        sorted_by_coverage = sorted(family_list, key = lambda x : -(x[cov_index]/(x[size_index])))
    else:
        sorted_by_coverage = sorted(family_list, key = lambda x : -(x[cov_index]/(x[size_index] + 1)))
    
    # the actual size of each split is given by the sum of the family sizes
    n = sum([i[0] for i in family_list])
    n1 = int(n * pool_sizes["test"])
    n2 = int(n * pool_sizes["val"])
    
    # concatenate every third family together and 
    # create pools to optimize splits form
    list_1 = sorted_by_coverage[::3]
    list_2 = sorted_by_coverage[1::3]
    list_3 = sorted_by_coverage[2::3]
    concat_list = list_1 + list_2 + list_3
    
    test_indx = find_index_by_size(concat_list, n1)
    val_indx = find_index_by_size(concat_list, n2, start_index = test_indx)
    
    
    test_pool = concat_list[0:test_indx]
    val_pool = concat_list[test_indx:val_indx]
    train_pool = concat_list[val_indx:]
    
    hit_proportion = num_hits / n
    
    pools = [test_pool, val_pool, train_pool]
    
    proportions = [n * proportions["test"], n * proportions["val"]]
    
    splits_out = []
    for i in range(len(proportions)):
        print("-" * 20)
        print("start proportions i =", i)
        print("len current pool", len(pools[i]))
        print("size current pool", sum([j[0] for j in pools[i]]))
        if balance_psi:
            print("balancing psi")
            pools_psi_sorted = sorted(pools[i], key = lambda x : x[3])
            if mean_psi is not None and i < len(mean_psi):
                print('mean psi condition passed')
                num_below_mean, num_above_mean, mean = mean_psi[i]
                diff = num_below_mean - num_above_mean
                total_pool_limit = int(proportions[i])
                if diff > 0:
                    psi_limit_high = (total_pool_limit // 2) + (diff // 2) 
                    psi_limit_low = (total_pool_limit // 2) - (diff // 2) 
                else:
                    psi_limit_high = (total_pool_limit // 2) - (diff // 2) 
                    psi_limit_low = (total_pool_limit // 2) + (diff // 2)
                if psi_limit_high + psi_limit_low < total_pool_limit:
                    if psi_limit_low < psi_limit_high:
                        psi_limit_low += 1
                    else:
                        psi_limit_high += 1
            else:
                print('mean psi condition not given')
                psi_sum = sum([pools[i][j][3] for j in range(len(pools[i]))])
                mean = psi_sum / len(pools[i])
                size_pools = sum([pools[i][j][0] for j in range(len(pools[i]))])
                psi_limit_high = int(proportions[i]) // 2
                psi_limit_low = int(proportions[i]) // 2
            mean_point = find_index_by_value(pools_psi_sorted, mean)
            pools_low_psi = pools_psi_sorted[:mean_point]
            pools_high_psi = pools_psi_sorted[mean_point:]
            size_pools_low_psi = int(sum(j[0] for j in pools_low_psi))
            size_pools_high_psi = int(sum(j[0] for j in pools_high_psi))
            p_low = np.array([j[3] for j in pools_low_psi])
            print('starting knapsack for high psi pool and low psi pool separately')
            m1 = knapsack2D(pools_high_psi, max_sizeA = psi_limit_high, max_sizeB = num_hits, score_index = score_index, A = A, B = B)
            m2 = knapsack2D(pools_low_psi, max_sizeA = psi_limit_low, max_sizeB = num_hits, score_index = score_index, A = A, B = B)
            print("perform traceback for split high and low pools")
            print("pools high psi[0]", pools_high_psi[0])
            print("pools low psi[0]", pools_low_psi[0])
            print("len pools high psi", len(pools_high_psi))
            print("len pools low psi", len(pools_low_psi))
            print("m1.shape", m1.shape)
            print("m2.shape", m2.shape)
            chosen1 = traceback2D(m1, pools_high_psi, A = A, B = B)
            chosen2 = traceback2D(m2, pools_low_psi, A = A, B = B)
            print(".")
            high_psi_samples = [pools_high_psi[j] for j in chosen1]
            low_psi_samples = [pools_low_psi[j] for j in chosen2]
            print("..")
            p_high_n = sum(j[0] for j in high_psi_samples)
            p_low_n = sum(j[0] for j in low_psi_samples)
            print("...")
            unchosen1 = [pools_high_psi[j] for j in range(len(pools_high_psi)) if j not in chosen1]
            unchosen2 = [pools_low_psi[j]  for j in range(len(pools_low_psi))  if j not in chosen2] 
            print("....")
            chosen1_families = set(j[2] for j in high_psi_samples)
            chosen2_families = set(j[2] for j in low_psi_samples)
            print(".....")
            unchosen2_families = set(j[2] for j in unchosen2)
            unchosen1_families = set(j[2] for j in unchosen1)
            print("made unchosen data. Adding samples to reach psi limits")
            print("number of families chosen", len(chosen1_families) + len(chosen2_families))
            print("number of families unchosen", len(unchosen1_families) + len(unchosen2_families))
            
            if mean_psi is not None:
                if p_high_n < psi_limit_high:
                    pools_extra_psi = unchosen1 + unchosen2
                    m3 = knapsack2D(pools_extra_psi, max_sizeA = psi_limit_high - p_high_n, max_sizeB = num_hits)
                    chosen3 = traceback2D(m3, pools_extra_psi, A = A, B = B)
                    high_psi_samples += [pools_extra_psi[j] for j in chosen3]
                    unchosen3 = [pools_extra_psi[j] for j in range(len(pools_extra_psi)) if j not in chosen3]
                elif p_low_n < psi_limit_low:
                    pools_extra_psi = unchosen1 + unchosen2
                    m3 = knapsack2D(pools_extra_psi, max_sizeA = psi_limit_low - p_low_n, max_sizeB = num_hits)
                    chosen3 = traceback2D(m3, pools_extra_psi, A = A, B = B)
                    low_psi_samples += [pools_extra_psi[j] for j in chosen3]
                    unchosen3 = [pools_extra_psi[j] for j in range(len(pools_extra_psi)) if j not in chosen3]
                else:
                    unchosen3 = unchosen1 + unchosen2
            else:
                unchosen3 = unchosen1 + unchosen2
            print("add current split to output")
            splits_out.append(high_psi_samples + low_psi_samples)
            pools[i + 1] += unchosen3
        elif not balance_psi:
            print("not balancing psi")
            size_of_curr_pool = sum(k[0] for k in pools[i])
            m = knapsack2D(pools[i], max_sizeA = int(proportions[i]), max_sizeB = num_hits, score_index = score_index, A = A, B = B)
            chosen = traceback2D(m, pools[i], A = A, B = B)
            families_chosen = [pools[i][j] for j in chosen]
            families_not_chosen = [pools[i][j] for j in range(len(pools[i])) if j not in chosen]
            splits_out.append([pools[i][j] for j in chosen])
            pools[i + 1] += [pools[i][j] for j in range(len(pools[i])) if j not in chosen]
        print()
    print("returning")
    splits_out.append(pools[2])
    return splits_out
