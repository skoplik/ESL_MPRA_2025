## Import Project
if __name__ == "__main__":
    import sys
    import os
    import getopt
    opts, args = getopt.getopt(sys.argv[1:],"", [
    "project_name="
    ]) # see if project name is provided, else use default project name
    opts = dict(opts)
    try:
        # try to import (e.g. if PYTHONPATH is set or installed via pip install -e)
        import src 
    except ModuleNotFoundError:
        # search parent directories for project name
        project_name = opts.get("--project_name", "P000--Exon_Skipping_Library")
        node = os.getcwd()
        for i in range(5): # max depth
            tree, node = os.path.split(node)
            if node == project_name:
                break 
            else:
                node = tree
        PROJECT_ROOT = os.path.join(tree, node)
        if node != project_name:
            # couldn't find project folder 
            raise ModuleNotFoundError(f"Could not find project directory {PROJECT_NAME}")
        if PROJECT_ROOT not in sys.path:
            sys.path.append(PROJECT_ROOT)

import pandas as pd
import csv
import numpy as np
from collections import defaultdict
from mtsplice_retraining.data_pipeline.mmsplice_data_split import get_split_families_mm
from mtsplice_retraining.data_pipeline.knapsack.knapsack2d import knapsack2D, traceback2D
from mtsplice_retraining.data_pipeline.knapsack.knapsack_helper import find_index_by_value, find_index_by_size, get_split_families_2D_knapsack
from mtsplice_retraining.data_pipeline.nan_handling.nan_helper_tsplice import get_nan_size_psi_and_coverage, get_not_nan_num
from mtsplice_retraining.data_pipeline.data_split_one_hot_tsplice import get_one_hots_from_splits, process_y
from src.esl import logit, clip, cell_types, cell_types2, cell_type_psi
from src.load_data import read_csvs, process_psi_csv, VARIANT_PATH, PSI_PATH, SUPERTABLE_PATH
from src.esl import SMN2_INTRON6, SMN2_INTRON7
import pickle
import getopt
import sys

# Updated functions for getting MTSplice split.
# This encapsulates the < 100nt filter from MMSplice

def mtsplice_baseline_csv(file_path):
    data=[]
    with open(file_path, 'r', newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:    
            data.append(row)
    df = {i:[]for i in data[0]}
    for i in range(1, len(data)):
        for j in range(len(data[0])):
            variable = data[0][j]
            if not data[i][j]:
                df[variable].append(np.nan)
            else:
                convert = int if variable == "Reference" else float
                df[variable].append(convert(data[i][j]))
    return  pd.DataFrame(df)


def get_mmsplice_split(merged_PSI_cov_logit_df, in_all_cells_matchedrefvar, splice_vs_super_double_variants):
    train_mm, val_mm, test_mm = get_split_families_mm(merged_PSI_cov_logit_df, splice_vs_super_double_variants)
    return train_mm, val_mm, test_mm


def get_mtsplice_split(merged_PSI_cov_logit_df, in_all_cells_matchedrefvar, splice_vs_super_double_variants):
    train_mm, val_mm, test_mm = get_split_families_mm(merged_PSI_cov_logit_df, splice_vs_super_double_variants)
    return train_mm, val_mm, test_mm


def get_data_splits(merged_PSI_cov_logit_df,\
                    in_all_cells_matchedrefvar,\
                    splice_vs_super_double_variants):
    train_m, val_m, test_m = get_mmsplice_split(merged_PSI_cov_logit_df,\
                                                in_all_cells_matchedrefvar,\
                                                splice_vs_super_double_variants)
    train_t,\
    val_t,\
    test_t,\
    has_ref,\
    is_none,\
    val_samples_to_add,\
    test_samples_to_add = get_tsplice_split(in_all_cells_matchedrefvar,\
                                            splice_vs_super_double_variants,\
                                            (train_m, val_m, test_m))
    train_c, val_c, test_c = get_combined_split(has_ref,\
                                                is_none,\
                                                in_all_cells_matchedrefvar,\
                                                train_m,\
                                                val_m,\
                                                test_m,\
                                                val_samples_to_add,\
                                                test_samples_to_add,\
                                               ) 
    return (train_m, val_m, test_m),\
            (train_t, val_t, test_t),\
            (train_c, val_c, test_c)


def one_hot_encode_filtered_mtsplice_split(dataset_mt,
                                           splice_vs_super,
                                           in_all_cells_matchedrefvar,
                                           ref_eventID_supertableRef_dict):

    # Combined model logit
    train_c, val_c, test_c = dataset_c # unpack combined splits
    x_train_logit_mtsplice, \
    x_val_logit_mtsplice, \
    x_test_logit_mtsplice, \
    y_train_logit_mtsplice, \
    y_val_logit_mtsplice, \
    y_test_logit_mtsplice = get_one_hots_from_splits(train_c,\
        val_c,\
        test_c,\
        in_all_cells_matchedrefvar=in_all_cells_matchedrefvar,\
        cell_type_scale="ref_logit",\
        split_ref_and_var = True,\
        ref_eventID_supertableRef_dict = ref_eventID_supertableRef_dict,\
    )

    y_train_logit_mtsplice = process_y(y_train_logit_mtsplice)
    y_val_logit_mtsplice = process_y(y_val_logit_mtsplice)
    y_test_logit_mtsplice = process_y(y_test_logit_mtsplice)
    arrs = []
    for i in range(len(x_val_logit_mtsplice)):
        if isinstance(x_val_logit_mtsplice, list):
            avg_index = i
    for i in [x_train_logit_mtsplice, x_val_logit_mtsplice, x_test_logit_mtsplice]:
        size = i[0].shape[0]
        arr = np.array(i[avg_index])
        arr = np.reshape(arr, (size, 1))
        arrs.append(arr)
    x_train_logit_mtsplice[avg_index] = arrs[0]
    x_val_logit_mtsplice[avg_index] = arrs[1]
    x_test_logit_mtsplice[avg_index] = arrs[2]


    # Get reference numbers to compare to baseline later
    train_ref_nums, val_ref_nums, test_ref_nums = get_one_hots_from_splits(\
        train_c,\
        val_c,\
        test_c,\
        in_all_cells_matchedrefvar=in_all_cells_matchedrefvar,\
        cell_type_scale="ref_logit",\
        split_ref_and_var = True,\
        return_ref_nums = True,\
        ref_eventID_supertableRef_dict = ref_eventID_supertableRef_dict,\
    )

    # Drop NaN from combined dataset
    val_non_nan = np.sum(np.isnan(y_val_logit_mtsplice), axis = 1) == 0
    test_non_nan = np.sum(np.isnan(y_test_logit_mtsplice), axis = 1) == 0

    x_val_logit_mtsplice = [i[val_non_nan] for i in x_val_logit_mtsplice]
    x_test_logit_mtsplice = [i[test_non_nan] for i in x_test_logit_mtsplice]
    y_val_logit_mtsplice = y_val_logit_mtsplice[val_non_nan]
    y_test_logit_mtsplice = y_test_logit_mtsplice[test_non_nan]

    # drop NaN in train only if every delta logit value is NaN, 8 sequences total
    train_non_nan = np.sum(np.isnan(y_train_logit_mtsplice), axis = 1) != 5
    x_train_logit_mtsplice = [i[train_non_nan] for i in x_train_logit_mtsplice]
    y_train_logit_mtsplice = y_train_logit_mtsplice[train_non_nan]

    train_ref_nums = np.array(train_ref_nums)[train_non_nan]
    val_ref_nums = np.array(val_ref_nums)[val_non_nan]
    test_ref_nums = np.array(test_ref_nums)[test_non_nan]
    
    data = {}
    data['x_train_logit_tsplice'] = x_train_logit_tsplice
    data['x_train_logit_mtsplice'] = x_train_logit_mtsplice
    data['x_val_logit_tsplice'] = x_val_logit_tsplice
    data['x_val_logit_mtsplice'] = x_val_logit_mtsplice
    data['x_test_logit_tsplice'] = x_test_logit_tsplice
    data['x_test_logit_mtsplice'] = x_test_logit_mtsplice
    data['y_train_logit_tsplice'] = y_train_logit_tsplice
    data['y_train_logit_mtsplice'] = y_train_logit_mtsplice
    data['y_val_logit_tsplice'] = y_val_logit_tsplice
    data['y_val_logit_mtsplice'] = y_val_logit_mtsplice
    data['y_test_logit_tsplice'] = y_test_logit_tsplice
    data['y_test_logit_mtsplice'] = y_test_logit_mtsplice
    data['train_ref_nums'] = train_ref_nums
    data['val_ref_nums'] = val_ref_nums
    data['test_ref_nums'] = test_ref_nums
    return data


def get_cell_ref_psi_from_refs(ref_nums, in_all_cells_matchedrefvar, ref_eventID_supertableRef_dict):
    """
    given refs (sample_ids) get all
    the corresponding psi values (for all cell lines)
    for the reference sequence
    """
    ref_to_id = dict(zip(in_all_cells_matchedrefvar['Reference'], in_all_cells_matchedrefvar['event_id']))
    all_refs = in_all_cells_matchedrefvar['Reference'].tolist()
    in_all_cells_matchedrefvar.reset_index()
    ref_to_indx = {all_refs[i] : i for i in range(len(all_refs))}
    cell_type_psi_numpy = in_all_cells_matchedrefvar[(cell_type_psi)].to_numpy()
    cell_ref_psi = np.zeros((len(ref_nums), 5))
    for i in range(len(ref_nums)):
        # get the event_id 
        event_id = ref_to_id[ref_nums[i]]
        # get the correct referenc
        ref = ref_eventID_supertableRef_dict[event_id]
        # get the psi values
        psi_indx = ref_to_indx[int(ref)]
        psi = cell_type_psi_numpy[psi_indx]
        cell_ref_psi[i] += psi
    return cell_ref_psi


def get_cell_psi_from_refs(ref_nums, in_all_cells_matchedrefvar, ref_eventID_supertableRef_dict):
    """
    given refs (sample_ids) get all
    the corresponding psi values (for all cell lines)
    for both the variant and its reference 
    if it is a reference, will return the same for both
    """
    ref_to_id = dict(zip(in_all_cells_matchedrefvar['Reference'], in_all_cells_matchedrefvar['event_id']))
    all_refs = in_all_cells_matchedrefvar['Reference'].tolist()
    in_all_cells_matchedrefvar.reset_index()
    ref_to_indx = {all_refs[i] : i for i in range(len(all_refs))}
    cell_type_psi_numpy = in_all_cells_matchedrefvar[(cell_type_psi)].to_numpy()
    cell_ref_psi = np.zeros((len(ref_nums), len(cell_type_psi)))
    cell_var_psi = np.zeros((len(ref_nums), len(cell_type_psi)))
    for i in range(len(ref_nums)):
        # get the event_id 
        event_id = ref_to_id[ref_nums[i]]
        # get the correct referenc
        ref = ref_eventID_supertableRef_dict[event_id]
        # get the psi values
        psi_ref_indx = ref_to_indx[int(ref)]
        psi_var_indx = ref_to_indx[int(ref_nums[i])]
        psi_ref = cell_type_psi_numpy[psi_ref_indx]
        psi_var = cell_type_psi_numpy[psi_var_indx]
        cell_ref_psi[i] += psi_ref
        cell_var_psi[i] += psi_var
    return cell_ref_psi, cell_var_psi


def remake_ref_psi_cols(splice_vs_super, ref_psi, var_psi):
    """
    remaking these columns are necessary to get
    the cell type effect for the variant - reference
    (i.e. what MTSplice outputs)
    """
    for i, cell_psi in enumerate(cell_type_psi):
        splice_vs_super[f"{cell_psi}_ref"] = ref_psi[:,i]
        splice_vs_super[f"{cell_psi}_var"] = var_psi[:,i]
    return splice_vs_super


def tsplice_data_pipeline(splice_vs_super_path = VARIANT_PATH,\
                          merged_PSI_path = PSI_PATH,\
                          supertable_file = SUPERTABLE_PATH,\
                          cell_type_scale = "logit",\
                          avg_scale = "logit",\
                          debug = False): 

    if debug:
        print("Reading CSVs")
    splice_vs_super_double_variants,\
    merged_PSI_cov_logit_df,\
    supertable = read_csvs(splice_vs_super_path, merged_PSI_path, supertable_file)
    if debug:
        print("processing psi data to get in all_cells_matchedrefvar")
    in_all_cells_matchedrefvar, in_all_cells_matchedrefvar_no_nan = process_psi_csv(merged_PSI_cov_logit_df)

    if debug:
        print("getting supertable seq dict")
    supertable_seq_dict = get_supertable_seq_dict(supertable_file)

    if debug:
        print("Getting data splits") 
    dataset_m, dataset_t, dataset_c = get_data_splits(merged_PSI_cov_logit_df,\
                                  in_all_cells_matchedrefvar,\
                                  splice_vs_super_double_variants)

    if debug:
        print("getting final, one hot encoded data")
    ref_eventID_supertableRef_dict = supertable["ref_eventID_supertableRef_dict"]
    data = one_hot_encode_splits(dataset_m,\
                                 dataset_t,\
                                 dataset_c,\
                                 in_all_cells_matchedrefvar,\
                                 ref_eventID_supertableRef_dict,\
                                 cell_type_scale = cell_type_scale,\
                                 avg_scale = avg_scale)
    if debug:
        breakpoint()
    return data


def process_sequences_mm_mt(exon, intron1, intron2):
    acceptor_intron = (SMN2_intron6 + intron1)[-100:-6]
    acceptor = (SMN2_intron6 + intron1)[-50:] + exon[:3]
    donor = exon[-5:] + intron2[:13]
    donor_intron = (intron2 + SMN2_intron7)[6:100]
    if len(exon) < 100:
        exon_L = exon + "m"*(100-len(exon))
        exon_R = "m"*(100-len(exon)) + exon
    else:
        exon_L = exon[:100]
        exon_R = exon[-100:]
    assert len(exon_L) == len(exon_R) == 100
    if len(intron1) < 300:
        intron1 = (SMN2_intron6 + intron1)[-300:]
    if len(intron2) < 300:
        intron2 = (intron2 + SMN2_intron7)[:300]
    assert len(intron1) == len(intron2) == 300, f"{len(intron1)},  {len(intron2)}"
    seqL = intron1 + exon_L
    seqR = exon_R + intron2
    return seqL, seqR, acceptor_intron, acceptor, exon, donor, donor_intron


# run process_sequences on exon and introns to 
# get the appropriate donor and acceptors
def get_seqs(x):
    seqL, seqR, acceptor_intron, acceptor, exon, donor, donor_intron = process_sequences_mt(x["exon"], x["intron1"], x["intron2"])
    x["seqL"] = seqL
    x["seqR"] = seqR
    x["acceptor_intron"] = acceptor_intron
    x["acceptor"] = acceptor
    x["exon"] = exon
    x["donor"] = donor
    x["donor_intron"] = donor_intron
    return 


### Move this later
def get_sequences_from_families_mt(dataset,
                                   splice_vs_super_full,
                                   in_all_cells_matchedrefvar,
                                   ref_eventID_supertableRef_dict,
                                   var_supertable_dict,
                                   ref_supertable_dict):
    # ref_eventID_supertableRef_dict : event_id ->  reference num
    # var_supertable_dict : reference num -> (exon, intron1, intron2)
    # var_supertable_dict : reference num -> list(variant nums)
    # dataset : list of families in the form (family_size, total_avg_coverage, event_id)

    # get variant nums from splice_vs_super (filtered)
    # but pull the values from in_all_cells_matchedrefvar (because it has reference values)
    # check if the delta_logit_pooled column is expected
    # furthermore, return the clipped PSI values for testing how other scalings look 
    # (i.e. after training)

    # TODO: Add Seq_L and Seq_R to splice types?
    grouped_df = splice_vs_super_full.groupby("event_id").groups
    seqs_df = in_all_cells_matchedrefvar
    seqs_df = prepare_seqs(seqs_df)
    splice_types = ["acceptor_intron", "acceptor", "exon", "donor", "donor_intron"]
    seqs_df = seqs_df[["intron1", "intron2"] + splice_types]
    seqs_df.apply(get_seqs, axis = 1)
    outputs = [f"{cell}_delta_logit_pooled" for cell in cell_types2]
    for i in outputs:
        seqs_df[i] = splice_vs_super_full[i]
    info_out = splice_types + outputs
    data_variants = {}
    logits_ = []
    mmsplice_predictions = []
    for i in splice_types:
        data_variants[i] = []
    data_refs = {}
    for i in splice_types:
        data_refs[i] = []
    indx = 0
    for data in dataset:
        event_id = data[2]
        ref_num  = ref_eventID_supertableRef_dict[event_id]
        variant_nums = grouped_df[event_id]
        variants_family_df = seqs_df.iloc[variant_nums][info_out]
        ref_seq = process_sequences(*ref_supertable_dict[ref_num])
        curr_ref_seq_by_type = {}
        for i in range(len(ref_seq)):
            curr_ref_seq_by_type[splice_types[i]] = ref_seq[i]
        # populate the data with the sequences
        for i in splice_types:
            data_variants[i] += variants_family_df[i].tolist()
            data_refs[i] += [curr_ref_seq_by_type[i]] * len(variant_nums)
        logits_ += variants_family_df["Supertable_Mean_Delta_Logit_PSI"].tolist()
        mmsplice_predictions += variants_family_df["MMSplice_Delta_Logit_PSI"].tolist()    
    return data_variants, data_refs, logits_, mmsplice_predictions 


def check_if_has_variants(df):
    """
    checks the output seqs for 
    MTSplice to see if the variant
    is within the window
    """
    return


if __name__ == "__main__":
    print("testing data pipeline")
    splice_vs_super_double_variants, merged_PSI_cov_logit_df, supertable = read_csvs()
    in_all_cells_matchedrefvar, in_all_no_nan = process_psi_csv(merged_PSI_cov_logit_df)

    train_mt, val_mt, test_mt = get_mtsplice_split(merged_PSI_cov_logit_df, in_all_cells_matchedrefvar, splice_vs_super_double_variants)

    ref_nums = splice_vs_super_double_variants["Reference"]
    ref_eventID_supertableRef_dict=supertable["ref_eventID_supertableRef_dict"]
    ref_psi, var_psi=get_cell_psi_from_refs(ref_nums, in_all_cells_matchedrefvar, ref_eventID_supertableRef_dict)
    breakpoint()

    splice_vs_super_double_variants = remake_ref_psi_cols(splice_vs_super_double_variants, ref_psi=ref_psi, var_psi=var_psi)
    

    for i, cell in enumerate(cell_types2):
        dlogit = splice_vs_super_double_variants[f"{cell}_delta_logit_pooled"]
        var = splice_vs_super_double_variants[f"{cell_types[i]}_PSI_var"]
        ref = splice_vs_super_double_variants[f"{cell_types[i]}_PSI_ref"]
        dlogit = dlogit[np.isfinite(dlogit)]
        dlogit_recalc = logit(var) - logit(ref)
        dlogit_recalc = dlogit_recalc[np.isfinite(dlogit_recalc)]
        print(np.isclose(dlogit, dlogit_recalc).mean())
        
    breakpoint()
    print("done")
