import numpy as np
"""
One-hot encode subsequences that are expected for each module.
To keep this step consistent with MMSplice's repo, using kipoiseq for encoding.
kipoiseq = https://github.com/kipoi/kipoiseq
It's right padding with N's to achieve the same length given a seq_vec.
"""
import kipoiseq.transforms.functional as F

from src.esl import SMN2_INTRON6 as SMN2_intron6
from src.esl import SMN2_INTRON7 as SMN2_intron7
from src.esl import cell_types, cell_types2, cell_type_psi


"""
Taken from MMSplice.utils
"""
def encodeDNA(seq_vec):
    max_len = max(map(len, seq_vec))
    return np.array([
        F.one_hot(F.pad(seq, max_len, anchor="start"), neutral_value=0, neutral_alphabet=['N', '*', 'm'])
        for seq in seq_vec
    ])


# process the sequences as defined by MTSplice
def process_sequences_mt(exon, intron1, intron2):
    original_sizes = (len(exon), len(intron1), len(intron2))
    if len(exon) < 100:
        exon_L = exon + "m"*(100-len(exon))
        exon_R = "m"*(100-len(exon)) + exon
    else:
        exon_L = exon[:100]
        exon_R = exon[-100:]
    assert len(exon_L) == len(exon_R) == 100
    if len(intron1) + len(SMN2_intron6) < 300: # if intron is really short
        padding = 300 - len(intron1) - len(SMN2_intron6)
        intron1 = "N" * padding + SMN2_intron6 + intron1
    elif len(intron1) < 300:
        intron1 = (SMN2_intron6 + intron1)[-300:]
    if len(intron2) < 300:
        intron2 = (intron2 + SMN2_intron7)[:300]
    if len(intron1) > 300:
        intron1 = intron1[-300:]
    if len(intron2) > 300:
        intron2 = intron2[:300]
    assert len(intron1) == len(intron2) == 300, f"Expected processed introns 1, 2 to be of size 300, 300 but got {len(intron1)},  {len(intron2)}. The original inputs to process_sequences_mt has the following sizes: exon size = {original_sizes[0]}, intron1 size = {original_sizes[1]}, intron2 size = {original_sizes[2]}"
    seqL = intron1 + exon_L
    seqR = exon_R + intron2
    return seqL, seqR


# necessary to make columns into empty strings for assignment later
def prepare_seqs(x):
    x["seqL"] = ""
    x["seqR"] = ""
    x["acceptor_intron"] = ""
    x["acceptor"] = ""
    x["donor"] = ""
    x["donor_intron"] = ""
    return x


def process_sequences_mm(exon, intron1, intron2):
    acceptor_intron = (SMN2_intron6 + intron1)[-100:-6]
    acceptor = (SMN2_intron6 + intron1)[-50:] + exon[:3]
    donor = exon[-5:] + intron2[:13]
    donor_intron = (intron2 + SMN2_intron7)[6:100]
    return (acceptor_intron, acceptor, exon, donor, donor_intron)


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
    seqL, seqR, acceptor_intron, acceptor, exon, donor, donor_intron = process_sequences_mm_mt(x["exon"], x["intron1"], x["intron2"])
    x["seqL"] = seqL
    x["seqR"] = seqR
    x["acceptor_intron"] = acceptor_intron
    x["acceptor"] = acceptor
    x["exon"] = exon
    x["donor"] = donor
    x["donor_intron"] = donor_intron
    return 


def prepare_df_for_var_ref_matching(in_all_cells_matchedrefvar):
    grouped_df = in_all_cells_matchedrefvar.reset_index().groupby("event_id").groups
    seqs_df = in_all_cells_matchedrefvar.reset_index()
    refs = seqs_df["Reference"].tolist()
    ref_to_idx = {int(refs[i]) : i for i in range(len(refs))}
    # seqs_df = seqs_df.apply(prepare_seqs, axis = 1)
    splice_types = ["acceptor_intron", "acceptor", "exon", "donor", "donor_intron", "seqL", "seqR"]
    # TODO: Find a more efficient way to do this (probably without using pandas)
    seqs_df = seqs_df[["exon", "intron1", "intron2"]]
    seqs_df = seqs_df.apply(prepare_seqs, axis = 1)
    seqs_df.apply(get_seqs, axis = 1)
    seqs_df = seqs_df[["intron1", "intron2"] + splice_types]
    return seqs_df, grouped_df, ref_to_idx


def get_sequences_from_families_mt(dataset,\
                                   cell_type_scale="psi",\
                                   avg_scale = "logit",\
                                   split_ref_and_var = False,\
                                   return_ref_nums = False,\
                                   seqs_df = None,\
                                   in_all_cells_matchedrefvar = None,\
                                   grouped_df = None,\
                                   ref_to_idx = None,\
                                   ref_eventID_supertableRef_dict = None):
    if split_ref_and_var:
        splice_types = ["acceptor_intron", "acceptor", "exon", "donor", "donor_intron", "seqL", "seqR"]
    else:
        splice_types = ["seqL", "seqR"]
    if avg_scale == "logit": # should be adding the mean logit to the logit predictions
        scaling_factor_key = "Avg_logit"
    elif avg_scale == "psi": # should not be the case but useful for debugging
        print("Warning: Using PSI as scaling factor for model outputs")
        scaling_factory_key = "Avg_PSI"
    if cell_type_scale == "logit":
        outputs = [f"{i}_delta_logit_mean_cell" for i in cell_types] + [scaling_factor_key]
    elif cell_type_scale == "psi":
        outputs = [f"{i}_PSI" for i in cell_types] + [scaling_factor_key]
    elif cell_type_scale == "ref_logit":
        cell_type_values = [f"{i}_logit" for i in cell_types]
        outputs = cell_type_values + [scaling_factor_key]
    if return_ref_nums:
        outputs.append("Reference")
    for i in outputs:
        seqs_df[i] = in_all_cells_matchedrefvar.reset_index()[i]
  
    info_out = splice_types + outputs
    data_seqs = {}
    for i in splice_types:
        if split_ref_and_var:
            data_seqs[i] = [[], []]
        else:
            data_seqs[i] = []
    data_logits = {i: [] for i in outputs}
    for t in dataset:
        event_id = t[2]
        sample_nums = grouped_df[event_id]
        if split_ref_and_var:
            ref_num  = int(ref_eventID_supertableRef_dict[event_id])
            try:
                ref_idx = ref_to_idx[ref_num]
            except KeyError as e:
                print(event_id)
                raise e
            variant_nums = [i for i in sample_nums if i != ref_to_idx[ref_num]]
            variants_df = seqs_df.iloc[variant_nums][info_out]
            ref_df = seqs_df.iloc[ref_idx][info_out]
            for sqtype in splice_types:
                data_seqs[sqtype][0] += [ref_df[sqtype]] * len(variant_nums)
                data_seqs[sqtype][1] += variants_df[sqtype].tolist()
            for logit in outputs:
                ref_logit = ref_df[logit]
                variant_logits = variants_df[logit]
                if "Avg" not in logit:
                    data_logits[logit] += (variant_logits).tolist()
                else:
                    data_logits[logit] += [ref_logit] * len(variant_nums)
        else:
            seq_logit_df = seqs_df.iloc[sample_nums][info_out]
            for i in splice_types:
                data_seqs[i] += seq_logit_df[i].tolist()
            for logit in outputs:
                data_logits[logit] += seq_logit_df[logit].tolist()
    return data_seqs, data_logits


# one hot encode all the sequence types in 
# each dataset using a dictionary to keep track of 
# the sequences
def one_hot_encode_all_mt(train, val, test, split_ref_and_var = False):
    # e.g. train = (train_variants, train_refs)
    datasets = ("train", "val", "test")
    seq_types = [i for i in train]
    seqs_one_hot = {}
    dataset_key = {"train":train, "val":val, "test":test}
    if split_ref_and_var:
        reference_keys = ("_ref", "_variant")
    else:
        reference_keys = ("", )
    for dataset in datasets:
        for seq_type in seq_types:
            for ref_key in range(len(reference_keys)):
                if split_ref_and_var:
                    seqs_one_hot[f"{dataset}_{seq_type}{reference_keys[ref_key]}"] = \
                    encodeDNA(dataset_key[dataset][seq_type][ref_key])
                else:
                    try:
                        seqs_one_hot[f"{dataset}_{seq_type}"] = \
                        encodeDNA(dataset_key[dataset][seq_type])
                    except IndexError:
                        breakpoint()
    return seqs_one_hot


# Create the final input and output one hots for the model.
def get_one_hots_from_splits(train, val, test, in_all_cells_matchedrefvar=None, seqs_df = None, cell_type_scale = "psi", avg_scale = "logit", split_ref_and_var = False, return_ref_nums = False, ref_eventID_supertableRef_dict = None):
    # get all the variants and corresponding reference for each family,
    # repeating for each of test, val and train splits
    # logits includes the mean logits and the target tissue-specific logits.
    # could also be in the psi scale depending on what scaling_factor was.
    if seqs_df is None:
        assert in_all_cells_matchedrefvar is not None
        seqs_df, grouped_df, ref_to_idx = prepare_df_for_var_ref_matching(in_all_cells_matchedrefvar)

    test_seqs, test_logits = get_sequences_from_families_mt(\
         dataset = test,\
         cell_type_scale = cell_type_scale,\
         avg_scale = avg_scale,\
         split_ref_and_var = split_ref_and_var,\
         return_ref_nums = return_ref_nums,\
         seqs_df = seqs_df,\
         in_all_cells_matchedrefvar = in_all_cells_matchedrefvar,\
         grouped_df = grouped_df,\
         ref_to_idx = ref_to_idx,\
         ref_eventID_supertableRef_dict = ref_eventID_supertableRef_dict,\
         )
    val_seqs, val_logits = get_sequences_from_families_mt(\
          dataset = val,\
          cell_type_scale = cell_type_scale,\
          avg_scale = avg_scale,\
          split_ref_and_var = split_ref_and_var,\
          return_ref_nums = return_ref_nums,\
          seqs_df = seqs_df,\
          in_all_cells_matchedrefvar = in_all_cells_matchedrefvar,\
          grouped_df = grouped_df,\
          ref_to_idx = ref_to_idx,\
          ref_eventID_supertableRef_dict = ref_eventID_supertableRef_dict,\
          )
    train_seqs, train_logits = get_sequences_from_families_mt(\
           dataset = train,\
           cell_type_scale = cell_type_scale,\
           avg_scale = avg_scale,\
           split_ref_and_var = split_ref_and_var,\
           return_ref_nums = return_ref_nums,\
           seqs_df = seqs_df,\
           in_all_cells_matchedrefvar = in_all_cells_matchedrefvar,\
           grouped_df = grouped_df,\
           ref_to_idx = ref_to_idx,\
           ref_eventID_supertableRef_dict = ref_eventID_supertableRef_dict,\
           )
    if avg_scale == "logit":
        scaling_factor_key = "Avg_logit"
    elif avg_scale == "psi":
        scaling_factor_key = "Avg_PSI"
    # one hot encode every sequence in all the datasets. This is an internal function
    # that uses positional arguments to put the correct sequences into the dictionary keys 
    # the idea is to run get_one_hots_from_splits directly from the train, val and test families
    seqs_one_hot = one_hot_encode_all_mt(train_seqs, val_seqs, test_seqs, split_ref_and_var)
    if split_ref_and_var:
        keys = {data:[f"{data}_acceptor_intron_ref",
            f"{data}_acceptor_ref",
            f"{data}_exon_ref",
            f"{data}_donor_ref",
            f"{data}_donor_intron_ref",
            f"{data}_acceptor_intron_variant",
            f"{data}_acceptor_variant",
            f"{data}_exon_variant",
            f"{data}_donor_variant",
            f"{data}_donor_intron_variant",
            f"{data}_seqL_ref",
            f"{data}_seqR_ref",
            f"{data}_seqL_variant",
            f"{data}_seqR_variant"] for data in ("train", "val", "test")}
    else:
        keys = {data:[f"{data}_seqL", f"{data}_seqR"] for data in ("train", "val", "test")}
    if return_ref_nums:
        return train_logits["Reference"], val_logits["Reference"], test_logits["Reference"]
    x_train = [seqs_one_hot[k] for k in keys["train"]] + [train_logits[scaling_factor_key]]
    x_val = [seqs_one_hot[k] for k in keys["val"]] + [val_logits[scaling_factor_key]]
    x_test = [seqs_one_hot[k] for k in keys["test"]] + [test_logits[scaling_factor_key]]
    y_train = [
        train_logits[i] for i in train_logits if i != scaling_factor_key
    ]
    y_val = [
        val_logits[i] for i in val_logits if i != scaling_factor_key
    ]
    y_test = [
        test_logits[i] for i in test_logits if i != scaling_factor_key
    ]
    
    return x_train, x_val, x_test, y_train, y_val, y_test
    
    
# process the outputs into the correct shape for 
# the model
def process_y(y):
    y = np.array(y)
    y = y.T
    return y
