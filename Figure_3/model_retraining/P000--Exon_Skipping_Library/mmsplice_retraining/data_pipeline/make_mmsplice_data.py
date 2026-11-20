import numpy as np
import pandas as pd
import sys
import os

"""
One-hot encode subsequences that are expected for each module.
To keep this step consistent with MMSplice's repo, using kipoiseq for encoding.
kipoiseq = https://github.com/kipoi/kipoiseq
It's right padding with N's to achieve the same length given a seq_vec.
"""
import kipoiseq.transforms.functional as F


if __name__ == "__main__":
     ## Import Project
    node = os.getcwd()
    for i in range(5): # max depth
        if "src" in os.listdir(node):
            tree, node = os.path.split(node)
            break
        tree, node = os.path.split(node)
        node = tree
    PROJECT_ROOT = os.path.join(tree, node)
    if not os.path.exists(PROJECT_ROOT) or "src" not in os.listdir(PROJECT_ROOT):
        raise ModuleNotFoundError("Could not find project root with 'src' directory")
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

from src.esl import cell_types, cell_types2, clip, logit
from src.esl import SMN2_INTRON6 as SMN2_intron6
from src.esl import SMN2_INTRON7 as SMN2_intron7


def calc_average_coverage(cov_df, cell_types):
    cell_types = [i + "_total_pooled" for i in cell_types]
    cell_types_ref = [i + "_ref" for i in cell_types]
    total = cov_df[cell_types].mean(axis=1)
    total_ref = cov_df[cell_types_ref].mean(axis = 1)
    return total, total_ref


def calc_average_psi(cov_df, cell_types):
    cell_types = [i + "_pooled_psi_raw" for i in cell_types]
    total = cov_df[cell_types].apply(clip).mean(axis=1)
    return total


def find_valid_mmsplice_variants(df):
    """
    check if variant locations are
    within the MMSplice input window
    """
    len_i1 = len(df["intron1"])
    len_ex = len(df["exon"])
    len_i2 = len(df["intron2"])
    valid_ranges = [
    [len_i1 - 100, len_i1 - 7], # acceptor intron
    [len_i1 - 50, len_i1 + min(len_ex - 1, 2)], # acceptor
    [max(len_i1 + len_ex - 5, len_i1), len_i1 + len_ex + min(len_i2, 12)], # donor
    [len_i1 + len_ex + 6, min(len_i1 + len_ex + len_i2, len_i1 + len_ex + 99)], # donor intron
    [len_i1, len_i1 + len_ex - 1] # exon
    ]
    snps = df["snp"].split(";")
    for snp in snps:
        loc, _ = snp.split(":")
        loc = int(loc)
        has_valid = False
        for (l,r) in valid_ranges:
            if ((loc <= r) and (loc >= l)):
                has_valid = True
                break
        if not has_valid:
            return False
    return True


def find_ref_margin_PSI(df):
    """
    check if the sequence has a ref PSI of 0 or 1
    in any cell line
    """
    ref_psi_cols = [f'{c}_pooled_psi_raw_ref'    for c in cell_types]
    margin_psi = df[ref_psi_cols].apply(lambda row: (row == 0).any() or (row == 1).any(), axis=1)
    return margin_psi


def get_non_margin_split(x, y, metadata, merged_df = None):
    """
    get x and y for a given split with ref PSI
    at 0 or 1 (for any cell type) dropped.

    modify split post-split generation
    for fine tuning purposes (i.e. only on train).
    """
    if "is_margin" not in metadata and merged_df is None:
        raise ValueError("Unable to resolve margin PSIs without 'is_margin' column or merged_df")
    if "is_margin" not in metadata:
        margin_psi = find_ref_margin_PSI(merged_df)
        merged_df["is_margin"] = margin_psi
        metadata = extract_sequences_from_metadata(merged_df, metadata)
    margin = metadata["is_margin"]
    non_margin = np.logical_not(margin) 
    x_new = [i[non_margin] for i in x]
    y_new = y[non_margin]
    return x_new, y_new


### Making splits
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


def extract_size_coverage_psi(x):
    return (len(x), np.sum(x["Avg_coverage"]) + np.mean(x["Avg_coverage_ref"]), np.mean(x["Avg_PSI"]), np.sum(x["Differential"]))
    

# Returns the families that are present in the dataset (splice_vs_super_full)
def filter_families_in_set(family_list, family_set):
    family_filtered = []
    for i in family_list:
        if i[2] in family_set:
            family_filtered.append(i)
    return family_filtered


def verify_exists_margin_info(metadata, opts):
    if "is_margin" in metadata:
        return
    elif "--merged_df_path" in opts:
        merged_df_path = opts["--merged_df_path"]
        return pd.read_csv(merged_df_path)
    return


def extract_sequences_from_metadata(merged_df, metadata):
    """
    Filter rows in df that are present
    in metadata
    """
    merged = merged_df.copy()
    meta = metadata.copy()
    merged["_key"] = list(zip(merged["var_id"], merged["intron1_size"]))
    meta["_key"] = list(zip(meta["var_id"], meta["intron1_size"]))
    out = (
        meta[["_key"]]
        .merge(merged, on="_key", how="left")
        .drop(columns="_key")
    )
    return out


def get_families_mmsplice(merged_df, keys = None):
    # filter out sequences without a reference
    # get either sequences that have a reference,
    # or is a reference itself
    # note that splice_vs_super_full does not have any references,
    # so lone reverences in coverage_df_filtered will either have a 
    # corresponding variant in the dataset or will be filtered
    avg_coverage, avg_coverage_ref = calc_average_coverage(merged_df, cell_types)
    avg_psi = calc_average_psi(merged_df, cell_types)
    merged_df["Avg_coverage"] = avg_coverage
    merged_df["Avg_coverage_ref"] = avg_coverage_ref
    merged_df["Avg_PSI"] = avg_psi
    eid_col = "event_id_161" if "event_id_161" in merged_df else "event_id"

    if "Differential" not in merged_df:
        merged_df["Differential"] = 0

    if keys is None:
        keys = ["Avg_coverage", "Avg_coverage_ref", "Avg_PSI", "Differential"]

    def extract(x):
        return (len(x), np.sum(x[keys[0]]) + np.mean(x[keys[1]]), np.mean(x[keys[2]]), np.sum(x[keys[3]]))
            
    fams = merged_df.groupby(eid_col)[keys].apply(extract)
    #              size       cov     eid   psi      diff
    family_list = [(i[1][0], i[1][1], i[0], i[1][2], i[1][3]) for i in fams.items()]

    # remove nans
    family_list_no_nan = []
    for i in family_list:
        if not np.isnan(i[1]):
            family_list_no_nan.append(i)
    family_list = family_list_no_nan

    family_set = set(merged_df[eid_col])
    family_list = filter_families_in_set(family_list, family_set)

    for i in family_list:
        if not i[0] > 0:
            if debug:
                breakpoint()
            else:
                raise ValueError("Error in making family list: family size is zero")
        assert i[0] > 0, i
    return family_list, merged_df


def visualize_splits(train, val, test):
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
    return splits_df


### Getting sequences and labels from splits
def process_sequences(exon, intron1, intron2):
    """
    process the sequences as defined by MMSplice.
    Gets the acceptor_intron, acceptor, donor and donor_intron
    inputs (along with the exon) for MMSplice from the 
    exon, intron1 and intron2 in the input dataframe
    """
    acceptor_intron = (SMN2_intron6 + intron1)[-100:-6]
    acceptor = (SMN2_intron6 + intron1)[-50:] + exon[:3]
    donor = exon[-5:] + intron2[:13]
    donor_intron = (intron2 + SMN2_intron7)[6:100]
    return (acceptor_intron, acceptor, exon, donor, donor_intron)


def prepare_seqs(x):
    """
    necessary to make columns into empty strings for assignment later
    """
    x["acceptor_intron"] = [""] * len(x)
    x["acceptor"] = [""] * len(x)
    x["donor"] = [""] * len(x)
    x["donor_intron"] = [""] * len(x)
    x["acceptor_intron_ref"] = [""] * len(x)
    x["acceptor_ref"] = [""] * len(x)
    x["donor_ref"] = [""] * len(x)
    x["donor_intron_ref"] = [""] * len(x)
    return x


def get_seqs(df):
    """
    run process_sequences on exon and introns to 
    get the appropriate donor and acceptors
    """
    acceptor_intron, acceptor, exon, donor, donor_intron = process_sequences(df["exon"], df["intron1"], df["intron2"])
    df["acceptor_intron"] = acceptor_intron
    df["acceptor"] = acceptor
    df["exon"] = exon
    df["donor"] = donor
    df["donor_intron"] = donor_intron
    # repeat for reference seqs
    if "exon_ref" in df:
        acceptor_intron_ref, acceptor_ref, exon_ref, donor_ref, donor_intron_ref = process_sequences(df["exon_ref"], df["intron1_ref"], df["intron2_ref"])
        df["acceptor_intron_ref"] = acceptor_intron_ref
        df["acceptor_ref"] = acceptor_ref
        df["exon_ref"] = exon_ref
        df["donor_ref"] = donor_ref
        df["donor_intron_ref"] = donor_intron_ref
    return df


def prepare_seq_df(merged_df, eid_col = "event_id_161"):
    """
    Create acceptor_intron, acceptor, donor and donor_intron
    columns in the merged_df dataframe, allowing for these
    to be quickly referenced to make the splits and metadata
    """
    merged_df = merged_df.reset_index()
    grouped_df = merged_df.groupby(eid_col).groups
    seqs_df = merged_df
    seqs_df = prepare_seqs(seqs_df)
    splice_types = ["acceptor_intron", "acceptor", "exon", "donor", "donor_intron"]
    splice_types += [i + "_ref" for i in splice_types]
    seqs_df = seqs_df.apply(get_seqs, axis = 1)
    outputs = ["mean_dlogit", "var_id", "ref_id"]
    for i in outputs:
        seqs_df[i] = merged_df[i]
    return seqs_df, grouped_df


def get_all_metadata_cols(df):
    """
    Get non-psi and non-sequence information
    """
    cols = []
    cell_set = set(cell_types) | set(cell_types2)
    for i in df.keys():
        if (("intron" in i and "_len" not in i) or
           ("exon" in i and "_len" not in i and "gene" not in i) or
           (i.split("_")[0] in cell_set) or
           ("logit" in i) or 
           ("donor" in i) or
           ("acceptor" in i)):
            continue
        cols.append(i)
    return cols


def get_sequences_from_families_MMSplice(dataset, seqs_df, grouped_df, metadata_output = None, eid_col = "event_id_161"):
    """
    Gets sequences, Y values (delta logit) and other identifying
    information (ref number, intron1 sizes) according to the 
    exon families in the given split.

    dataset -> a list of tuples representing each family in
               the split. The element at index 2 is always the 
               event_id. This may contain alternate isoforms
               that are identified by the combination of variant
               id and intron1 size.
    seqs_df -> a processed version of the dataframe containing
               the MMSplice inputs.
    grouped_df -> a grouping of the dataframe mapping each event_id
                  to indexes in seqs_df
    metadata_output -> column names to pull from dataframe to form metadata for split

    Outputs:
    data_variants -> dictionary containing MMSplice input sequences for
                     each variant
    data_refs -> same as data_variants but for the corresponding ref sequence
    logits_ -> delta logit values (var - ref) of the corresponding sequences
    metadata -> other information pulled for each sample in the split
    """
    if metadata_output is None:
        metadata_output = get_all_metadata_cols(seqs_df)
    outputs = ["mean_dlogit"] + metadata_output
    splice_types = ["acceptor_intron", "acceptor", "exon", "donor", "donor_intron"]
    splice_types += [i + "_ref" for i in splice_types]
    info_out = splice_types + outputs
    data_variants = {}
    metadata = {i : [] for i in outputs if i != "mean_dlogit"}
    metadata[eid_col] = []
    logits_ = []
    ref_nums = []
    var_nums = []
    i1_sizes = []
    is_margin = []
    ev_ids = []
    data_refs = {}
    for i in splice_types:
        if "_ref" in i:
            data_refs[i] = []
        else:
            data_variants[i] = []
    indx = 0
    for data in dataset:
        event_id = data[2]
        event_id_indexes = grouped_df[event_id]
        try:
            family_df = seqs_df.iloc[event_id_indexes][info_out]
        except IndexError:
            breakpoint()
        # populate the data with the sequences
        for i in splice_types:
            if "_ref" in i:
                data_refs[i]     += family_df[i].tolist()
            else:
                data_variants[i] += family_df[i].tolist()
        logits_  += family_df["mean_dlogit"].tolist()
        for i in metadata_output:
            try:
                metadata[i] += family_df[i].tolist()
            except KeyError:
                breakpoint()
        # metadata[eid_col]   += [event_id] * len(family_df)
    return data_variants, data_refs, logits_, metadata


### One hot encoding and organizing
def encodeDNA(seq_vec, max_len = None):
    """
    Taken from MMSplice.utils. 
    Modified to accept max_len parameter.
    This is useful for padding changing exon
    sizes.
    """
    if max_len is None:
        max_len = max(map(len, seq_vec))
    return np.array([
        F.one_hot(F.pad(seq, max_len, anchor="start"), neutral_value=0, neutral_alphabet=['N', '*'])
        for seq in seq_vec
    ])


def one_hot_encode_all(train, val, test, max_exon_len = 90):
    """
    one hot encode all the sequence types in 
    each dataset using a dictionary to keep track of 
    the sequences. Each train, val, test
    is a list of two dictionaries, corresponding to the
    variant and reference sequences in any order.
   
    dictionary keys correspond to the MMSplice inputs.
    If the sequence corresponded to a reference,
    then each key will have a _ref suffix.
    """
    # e.g. train = (train_variants, train_refs)
    datasets = ("train", "val", "test")
    assert [i for i in train[0]] == [i for i in val[0]] == [i for i in test[0]]
    seqs_one_hot = {}
    dataset_key = {"train":train, "val":val, "test":test}
    for dataset in datasets:
        for i in range(len(dataset_key[dataset])):
            seq_types = dataset_key[dataset][i].keys()
            for seq_type in seq_types:
                try:
                    if seq_type == "exon" or seq_type == "exon_ref":
                        seqs_one_hot[f"{dataset}_{seq_type}"] = \
                                encodeDNA(dataset_key[dataset][i][seq_type], max_len = max_exon_len)
                    else:
                        seqs_one_hot[f"{dataset}_{seq_type}"] = \
                                encodeDNA(dataset_key[dataset][i][seq_type])
                except KeyError as e:
                    breakpoint()
                except AssertionError as e:
                    breakpoint()
    return seqs_one_hot


def process_y(y):
    """
    process the outputs into the correct shape for 
    the model
    """
    y = np.array(y)
    y = y.T
    if len(y.shape) == 1:
        y = np.reshape(y, (len(y), 1))
    return y


def get_one_hots_from_splits(train, val, test, merged_df):
    """
    pipeline to turn splits into the final one-hot encoded
    sequences and corresponding Y-values and metadata.

    Each train, val, test is a list of tuples representing
    each exon family in that split. 
    
    Each split is passed to get_sequences_from_families_MMSplice
    to get the string sequences, then passed into one_hot_encode_all.
    Each sequence one hot is put into a dictionary (seqs_one_hot)
    and then referenced by the correct dataset (train, val, test)
    to output the final X values. The Y-values are also organized, and
    metadata (id numbers, intron sizes, etc.) are also returned.
    """
    eid_col = "event_id_161" if "event_id_161" in merged_df else "event_id"
    # get all the variants and corresponding reference for each family,
    # repeating for each of test, val and train splits
    seqs_df, grouped_df = prepare_seq_df(merged_df)
    test_variants, test_refs, test_logits, test_metadata = get_sequences_from_families_MMSplice(test,  seqs_df, grouped_df, eid_col = eid_col)
    val_variants, val_refs, val_logits, val_metadata = get_sequences_from_families_MMSplice(val,   seqs_df, grouped_df, eid_col = eid_col)
    train_variants, train_refs, train_logits, train_metadata = get_sequences_from_families_MMSplice(train, seqs_df, grouped_df, eid_col = eid_col)

    max_exon_len = merged_df["exon"].apply(len).max()
    # one hot encode every sequence in all the datasets. This is an internal function
    # that uses positional arguments to put the correct sequences into the dictionary keys 
    # the idea is to run get_one_hots_from_splits directly from the train, val and test families
    seqs_one_hot = one_hot_encode_all((train_variants, train_refs), (val_variants, val_refs), (test_variants, test_refs), max_exon_len = max_exon_len)
    x_train = [
        seqs_one_hot["train_acceptor_intron_ref"],
        seqs_one_hot["train_acceptor_ref"],
        seqs_one_hot["train_exon_ref"],
        seqs_one_hot["train_donor_ref"],
        seqs_one_hot["train_donor_intron_ref"],
        seqs_one_hot["train_acceptor_intron"],
        seqs_one_hot["train_acceptor"],
        seqs_one_hot["train_exon"],
        seqs_one_hot["train_donor"],
        seqs_one_hot["train_donor_intron"],
    ]
    x_val = [
        seqs_one_hot["val_acceptor_intron_ref"],
        seqs_one_hot["val_acceptor_ref"],
        seqs_one_hot["val_exon_ref"],
        seqs_one_hot["val_donor_ref"],
        seqs_one_hot["val_donor_intron_ref"],
        seqs_one_hot["val_acceptor_intron"],
        seqs_one_hot["val_acceptor"],
        seqs_one_hot["val_exon"],
        seqs_one_hot["val_donor"],
        seqs_one_hot["val_donor_intron"],
    ]
    x_test = [
        seqs_one_hot["test_acceptor_intron_ref"],
        seqs_one_hot["test_acceptor_ref"],
        seqs_one_hot["test_exon_ref"],
        seqs_one_hot["test_donor_ref"],
        seqs_one_hot["test_donor_intron_ref"],
        seqs_one_hot["test_acceptor_intron"],
        seqs_one_hot["test_acceptor"],
        seqs_one_hot["test_exon"],
        seqs_one_hot["test_donor"],
        seqs_one_hot["test_donor_intron"],
    ]

    y_train = process_y(train_logits)
    y_val   = process_y(val_logits)
    y_test  = process_y(test_logits)

    one_hots = {"x_train"        : x_train,
               "x_val"          : x_val,
               "x_test"         : x_test,
               "y_train"        : y_train,
               "y_val"          : y_val,
               "y_test"         : y_test,
               "metadata_train" : train_metadata,
               "metadata_val"   : val_metadata,
               "metadata_test"  : test_metadata,
    }
    return one_hots
