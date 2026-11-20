import pandas as pd
import csv
import numpy as np
from collections import defaultdict
import sys
import os

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(PROJECT_ROOT)

from src.data_helpers.knapsack.knapsack_helper import (
    get_split_families_balanced_psi
)
from mtsplice_retraining.data_pipeline.mmsplice_data_split import (
    get_split_families_mm
)
from src.data_helpers.knapsack.knapsack2d import (
    knapsack2D,
    traceback2D
)
from src.data_helpers.knapsack.knapsack_helper import (
    find_index_by_value,
    find_index_by_size,
    get_split_families_2D_knapsack
)
from src.data_helpers.nan_handling.nan_helper_tsplice import (
    get_nan_size_psi_and_coverage,
    get_not_nan_num
)
from mtsplice_retraining.data_pipeline.data_split_one_hot_tsplice import (
    get_one_hots_from_splits,
    process_y,
    one_hot_encode_all_mt
)
from src.esl import (
    logit,
    clip,
    cell_types,
    cell_types2,
    SMN2_INTRON6,
    SMN2_INTRON7,
)
from src.load_data import (
    read_csvs,
    process_psi_csv,
    load_splits_dir,
    read_supertable
)
from mmsplice_retraining.data_pipeline.make_mmsplice_data import (
    process_sequences as process_sequences_mm
)
from mmsplice_retraining.data_pipeline.saving import psi_distribution_plot
import pickle
import getopt
import matplotlib.pyplot as plt

cell_type_psi = ['HEK293_PSI', 'HeLa_PSI', 'K562_PSI', 'MCF7_PSI', 'HMC3_PSI']


def save_pickle(data, path, log=True):
    """helper function to save data"""
    with open(path, "wb") as f:
        pickle.dump(data, f)
    if log:
        print("Saved: {path}")
    return


def remove_split_nans(split,
        cell_types_=cell_types2,
        keys=None,
        prefix="",
        suffix=""
    ):
    """
    removes rows with any number of nans
    for a specific cell type key. The keys
    may be either passed in in full or
    build from cell_types_ using a prefix
    and/or a suffix.
    """
    if keys is None:
        if prefix and suffix:
            keys = [prefix + i + suffix for i in cell_types_]
        elif prefix:
            keys = [prefix + i for i in cell_types_]
        elif suffix:
            keys = [i + suffix for i in cell_types_]
        else:
            raise ValueError(
                "Could not identify cell keys "
                "to remove nans from. "
                "Expected prefix, suffix or keys to be passed in."
            )
    num_nans = split[keys].apply(np.isnan).sum(axis=1)
    return split[num_nans == 0]


def mtsplice_distribution_plot(
        train,
        val,
        test,
        data_dir,
        scale="PSI",
        save=True
    ):
    """
    makes plot of distribution of dlogits
    for each cell type for each split.
    """
    # PSI distributions of train, val and test
    fig, ax = plt.subplots(3, 5, figsize=(15, 20))
    for i in range(5):
        ax[0][i].hist(train[:,i])
        ax[0][i].set_title(f"Training Set {cell_types[i]}", fontsize=10)
        ax[0][i].set_xlabel(scale, fontsize=10)
        ax[0][i].set_ylabel("Frequency", fontsize=10)
        ax[1][i].hist(val[:,i])
        ax[1][i].set_title(f"Validation Set {cell_types[i]}", fontsize=10)
        ax[1][i].set_xlabel(scale, fontsize=10)
        ax[1][i].set_ylabel("Frequency", fontsize=10)
        ax[2][i].hist(test[:,i])
        ax[2][i].set_title(f"Test Set {cell_types[i]}", fontsize=10)
        ax[2][i].set_xlabel(scale, fontsize=10)
        ax[2][i].set_ylabel("Frequency", fontsize=10)
    # fig.supxlabel(scale, fontsize = 20)
    # fig.supylabel("Frequency", fontsize = 20)
    fig.suptitle(
        f"Distribution of {scale} Across Data Splits",
        fontsize=20, 
        y=1.01
    )
    save_dist_name = scale.lower().replace(" ", "_")
    plt.tight_layout()
    if save:
        plot_name = f"{save_dist_name}_distribution"
        path = os.path.join(data_dir, plot_name)
        plt.savefig(path + ".pdf", bbox_inches = "tight")
        plt.savefig(path + ".png", bbox_inches = "tight")
    return


def make_metadata(
        split,
        metadata,
        extra_keys=None
    ):
    """
    make new metadata for TSplice
    and MTSplice using keys from
    the MMSplice metadata.
    """
    metadata_new = {}
    if extra_keys is None:
        extra_keys = []
    for e in extra_keys:
        metadata_new[e] = split[e].tolist()
    if "metadata_train" in metadata:
        metadata = metadata["metadata_train"].keys()
    for key in metadata:
        if key in split:
            metadata_new[key] = split[key].tolist()
        elif key.endswith("len"):
            metadata_new[key] = split[key.split("_")[0]].apply(len).tolist()
    return pd.DataFrame(metadata_new)


def split_all_samples_df(
        all_samples_df, 
        train_fams,
        val_fams,
        test_fams,
        train_refs,
        val_refs,
        test_refs
    ):
    """
    separate references from variants
    using all_samples_df as input.
    Does so separately for each split
    using the MMSplice splits.
    """
    train_eids = set([i[2] for i in train_fams])
    val_eids   = set([i[2] for i in val_fams])
    test_eids  = set([i[2] for i in test_fams])
    if "num_nans" not in all_samples_df:
        all_samples_df["num_nans"] = all_samples_df[
            [
                f"{c}_pooled_psi_clipped"
                for c in cell_types2
            ]
        ].isna().sum(axis = 1)
    samples_train = all_samples_df[
        all_samples_df["event_id_161"]
        .isin(train_eids)
    ]
    samples_val = all_samples_df[
        (all_samples_df["event_id_161"].isin(val_eids)) 
        & (all_samples_df["num_nans"] == 0)
    ]
    samples_test = all_samples_df[
        (all_samples_df["event_id_161"].isin(test_eids))
        & (all_samples_df["num_nans"] == 0)
    ]
    train_refs_set = set(train_refs)
    val_refs_set = set(val_refs)
    test_refs_set = set(test_refs)
    samples_train2 = all_samples_df[
        all_samples_df["Reference"]
        .isin(train_refs_set)
    ]
    samples_val2 = all_samples_df[
        all_samples_df["Reference"]
        .isin(val_refs_set)
    ]
    samples_test2 = all_samples_df[
        all_samples_df["Reference"]
        .isin(test_refs_set)
    ]
    samples_train_combined = pd.concat([samples_train, samples_train2])
    samples_val_combined = pd.concat([samples_val, samples_val2])
    samples_test_combined = pd.concat([samples_test, samples_test2])
    return (
        samples_train_combined,
        samples_val_combined,
        samples_test_combined
    )


def add_to_metadata(family, metadata, all_samples_df, split):
    return metadata


def update_metadata(
        metadata,
        all_samples_df,
        train_t,
        val_t,
        test_t
    ):
    for i in train_t:
        add_to_metadata(
            i,
            metadata["metadata_train"],
            all_samples_df,
            "Train"
        )
    for i in val_t:
        add_to_metadata(
            i,
            metadata["metadata_val"],
            all_samples_df,
            "Val"
        )
    for i in test_t:
        add_to_metadata(
            i,
            metadata["metadata_test"],
            all_samples_df,
            "Test"
        )
    return


def mtsplice_baseline_csv(file_path):
    data=[]
    with open(file_path, "r", newline="") as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")
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


def get_mmsplice_split(
        merged_PSI_cov_logit_df,
        in_all_cells_matchedrefvar,
        splice_vs_super_double_variants
    ):
    (train_mm,
    val_mm,
    test_mm) = get_split_families_mm(
        merged_PSI_cov_logit_df,
        splice_vs_super_double_variants
    )
    return train_mm, val_mm, test_mm


def merged_df_to_all_cells_df(df):
    """
    Changes df in the format (ref_id, var_id) for each row
    into separate ref_id rows, where snp may be none
    Useful for training non-variant effect models (e.g.
    TSplice)
    """
    ref_suffix = "_ref"

    # Identify columns
    ref_value_cols = [c for c in df.columns if c.endswith(ref_suffix)]
    base_cols = [c[:-len(ref_suffix)] for c in ref_value_cols]

    var_value_cols = base_cols

    shared_cols = [
        c for c in df.columns
        if c not in ref_value_cols
        and c not in var_value_cols
        and c not in ["ref_id", "var_id"]
    ]

    # --- reference rows ---
    ref_df = df[["ref_id"] + ref_value_cols + shared_cols].copy()
    ref_df = ref_df.rename(columns={"ref_id": "Reference"})
    ref_df = ref_df.rename(columns=dict(zip(ref_value_cols, base_cols)))
    ref_df["snp"] = "none"
    ref_df = ref_df.drop_duplicates(subset=["Reference", "intron1_len"])

    # --- variant rows ---
    var_df = df[["var_id"] + var_value_cols + shared_cols].copy()
    var_df = var_df.rename(columns={"var_id": "Reference"})

    # Concatenate
    return pd.concat([ref_df, var_df], ignore_index=True)


def find_samples_to_add(all_samples_df, metadata, drop_nan = True):
    """
    Searches existing families from MMSplice splits
    and add samples from that family to the correct split.
    Specifically, if a sample from MMSplice is dropped 
    for whatever reason (e.g. variant was outside the scoring
    window), it may be added here.

    drop_nan is used for val and test splits, where only 
    samples without nans are allowed
    """
    samples_to_add = []
    grouped_metadata = metadata.groupby("event_id_161").groups
    grouped_df = all_samples_df.groupby("event_id_161").groups
    var_ids = metadata["var_id"].tolist()
    eids = metadata["event_id_161"].tolist()
    for event_id in metadata["event_id_161"].drop_duplicates():
        mm_splice_samples = set(
            metadata.iloc[grouped_metadata[event_id]]
            ["var_id"]
            .tolist()
        )
        remaining = (
            all_samples_df.iloc[grouped_df[event_id]]
            ["Reference"]
            .tolist()
        )
        remainin_psi = (
            all_samples_df.iloc[grouped_df[event_id]]
            [[
                f"{c}_pooled_psi_clipped"
                for c in cell_types2
            ]]
        )
        num_nans = remainin_psi.isna().sum(axis = 1).tolist()
        snps_by_family = (
            all_samples_df.iloc[grouped_df[event_id]]
            ["snp"]
            .tolist()
        )
        for s in range(len(remaining)):
            if (
                remaining[s] not in mm_splice_samples 
                and (
                    (num_nans[s] == 0) or not drop_nan
                )
                and (num_nans[s] < 5)
            ):
            
                samples_to_add.append(remaining[s])
    return samples_to_add


def check_for_new_fams_tsplice(split_df, metadata):
    """
    see if all_samples_df has families
    not originally in MMSplice dataset
    (e.g. those without a reference)
    """
    mmsplice_eids = []
    for i in metadata:
        mmsplice_eids += metadata[i]["event_id_161"].tolist()
    mmsplice_eids_unique = set(mmsplice_eids)
    split_df_eids = split_df["event_id_161"].tolist()
    new_eids = []
    for eid in split_df_eids:
        if eid not in mmsplice_eids_unique:
            new_eids.append(eid) 
    return new_eids


def update_tsplice_split_from_metadata(
        all_samples_df,
        metadata,
        split_df
    ):
    """
    Uses metadata and all_samples_df
    to get extra samples:
        A. Families with a NaN reference or all NaN variants
        B. Variants where snp was outside the MMSplice scoring window
    If the number of samples to be added allows 
    for further splitting families,
    (i.e. samples from B didn't unbalance the splits too much)
    knapsack will be re-run to optimize coverage

    all_samples_df -> NO_DELTAS dataframe, 
                      including samples with NaN variant effect
    metadata -> a dictionary with keys 
                'metadata_train', 'metadata_val', 'metadata_test'
                where each is a pandas dataframe
                with metadata about a split
    """
    event_ids = set()
    grouped_full = all_samples_df.groupby("event_id_161").groups
    test_samples_to_add = []
    val_samples_to_add = []
    train_samples_to_add = []

    test_samples_to_add = find_samples_to_add(
        all_samples_df,
        metadata["metadata_test"]
    )
    val_samples_to_add = find_samples_to_add(
        all_samples_df,
        metadata["metadata_val"]
    )
    train_samples_to_add = find_samples_to_add(
        all_samples_df,
        metadata["metadata_train"],
        drop_nan = False
    )
    test_samples_to_add = list(dict.fromkeys(test_samples_to_add))
    val_samples_to_add = list(dict.fromkeys(val_samples_to_add))
    train_samples_to_add = list(dict.fromkeys(train_samples_to_add))

    if len(val_samples_to_add) < len(test_samples_to_add):
        val_samples_needed = len(test_samples_to_add) - len(val_samples_to_add)
        test_samples_needed = 0
    elif len(test_samples_to_add) < len(val_samples_to_add):
        test_samples_needed = len(val_samples_to_add) - len(test_samples_to_add)
        val_samples_needed = 0

    train_set = set(metadata["metadata_train"].event_id_161.tolist())
    val_set = set(metadata["metadata_val"].event_id_161.tolist())
    test_set = set(metadata["metadata_test"].event_id_161.tolist())
    
    train_split = split_df[split_df["event_id_161"].isin(train_set)]
    val_split = split_df[split_df["event_id_161"].isin(val_set)]
    test_split = split_df[split_df["event_id_161"].isin(test_set)]

    train_size = len(train_split)
    val_size = len(val_split)
    test_size = len(test_split)

    missing_fams = check_for_new_fams_tsplice(all_samples_df, metadata)
    missing_fams_set = set(missing_fams)
    extra_families_df = all_samples_df[
        all_samples_df["event_id_161"]
        .isin(missing_fams_set)
    ]
    extra_families_ref_set = set(extra_families_df["Reference"].tolist())
    extra_families = (
        extra_families_df
        .groupby("event_id_161")
        .apply(get_nan_size_psi_and_coverage)
    )
    extra_families_list = [
        (
            i[1][0],
            i[1][1],
            i[0],
            i[1][2],
            i[1][3],
            i[1][4],
            i[1][5]
        )
        for i in extra_families.items()
    ]
    snps = all_samples_df["snp"].tolist()
    eid = all_samples_df["event_id_161"].tolist()
    i1 = all_samples_df["intron1_size"].tolist()
    vals = all_samples_df[cell_type_psi].reset_index()
    is_none = {}
    for i in range(len(snps)):
        if snps[i] == "none":
            is_none[eid[i]] = np.sum(np.isnan(vals.iloc[i]))

    snps_set = (
        extra_families_df
        .groupby("event_id_161")
        ["snp"]
        .apply(set)
    )

    snps_set = {i[0] : i[1] for i in snps_set.items()}

    no_ref = []
    has_ref= []
    for i in extra_families_list:
        if "none" in snps_set[i[2]]:
            has_ref.append(i)
        else:
            no_ref.append(i)

    no_ref_families_no_nan = [
        (
            i[4],
            i[5],
            i[2],
            i[3],
            i[6],
            i[0],
            i[1]
        ) 
        for i in no_ref
        if i[4] > 0
    ]
    no_ref_families_all_nan = [i for i in no_ref if i[4] == 0]

    total_extra_size = sum(i[0] for i in no_ref)
    extra_non_nan_size = sum(i[0] for i in no_ref_families_no_nan)

    train_t_initial = [
        i for i in extra_families_list
        if i[4] == 0
    ]
    train_size += sum([i[0] for i in train_t_initial])
    # prepare samples for knapsack
    extra_families_list = [
        i for i in extra_families_list
        if i[4] != 0
    ]
    test_t, val_t, train_t = get_split_families_balanced_psi(
        extra_families_list,
        proportions=None,
        size_index=4,
        cov_index=5,
        score_index=5,
        pool_sizes={"test" : 0.33, "val" : 0.32},
        num_hits=1,
        A=4,
        B=6,
        balance_psi=True,
        fam_size_has_ref=True
    )
    train_t += train_t_initial
    return (
        train_t,
        val_t,
        test_t,
        train_samples_to_add,
        val_samples_to_add,
        test_samples_to_add
    )


def get_tsplice_split(
        in_all_cells_matchedrefvar,
        splice_vs_super_double_variants,
        dataset_m
    ):
    grouped_full = in_all_cells_matchedrefvar.groupby("event_id").groups
    grouped_df = splice_vs_super_double_variants.groupby("event_id").groups
    train_mm, val_mm, test_mm = dataset_m
    test_samples_to_add = []
    event_ids = set()
    test_nans_num = 0
    for i in test_mm:
        event_id = i[2]
        assert event_id not in event_ids
        event_ids.add(event_id)
        mm_splice_samples = set(
            splice_vs_super_double_variants
            .iloc[grouped_df[event_id]]
            ["Reference"]
            .tolist()
        )
        remaining = (
            in_all_cells_matchedrefvar
            .iloc[grouped_full[event_id]]
            ["Reference"]
            .tolist()
        )
        assert len(remaining) >= len(mm_splice_samples)
        remainin_psi = (
            in_all_cells_matchedrefvar
            .iloc[grouped_full[event_id]]
            [[f"{i}_PSI" for i in cell_types]]
        )
        num_nans = remainin_psi.isna().sum(axis = 1).tolist()
        snps_by_family = (
            in_all_cells_matchedrefvar
            .iloc[grouped_full[event_id]]
            ["snp"]
            .tolist()
        )
        for s in range(len(remaining)):
            if remaining[s] not in mm_splice_samples and (num_nans[s] == 0):
                test_samples_to_add.append(remaining[s])
            # nans present in original mmsplice split for psi values
            elif remaining[s] in mm_splice_samples: 
                if num_nans[s] != 0:
                    test_nans_num += 1
            
    val_samples_to_add = []
    val_nans_num = 0
    for i in val_mm:
        event_id = i[2]
        assert event_id not in event_ids
        event_ids.add(event_id)
        mm_splice_samples = set(
            splice_vs_super_double_variants
            .iloc[grouped_df[event_id]]
            ["Reference"]
            .tolist()
        )
        remaining = (
            in_all_cells_matchedrefvar
            .iloc[grouped_full[event_id]]
            ["Reference"]
            .tolist()
        )
        assert len(remaining) >= len(mm_splice_samples)
        remainin_psi = (
            in_all_cells_matchedrefvar
            .iloc[grouped_full[event_id]]
            [[f"{i}_PSI" for i in cell_types]]
        )
        num_nans = remainin_psi.isna().sum(axis = 1).tolist()
        snps_by_family = (
            in_all_cells_matchedrefvar
            .iloc[grouped_full[event_id]]
            ["snp"]
            .tolist()
        )
        for s in range(len(remaining)):
            if (
                (remaining[s] not in mm_splice_samples)
                and (num_nans[s] == 0)
            ):
                val_samples_to_add.append(remaining[s])
            # nans present in original mmsplice split for psi values
            elif remaining[s] in mm_splice_samples:
                if num_nans[s] != 0:
                    val_nans_num += 1


    train_samples_to_add = []
    for i in train_mm:
        event_id = i[2]
        assert event_id not in event_ids
        event_ids.add(event_id)
        mm_splice_samples = set(
            splice_vs_super_double_variants
            .iloc[grouped_df[event_id]]["Reference"]
            .tolist()
        )
        remaining = (
            in_all_cells_matchedrefvar
            .iloc[grouped_full[event_id]]
            ["Reference"]
            .tolist()
        )
        assert len(remaining) >= len(mm_splice_samples)
        snps_by_family = (
            in_all_cells_matchedrefvar
            .iloc[grouped_full[event_id]]
            ["snp"]
            .tolist()
        )
        for s in range(len(remaining)):
            if remaining[s] not in mm_splice_samples:
                train_samples_to_add.append(remaining[s])

    if (
        len(val_samples_to_add) - val_nans_num 
        < len(test_samples_to_add) - test_nans_num
    ):
        val_samples_needed = (
            len(test_samples_to_add)
            - len(val_samples_to_add)
            - test_nans_num
            + val_nans_num
        )
        test_samples_needed = 0
    elif len(test_samples_to_add) < len(val_samples_to_add):
        test_samples_needed = (
            len(val_samples_to_add)
            - len(test_samples_to_add) 
            - vals_nans_num
            + test_nans_num
        )
        val_samples_needed = 0


    extra_families_df = in_all_cells_matchedrefvar[
        ~in_all_cells_matchedrefvar["event_id"]
        .isin(event_ids)
    ]

    extra_families_ref_set = set(extra_families_df["Reference"].tolist())

    extra_families = (
        extra_families_df
        .groupby("event_id")
        .apply(get_nan_size_psi_and_coverage)
    )

    extra_families_list = [
        (
            i[1][0],
            i[1][1],
            i[0],
            i[1][2],
            i[1][3],
            i[1][4],
            i[1][5]
        )
        for i in extra_families.items()
    ]

    snps = in_all_cells_matchedrefvar["snp"].tolist()
    eid = in_all_cells_matchedrefvar["event_id"].tolist()
    vals = in_all_cells_matchedrefvar[cell_type_psi].reset_index()
    is_none = {}
    for i in range(len(snps)):
        if snps[i] == "none":
            is_none[eid[i]] = np.sum(np.isnan(vals.iloc[i]))

    snps_set = extra_families_df.groupby("event_id")["snp"].apply(set)

    snps_set = {i[0] : i[1] for i in snps_set.items()}

    no_ref = []
    has_ref= []
    for i in extra_families_list:
        if "none" in snps_set[i[2]]:
            has_ref.append(i)
        else:
            no_ref.append(i)


    no_ref_families_no_nan = [
        (
            i[4],
            i[5],
            i[2],
            i[3],
            i[6],
            i[0],
            i[1]
        ) 
        for i in no_ref
        if i[4] > 0
    ]
    no_ref_families_all_nan = [i for i in no_ref if i[4] == 0]

    total_extra_size = sum(i[0] for i in no_ref)
    extra_non_nan_size = sum(i[0] for i in no_ref_families_no_nan)


    # adding any samples to train or test will make the 
    # proportions even worse, so
    # don't need to run knapsack. Just add all families to val

    # families with a nan value in at least one cell in all variants
    train_t = train_mm + no_ref_families_all_nan
    # families where at least one variant has PSIs for all cell types
    val_t = val_mm + no_ref_families_no_nan 
    test_t = test_mm

    # true train size:
     # number of samples in mm split,
     # plus the number of samples in no_ref_families_all_nan, plus
        # number of extra variants in train split from mmsplice
    true_train_size_t = sum(i[0] for i in train_t) + len(train_samples_to_add)
    print("true train size", true_train_size_t)
    # true val size:
     # number of samples in mm split, 
     # minus nans in mmsplit (1011),
     # plus the number of samples in no_ref_families_no_nan,
        # minus the number of nan variants in no_ref_families_no_nan
        # plus number of extra non-nan val variants from mmsplice split 
    true_val_size_t = (
        sum(i[0] for i in val_mm)
        - 1011
        + sum(i[0] for i in no_ref_families_no_nan)
        + len(val_samples_to_add)
    )
    print("true val size", true_val_size_t)
    # true test size:
     # number of samples in mm split, minus nans in mmsplit (1003)
     # plus number of extra non-nan test variants from mmsplice split
    true_test_size_t = (
        sum(i[0] for i in test_mm)
        - 1003
        + len(test_samples_to_add)
    )
    print("true test size", true_test_size_t)

    return (
        train_t,
        val_t,
        test_t,
        has_ref,
        is_none,
        val_samples_to_add,
        test_samples_to_add
    )



def tsplice_split_from_metadata(split_df, metadata):
    splice_types = ["seqL", "seqR"]
    if avg_scale == "logit": # should be adding the mean logit to the logit predictions
        scaling_factor_key = "Avg_logit"
    elif avg_scale == "psi": # should not be the case but useful for debugging
        print("Warning: Using PSI as scaling factor for model outputs")
        scaling_factory_key = "Avg_PSI"
    if cell_type_scale == "logit":
        outputs = (
            [f"{i}_delta_logit_mean_cell" for i in cell_types]
            + [scaling_factor_key]
        )
    elif cell_type_scale == "psi":
        outputs = (
            [f"{i}_PSI" for i in cell_types]
            + [scaling_factor_key]
        )
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
            variant_nums = [
                i for i in sample_nums
                if i != ref_to_idx[ref_num]
            ]
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


def get_tsplice_split_metadata(split_df, metadata):
    tsplice_df = {}
    for i in "train", "val", "test":
        mm_fams = set(metadata["metadata_"+i]["event_id_161"].tolist())
        tsplice_df[i] = split_df[split_df["event_id_161"].isin(mm_fams)]
    return tsplice_df


def get_combined_split(has_ref, is_none, in_all_cells_matchedrefvar, train_mm, val_mm, test_mm, val_samples_to_add, test_samples_to_add):

    # has nans only in the variants, and has at least one variant with no nan
    ref_families_no_nan = []
    ref_families_all_nan = []
    for i in has_ref: # index 2 is event id, index 4 is total number of samples with 1 or more nan in that family
        if i[4] > 1 and is_none[i[2]] == 0: # 1 because ref needs 0 nans, and need at least one variant
                                            # second condition ensures the reference has no nans
            ref_families_no_nan.append(i)
        else:
            ref_families_all_nan.append(i)

    total_extra_size = sum(i[0] for i in ref_families_all_nan)
    extra_non_nan_size = sum(i[0] for i in ref_families_no_nan)


    def num_zero(x):
        return np.sum(x == 0)


    in_all_cells_matchedrefvar['num_nan'] = in_all_cells_matchedrefvar[cell_type_psi].isna().sum(axis=1)
    variant_nan_counts_grouped = in_all_cells_matchedrefvar[in_all_cells_matchedrefvar["snp"] != "none"]\
                        .groupby('event_id')['num_nan'].apply(num_zero)

    # process mm splits to remove families with nan in reference within val / test splits
    # must do this now since later will remove all NaNs, but need at the reference value 
    # to compute target variable (delta logit from reference per cell type)
    val_mm_filtered = []
    test_mm_filtered = []
    group_psi = in_all_cells_matchedrefvar.groupby("event_id")[cell_type_psi]
    for i in val_mm:
        if is_none[i[2]] == 0 and variant_nan_counts_grouped[i[2]] > 0: # skip references with nans
            # check if there is at least one valid variant
            val_mm_filtered.append(i)
    for i in test_mm:
        if is_none[i[2]] == 0 and variant_nan_counts_grouped[i[2]] > 0:
            test_mm_filtered.append(i)

    test_added_size = sum(i[0] for i in test_mm_filtered) + len(test_samples_to_add)
    test_added_size

    val_added_size = sum(i[0] for i in val_mm_filtered) + len(val_samples_to_add)
    val_added_size

    ref_no_nan_size = sum(i[4] for i in ref_families_no_nan)
    ref_no_nan_size


    x = int((val_added_size + ref_no_nan_size - test_added_size) / 2)

    ref_families_no_nan_true_count = [
    (i[4], i[1], i[2], i[3], i[6], i[5], i[0])
    for i in ref_families_no_nan
    ]

    psis_train = ([i[3] for i in train_mm])
    psis_val = ([i[3] for i in val_mm_filtered])
    psis_test = ([i[3] for i in test_mm_filtered])

    proportions = {
    
    "test" : x/ref_no_nan_size,
    "train" : 0.0,
    "val": (ref_no_nan_size - x) / ref_no_nan_size
    
    }


    pool_sizes = {
        "test" : proportions["test"] * 2,
        "val" : 1.0,
        "train": 0.0
    }

    # psis most unbalanced in the test set
    mean_psi = [[np.sum(psis_test < np.mean(psis_test)), np.sum(psis_test >= np.mean(psis_test)), np.mean(psis_test)]]


    test_mm_filtered_split, \
    val_mm_filtered_split, \
    train_mm_filtered_split = get_split_families_2D_knapsack(ref_families_no_nan_true_count,\
                                   proportions = proportions,\
                                   pool_sizes = pool_sizes, num_hits = 1,\
                                    mean_psi = mean_psi)

    a = set(i[2] for i in train_mm_filtered_split)
    b = set(i[2] for i in test_mm_filtered_split)
    c = set(i[2] for i in val_mm_filtered_split)
    for i in test_mm_filtered_split:
        assert i[2] not in a
        assert i[2] not in c
    for i in val_mm_filtered_split:
        assert i[2] not in a
        assert i[2] not in b
    for i in train_mm_filtered_split:
        assert i[2] not in b
        assert i[2] not in c

    # like before, just add all non-nan samples to val since test size is already larger
    # and put samples with nan into the train set
    train_c = train_mm + ref_families_all_nan + train_mm_filtered_split
    val_c = val_mm_filtered + val_mm_filtered_split
    test_c = test_mm_filtered + test_mm_filtered_split
    return train_c, val_c, test_c


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


def one_hot_encode_splits(dataset_m,\
                          dataset_t,\
                          dataset_c,\
                          in_all_cells_matchedrefvar,\
                          ref_eventID_supertableRef_dict,\
                          cell_type_scale="logit",\
                          avg_scale="logit"):
    # Tsplice Logits
    train_t, val_t, test_t = dataset_t # unpack tsplice splits
    train_t = [(i[0], i[1], i[2]) for i in train_t]
    val_t = [(i[0], i[1], i[2]) for i in val_t]
    test_t = [(i[0], i[1], i[2]) for i in test_t]

    (x_train_logit_tsplice,
    x_val_logit_tsplice,
    x_test_logit_tsplice,
    y_train_logit_tsplice,
    y_val_logit_tsplice,
    y_test_logit_tsplice) = get_one_hots_from_splits(
        train_t,
        val_t,
        test_t,
        in_all_cells_matchedrefvar=in_all_cells_matchedrefvar,
        cell_type_scale = cell_type_scale,
        avg_scale = avg_scale,
        ref_eventID_supertableRef_dict = ref_eventID_supertableRef_dict,
    )
    y_train_logit_tsplice = process_y(y_train_logit_tsplice)
    y_val_logit_tsplice = process_y(y_val_logit_tsplice)
    y_test_logit_tsplice = process_y(y_test_logit_tsplice)
    arrs = []
    for i in [
        x_train_logit_tsplice,
        x_val_logit_tsplice,
        x_test_logit_tsplice
    ]:
        size = i[0].shape[0]
        arr = np.array(i[2])
        arr = np.reshape(arr, (size, 1))
        arrs.append(arr)
    x_train_logit_tsplice = [
        x_train_logit_tsplice[0],
        x_train_logit_tsplice[1],
        arrs[0]
    ]
    x_val_logit_tsplice = [
        x_val_logit_tsplice[0],
        x_val_logit_tsplice[1],
        arrs[1]
    ]
    x_test_logit_tsplice = [
        x_test_logit_tsplice[0],
        x_test_logit_tsplice[1],
        arrs[2]
    ]
    # drop NaN for tsplice labels
    val_non_nan = np.sum(np.isnan(y_val_logit_tsplice), axis = 1) == 0
    test_non_nan = np.sum(np.isnan(y_test_logit_tsplice), axis = 1) == 0

    x_val_logit_tsplice = [i[val_non_nan] for i in x_val_logit_tsplice]
    x_test_logit_tsplice = [i[test_non_nan] for i in x_test_logit_tsplice]
    y_val_logit_tsplice = y_val_logit_tsplice[
        np.sum(np.isnan(y_val_logit_tsplice), axis=1) == 0
    ]
    y_test_logit_tsplice = y_test_logit_tsplice[
        np.sum(np.isnan(y_test_logit_tsplice), axis=1) == 0
    ]

    # Combined model logit
    train_c, val_c, test_c = dataset_c # unpack combined splits
    (x_train_logit_mtsplice, 
    x_val_logit_mtsplice, 
    x_test_logit_mtsplice, 
    y_train_logit_mtsplice, 
    y_val_logit_mtsplice, 
    y_test_logit_mtsplice) = get_one_hots_from_splits(
        train_c,
        val_c,
        test_c,
        in_all_cells_matchedrefvar=in_all_cells_matchedrefvar,
        cell_type_scale="ref_logit",
        split_ref_and_var=True,
        ref_eventID_supertableRef_dict=ref_eventID_supertableRef_dict,
    )

    y_train_logit_mtsplice = process_y(y_train_logit_mtsplice)
    y_val_logit_mtsplice = process_y(y_val_logit_mtsplice)
    y_test_logit_mtsplice = process_y(y_test_logit_mtsplice)
    arrs = []
    for i in range(len(x_val_logit_mtsplice)):
        if isinstance(x_val_logit_mtsplice, list):
            avg_index = i
    for i in [
        x_train_logit_mtsplice,
        x_val_logit_mtsplice,
        x_test_logit_mtsplice
    ]:
        size = i[0].shape[0]
        arr = np.array(i[avg_index])
        arr = np.reshape(arr, (size, 1))
        arrs.append(arr)
    x_train_logit_mtsplice[avg_index] = arrs[0]
    x_val_logit_mtsplice[avg_index] = arrs[1]
    x_test_logit_mtsplice[avg_index] = arrs[2]


    # Get reference numbers to compare to baseline later
    train_ref_nums, val_ref_nums, test_ref_nums = get_one_hots_from_splits(
        train_c,
        val_c,
        test_c,
        in_all_cells_matchedrefvar=in_all_cells_matchedrefvar,
        cell_type_scale="ref_logit",
        split_ref_and_var=True,
        return_ref_nums=True,
        ref_eventID_supertableRef_dict = ref_eventID_supertableRef_dict,
    )

    # Drop NaN from combined dataset
    val_non_nan = np.sum(np.isnan(y_val_logit_mtsplice), axis = 1) == 0
    test_non_nan = np.sum(np.isnan(y_test_logit_mtsplice), axis = 1) == 0

    x_val_logit_mtsplice = [i[val_non_nan] for i in x_val_logit_mtsplice]
    x_test_logit_mtsplice = [i[test_non_nan] for i in x_test_logit_mtsplice]
    y_val_logit_mtsplice = y_val_logit_mtsplice[val_non_nan]
    y_test_logit_mtsplice = y_test_logit_mtsplice[test_non_nan]

    # drop NaN in train only if every delta logit 
    # value is NaN, 8 sequences total
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


def get_cell_ref_psi_from_refs(
        ref_nums,
        in_all_cells_matchedrefvar,
        ref_eventID_supertableRef_dict
    ):
    """
    """
    ref_to_id = dict(zip(
        in_all_cells_matchedrefvar['Reference'],
        in_all_cells_matchedrefvar['event_id']
    ))
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


def tsplice_data_pipeline(
        merged_data_path,
        supertable_file,
        mmsplice_data_dir=None,
        cell_type_scale="logit",
        avg_scale="logit",
        debug=False
    ): 
    if debug:
        print("Reading CSVs")
    merged_data = pd.read_csv(merged_data_path)
    supertable = read_supertable(supertable_file)
    if debug:
        print("processing psi data to get in all_cells_matchedrefvar")

    if debug:
        print("Getting data splits") 
    (dataset_m,
    dataset_t,
    dataset_c) = get_data_splits(
        merged_PSI_cov_logit_df,
        in_all_cells_matchedrefvar,
        splice_vs_super_double_variants
    )

    if debug:
        print("getting final, one hot encoded data")
    ref_eventID_supertableRef_dict = supertable[
        "ref_eventID_supertableRef_dict"
    ]
    data = one_hot_encode_splits(
        dataset_m,
        dataset_t,
        dataset_c,
        in_all_cells_matchedrefvar,
        ref_eventID_supertableRef_dict,
        cell_type_scale=cell_type_scale,
        avg_scale=avg_scale
    )
    if debug:
        breakpoint()
    return data




def read_mmsplice_metadata(mmsplice_data_dir):
    metadata_train = pd.read_csv(
        os.path.join(
            mmsplice_data_dir,
            "metadata_train.csv"
        )
    )
    metadata_val = pd.read_csv(
        os.path.join(
            mmsplice_data_dir,
            "metadata_val.csv"
        )
    )
    metadata_test = pd.read_csv(
        os.path.join(
            mmsplice_data_dir,
            "metadata_test.csv"
        )
    )
    metadata = {
        "train" : metadata_train,
        "val" : metadata_val,
        "test" : metadata_test
    }
    return metadata


def compute_variant_cell_logit(df):
    """
    Compute cell logits and 
    delta logits (cell - mean cell)
    according to TSplice paper
    (e.g. logit(mean) rather than mean(logit))
    (see https://doi.org/10.1101/2020.06.07.138453).
    """
    if "HEK_pooled_logit" in df:
        cell_types_ = cell_types2
    elif "HEK293_pooled_logit" in df:
        cell_types_ = cell_types
    else:
        raise ValueError("Neither HEK nor HEK293 logit found in df columns.")
    df["mean_cell_psi"] = df[[
        f"{cell}_pooled_psi_clipped"
        for cell in cell_types_
    ]].mean(axis=1)
    df["mean_cell_psi_ref"] = df[[
        f"{cell}_pooled_psi_clipped_ref"
        for cell in cell_types_
    ]].mean(axis=1)
    df["logit_mean_cell"] = df["mean_cell_psi"].apply(logit)
    df["logit_mean_cell_ref"] = df["mean_cell_psi_ref"].apply(logit)
    for cell in cell_types_:
        df[f"{cell}_cell_dlogit"] = (
            df[f"{cell}_pooled_logit"]
            - df["logit_mean_cell"]
        )
        df[f"{cell}_cell_dlogit_ref"] = (
            df[f"{cell}_pooled_logit_ref"]
            - df["logit_mean_cell_ref"]
        )
        df[f"{cell}_dlogit"] = (
            df[f"{cell}_pooled_logit"]
            - df[f"{cell}_pooled_logit_ref"]
        )
    return df


def compute_cell_logit(df):
    """
    Compute cell logits and 
    delta logits (cell - mean cell)
    according to TSplice paper
    (e.g. logit(mean) rather than mean(logit))
    (see https://doi.org/10.1101/2020.06.07.138453).
    Separately from variant logit since it uses 
    different keys.
    """
    if "HEK_pooled_psi_clipped" in df:
        cell_types_ = cell_types2
    elif "HEK293_pooled_psi_clipped" in df:
        cell_types_ = cell_types
    else:
        raise ValueError("neither HEK nor HEK293 found in df columns.")
    df["mean_cell_psi"] = df[[
        f"{cell}_pooled_psi_clipped"
        for cell in cell_types_
    ]].mean(axis=1)
    df["logit_mean_cell"] = df["mean_cell_psi"].apply(logit)
    for cell in cell_types_:
        df[f"{cell}_cell_dlogit"] = (
            df[f"{cell}_pooled_logit"]
            - df["logit_mean_cell"]
        )
    return df
        


def process_sequences_mt(exon, intron1, intron2):
    """
    separate full sequences 
    (161 variable region + SMN2 introns)
    into inputs for TSplice model
    and for the MMSplice model (hence, MTSplice)
    (see https://doi.org/10.1101/2020.06.07.138453
    https://biorxiv.org/lookup/doi/10.1101/438986).
    """
    original_sizes = (len(exon), len(intron1), len(intron2))
    if len(exon) < 100:
        exon_L = exon + "m"*(100-len(exon))
        exon_R = "m"*(100-len(exon)) + exon
    else:
        exon_L = exon[:100]
        exon_R = exon[-100:]
    assert len(exon_L) == len(exon_R) == 100
    if len(intron1) + len(SMN2_INTRON6) < 300: # if intron is really short
        padding = 300 - len(intron1) - len(SMN2_INTRON6)
        intron1 = "N" * padding + SMN2_INTRON6 + intron1
    elif len(intron1) < 300:
        intron1 = (SMN2_INTRON6 + intron1)[-300:]
    if len(intron2) < 300:
        intron2 = (intron2 + SMN2_INTRON7)[:300]
    if len(intron1) > 300:
        intron1 = intron1[-300:]
    if len(intron2) > 300:
        intron2 = intron2[:300]
    assert len(intron1) == len(intron2) == 300, (
        f"Expected processed introns 1, 2 to be of size "
        f"300, 300 but got {len(intron1)},  {len(intron2)}. "
        f"The original inputs to process_sequences_mt has "
        f"the following sizes: exon size = {original_sizes[0]}, "
        f"intron1 size = {original_sizes[1]}, "
        f"intron2 size = {original_sizes[2]}"
    )
    seqL = intron1 + exon_L
    seqR = exon_R + intron2
    return seqL, seqR


def prepare_tsplice_seqs(df):
    """
    separate full sequences 
    (161 variable region + SMN2 introns)
    into inputs for TSplice model
    (see https://doi.org/10.1101/2020.06.07.138453).
    """
    seqL, seqR = process_sequences_mt(
        exon=df["exon"],
        intron1=df["intron1"],
        intron2=df["intron2"],
    )
    df["seqL"] = seqL
    df["seqR"] = seqR
    return df


def prepare_tsplice_ref_seqs(df):
    """
    separate full sequences 
    (161 variable region + SMN2 introns)
    into inputs for TSplice model
    (see https://doi.org/10.1101/2020.06.07.138453).
    """
    seqL_ref, seqR_ref = process_sequences_mt(
        exon=df["exon_ref"],
        intron1=df["intron1_ref"],
        intron2=df["intron2_ref"],
    )
    df["seqL_ref"] = seqL_ref
    df["seqR_ref"] = seqR_ref
    return df


def prepare_mmsplice_seqs(df):
    """
    separate full sequences 
    (161 variable region + SMN2 introns)
    into inputs for MMSplice model
    (see https://biorxiv.org/lookup/doi/10.1101/438986).
    """
    (acceptor_intron,
    acceptor,
    exon,
    donor,
    donor_intron) = process_sequences_mm(
        exon=df["exon"],
        intron1=df["intron1"],
        intron2=df["intron2"],
    )
    df["acceptor_intron"] = acceptor_intron
    df["acceptor"] = acceptor
    df["donor"] = donor
    df["donor_intron"] = donor_intron
    (acceptor_intron_ref,
    acceptor_ref,
    exon_ref,
    donor_ref,
    donor_intron_ref) = process_sequences_mm(
        exon=df["exon_ref"],
        intron1=df["intron1_ref"],
        intron2=df["intron2_ref"],
    )
    df["acceptor_intron_ref"] = acceptor_intron_ref
    df["acceptor_ref"] = acceptor_ref
    df["donor_ref"] = donor_ref
    df["donor_intron_ref"] = donor_intron_ref
    return df


def tsplice_data_pipeline2(merged_data_path,
                           supertable_file,
                           mmsplice_data_dir,
                           all_samples_path,
                           add_no_deltas = True,
                           debug = False,):
    debug_print = print if debug else lambda *x : x
    merged_df = pd.read_csv(merged_data_path)
    debug_print("read merged_df")
    all_samples_df = pd.read_csv(all_samples_path)
    debug_print("read all samples df")
    supertable = read_supertable(supertable_file)
    debug_print("read supertable")
    merged_df = compute_variant_cell_logit(merged_df)
    debug_print("computed cell logits")
    merged_df = merged_df.apply(prepare_mmsplice_seqs, axis = 1)
    debug_print("prepared seqs")
    split_df = merged_df_to_all_cells_df(merged_df)
    result, metadata = load_splits_dir(mmsplice_data_dir, extension = ".csv")
    
    if add_no_deltas:
        debug_print("Adding no deltas")
        new_families = check_for_new_fams_tsplice(all_samples_df, metadata)
        # split_df = add_no_delta_samples(split_df, all_samples_df) # TODO
        dataset_m = metadata_to_splits(metadata) # TODO
        train_t, val_t, test_t, train_samples_to_add, val_samples_to_add, test_samples_to_add = update_tsplice_split_from_metadata(all_samples_df, metadata, split_df)
        train_split, val_split, test_split = split_all_samples_df(all_samples_df = all_sampled_df,
                                                                  train_fams = train_t,
                                                                  val_fams   = val_t,
                                                                  test_fams  = test_t,
                                                                  train_refs = train_samples_to_add,
                                                                  val_refs   = val_samples_to_add,
                                                                  test_refs  = test_samples_to_add,)    
        metadata_tsplice = update_metadata(metadata, all_samples_df, train=t_train, val=t_val, test=t_test)
    else:
        debug_print("NOT adding no deltas")
        metadata_tsplice = metadata

    split_df = split_df.apply(prepare_tsplice_seqs, axis = 1)
    x_tsplice, y_tsplice, tsplice_metadata = tsplice_splits_from_metadata(split_df, metadata_tsplice)
    debug_print("got split from metadata")
    x_mtsplice, y_mtsplice, mtsplice_metadata = mtsplice_splits_from_metadata(merged_df, metadata)
    debug_print("got mtsplice splits from metadata")
    tsplice_one_hots = get_one_hots_from_tsplice_splits(x_tsplice)
    debug_print("got tsplice one hots")
    mtsplice_one_hots = get_one_hots_from_mtsplice_splits(y_tsplice)
    debug_print("got mtsplice one hots")
    out = {"x_tsplice" : tsplice_one_hots,
           "y_tsplice" : y_tsplice,
           "metadata_tsplice" : tsplice_metadata,
           "metadata_mtsplice" : mtsplice_metadata,
    }
    return out


def remake_data(opts):
    """
    Function to check data inputs exist and remake data 
    if splits not provided or not found.
    """
    err = []
    if "--merged_data_path" not in opts:
        err.append("merged_data_path")
    if "--all_samples_path" not in opts:
        err.append("all_samples_path")
    if "--mmsplice_data_dir" not in opts:
        err.append("mmsplice_data_dir")
    if err:
        err_msg = (
            "Could not find tsplice data splits."
            + " Attempted to remake data, but "
            + ",".join(err)
            + " are missing from inputs."
        )
        raise ValueError(err_msg)    
    merged_data_path = opts["--merged_data_path"]
    all_samples_path = opts["--all_samples_path"]
    mmsplice_data_dir = opts["--mmsplice_data_dir"]
    tsplice_data, mtsplice_data = tsplice_data_pipeline3(
        merged_data_path,
        all_samples_path,
        mmsplice_data_dir,
        tsplice_output_dir="",
        mtsplice_output_dir="",
        save_tsplice_data=False,
        save_mtsplice_data=False,
        debug=False,
    )
    return tsplice_data, mtsplice_data


def tsplice_data_pipeline3(
        merged_data_path,
        all_samples_path,
        mmsplice_data_dir,
        tsplice_output_dir,
        mtsplice_output_dir,   
        save_tsplice_data=True,
        save_mtsplice_data=True,
        debug=False,
    ):
    """
    main pipeline to get tsplice and mtsplice
    one hot sequences and delta logit values and
    metadata.

    merged_data_path: 
        variant-reference pair data.
    all_samples_path:
        data where not all variants
        have a reference.
    mmsplice_data_dir:
        original dataset containing mmsplice splits
        this is necessary to keep families in the
        correct splits (MTSplice uses MMSplice)
        if we add non-matched samples to the TSplice
        training.
    tsplice_output_dir:
        will write x_train.p, x_val.p, x_test.p
        y_train.p, y_val.p, y_test.p and metadata:
        metadata_train.csv,
        metadata_val.csv,
        metadata_test.csv
        the x pickles has the one hot encoded sequences
        (a list of numpy arrays)
        and the y pickles contain the target variable
        (in dlogit cell - mean cell scale).
        Metadata contains information about each sample.
    mtsplice_output_dir:
        same as tsplice_output_dir but for mtsplice data.
        The target variable is in variant dlogit
        (variant - reference logit for each cell). 
    save_tsplice_data, save_mtsplice_data:
        whether or not to actually save the data to the 
        respective directories. Should only be false if
        remake_data is called (or if called manually).
    debug:
        enable breakpoints.
    """
    merged_df = pd.read_csv(merged_data_path)
    print("read merged_df")
    all_samples_df = pd.read_csv(all_samples_path)
    all_samples_df["Avg_PSI"] = all_samples_df[[
        f"{c}_pooled_psi_clipped"
        for c in cell_types2
    ]].mean(axis=1)
    all_samples_df["Avg_coverage"] = all_samples_df[[
        f"{c}_total_pooled"
        for c in cell_types2
    ]].mean(axis=1)
    for c in cell_types2:
        all_samples_df[f"{c}_PSI"] = all_samples_df[f"{c}_pooled_psi_clipped"]
    all_samples_df["HEK293_PSI"] = all_samples_df["HEK_pooled_psi_clipped"]
    all_samples_df["intron1_size"] = all_samples_df["intron1"].apply(len)
    all_samples_df["num_nans"] = all_samples_df[[
        f"{c}_pooled_psi_clipped"
        for c in cell_types2
    ]].isna().sum(axis=1)
    all_samples_df = compute_cell_logit(all_samples_df)
    print("read all samples df")
    merged_df = compute_variant_cell_logit(merged_df)
    print("computed cell logits")
    split_df = merged_df_to_all_cells_df(merged_df)
    print("made split df")
    result, metadata = load_splits_dir(mmsplice_data_dir, extension=".csv")
    print("loaded mmsplice data")
    # samples to add
    (train_t,
    val_t,
    test_t, 
    train_samples_to_add,
    val_samples_to_add, 
    test_samples_to_add) = update_tsplice_split_from_metadata(
        all_samples_df,
        metadata,
        split_df
    )
    print("got extra samples from knapsack")
    train_samples_all = (
        train_samples_to_add
        + metadata["metadata_train"]["var_id"].tolist()
        + metadata["metadata_train"]["ref_id"].tolist()
    )
    val_samples_all = (
        val_samples_to_add
        + metadata["metadata_val"]["var_id"].tolist()
        + metadata["metadata_val"]["ref_id"].tolist()
    )
    test_samples_all = (
        test_samples_to_add
        + metadata["metadata_test"]["var_id"].tolist()
        + metadata["metadata_test"]["ref_id"].tolist()
    )
    (train_split, 
    val_split,
    test_split) = split_all_samples_df(
        all_samples_df=all_samples_df,
        train_fams=train_t,
        val_fams=val_t,
        test_fams=test_t,
        train_refs=train_samples_all,
        val_refs=val_samples_all,
        test_refs=test_samples_all,
    )    

    if debug:
        breakpoint()
        " check remove nans tsplice "

    val_split = remove_split_nans(val_split, suffix="_cell_dlogit")
    test_split = remove_split_nans(test_split, suffix="_cell_dlogit")

    print("got splits")
    train_split = train_split.apply(prepare_tsplice_seqs, axis=1)
    val_split = val_split.apply(prepare_tsplice_seqs, axis=1)
    test_split = test_split.apply(prepare_tsplice_seqs, axis=1)

    tsplice_seqs = ["seqL", "seqR"]
    tsplice_ref_seqs = [i + "_ref" for i in tsplice_seqs]
    seqs = {"train" : {}, "val" : {}, "test" : {}}
    for i in tsplice_seqs:
        seqs["train"][i] = train_split[i].tolist()
        seqs["val"][i] = val_split[i].tolist()
        seqs["test"][i] = test_split[i].tolist()

    metadata_t = {}
    metadata_t["metadata_train"] = make_metadata(
        train_split,
        metadata,
        ["Reference", "wt_reference"]
    )
    metadata_t["metadata_val"] = make_metadata(
        val_split,
        metadata,
        ["Reference", "wt_reference"]
    )
    metadata_t["metadata_test"] = make_metadata(
        test_split,
        metadata,
        ["Reference", "wt_reference"]
    )
    print("made metadata")

    tsplice_one_hots = one_hot_encode_all_mt(
        train=seqs["train"],
        val=seqs["val"],
        test=seqs["test"],
        split_ref_and_var=False
    )
    print("got tsplice one hots")
    
    y_tsplice = {}

    y_tsplice["train_y"] = train_split[[
        f"{cell}_cell_dlogit"
        for cell in cell_types2
    ]].to_numpy()
    y_tsplice["val_y"]   = val_split[[
        f"{cell}_cell_dlogit"
        for cell in cell_types2
    ]].to_numpy()
    y_tsplice["test_y"]  = test_split[[
        f"{cell}_cell_dlogit" for cell in cell_types2
    ]].to_numpy()

    if debug:
        breakpoint()
        " check scale tsplice "

    tsplice_cell_means = {}
    tsplice_cell_means["train"] = train_split["mean_cell_psi"].to_numpy()
    tsplice_cell_means["train"] = np.reshape(
        tsplice_cell_means["train"],
        (len(tsplice_cell_means["train"]), 1)
    ) 
    tsplice_cell_means["val"] = val_split["mean_cell_psi"].to_numpy()
    tsplice_cell_means["val"] = np.reshape(
        tsplice_cell_means["val"],
        (len(tsplice_cell_means["val"]), 1)
    ) 
    tsplice_cell_means["test"] = test_split["mean_cell_psi"].to_numpy()
    tsplice_cell_means["test"] = np.reshape(
        tsplice_cell_means["test"],
        (len(tsplice_cell_means["test"]), 1)
    ) 
     
    tsplice_data_dir = tsplice_output_dir
    tsplice_data = {}
    for d in ("train", "val", "test"):
        tsplice_cell_mean = [tsplice_cell_means[d]]

        out_key_x = f"x_{d}"
        out_key_y = f"y_{d}"
        out_key_m = f"metadata_{d}"

        data_x = [
            tsplice_one_hots[f"{d}_{seq}"]
            for seq in tsplice_seqs
        ] + tsplice_cell_mean
        data_y = y_tsplice[f"{d}_y"]
        data_m = metadata_t[out_key_m]
        assert len(data_x[0]) == len(data_y) == len(data_m), (
            f"Expected same number of samples for TSplice {d} "
            f"X, Y and metadata, "
            f"but got\nX = {len(data_x[0])}\n"
            f"Y = {len(data_y)}\nmetadata = {len(data_m)}"
        )
        if save_tsplice_data:
            path_x = os.path.join(tsplice_data_dir, out_key_x + ".p")
            path_y = os.path.join(tsplice_data_dir, out_key_y + ".p")
            path_m = os.path.join(tsplice_data_dir, out_key_m + ".csv")
            save_pickle(
                data=[
                    tsplice_one_hots[f"{d}_{seq}"]
                    for seq in tsplice_seqs
                ] + tsplice_cell_mean,
                path=path_x
            )
            save_pickle(
                data=y_tsplice[f"{d}_y"],
                path=path_y
            )
            data_m.to_csv(path_m, index=False)
            print(f"Saved: {path_x}\n{path_y}\n{path_m}")
        tsplice_data[out_key_x] = data_x
        tsplice_data[out_key_y] = data_y
        tsplice_data[out_key_m] = data_m

    print("got final tsplice data")

    
    if debug:
        breakpoint()
        print("making MTSplice split")

    train_var_ids = metadata["metadata_train"]["var_id"].tolist()
    val_var_ids = metadata["metadata_val"]["var_id"].tolist()
    test_var_ids = metadata["metadata_test"]["var_id"].tolist()

    # Repeat for MTSplice split
    # here, just do the same splits as MMSplice and include the seqL, seqR 
    # and have {cell}_dlogit as the output variable

    merged_df = merged_df.apply(prepare_mmsplice_seqs, axis=1)
    merged_df = merged_df.apply(prepare_tsplice_seqs, axis=1)
    merged_df = merged_df.apply(prepare_tsplice_ref_seqs, axis=1)
    print("prepared seqs")

    train_split_mm = merged_df[merged_df.var_id.isin(train_var_ids)].copy()
    val_split_mm = merged_df[merged_df.var_id.isin(val_var_ids)].copy()
    test_split_mm = merged_df[merged_df.var_id.isin(test_var_ids)].copy()


    # Remove NaNs at the delta logit scale
    # this ensures we don't have NaNs with any scaling
    val_split_mm = remove_split_nans(
        val_split_mm,
        cell_types,
        suffix="_dlogit"
    )
    test_split_mm = remove_split_nans(
        test_split_mm,
        cell_types,
        suffix="_dlogit"
    )

    print("got mmsplice splits")

    mmsplice_seqs = [
        "acceptor_intron",
        "acceptor", 
        "exon",
        "donor",
        "donor_intron"
    ] 
    mmsplice_ref_seqs = [i + "_ref" for i in mmsplice_seqs]

    if debug:
        breakpoint()
        " check made seqs "

    mt_seqs = {
        "train" : {},
        "val" : {},
        "test" : {}
    }
    mtsplice_seqs = (
        mmsplice_ref_seqs
        + mmsplice_seqs
        + tsplice_ref_seqs
        + tsplice_seqs
    )
    for i in mtsplice_seqs:
        mt_seqs["train"][i] = train_split_mm[i].tolist()
        mt_seqs["val"][i] = val_split_mm[i].tolist()
        mt_seqs["test"][i] = test_split_mm[i].tolist()

    ref_psi_keys = [
        i for i in train_split_mm
        if "pooled_psi_clipped" in i
        and "_ref" in i
    ]
    metadata_mt = {}
    metadata_mt["metadata_train"] = make_metadata(
        train_split_mm, 
        metadata,
        ref_psi_keys
    )
    metadata_mt["metadata_val"] = make_metadata(
        val_split_mm,
        metadata,
        ref_psi_keys
    )
    metadata_mt["metadata_test"] = make_metadata(
        test_split_mm,
        metadata,
        ref_psi_keys
    )
    mtsplice_one_hots = one_hot_encode_all_mt(
        train=mt_seqs["train"],
        val=mt_seqs["val"],
        test=mt_seqs["test"],
        split_ref_and_var=False
    )
    y_mtsplice = {}
    # mtsplice target variable is the variant logit
    # according ot the MTSplice paper
    y_mtsplice["train_y"] = train_split_mm[[
        f"{cell}_pooled_logit"
        for cell in cell_types
    ]].to_numpy()
    y_mtsplice["val_y"] = val_split_mm[[
        f"{cell}_pooled_logit"
        for cell in cell_types
    ]].to_numpy()
    y_mtsplice["test_y"] = test_split_mm[[
        f"{cell}_pooled_logit"
        for cell in cell_types
    ]].to_numpy()
    ref_cell_psi = [f"{cell}_pooled_psi_clipped_ref" for cell in cell_types]
    train_split_mm["logit_mean_ref"] = (
        train_split_mm[ref_cell_psi]
        .mean(axis=1)
        .apply(logit)
    )
    val_split_mm["logit_mean_ref"] = (
        val_split_mm[ref_cell_psi]
        .mean(axis=1)
        .apply(logit)
    )
    test_split_mm["logit_mean_ref"] = (
        test_split_mm[ref_cell_psi]
        .mean(axis=1)
        .apply(logit)
    )
    ref_psi = {
        "train" : train_split_mm[ref_cell_psi].to_numpy(),
         "val" : val_split_mm[ref_cell_psi].to_numpy(),
         "test" : test_split_mm[ref_cell_psi].to_numpy(),
    }
    mtsplice_cell_means = {}
    mtsplice_cell_means["train"] = train_split_mm["logit_mean_ref"].to_numpy()
    mtsplice_cell_means["train"] = np.reshape(
        mtsplice_cell_means["train"],
        (len(mtsplice_cell_means["train"]), 1)
    )
    mtsplice_cell_means["val"] = val_split_mm["logit_mean_ref"].to_numpy()
    mtsplice_cell_means["val"] = np.reshape(
        mtsplice_cell_means["val"],
        (len(mtsplice_cell_means["val"]), 1)
    ) 
    mtsplice_cell_means["test"] = test_split_mm["logit_mean_ref"].to_numpy()
    mtsplice_cell_means["test"] = np.reshape(
        mtsplice_cell_means["test"],
        (len(mtsplice_cell_means["test"]), 1)
    ) 
    mtsplice_data_dir = mtsplice_output_dir
    mtsplice_data = {}
    for d in ("train", "val", "test"):
        mtsplice_cell_mean = [mtsplice_cell_means[d]]
        out_key_x = f"x_{d}"
        out_key_y = f"y_{d}"
        out_key_m = f"metadata_{d}"
        out_key_psi = f"psi_{d}"
        data_x = [
            mtsplice_one_hots[f"{d}_{seq}"]
            for seq in mtsplice_seqs
        ] + mtsplice_cell_mean
        data_y = y_mtsplice[f"{d}_y"]
        data_m = metadata_mt[out_key_m]
        assert len(data_x[0]) == len(data_y) == len(data_m), (
            "Expected same number of samples for MTSplice X, Y and metadata, "
            f"but got\nX = {len(data_x[0])}\n"
            f"Y = {len(data_y)}\nmetadata = {len(data_m)}"
        )
        if save_mtsplice_data:
            path_x = os.path.join(mtsplice_data_dir, out_key_x + ".p")
            path_y = os.path.join(mtsplice_data_dir, out_key_y + ".p")
            path_m = os.path.join(mtsplice_data_dir, out_key_m + ".csv")
            path_psi = os.path.join(mtsplice_data_dir, out_key_psi + ".p")
            save_pickle(
                data=data_x,
                path=path_x,
            )
            save_pickle(
                data=data_y,
                path=path_y,
            )
            save_pickle(
                data=ref_psi[d],
                path=path_psi,
            )
            metadata_mt["metadata_" + d].to_csv(path_m, index=False)
            print(f"Saved: {path_x}\n{path_y}\n{path_m}\n{path_psi}")
        mtsplice_data[out_key_x] = data_x
        mtsplice_data[out_key_y] = data_y
        mtsplice_data[out_key_m] = data_m
        mtsplice_data[out_key_psi] = ref_psi[d]

    print("got final mtsplice data")
    
    if save_mtsplice_data:
        mtsplice_distribution_plot(
            train=y_mtsplice["train_y"],
            val=y_mtsplice["val_y"],
            test=y_mtsplice["test_y"],
            data_dir=mtsplice_data_dir,
            scale="Variant Logit"
        )

    if debug:
        breakpoint()
        " final "
    return tsplice_data, mtsplice_data


def save_opts(opts):
    if "--debug" in opts:
        return
    import json
    d1 = opts["--tsplice_output_dir"]
    d1 = os.path.join(d1, "run_metadata.json")
    d2 = opts["--mtsplice_output_dir"]
    d2 = os.path.join(d2, "run_metadata.json")
    try:
        with open(d1, "w") as f:
            json.dump(opts, f)
    except Exception as e:
        print(f"could not save opts as json to {d1}:\n{e}")
    try:
        with open(d2, "w") as f:
            json.dump(opts, f)
    except Exception as e:
        print(f"could not save opts as json to {d2}:\n{e}")
    return
     


help_="""
New tsplice data pipeline:

read merged df
read metadata
make split df
set up dictionaries for get_tsplice_df
get tsplice_df using get_tsplice_splits_metadata
check if any NEW families added
get one hots (modify get_sequences...)
use MMSplice split for MTSplice split
but use Y values for cell specific stuff
and new sequences
"""


if __name__ == "__main__":
    print("testing data pipeline")
    opts, args = getopt.getopt(sys.argv[1:],"", [
        "merged_data_path=",
        "all_samples_path=",
        "mmsplice_data_dir=",
        "tsplice_output_dir=",
        "mtsplice_output_dir=",
        "slurm_job_id=",
        "debug",
        "help",
    ])
    opts = dict(opts)
    if "--help" in opts:
        print(help_)
        sys.exit(0)

    if "--debug" in opts:
        data_dir = os.path.join(
            PROJECT_ROOT,
            "..",
            "..",
            "Figures_SK",
            "General_preprocessing",
            "output_5_23_2026"
        )
        merged_data_path = os.path.join(
            data_dir,
            "merged_05_23_2026.csv"
        )
        all_samples_path = os.path.join(
            data_dir, 
            "05_23_20261_e-2_ALL_WTS_VARS_NO_DELTAS_with_SI.csv"
        )
        mmsplice_data_dir = os.path.join(
            PROJECT_ROOT,
            "models",
            "data",
            "mmsplice14"
        )
        tsplice_output_dir = os.path.join(
            PROJECT_ROOT, 
            "models",
            "data",
            "merged_mm_mt_splice_logit_tsplice_split2",
        )
        mtsplice_output_dir = os.path.join(
            PROJECT_ROOT, 
            "models",
            "data",
            "merged_mm_mt_splice_logit_mtsplice_split2",
        )
        debug = True
    else:
        merged_data_path = opts["--merged_data_path"]
        all_samples_path = opts["--all_samples_path"]
        mmsplice_data_dir = opts["--mmsplice_data_dir"]
        tsplice_output_dir = opts["--tsplice_output_dir"]
        mtsplice_output_dir = opts["--mtsplice_output_dir"]
        slurm_job_id = opts.get("--slurm_job_id", None)
        debug = False

    print(opts)

    tsplice_data, mtsplice_data = tsplice_data_pipeline3(
        merged_data_path,
        all_samples_path,
        mmsplice_data_dir,
        tsplice_output_dir,
        mtsplice_output_dir,   
        save_tsplice_data=not debug,
        save_mtsplice_data=not debug,
        debug=debug,
    )
    save_opts(opts)
    print("done")

