help_="""
Retrain the MMSplice architecture for ESL cell line data.

Made by Gabriel Fonseca and Angela Yu

Revised version for filtering mismatched variants event-ids

and also for running with sbatch

GPL License

"""

import getopt
import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:],"", [
    "data_splits_dir=",
    "load_splits=",
    "merged_df_path=",
    "drop_margin_psi=",
    "balance_mode=",
    "loss=",
    "n_epochs=",
    "learning_rate=",
    "hyperparameter_tune=",
    "min_l1=",
    "max_l1=",
    "min_l2=",
    "max_l2=",
    "min_lr=",
    "max_lr=",
    "min_dropout=",
    "max_dropout=",
    "min_batchsize=",
    "max_batchsize=",
    "early_stop=",
    "hyperband_max_epochs=",
    "hyperband_patience=",
    "hyperband_retrain_epochs=",
    "hyperband_retrain_patience=",
    "freeze_prior_layer=",
    "freeze_batchnorm=",
    "fine_tune_non_margin_psi=",
    "fine_tune_epochs=",
    "sample_weights=",
    "slurm_job_id=",
    "config_file=",
    "make_data",
    "debug",
    "help", 
    ])
    opts = dict(opts)
    if "--help" in opts:
        print(help_)
        sys.exit(0)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import scipy.stats as stats
from scipy.stats import pearsonr
import seaborn as sns
import json
from sklearn import preprocessing
import random
import pickle
from collections import defaultdict
import os
import keras_tuner as kerastuner
from toolz import interleave
from itertools import chain
from scipy.stats import linregress
try:
    from tensorflow.keras.optimizers import Adam
except ModuleNotFoundError:
    from keras.optimizers import Adam
try:
    from keras.src.models.functional import Functional
except ModuleNotFoundError:
    from tensorflow.python.keras.engine.functional import Functional
try:
    from tensorflow.keras import regularizers
except ModuleNotFoundError:
    from keras import regularizers
import getopt


def updir(n, path = ""):
    if not path:
        try:
            path = os.path.dirname(__file__)
        except NameError:
            path = os.getcwd()
    for i in range(n):
        path = os.path.dirname(path)
    return path
    

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
    raise ModuleNotFoundError("Could not find project "
                              "root with 'src' directory")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

if __name__ == "__main__":
    from src.utils.parsing import ErrorWrapper, bool_map
    opts = dict(opts)
    opts = ErrorWrapper(opts, max_printout_size=300)
    if "--make_data" in opts:
        logging.info("[MAKE DATA]")
        make_data = True
    else:
        make_data = False

    if "--debug" in opts:
        debug = True
        logging.info("[DEBUG]")
    else:
        debug = False

    if debug or make_data:
        if make_data:
            load_splits = False
        else:
            load_splits = True
        if debug:
            merged_df_path = os.path.join(
                updir(3),
                "Figures_SK",
                "General_preprocessing",
                "output_5_23_2026",
                "merged_05_23_2026.csv",
            )
            data_splits_dir = os.path.join(
                PROJECT_ROOT,
                "models",
                "data",
                "mmsplice14",
            )
            drop_margin_psi = False
            balance_mode = "dlogit"
        else:
            merged_df_path = opts["--merged_df_path"]
            data_splits_dir = opts["--data_splits_dir"]
            drop_margin_psi = bool_map(opts["--drop_margin_psi"])
            balance_mode = opts["--balance_mode"] # PSI or dLogit or dPSI

        fine_tune_non_margin_psi = False
        fine_tune_epochs = 5
        freeze_prior_layer = False
        freeze_batchnorm = False
        hyperparameter_tune = False
        learning_rate = 0.001
        n_epochs = 1
        min_l1 = 0.0
        max_l1 = 0.03
        min_l2 = 0.0
        max_l2 = 0.03
        min_lr = 1e-7
        max_lr = 1e-4
        min_dropout = 0.1
        max_dropout = 0.4
        min_batchsize = 16
        max_batchsize = 48
        early_stop  = True
        hyperband_max_epochs     = 20
        hyperband_retrain_epochs = 15
        hyperband_patience = 5
        hyperband_retrain_patience = 20
        loss="mse"
        sample_weights = "none"

    else: # normal mode, read in all parameters
        load_splits = bool_map(opts["--load_splits"])
        if load_splits:
            data_splits_dir = opts["--data_splits_dir"]
        else:
            merged_df_path = opts["--merged_df_path"]
            drop_margin_psi = bool_map(opts["--drop_margin_psi"])
        hyperparameter_tune = bool_map(opts["--hyperparameter_tune"])
        loss = opts["--loss"]
        sample_weights = opts["--sample_weights"]
        fine_tune_non_margin_psi = bool_map(opts["--fine_tune_non_margin_psi"])
        if fine_tune_non_margin_psi:
            fine_tune_epochs = int(opts["--fine_tune_epochs"])
        freeze_prior_layer = bool_map(opts["--freeze_prior_layer"])
        freeze_batchnorm = bool_map(opts["--freeze_batchnorm"])
        if hyperparameter_tune:
            min_l1 = float(opts["--min_l1"])
            max_l1 = float(opts["--max_l1"])
            min_l2 = float(opts["--min_l2"])
            max_l2 = float(opts["--max_l2"])
            min_lr = float(opts["--min_lr"])
            max_lr = float(opts["--max_lr"])
            min_dropout = float(opts["--min_dropout"])
            max_dropout = float(opts["--max_dropout"])
            min_batchsize = int(opts["--min_batchsize"])
            max_batchsize = int(opts["--max_batchsize"])
            early_stop  = bool_map(opts["--early_stop"])
            if early_stop:
                hyperband_retrain_patience = int(
                    opts["--hyperband_retrain_patience"]
                )
            hyperband_max_epochs = int(opts["--hyperband_max_epochs"])
            hyperband_retrain_epochs = int(opts["--hyperband_retrain_epochs"])
            hyperband_patience = int(opts["--hyperband_patience"])
        else:
            n_epochs = int(opts["--n_epochs"]) 
            learning_rate = float(opts["--learning_rate"])

    slurm_job_id = opts.get("--slurm_job_id", None)
    config_file = opts.get("--config_file", None)
    opts.check_inputs()
    print(opts)


# custom functions
from src.filter_data import filter_variant_size_and_seq
from mmsplice_retraining.mmsplice_model.mmsplice_build import (
    MMSplice_build_subtree,
    get_hyperband_mmsplice,
)
from mmsplice_retraining.mmsplice_model.mmsplice_build import (
    get_saved_model as get_saved_mmsplice,
)
from src.data_helpers.knapsack.knapsack_helper import (
    get_split_families_balanced_psi,
)
from src.esl import (
    logit,
    expit,
    clip,
)
from src.esl import cell_types
from src.data_helpers.knapsack.knapsack2d import (
    knapsack2D,
    traceback2D,
)
from src.utils.check_version import check_version
from src.utils.overwrite_protection import get_new_dir
from src.load_data import load_splits_dir
from src.model_eval import (
    r2,
    calc_r,
    mse,
)
from mmsplice_retraining.data_pipeline.saving import (
    save_splits,
    psi_distribution_plot,
)
from mmsplice_retraining.data_pipeline.make_mmsplice_data import (
    get_families_mmsplice,
    find_valid_mmsplice_variants,
    find_ref_margin_PSI,
)
from mmsplice_retraining.data_pipeline.make_mmsplice_data import (
    visualize_splits,
    get_one_hots_from_splits,
    get_non_margin_split,
)
from src.train_utils import weigh_samples

#from tensorflow import keras
#import keras
import tensorflow as tf
np.random.seed(1337)


def create_directory(directory):
    """
    Create the output directory if it does not already exist.
    Return: The name of the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from layers import (
    normalize_data_format,
    ConvDNA)  # soft linked to MMSplice's layers.py
from 
    mmsplice_retraining.\
    mmsplice_model.\
    global_average_pooling1d_mask0 import GlobalAveragePooling1D_Mask0

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    concatenate,
    Subtract,
    Dense,
    Layer,
)
from tensorflow.keras import (
    Model,
    Input,
)

# Create two copies, one acting as the baseline
mm_splice_saved = get_saved_mmsplice()
merged_MMSplice_full = get_saved_mmsplice()

def freeze_batchnorm_func():
    for layer in merged_MMSplice_full.layers:
        if isinstance(layer, Functional):
            for layer_ in layer.layers:
                if isinstance(layer_, tf.keras.layers.BatchNormalization):
                    layer_.trainable = False
    return


if freeze_prior_layer:
    merged_MMSplice_full.layers[-1].trainable = False
if freeze_batchnorm:
    freeze_batchnorm_func()

logging.info("merged_MMSplice_full.summary()")
merged_MMSplice_full.summary()
logging.info(
    f"merged_MMSplice_full.layers[-1].get_weights() = "
    f"{merged_MMSplice_full.layers[-1].get_weights()}"
)


def hyperband_mmsplice(hp):
    """
    Reload the pre-trained model for fine-tuning with 
    new hyperparameters
    """
    for i in merged_MMSplice_full.layers:
        if isinstance(i, Functional):
            for layer in i.layers:
                if layer.weights:
                    layer.set_weights(
                        mm_splice_saved.get_layer(i.name)
                        .get_layer(layer.name)
                        .get_weights()
                    )
                    hp_l1 = hp.Float(
                        "l1_" + layer.name,
                        min_value=min_l1,
                        max_value=max_l1,
                        step=(max_l1 - min_l1)/10
                    )
                    hp_l2 = hp.Float(
                        "l2_" + layer.name,
                        min_value=min_l2,
                        max_value=max_l2,
                        step=(max_l2 - min_l2)/10
                    )
                    layer.kernel_regularizer = regularizers.l1_l2(
                        l1=hp_l1,
                        l2=hp_l2
                    )
                if "dropout" in layer.name:
                    hp_drop = hp.Float(
                        "drop_rate_" + layer.name,
                        min_value=min_dropout,
                        max_value=max_dropout,
                        step=(max_dropout - min_dropout)/20
                    )
                    layer.rate = hp_drop
        elif "dropout" in i.name:
            hp_drop = hp.Float(
                "drop_rate_" + i.name,
                min_value=min_dropout,
                max_value=max_dropout,
                step=(max_dropout - min_dropout)/20
            )
            i.rate = hp_drop
        elif i.weights:
            i.set_weights(
                mm_splice_saved.get_layer(i.name).get_weights()
            )
            hp_l1 = hp.Float(
                "l1_" + i.name,
                min_value=min_l1,
                max_value=max_l1,
                step=(max_l1 - min_l1)/10
            )
            hp_l2 = hp.Float(
                "l2_" + i.name,
                min_value=min_l2,
                max_value=max_l2,
                step=(max_l2 - min_l2)/10
            )
            i.kernel_regularizer = regularizers.l1_l2(
                l1=hp_l1,
                l2=hp_l2
            )
    hp_lr = hp.Float(
        "learning_rate",
        min_value=min_lr,
        max_value=max_lr,
        step=((max_lr-min_lr)/10
    ))
    # use epsilon = 1e-07 due to numerical stability issues
    adam = Adam(
        learning_rate=hp_lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    # MSE seems to perform better with the new dataset
    merged_MMSplice_full.compile(loss=loss, optimizer=adam)
    return merged_MMSplice_full


### Load or make splits
## Set up data directories
run_metadata = {}
mmsplice_models_dir = os.path.join(
    PROJECT_ROOT,
    "models",
    "mmsplice_revised",
    "mmsplice"
)
output_models_dir, model_number = get_new_dir(mmsplice_models_dir)
output_figures_dir = (os.path.join(output_models_dir, "figures"))

if not load_splits:
    data_dir = os.path.join(PROJECT_ROOT, "models", "data", "mmsplice9")
    data_dir, data_number = get_new_dir(data_dir)
    logging.info(f"NEW data dir {data_dir}")
else:
    data_dir = data_splits_dir
    import re
    data_number = re.findall(r'\d+', os.path.split(data_splits_dir)[-1])[0]
    if not data_number:
        data_number = 1
    else:
        data_number = int(data_number)

# write dynamically created output_models_dir
# or data_dir (depending on whether the script
# will make the data or train the model)
# to a file that the slurm job can find.
if slurm_job_id is not None:
    if not make_data:
        dir_reference = "data_dir"
        dir_to_write = data_dir
    else:
        dir_reference = "output_models_dir"
        dir_to_write = output_models_dir
    slurm_path = os.path.join(
        PROJECT_ROOT,
        "bash_scripts",
        "run_directories",
        dir_reference,
        f"dir_{slurm_job_id}.txt"
    )
    with open(slurm_path, "w") as f:
        f.write(dir_to_write)

if debug:
    output_models_dir = create_directory("mmsplice_revised_debug")
    output_figures_dir = create_directory(
        os.path.join(
            "mmsplice_revised_debug",
            "figures"
        )
    )
elif not debug:
    create_directory(output_models_dir)
    logging.info(f"Created output_models_dir {output_models_dir}")
    create_directory(output_figures_dir)
    logging.info(f"Created output_figures_dir {output_figures_dir}")

## Save directory information for later reference
run_metadata["data_dir"] = data_dir
run_metadata["figures_dir"] = output_figures_dir
run_metadata["data_number"] = data_number
run_metadata["model_number"] = model_number
run_metadata["config_file"] = config_file
run_metadata["slurm_job_id"] = slurm_job_id

## Execute making or loading splits
if not load_splits:
    merged_df = pd.read_csv(merged_df_path)
    mmsplice_valid = merged_df.apply(find_valid_mmsplice_variants, axis=1)
    merged_df = merged_df[mmsplice_valid]
    if drop_margin_psi:
        margin_psi = find_ref_margin_PSI(merged_df)
        merged_df["is_margin"] = margin_psi
        merged_df = merged_df[~margin_psi]
    else:
        margin_psi = find_ref_margin_PSI(merged_df)
        merged_df["is_margin"] = margin_psi
    # drop NaN delta logits
    # equivalent to dropping row where all delta columns
    # were nan
    merged_df = merged_df[~merged_df["mean_dlogit"].isna()]
    fam_list_keys = [
        "Avg_coverage",
        "Avg_coverage_ref",
        "Avg_PSI",
        "Differential",
    ]
    balance_psi = True
    if balance_mode.lower() == "psi":
        dist_scale = "PSI"
        pass
    elif balance_mode.lower() == "dlogit":
        merged_df["abs_mean_dlogit"] = np.abs(merged_df["mean_dlogit"])
        fam_list_keys[2] = "abs_mean_dlogit"
        dist_scale = "Mean Delta Logit"
    elif balance_model.lower() == "dpsi":
        raise NotImplementedError("Implement delta PSI balancing if relevant")
    else:
        balance_psi = False
    family_list, merged_df = get_families_mmsplice(
        merged_df, 
        keys=fam_list_keys
    )
    test, val, train = get_split_families_balanced_psi(
        family_list,
        proportions=None,
        score_index=1,
        pool_sizes={
            "test" : 0.33, 
            "val" : 0.32,
        },
        num_hits=1,
        A=0,
        B=4,
        balance_psi=balance_psi,
    )
    # check that every sample was chosen
    split_total, fam_list_total = 0, 0
    for data in (train, val, test):
        for fam in data:
            split_total += fam[0]
    for fam in family_list:
        fam_list_total += fam[0]
    assert split_total == fam_list_total, (
        "Split total doesn't match family list total"
    )
    # Visualizing the splits
    splits_df = visualize_splits(train, val, test)
    print(splits_df)
    logging.info("getting one hots from splits")
    one_hot_splits = get_one_hots_from_splits(
        train,
        val,
        test,
        merged_df
    )
    x_train = one_hot_splits["x_train"]
    x_val = one_hot_splits["x_val"]
    x_test = one_hot_splits["x_test"]
    y_train = one_hot_splits["y_train"]
    y_val = one_hot_splits["y_val"]
    y_test = one_hot_splits["y_test"]
    metadata_train = one_hot_splits["metadata_train"]
    metadata_val = one_hot_splits["metadata_val"]
    metadata_test = one_hot_splits["metadata_test"]
    assert len(x_train) == len(x_val) == len(x_test), (
        f"\nlen x_train: {len(x_train)}\n"
        f"len x_train: {len(x_train)}\n" 
        f"len x_val: {len(x_val)}\n"
        f"len x_test: {len(x_test)}\n"
    )
    for i in range(len(x_train)):
        shape_check = (
            x_train[i].shape[1] 
            == x_val[i].shape[1] 
            == x_test[i].shape[1]
        )
        assert shape_check, (
            f"\nx_train {i} shape: {x_train[i].shape[i]}\n"
            f"x_train {i} shape: {x_train[i].shape[i]}\n" 
            f"x_val {i} shape: {x_val[i].shape[i]}\n"
            f"x_test {i} shape: {x_test[i].shape[i]}\n"
        ) 
    for i in range(len(x_train)):
        shape_check = (
            x_train[i].shape[2]
            == x_val[i].shape[2] 
            == x_test[i].shape[2]
        )
        assert shape_check, (
            f"\nx_train {i} shape: {x_train[i].shape[i]}\n"
            f"x_train {i} shape: {x_train[i].shape[i]}\n" 
            f"x_val {i} shape: {x_val[i].shape[i]}\n"
            f"x_test {i} shape: {x_test[i].shape[i]}\n"
        ) 
    if debug:
        breakpoint()
        " about to make new data dir and save data "
    create_directory(data_dir)
    logging.info(f"Created data_dir {data_dir}")
    if make_data:
        save_splits(one_hot_splits, data_dir)
        psi_distribution_plot(
            train,
            val,
            test,
            data_dir,
            scale=dist_scale,
        )
        del run_metadata["figures_dir"]
        del run_metadata["model_number"]
        run_metadata["merged_df_path"] = merged_df_path
        metadata_json = os.path.join(
            data_dir,
            "run_metadata.json"
        )
        with open(metadata_json, "w") as f:
            json.dump(run_metadata, f)
        sys.exit(0)
else:
    loaded, dict_data = load_splits_dir(data_dir)
    if loaded["x_train"]:
        data = dict_data
        x_train = data["x_train"]
        x_val = data["x_val"]
        x_test = data["x_test"]
        y_train = data["y_train"]
        y_val = data["y_val"]
        y_test = data["y_test"]
        metadata_train = data["metadata_train"]
        metadata_val = data["metadata_val"]
        metadata_test = data["metadata_test"]
    elif debug:
        breakpoint()
        " error loading splits "
    else:
        raise FileNotFoundError(f"Could not load split from {data_dir}")
    logging.info(
        f"Loaded data from {data_dir}\n"
        f"Train size: {y_train.shape[0]}\n"
        f"Val size: {y_val.shape[0]}\n"
        f"Test size: {y_test.shape[0]}"
    )


### Get baseline predictions
logging.info("Getting baseline predictions")

get_predictions = True
if debug:
    breakpoint()
    " set get_predictions = False to skip getting baseline predictions "

if get_predictions:
    pr_train_base = merged_MMSplice_full.predict(x_train)
    pr_val_base = merged_MMSplice_full.predict(x_val)
    pr_test_base = merged_MMSplice_full.predict(x_test)
else:
    pr_train_base = np.random.random(y_train.shape)
    pr_val_base = np.random.random(y_val.shape)
    pr_test_base = np.random.random(y_test.shape)



if debug:


    def check_batchnorm():
        for layer in merged_MMSplice_full.layers:
            if isinstance(layer, Functional):
                for layer_ in layer.layers:
                    if isinstance(layer_, tf.keras.layers.BatchNormalization):
                        print(layer_.moving_mean.numpy()[:5])
        return


    predictions = []


    def get_predictions(train=True, val=True, test=True, verbose=1):
        curr = {}
        if train:
            train_pr = merged_MMSplice_full.predict(x_train, verbose = verbose)
            train_r = pearsonr(train_pr, y_train)[0][0]
            train_mse = mse(train_pr, y_train)
            curr["train_r"] = train_r
            curr["train_mse"] = train_mse
        if val:
            val_pr = merged_MMSplice_full.predict(x_val, verbose = verbose)
            val_r = pearsonr(val_pr, y_val)[0][0]
            val_mse = mse(val_pr, y_val)
            curr["val_r"] = val_r
            curr["val_mse"] = val_mse
        if test:
            test_pr = merged_MMSplice_full.predict(x_test, verbose = verbose)
            test_r = pearsonr(test_pr, y_test)[0][0]
            test_mse = mse(test_pr, y_test)
            curr["test_r"] = test_r
            curr["test_mse"] = test_mse
        for i in curr:
            print(i.ljust(12), str(np.round(curr[i], 3)))
        predictions.append(curr)
        return


    breakpoint()
    " start training "




if sample_weights == "none":
    sample_weight = np.ones(y_train.shape)
elif sample_weights == "high":
    sample_weight = weigh_samples(y_train)

if not hyperparameter_tune:
    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    merged_MMSplice_full.compile(loss=loss, optimizer=adam)
    skip_train = False
    if debug:
        breakpoint()
        " set skip_train to True to skip training "
    if not skip_train:
        merged_MMSplice_full.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            epochs=n_epochs,
            sample_weight=sample_weight,
        )
else:
    if debug:
        breakpoint()
    else:
        hyperparam_dir = os.path.join(
            output_models_dir,
            "hyperparam",
        )
        assert not os.path.exists(hyperparam_dir), (
            f"Hyperparameter dir {hyperparam_dir} "
            f"already exists!"
        )



    if "--min_batchsize" in opts:
        class BatchHyperband(kerastuner.Hyperband):
            def run_trial(self, trial, *args, **kwargs):
                # Define and inject the batch size search space
                kwargs["batch_size"] = trial.hyperparameters.Int(
                    "batch_size",
                    min_batchsize,
                    max_batchsize,
                    step=((max_batchsize - min_batchsize)//10)
                )
                return (
                    super(BatchHyperband, self).run_trial(
                        trial,
                        *args,
                        **kwargs
                ))

        tuner = BatchHyperband(
            hyperband_mmsplice,
            objective='val_loss',
            max_epochs=hyperband_max_epochs,
            factor=3,
            directory=os.path.join(output_models_dir, "hyperparam"),
        )

    else:
        tuner = kerastuner.Hyperband(
            hyperband_mmsplice,
            objective='val_loss',
            max_epochs=hyperband_max_epochs,
            factor=3,
            directory=os.path.join(output_models_dir, "hyperparam"),
        )
    
    
    def load_weights_wrapper(tuner):
        """
        fix for tensorlflow 2.17
        adds .weights.h5 to saved checkpoints.
        """ 
        old_load_func = tuner._get_checkpoint_fname
        def new_load_func(trial_id):
            return old_load_func(trial_id) + ".weights.h5"
        return new_load_func


    tuner._get_checkpoint_fname = load_weights_wrapper(tuner)


    tuner.search_space_summary()
    
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", 
        min_delta=0.001,
        patience=hyperband_patience
    )
    
    tuner.search(
        x_train,
        y_train,
        epochs=hyperband_max_epochs,
        validation_data=(x_val, y_val),
        callbacks=[stop_early],
    )
    
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    hps = ""
    for value in best_hps.values:
        hps += str(value) + " " + str(best_hps.get(value)) + "\n"
    logging.info(f"best hyperparameters:\n{hps}") 
    
    # Build the model with the optimal hyperparameters
    # and train until it overfits 
    # then revert to best validation set epoch (if early stop = True)
    model = tuner.hypermodel.build(best_hps)
    " monitor to restore best epoch "
    if early_stop:
        stop_early_save_best = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=hyperband_retrain_patience,
            restore_best_weights=True
        )
        history = model.fit(
            x_train,
            y_train,
            epochs=hyperband_retrain_epochs,
            validation_data=(x_val, y_val),
            callbacks=[stop_early_save_best],
            sample_weight=sample_weight
        )
        val_acc_per_epoch = history.history["val_loss"]
        best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
        logging.info(
            f"Best epoch: {best_epoch}, "
            f"Val Loss: {min(val_acc_per_epoch)}"
        )
        if fine_tune_non_margin_psi:
            x_train_non_margin, y_train_non_margin = get_non_margin_split(
                x_train,
                y_train,
                metadata_train
            )
            if sample_weights == "none":
                margin_sample_weight = np.ones(y_train_non_margin.shape)
            elif sample_weights == "high":
                margin_sample_weight = weigh_samples(y_train_non_margin)

            logging.info("fine tuning non-margin psi")
            
            history = model.fit(
                x_train_non_margin,
                y_train_non_margin,
                epochs=fine_tune_epochs,
                validation_data=(x_val, y_val),
                callbacks=[stop_early_save_best],
                sample_weight=margin_sample_weight
            )
    else:
        history = model.fit(
            x_train,
            y_train,
            epochs=hyperband_retrain_epochs,
            validation_data=(x_val, y_val),
            sample_weight=sample_weight
        )
        val_acc_per_epoch=history.history["val_loss"]
        if fine_tune_non_margin_psi:
            x_train_non_margin, y_train_non_margin = get_non_margin_split(
                x_train,
                y_train,
                metadata_train
            )
            if sample_weights == "none":
                margin_sample_weight = np.ones(y_train_non_margin.shape)
            elif sample_weights == "high":
                margin_sample_weight = weigh_samples(y_train_non_margin)
            logging.info("fine tuning non-margin psi")
            history = model.fit(
                x_train_non_margin,
                y_train_non_margin,
                epochs=hyperband_retrain_epochs,
                validation_data=(x_val, y_val),
                sample_weight=margin_sample_weight
            )
    

    if debug:
        breakpoint()
        " testing reload weights "

    # ensure hyperparameter tuned layer weights 
    # are passed onto the final model
    for layer in model.layers:
        name = layer.name
        merged_MMSplice_full.get_layer(name).set_weights(layer.weights)

if debug:
    breakpoint()
    " finished training "

# Save reimplemented MMSplice model
if model_number > 1:
    model_save_num = ""
else:
    model_save_num = model_number
model_name = (
    f"MMSplice_delta_logit_PSI_model_retrained_revised"
    f"{model_save_num}.hdf5"
)
model_save_path = os.path.join(
    output_models_dir,
    model_name
)
assert not os.path.exists(model_save_path), (
    f"model save path {model_save_path} exists"
)

if check_version(tf.__version__, '2.17') == "less":
    logging.info("Tensorflow version < 2.17")
    from tensorflow.keras.layers import (
        Dense,
        Input,
    )
    from tensorflow.keras.models import Model
    def save_model(model, path):
        model.save(path)
        return
else:
    logging.info("Tensorflow version >= 2.17")
    from keras.src.layers import (
        Dense,
        Input,
    )
    from keras.src.models import Model
    from keras.src.saving.saving_api import save_model

if not debug:
    try:
        # reload model layers to fix 'name already exists' error
        merged_MMSplice_full = Model(
            inputs=merged_MMSplice_full.inputs,
            outputs=merged_MMSplice_full.outputs
        )
        save_model(merged_MMSplice_full, model_save_path)
        logging.info(f"SAVED model to {model_save_path}")
        keras_save_path = model_save_path.replace("hdf5", "keras")
        save_model(merged_MMSplice_full, keras_save_path)
        logging.info(f"SAVED model to {keras_save_path}")
    except ValueError as e:
        import traceback
        traceback.print_exc()
        if not os.path.exists(model_save_path):
            raise FileNotFoundError(f"Model save failure at {model_save_path}")


def merge_train_val_test(train, val, test, y_train, y_val, y_test):
    """
    Merge datasets for aggregate predictions and
    plots
    """
    merged_X = [None] * len(train)
    for i in range(len(train)):
        merged_X[i] = np.concatenate([train[i], val[i], test[i]])
    merged_Y = np.concatenate([y_train, y_val, y_test])    
    return merged_X, merged_Y


X,Y = merge_train_val_test(
    x_train,
    x_val,
    x_test,
    y_train,
    y_val,
    y_test
)
logging.info("Getting retrained predictions")
pr_train_rt = merged_MMSplice_full.predict(x_train)
pr_val_rt = merged_MMSplice_full.predict(x_val) 
pr_test_rt = merged_MMSplice_full.predict(x_test)
logging.info(
    f"{'baseline train'.ljust(15)} "
    f"r={calc_r(pr_train_base, y_train)[0]} "
    f"mse={mse(pr_train_base, y_train)}"
)
logging.info(
    f"{'baseline val'.ljust(15)} "
    f"r={calc_r(pr_val_base, y_val)[0]} "
    f"mse={mse(pr_val_base, y_val)}"
)
logging.info(
    f"{'baseline test'.ljust(15)} "
    f"r={calc_r(pr_test_base, y_test)[0]} "
    f"mse={mse(pr_test_base, y_test)}"
)
logging.info(
    f"{'retrained train'.ljust(15)} "
    f"r={calc_r(pr_train_rt, y_train)[0]} "
    f"mse={mse(pr_train_rt, y_train)}"
)
logging.info(
    f"{'retrained val'.ljust(15)} "
    f"r={calc_r(pr_val_rt, y_val)[0]} "
    f"mse={mse(pr_val_rt, y_val)}"
)
logging.info(
    f"{'retrained test'.ljust(15)} "
    f"r={calc_r(pr_test_rt, y_test)[0]} "
    f"mse={mse(pr_test_rt, y_test)}"
)
Y_baseline = np.concatenate([
        pr_train_base, 
        pr_val_base, 
        pr_test_base
    ], 
    axis=0
)
logging.info(
    f"baseline model aggregate data "
    f"r2 {r2(Y_baseline[:,0], Y[:,0])}"
)

if debug:
    breakpoint()
    " start plotting "

### Plotting

# Parameters for plotting model results #
pd.set_option("display.max_colwidth",100)
sns.set(style="ticks", color_codes=True)
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.labelpad'] = 5
plt.rcParams['axes.linewidth']= 1.5
plt.rcParams['xtick.labelsize']= 14
plt.rcParams['ytick.labelsize']= 14
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'

r_train = calc_r(pr_train_rt[:,0], y_train[:,0])
r_val = calc_r(pr_val_rt[:,0], y_val[:,0])
r_test = calc_r(pr_test_rt[:,0], y_test[:,0])
sns.scatterplot(
    x=pr_train_rt[:,0], 
    y=y_train[:,0],
    alpha=0.1,
    label="Train",
)
sns.scatterplot(
    x=pr_test_rt[:,0],
    y=y_test[:,0],
    alpha=0.1, 
    label="Test"
)
plt.xlabel('MMSplice Δlogit PSI Prediction')
plt.ylabel('Mean Δlogit PSI Across Cell Types')
plt.title(
    f"Retrained MMSplice Δlogit PSI\n"
    f"Train r={round(r_train,3)}, "
    f"Test r={round(r_test,3)}"
)
leg = plt.legend(loc="upper right", fontsize=12)
try:
    for lh in leg.legendHandles:
        lh.set_alpha(1)
except AttributeError:
    for lh in leg.legend_handles:
        lh.set_alpha(1)
plt.tight_layout()
plt.savefig(
    os.path.join(
        output_figures_dir, 
        "prediction_observation_scatter.png"
    )
)
plt.savefig(
    os.path.join(
        output_figures_dir,
        "prediction_observation_scatter.pdf"
    )
)
saved_path = os.path.join(
    output_figures_dir,
    "prediction_observation_scatter.png"
)
logging.info(f"saved figure to {saved_path}")
plt.close()

logging.info(f"retrained delta logit PSI correlation, test =  {r_test}")
logging.info(f"retrained delta logit PSI correlation, train = , {r_train}")

r_train_baseline = calc_r(pr_train_base[:,0], y_train[:,0])
r_val_baseline = calc_r(pr_val_base[:,0], y_val[:,0])
r_test_baseline = calc_r(pr_test_base[:,0], y_test[:,0])

fig, ax = plt.subplots(3, figsize=(15,29))

sns.scatterplot(
    x=pr_train_rt[:,0],
    y=y_train[:,0],
    alpha=0.1,
    ax=ax[0]
)
sns.scatterplot(
    x=pr_train_base[:,0],
    y=y_train[:,0],
    alpha=0.1,
    ax=ax[0]
)
sns.scatterplot(
    x=pr_val_rt[:,0],
    y=y_val[:,0],
    alpha=0.1,
    ax=ax[1]
)
sns.scatterplot(
    x=pr_val_base[:,0],
    y=y_val[:,0],
    alpha=0.1,
    ax=ax[1]
)
sns.scatterplot(
    x=pr_test_rt[:,0],
    y=y_test[:,0],
    alpha=0.1,
    ax=ax[2]
)
sns.scatterplot(
    x=pr_test_base[:,0],
    y=y_test[:,0],
    alpha=0.1,
    ax=ax[2]
)
data_names = ["Train", "Validation", "Test"]
r_vals = [
    (r_train, r_train_baseline), 
    (r_val, r_val_baseline),
    (r_test, r_test_baseline)
]
for i in range(3):
    ax_title = (
        f"Retrained MMSplice Δlogit PSI vs Pretrained "
        f"on {data_names[i]} Set "
        f"(Retrained r={round(r_vals[i][0],3)}, "
        f"Baseline r = {round(r_vals[i][1],3)})"
    )
    ax[i].set_title(ax_title, fontsize=20)
    ax[i].set_xlabel(
        "Predicted Mean Δlogit PSI Across Cells",
        fontsize=20
    )
    ax[i].set_ylabel(
        "True Mean Δlogit PSI Across Cells", 
        fontsize=20
    )
leg = fig.legend(
    ["Retrained", "Baseline"],  
    loc="upper right", 
    fontsize=12
)
try:
    for lh in leg.legendHandles:
        lh.set_alpha(1)
except AttributeError:
    for lh in leg.legend_handles:
        lh.set_alpha(1)

fig_title = (
    "Comparison of Retrained \nMMSplice to Baseline Model"
)
fig.suptitle(fig_title, fontsize=20)
save_path = os.path.join(
    output_figures_dir,
    "mmsplice_vs_pretrained.png"
)
plt.savefig(save_path)
plt.savefig(save_path.replace(".png", ".pdf"))
logging.info(f"saved figure to {save_path}")
plt.close()

fig, ax = plt.subplots(3, 2, figsize=(22, 30))

sns.scatterplot(
    x=pr_train_rt[:,0],
    y=y_train[:,0],
    alpha =0.1,
    ax=ax[0][0]
)
sns.scatterplot(
    x=pr_train_base[:,0],
    y=y_train[:,0],
    alpha=0.1,
    ax=ax[0][1]
)
sns.scatterplot(
    x=pr_val_rt[:,0],
    y=y_val[:,0],
    alpha=0.1,
    ax=ax[1][0]
)
sns.scatterplot(
    x=pr_val_base[:,0],
    y=y_val[:,0],
    alpha=0.1,
    ax=ax[1][1]
)
sns.scatterplot(
    x=pr_test_rt[:,0],
    y=y_test[:,0],
    alpha=0.1,
    ax=ax[2][0]
)
sns.scatterplot(
    x=pr_test_base[:,0],
    y=y_test[:,0],
    alpha=0.1,
    ax=ax[2][1]
)
data = [
    (pr_train_rt, pr_train_base),
    (pr_val_rt, pr_val_base), 
    (pr_test_rt, pr_test_base)
]
data_names = ["Train", "Validation", "Test"]
model_names = ["Retrained", "Baseline"]
r_vals = [
    (r_train, r_train_baseline),
    (r_val, r_val_baseline),
    (r_test, r_test_baseline)
]
for i in range(3):
    for j in range(2):
        ax_title = (
            f"{data_names[i]} Set "
            f"{model_names[j]}\n "
            f"n={len(data[i][j])}, "
            f"r={round(r_vals[i][j],3)}"
        )
        ax[i][j].set_title(ax_title, fontsize=20)
        ax[i][j].set_xlabel(
            "Predicted Mean Δlogit PSI Across Cells", 
            fontsize=20
        )
        ax[i][j].set_ylabel(
            "True Mean Δlogit PSI Across Cells",
            fontsize=20
        )
fig_title = "Comparison of Retrained MMSplice to Baseline Model"
fig.suptitle(
    fig_title,
    fontsize=20,
    x=0.5,
    y=1.01
)
fig.tight_layout()
save_path = os.path.join(
    output_figures_dir,
    "mmsplice_vs_pretrained_split.png"
)
plt.savefig(save_path)
plt.savefig(save_path.replace(".png", ".pdf"))
logging.info(f"saved figure to {save_path}")
plt.close()

preds = np.concatenate([
        pr_train_rt,
        pr_val_rt,
        pr_test_rt
    ],
    axis=0
)
true = np.concatenate([
        y_train,
        y_val,
        y_test
    ],
    axis=0
)
## Aggregate plots
r_val = calc_r(preds[:,0], true[:,0])
sns.scatterplot(x=preds[:,0], y=true[:,0], alpha =0.1)
plt.xlabel('Δlogit PSI Prediction')
plt.ylabel('Mean Δlogit PSI Across Cell Types')
fig_title = (
    f"Retrained Full Dataset Δlogit PSI\n"
    f"n={len(true)}, r={round(r_val, 3)}"
) 
plt.title(fig_title)
plt.tight_layout()

save_path = os.path.join(
    output_figures_dir,
    "retrained_predictions_scatter_full_data.png"
)
plt.savefig(save_path)
plt.savefig(save_path.replace(".png", ".pdf"))
plt.close()

logging.info(f"retrained aggregate delta logit PSI correlation= {r_val}")
logging.info(f"retrained aggregate delta logit PSI R^2= {r_val ** 2}")

preds_old = np.concatenate([
        pr_train_base,
        pr_val_base,
        pr_test_base
    ],
    axis=0
)

sns.scatterplot(x=preds_old[:,0], y=true[:,0], alpha =0.1)
r_val = calc_r(preds_old[:,0], true[:,0])
plt.xlabel('Δlogit PSI Prediction')
plt.ylabel('Mean Δlogit PSI Across Cell Types')
plt.title(
    f"Baseline Full Dataset Δlogit PSI\nn={len(true)}, "
    f"r={round(r_val, 3)}"
)
plt.tight_layout()

baseline_full = "mmsplice_baseline_predictions_scatter_full_data.png"
models_path = os.path.join(output_figures_dir, baseline_full)
data_path = os.path.join(data_dir, baseline_full)

plt.savefig(data_path)
plt.savefig(data_path.replace(".png", ".pdf"))
logging.info(f"saved baseline full figure to {data_path}")
plt.savefig(models_path)
plt.savefig(models_path.replace(".png", ".pdf"))
logging.info(f"saved baseline full figure to {models_path}")
plt.close()

logging.info(f"baseline aggregate delta logit PSI correlation= {r_val}")
logging.info(f"baseline aggregate delta logit PSI R^2= {r_val ** 2}")

metadata_json = os.path.join(
    output_models_dir,
    "run_metadata.json"
)
with open(metadata_json, "w") as f:
    json.dump(run_metadata, f)

predictions_df = {
    "Variant_ID" : [],
    "Reference_ID" : [],
    "Intron1 Size" : [],
    "Baseline_MMSplice_Predicted_Delta_Logit" : [],
    "Retrained_MMSplice_Predicted_Delta_Logit" : [],
    "True_Delta_Logit" : [],
    "Event ID" : [],
    "Split" : [],
}

def add_predictions(
        predictions_df,
        retrained,
        baseline,
        y,
        metadata,
        split_name
    ):
    """
    Helper functions to build prediction dataframe.
    gathers IDs, predictions and true value into a
    single dataframe. split_name (Train, Val, Test)
    is used to identify which spits the predictions
    originated from.
    """
    var_ids = metadata["var_id"].tolist()
    ref_ids = metadata["ref_id"].tolist()
    intron1_sizes = metadata["intron1_len"].tolist()
    if "event_id_161" in metadata:
        event_ids = metadata["event_id_161"].tolist()
    else:
        event_ids = metadata["event_id"].tolist()
    dlogit_key = "MMSplice_Predicted_Delta_Logit"
    rt_dlogit_key = f"Retrained_{dlogit_key}"
    base_dlogit_key = f"Baseline_{dlogit_key}"
    for i in range(len(retrained)):
        predictions_df[rt_dlogit_key].append(retrained[i,0])
        predictions_df[base_dlogit_key].append(baseline[i,0])
        predictions_df["Variant_ID"].append(var_ids[i])
        predictions_df["Reference_ID"].append(ref_ids[i])
        predictions_df["Intron1 Size"].append(intron1_sizes[i])
        predictions_df["Event ID"].append(event_ids[i])
        predictions_df["True_Delta_Logit"].append(y[i,0])
        predictions_df["Split"].append(split_name)
    return predictions_df


predictions_df = add_predictions(
    predictions_df=predictions_df,
    retrained=pr_train_rt,
    baseline=pr_train_base,
    y=y_train,
    metadata=metadata_train,
    split_name="Train",
)
predictions_df = add_predictions(
    predictions_df=predictions_df,
    retrained=pr_val_rt,
    baseline=pr_val_base,
    y=y_val,
    metadata=metadata_val,
    split_name="Val",
)
predictions_df = add_predictions(
    predictions_df=predictions_df,
    retrained=pr_test_rt,
    baseline=pr_test_base,
    y=y_test,
    metadata=metadata_test,
    split_name="Test",
)
predictions_df = pd.DataFrame(predictions_df)
predictions_path = os.path.join(
    output_models_dir,
    "predictions.csv"
)
if debug:
    breakpoint()
    " check predictions path "
predictions_df.to_csv(predictions_path, index=False)
logging.info(
    f"Saved predictions dataframe to "
    f"{predictions_path}"
)

if debug:
    breakpoint()
    " final breakpoint "

logging.info("Done")
