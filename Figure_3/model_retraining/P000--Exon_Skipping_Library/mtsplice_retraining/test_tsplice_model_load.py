"""
plotting.Retrain the MMSplice architecture for ESL cell line data.
"""


# TODO: add potential opts:
    # Optimizer (e.g. with adam learning rate),
    # currently using adam in hyperband
    # log the optimizer used


import getopt
import sys
import os
import logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)


### Import Project
node = os.getcwd()
for i in range(5): # max depth
    if "src" in os.listdir(node):
        tree, node = os.path.split(node)
        break
    tree, node = os.path.split(node)
    node = tree
PROJECT_ROOT = os.path.join(tree, node)
if not os.path.exists(PROJECT_ROOT) or "src" not in os.listdir(PROJECT_ROOT):
    raise ModuleNotFoundError(
        "Could not find project root with 'src' directory"
    )
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# define input arguments
input_args = [
    "mtsplice_model_number=", 
    "mmsplice_model_number=", 
    "load_data=", 
    "load_ascot_data=", 
    "ascot_data_dir=", 
    "tsplice_data_dir=", 
    "mtsplice_data_dir=", 
    "output_models_dir=",
    "output_figures_dir=",
    "baseline_mmsplice_path=", 
    "test_loaded_weights=", 
    "tsplice_n_epochs=", 
    "tsplice_loss=", 
    "tsplice_learning_rate=", 
    "tsplice_batch_size=", 
    "tsplice_cell_weights=", 
    "tsplice_early_stop=", 
    "tsplice_patience=", 
    "mtsplice_n_epochs=", 
    "mtsplice_loss=", 
    "mtsplice_learning_rate=", 
    "mtsplice_batch_size=", 
    "mtsplice_cell_weights=", 
    "mtsplice_early_stop=", 
    "mtsplice_patience=", 
    "gpu_memory=", 
    "weight_initializer=", 
    "hyperparameter_tune=", 
    "hyperband_n_epochs=", 
    "hyperband_max_n_epochs=", 
    "hyperband_patience=", 
    "hyperband_retrain_n_epochs=", 
    "hyperband_retrain_patience=", 
    "hyperband_lr_bounds=", 
    "freeze_mmsplice_layers=",
    "tsplice_natural_scale=", 
    "tsplice_output_cell_scale=", 
    "nan_mask_strategy=", 
    "output_plot_scale=",
    "merged_data_path=",
    "all_samples_data_path=",
    "mmsplice_data_dir=",
    "slurm_job_id=",
    "make_data", 
    "debug", 
    "help", 
]

opts, args = getopt.getopt(sys.argv[1:],"", input_args)
opts = dict(opts)
if "--help" in opts:
    dir_ = os.path.dirname(__file__)
    readme = os.path.join(dir_, "README")
    with open(readme, "r") as f:
        help_ = f.read()
    print(help_)
    sys.exit(0)

from src.utils.parsing import bool_map, ErrorWrapper
opts = ErrorWrapper(opts)
if "--debug" in opts:
    if "--make_data" in opts:
        raise NotImplementedError("Cannot make data in debug mode")
    if len(opts) > 1:
        raise ValueError("--debug option cannot be used with other arguments")
    mtsplice_model_number = "1"
    load_data = True
    load_ascot_data = False
    output_models_dir = os.path.join(PROJECT_ROOT, "models")
    data_parent_dir = os.path.join(PROJECT_ROOT, "models", "data")
    ascot_data_dir = os.path.join(
        data_parent_dir,
        "tsplice_ascot_esl_data3",
    )
    tsplice_data_dir = os.path.join(
        data_parent_dir,
        "merged_mm_mt_splice_logit_tsplice_split2"
    )
    mtsplice_data_dir = os.path.join(
        data_parent_dir,
        "merged_mm_mt_splice_logit_mtsplice_split2"
    )
    test_loaded_weights = False
    tsplice_n_epochs = 15
    tsplice_loss = "mse"
    baseline_mmsplice_path = os.path.join(
        os.path.dirname(PROJECT_ROOT),
        "MMSplice_MTSplice"
    )
    mmsplice_model_number = "139"
    mtsplice_n_epochs = 2
    mtsplice_learning_rate = 9.01e-6
    mtsplice_batch_size = 16
    tsplice_cell_weights=";".join([
        "proportional",
        "inverse",
        "1.19,1.19,0.24,1.19,1.19",
        "1.81494199,0.35241592,0.4063303,2.05465131,0.37166048",
    ])
    mtsplice_cell_weights="proportional"
    mtsplice_early_stop = False
    mtsplice_patience = 10
    mtsplice_baseline_predictions_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "data",
        "mmsplice_with_measured_filtered",
        "mtsplice_with_measured_unfiltered.tsv"
    )
    gpu_memory = 96
    initializer = None
    hyperparameter_tune = True
    hyperband_n_epochs = 2
    hyperband_max_n_epochs = 5 
    hyperband_lr_max = 0.01
    hyperband_lr_min = 1e-6
    hyperband_patience = 1
    hyperband_retrain_n_epochs = 5
    hyperband_retrain_patience = 1
    freeze_mmsplice_layers = True
    tsplice_natural_scale = False
    tsplice_output_cell_scale = "logit"
    output_plot_scale = ["dlogit"]
    nan_mask_strategy = "inpute_zero"
    slurm_job_id = None
    debug = True
    logging.info("DEBUG MODE")
elif "--make_data" not in opts:
    mtsplice_model_number = opts.get("--mtsplice_model_number", "1")
    mmsplice_model_number = opts["--mmsplice_model_number"]
    load_data = bool_map(opts["--load_data"])
    load_ascot_data = bool_map(opts["--load_ascot_data"])
    if load_data:
        try:
            tsplice_data_dir = opts["--tsplice_data_dir"]
            mtsplice_data_dir = opts["--mtsplice_data_dir"]
        except KeyError as e:
            logging.error(
                "tsplice_data_dir and mtsplice_data_dir "
                "input is necessary if load_data is true"
            )
            sys.exit(1)
    if load_ascot_data:
        try:
            ascot_data_dir = opts["--ascot_data_dir"]
        except KeyError as e:
            logging.error(
                "ascot_data_dir input is "
                "necessary if load_ascot_data is true")
            sys.exit(1)
    output_models_dir = opts["--output_models_dir"]
    test_loaded_weights = bool_map(opts["--test_loaded_weights"])
    tsplice_n_epochs = int(opts["--tsplice_n_epochs"])
    tsplice_loss = opts.get("--tsplice_loss", "huber")
    baseline_mmsplice_path = opts.get("--baseline_mmsplice_path", "")
    mtsplice_n_epochs = int(opts["--mtsplice_n_epochs"])
    mtsplice_learning_rate = float(opts["--mtsplice_learning_rate"])
    mtsplice_batch_size = int(opts["--mtsplice_batch_size"])
    tsplice_cell_weights = opts["--tsplice_cell_weights"]
    mtsplice_cell_weights = opts["--mtsplice_cell_weights"]
    mtsplice_early_stop = bool_map(opts["--mtsplice_early_stop"])
    if mtsplice_early_stop:
        mtsplice_patience = int(opts.get("--mtsplice_patience", 10))
    gpu_memory = (opts.get("--gpu_memory", None))
    if gpu_memory is not None:
        gpu_memory = int(gpu_memory)
    weight_initializer = opts.get("--weight_initializer", None)
    hyperparameter_tune = bool_map(opts["--hyperparameter_tune"])
    if hyperparameter_tune:
        hyperband_n_epochs = int(opts["--hyperband_n_epochs"])
        hyperband_max_n_epochs = int(opts["--hyperband_max_n_epochs"])
        hyperband_patience = int(opts["--hyperband_patience"])
        hyperband_retrain_n_epochs = int(opts["--hyperband_retrain_n_epochs"])
        hyperband_retrain_patience = int(opts["--hyperband_retrain_patience"])
        hyperband_lr = opts.get(
            "--hyperband_lr_bounds",
            "0.00001,0.01",
        )
        hyperband_lr_min, hyperband_lr_max = hyperband_lr.split(",")
        hyperband_lr_min = float(hyperband_lr_min)
        hyperband_lr_max = float(hyperband_lr_max)
    freeze_mmsplice_layers = bool_map(opts["--freeze_mmsplice_layers"])
    tsplice_natural_scale = bool_map(opts["--tsplice_natural_scale"])
    tsplice_output_cell_scale = opts.get("--tsplice_output_cell_scale", None)
    output_plot_scale = opts["--output_plot_scale"].split(",")
    nan_mask_strategy = opts["--nan_mask_strategy"]
    assert (tsplice_output_cell_scale == "logit" 
            or tsplice_output_cell_scale == "psi"), (
                f"--tsplice_output_cell_scale must either be "
                f"'logit' or 'psi', got "
                f"'{tsplice_output_cell_scale}'"
            )
    slurm_job_id = opts.get("--slurm_job_id", None)
    debug = False
else:
    raise NotImplementedError(
        "Please use bash script (make_mtsplice_data_splits.sh) "
        "to make data splits, since it supports saving .err and .out files"
    )
    sys.exit(0)
    
logging.info(f"\n{opts.__str__()}\n")

opts.check_inputs()
try:
    import mmsplice
except ModuleNotFoundError:
    sys.path.append(baseline_mmsplice_path) #TODO: Match with MMSplice rt

try:
    import mmsplice
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Failed to import mmsplice. Neither importing"
        "directly via import mmsplice or appending"
        "baseline_mmsplice_path from opts to sys path"
        "worked.\nThe input baseline_mmsplice_path was"
        f"{baseline_mmsplice_path}\n"
        "note that the default value is an empty string,"
        "so ensure --baseline_mmsplice_path was passed in"
        "or add the module to the python environment."
    )
# finally import mmsplice
import mmsplice

# import common packages
import pandas as pd
import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# custom functions
from src.utils.check_version import check_version
from src.utils.overwrite_protection import get_new_dir
from src.load_data import load_splits_dir
from src.esl import (
    clip,
    logit,
    expit,
    cell_types,
    cell_types2,
    Expit,
)
from src.model_eval import (
    r2,
    calc_r,
    mse
)
from src.data_helpers.nan_handling.nan_helper_tsplice import (
    NanMask,
    preprocess_nan_masks,
    add_nan_mask_layer
)
# mmsplice functions
from mmsplice_retraining.mmsplice_model.mmsplice_build import (
    get_saved_model as get_saved_mmsplice
)
from mmsplice_retraining.mmsplice_model.mmsplice_build import (
    get_hyperband_mmsplice
)
# tsplice and mtsplice functions
from mtsplice_retraining.data_pipeline.get_tsplice_data import (
    remake_data,
)
from mtsplice_retraining.tsplice_model.cell_type_weight_balancer import (
    get_cell_type_weights,
    parse_cell_type_weights
)
from mtsplice_retraining.mtsplice_model.mt_splice_combiner import (
    combine_models_scale
)
from mtsplice_retraining.model_utils.initializer import zero_init
import mtsplice_retraining.tsplice_model.tsplice_hyperband as tsplice_hyperband
from mtsplice_retraining.model_evaluation.r2_calc import (
    calc_r2_with_nan,
    calc_r2_with_nan_tsplice
)
from mtsplice_retraining.model_evaluation import model_evaluation
from mtsplice_retraining.plotting import save_model_predictions
import mtsplice_retraining.plotting.plotting as plotting
from mtsplice_retraining.model_evaluation.mtsplice_predictor import (
    MTSplicePredictor
)

# import version-specific tensorflow modules
import tensorflow as tf
if check_version(tf.__version__, "2.17") == "less":
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
        Input
    )
    from keras.src.models import Model
    from keras.src.saving.saving_api import save_model
try:
    from keras.src.models.functional import Functional
except ModuleNotFoundError:
    from tensorflow.python.keras.engine.functional import Functional
from tensorflow.keras import regularizers
import keras_tuner as kerastuner
from tensorflow.keras.optimizers import Adam

try: 
    from OSU import create_directory
except ModuleNotFoundError:
    def create_directory(path):
        if path[0] == "/":
            # absolute path
            curr_path = "/"
            path = path.split("/")[1:]
        else:
            curr_path = ""
            path = path.split("/")
        for i in range(len(path) - 1):
            curr_path = os.path.join(curr_path, path[i])
            if not os.path.exists(curr_path):
                os.mkdir(curr_path)
        return os.path.join(curr_path, path[-1])



logging.info(f"cell types: {cell_types}")


### Defining model

    
def get_single_model_custom_loss(
        model_indx=0,
        scale="psi",
        mt_splice=None,
        scale_output=False
    ):
    """
    Loads models from the MTSplice Github,
    and redifines them for the specified scale (psi or logit)
    and extracts one of the output heads of the ensemble
    using model_indx.
    """
    if mt_splice is None:
        mt_splice = mmsplice.mtsplice.MTSplice()
    in1 = Input(shape=(400, 4), name = f"seqL")
    in2 = Input(shape=(400, 4), name = f"seqR")
    in_mean = Input(shape=(1,))
    
    model = mt_splice.mtsplice_models[model_indx]
    dense = model.layers[-1]
    assert "dense" in dense.name
    new_dense = Dense(5, name = "cell_type_dense")
    new_dense_out = new_dense(dense.input)
    new_model = Model(
        inputs=model.inputs,
        outputs=[new_dense_out],
        name="tsplice_delta_logit_model"
    )
    final_out_delta_logit = new_model([in1, in2])
    
    if scale == "psi":
        final_out_cell_logit = final_out_delta_logit + in_mean
        final_out_cell_psi = Expit(name="expit")(final_out_cell_logit)
        final_model = Model(
            inputs=[in1, in2, in_mean],
            outputs=[final_out_cell_psi]
        )
    elif scale == "logit":
        if isinstance(final_out_delta_logit, list):
            final_out_delta_logit = final_out_delta_logit[0]
        if scale_output:
            final_out_delta_logit_scaled = final_out_delta_logit + in_mean
            final_model = Model(
                inputs=[in1, in2, in_mean],
                outputs=[final_out_delta_logit_scaled]
            )
        else:
            final_model = Model(
                inputs=[in1, in2],
                outputs=[final_out_delta_logit]
            )
    return final_model
 

# get all models in ensemble
# load mtsplice so we dont reload it for each ensemble head
mt_splice = mmsplice.mtsplice.MTSplice() 
TSplice_Model_0 = get_single_model_custom_loss(
    model_indx=0,
    scale=tsplice_output_cell_scale,
    mt_splice=mt_splice
)
TSplice_Model_1 = get_single_model_custom_loss(
    model_indx=1,
    scale=tsplice_output_cell_scale,
    mt_splice=mt_splice
)
TSplice_Model_2 = get_single_model_custom_loss(
    model_indx=2,
    scale=tsplice_output_cell_scale,
    mt_splice=mt_splice
)
TSplice_Model_3 = get_single_model_custom_loss(
    model_indx=3,
    scale=tsplice_output_cell_scale,
    mt_splice=mt_splice
)


def get_tissue_indexes(keys):
    """Get baseline weights for the final dense layer"""
    return [tissue_index_map[i] for i in keys]


TISSUES = [
    "Retina - Eye", "RPE/Choroid/Sclera - Eye", "Subcutaneous - Adipose",
    "Visceral (Omentum) - Adipose", "Adrenal Gland", "Aorta - Artery",
    "Coronary - Artery", "Tibial - Artery", "Bladder", "Amygdala - Brain",
    "Anterior cingulate - Brain", "Caudate nucleus - Brain",
    "Cerebellar Hemisphere - Brain", "Cerebellum - Brain", "Cortex - Brain",
    "Frontal Cortex - Brain", "Hippocampus - Brain", "Hypothalamus - Brain",
    "Nucleus accumbens - Brain", "Putamen - Brain",
    "Spinal cord (C1) - Brain", "Substantia nigra - Brain",
    "Mammary Tissue - Breast", "EBV-xform lymphocytes - Cells",
    "Leukemia (CML) - Cells", "Xform. fibroblasts - Cells",
    "Ectocervix - Cervix", "Endocervix - Cervix", "Sigmoid - Colon",
    "Transverse - Colon", "Gastroesoph. Junc. - Esophagus",
    "Mucosa - Esophagus", "Muscularis - Esophagus", "Fallopian Tube",
    "Atrial Appendage - Heart", "Left Ventricle - Heart", "Cortex - Kidney",
    "Liver", "Lung", "Minor Salivary Gland", "Skeletal - Muscle",
    "Tibial - Nerve", "Ovary", "Pancreas", "Pituitary", "Prostate",
    "Not Sun Exposed - Skin", "Sun Exposed (Lower leg) - Skin",
    "Ileum - Small Intestine", "Spleen", "Stomach", "Testis", "Thyroid",
    "Uterus", "Vagina", "Whole Blood"
]

# Get the weigths by aggregating across the relevant cell types
# for the 5 cancer cell lines
tissue_index_map = {}
for i in range(len(TISSUES)):
    tissue_index_map[TISSUES[i]] = i
HEK_tissues = ["Cortex - Kidney"]
HeLa_tissues = ["Uterus", "Ectocervix - Cervix", "Endocervix - Cervix"]
K562_tissues = ["Leukemia (CML) - Cells"]
MCF7_tissues = ["Mammary Tissue - Breast"]
# average over all brain tissues
HMC3_tissues = [tissue for tissue in TISSUES if "Brain" in tissue] 
tissue_indexes = [
    get_tissue_indexes(i) 
    for i in [
        HEK_tissues,
        HeLa_tissues,
        K562_tissues,
        MCF7_tissues,
        HMC3_tissues
    ]
]


def get_baseline_wts(dense_wts):
    """
    extracts baseline weights for the final dense
    (weights and bias), according to the tissue
    indexes. The shape changes from (32, 56)
    to (32, 5), where 32 is the dimensions of 
    the previous hidden layer and 56 or 5 are
    the numbers of target cell types.
    """
    wts = np.zeros((32, 5))
    bias = np.zeros((5,))
    for i in range(len(tissue_indexes)):
        outputs = []
        for j in tissue_indexes[i]:
            # get the current vector mapping 32 inputs to 1 tissue type
            wts[:, i] += (dense_wts.weights[0][:,j])
            bias[i] += dense_wts.weights[1][j]
        # average across all the tissues of that cell type
        # (e.g. brain tissues for HMC3)
        wts[:, i] /= len(tissue_indexes[i])
        bias[i] /= len(tissue_indexes[i])        
    return wts, bias


extracted_wts = [
    get_baseline_wts(mt_splice.mtsplice_models[i].layers[-1]) 
    for i in range(len(mt_splice.mtsplice_models))
]

# find the appropriate dense layers and set weights to 
# the extracted weights
for i in TSplice_Model_0.layers:
    if "model" in i.name or "functional" in i.name:
        tsplice_model_logit_0 = i
        i.layers[-1].set_weights(extracted_wts[0])
        break
for i in TSplice_Model_1.layers:
    if "model" in i.name or "functional" in i.name:
        tsplice_model_logit_1 = i
        i.layers[-1].set_weights(extracted_wts[1])
        break
for i in TSplice_Model_2.layers:
    if "model" in i.name or "functional" in i.name:
        tsplice_model_logit_2 = i
        i.layers[-1].set_weights(extracted_wts[2])
        break
for i in TSplice_Model_3.layers:
    if "model" in i.name or "functional" in i.name:
        tsplice_model_logit_3 = i
        i.layers[-1].set_weights(extracted_wts[3])
        break

# define optimizers for tsplice
adam0 = "adam"
adam1 = "adam"
adam2 = "adam"
adam3 = "adam"

### Load data from pickle
## define directories to save / load  models and data

# check for existing directory to prevent overwrite
tsplice_model_dir_load = os.path.join(
    output_models_dir,
    "mtsplice_revised", # TODO remove this
    f"mtsplice_ensemble{mtsplice_model_number}",
)


tsplice_model_dir_save, new_ensemble_model_number = get_new_dir(
    tsplice_model_dir_load
)

if not debug:
    tsplice_model_dir_save = create_directory(tsplice_model_dir_save)
    logging.info(
        f"Initialized new directory "
        f"for saving tsplice models: {tsplice_model_dir_save}"
    )
else:
    logging.info(
        f"[DEBUG] directory for saving "
        f"tsplice models would be: {tsplice_model_dir_save}"
    )
    tsplice_model_dir_save = os.path.join(
        output_models_dir,
        "mtsplice_revised",
        "debug"
    )
    tsplice_model_dir_save = create_directory(tsplice_model_dir_save)


# write dynamically created tsplice_model_dir_save
# or data_dir (depending on whether the script
# will make the data or train the model)
# to a file that the slurm job can find.
if slurm_job_id is not None:
    dir_reference = "tsplice_model_dir_save"
    dir_to_write = tsplice_model_dir_save
    slurm_path = os.path.join(
        PROJECT_ROOT,
        "bash_scripts",
        "run_directories",
        dir_reference,
        f"dir_{slurm_job_id}.txt"
    )
    with open(slurm_path, "w") as f:
        f.write(dir_to_write)


remade_data = False
if load_data:
    logging.info(f"loading tsplice pickle data from {tsplice_data_dir}")
    loaded, tsplice_data = load_splits_dir(tsplice_data_dir)
    if loaded["x_train"]:
        x_train_logit_tsplice = tsplice_data["x_train"]
        x_val_logit_tsplice = tsplice_data["x_val"]
        x_test_logit_tsplice = tsplice_data["x_test"]
        y_train_logit_tsplice = tsplice_data["y_train"]
        y_val_logit_tsplice = tsplice_data["y_val"]
        y_test_logit_tsplice = tsplice_data["y_test"]
        t_metadata_train = tsplice_data["metadata_train"]
        t_metadata_val = tsplice_data["metadata_val"]
        t_metadata_test = tsplice_data["metadata_test"]
        could_find_tsplice_data = True
        train_size = y_train_logit_tsplice.shape[0]
        val_size = y_val_logit_tsplice.shape[0]
        test_size = y_test_logit_tsplice.shape[0]
        full_size = train_size + val_size + test_size
        train_percent = f"({100*train_size/full_size})%"
        val_percent = f"({100*val_size/full_size})%"
        test_percent = f"({100*test_size/full_size})%"
        logging.info(
            f"Loaded TSplice data.\n"
            f"train size: {str(train_size).ljust(6)} {train_percent}\n"
            f"val size:   {str(val_size).ljust(6)}   {val_percent}\n"
            f"test size:  {str(test_size).ljust(6)}  {test_percent}"
        )
    else:
        could_find_tsplice_data = False
        logging.warning("Could not load tsplice data from pickle")
if not load_data or not could_find_tsplice_data:
    # re-generate the data if load_data is false (from opts)
    # or if there was an error in the pickle data path
    tsplice_data, mtsplice_data = remake_data(opts)
    remade_data = True
    x_train_logit_tsplice = tsplice_data["x_train"]
    x_val_logit_tsplice = tsplice_data["x_val"]
    x_test_logit_tsplice = tsplice_data["x_test"]
    y_train_logit_tsplice = tsplice_data["y_train"]
    y_val_logit_tsplice = tsplice_data["y_val"]
    y_test_logit_tsplice = tsplice_data["y_test"]


# test_tsplice_no_nan = np.sum(np.isnan(y_test_logit_tsplice), axis=1) == 0    
# val_tsplice_no_nan = np.sum(np.isnan(y_val_logit_tsplice), axis=1) == 0    
# y_test_logit_tsplice = y_test_logit_tsplice[test_tsplice_no_nan]
# x_test_logit_tsplice = [i[test_tsplice_no_nan] for i in x_test_logit_tsplice]
# y_val_logit_tsplice = y_val_logit_tsplice[val_tsplice_no_nan]
# x_val_logit_tsplice = [i[val_tsplice_no_nan] for i in x_val_logit_tsplice]


if load_data:
    logging.info(f"loading mtsplice pickle data from {mtsplice_data_dir}")
    loaded, mtsplice_data = load_splits_dir(mtsplice_data_dir)
    if loaded["x_train"]:
        x_train_logit_mtsplice = mtsplice_data["x_train"]
        x_val_logit_mtsplice = mtsplice_data["x_val"]
        x_test_logit_mtsplice = mtsplice_data["x_test"]
        y_train_logit_mtsplice = mtsplice_data["y_train"]
        y_val_logit_mtsplice = mtsplice_data["y_val"]
        y_test_logit_mtsplice = mtsplice_data["y_test"]
        x_train_ref_psi = mtsplice_data["psi_train"]
        x_val_ref_psi = mtsplice_data["psi_val"]
        x_test_ref_psi = mtsplice_data["psi_test"]
        metadata_train = mtsplice_data["metadata_train"]
        metadata_val = mtsplice_data["metadata_val"]
        metadata_test = mtsplice_data["metadata_test"]
        could_find_mtsplice_data = True
        train_size = y_train_logit_mtsplice.shape[0]
        val_size = y_val_logit_mtsplice.shape[0]
        test_size = y_test_logit_mtsplice.shape[0]
        full_size = train_size + val_size + test_size
        train_percent = f"({100*train_size/full_size})%"
        val_percent = f"({100*val_size/full_size})%"
        test_percent = f"({100*test_size/full_size})%"
        logging.info(
            f"Loaded TSplice data.\n"
            f"train size: {str(train_size).ljust(6)} {train_percent}\n"
            f"val size:   {str(val_size).ljust(6)}   {val_percent}\n"
            f"test size:  {str(test_size).ljust(6)}  {test_percent}"
        )
    else:
        could_find_mtsplice_data = False
        logging.warning("Could not load mtsplice data from pickle")

# Failed to load data, so remake it as necessary
if not load_data or not could_find_mtsplice_data:
    if not remade_data:
        tsplice_data, mtsplice_data = remake_data(opts)
        remade_data = True
    x_train_logit_mtsplice = mtsplice_data["x_train"]
    x_val_logit_mtsplice = mtsplice_data["x_val"]
    x_test_logit_mtsplice = mtsplice_data["x_test"]
    y_train_logit_mtsplice = mtsplice_data["y_train"]
    y_val_logit_mtsplice = mtsplice_data["y_val"]
    y_test_logit_mtsplice = mtsplice_data["y_test"]


# def filter_nans(y):
    # nans = np.isnan(y)
    # num_nans = np.sum(nans, axis=1)
    # no_nan = num_nans == 0
    # return no_nan

# test_mtsplice_no_nan = filter_nans(y_test_logit_mtsplice)
# val_mtsplice_no_nan = filter_nans(y_val_logit_mtsplice)
# x_val_logit_mtsplice = [
    # i[val_mtsplice_no_nan] 
    # for i in x_val_logit_mtsplice
# ]
# x_test_logit_mtsplice = [
    # i[test_mtsplice_no_nan] 
    # for i in x_test_logit_mtsplice
# ]
# y_val_logit_mtsplice = y_val_logit_mtsplice[val_mtsplice_no_nan]
# y_test_logit_mtsplice = y_test_logit_mtsplice[test_mtsplice_no_nan]


### Load in original TSplice data (ASCOT)
if load_ascot_data:
    logging.info(f"Loading ASCOT data from {ascot_data_dir}")
    got_loaded_ascot, ascot = load_splits_dir(
        ascot_data_dir,
        extension="p"
    )
    if got_loaded_ascot["train_x"]:
        x_train_ascot = ascot["train_x"]
        x_val_ascot = ascot["val_x"]
        x_test_ascot = ascot["test_x"]
        y_train_logit_ascot = ascot["train_y_dlogit"]
        y_val_logit_ascot = ascot["val_y_dlogit"]
        y_test_logit_ascot = ascot["test_y_dlogit"]
        y_train_psi_ascot = ascot["train_y_psi"]
        y_val_psi_ascot = ascot["val_y_psi"]
        y_test_psi_ascot = ascot["test_y_psi"]
        y_train_logit_cell_mean = logit(
            np.nanmean(
                y_train_psi_ascot[0], axis=1
            )
        )
        y_val_logit_cell_mean = logit(
            np.nanmean(
                y_val_psi_ascot[0], axis=1
            )
        )
        y_test_logit_cell_mean = logit(
            np.nanmean(
                y_test_psi_ascot[0], axis=1
            )
        )

        # Reshape into the same shape as the tsplice ESL data
        # (with different number of samples)
        x_train_size = x_train_ascot[0].shape[0]
        x_val_size = x_val_ascot[0].shape[0]
        x_test_size = x_test_ascot[0].shape[0]
        y_train_size = y_train_logit_ascot[0].shape[0]
        y_val_size = y_val_logit_ascot[0].shape[0]
        y_test_size = y_test_logit_ascot[0].shape[0]


        def reshape_ascot(x_ascot, tsplice_size):
            for i in range(len(x_ascot)):
                shape = (tsplice_size, 400, 4)
                if len(x_ascot[i].shape) == 4:
                    x_ascot[i] = np.reshape(x_ascot[i], shape)
            return x_ascot


        def reshape_y(y):
            if isinstance(y, list) and len(y) == 1:
                y = y[0]
            if len(y.shape) == 1:
                y.shape = np.rehspae(y, (y_size, 1))
            return y


        x_train_ascot = reshape_ascot(x_train_ascot, x_train_size)
        x_val_ascot = reshape_ascot(x_val_ascot, x_val_size)
        x_test_ascot = reshape_ascot(x_test_ascot, x_test_size)

        y_train_logit_cell_mean = reshape_y(
            y_train_logit_cell_mean,
            y_train_size
        )
        y_val_logit_cell_mean = reshape_y(
            y_val_logit_cell_mean,
            y_val_size
        )
        y_test_logit_cell_mean = reshape_y(
            y_test_logit_cell_mean,
            y_test_size
        )


        def fold_in_ascot(ascot, tsplice):
            folded = [None, None]
            for i in range(len(folded)):
                folded[i] = np.vstack(
                    [tsplice[i], ascot[i]]
                )
            return folded


        x_train_logit_tsplice = fold_in_ascot(
            x_train_ascot,
            x_train_logit_tsplice
        )
        x_val_logit_tsplice = fold_in_ascot(
            x_val_ascot,
            x_val_logit_tsplice
        )
        x_test_logit_tsplice = fold_in_ascot(
            x_test_ascot,
            x_test_logit_tsplice
        )
        y_train_logit_tsplice = fold_in_ascot(
            y_train_logit_ascot,
            y_train_logit_tsplice
        )
        y_val_logit_tsplice = fold_in_ascot(
            y_val_logit_ascot,
            y_val_logit_tsplice
        )
        y_test_logit_tsplice = fold_in_ascot(
            y_test_logit_ascot,
            y_test_logit_tsplice
        )

    
        train_size = len(x_train_logit_tsplice[0])
        val_size = len(x_val_logit_tsplice[0])
        test_size = len(x_test_logit_tsplice[0])
        total_size = train_size + val_size + test_size
        train_percent = round(100 * train_size / total_size, 3)
        val_percent = round(100 * val_size / total_size, 3)
        test_percent = round(100 * test_size / total_size, 3)
        logging.info(
            f"Loaded ASCOT data.\n"
            f"\n"
            f"train size: {train_size} ({train_percent}%)\n"
            f"val size:   {val_size}   ({val_percent}%)\n"
            f"test size:  {test_size}  ({test_percent}%)\n"
        )

# split sequences and scaling input 
train_seqs_tsplice = [
    x_train_logit_tsplice[0],
    x_train_logit_tsplice[1]
] 
train_logit_mean_cell_tsplice = x_train_logit_tsplice[2]

val_seqs_tsplice = [
    x_val_logit_tsplice[0],
    x_val_logit_tsplice[1]
]
val_logit_mean_cell_tsplice = x_val_logit_tsplice[2]

test_seqs_tsplice = [
    x_test_logit_tsplice[0],
    x_test_logit_tsplice[1]
]
test_logit_mean_cell_tsplice = x_test_logit_tsplice[2]

if tsplice_natural_scale:
    logging.info("using natual scale for training/predictions with Tsplice")
    # assuming data is aways dlogit (logit cell - logit(mean(cell)))
    y_train_logit_tsplice += train_logit_mean_cell_tsplice
    y_val_logit_tsplice   += val_logit_mean_cell_tsplice
    y_test_logit_tsplice  += test_logit_mean_cell_tsplice
    x_train_tsplice_input = [
        train_seqs_tsplice[0],
        train_seqs_tsplice[1], 
        train_logit_mean_cell_tsplice
    ]
    x_val_tsplice_input = [
        val_seqs_tsplice[0],
        val_seqs_tsplice[1],
        val_logit_mean_cell_tsplice
    ]
    x_test_tsplice_input = [
        test_seqs_tsplice[0],
        test_seqs_tsplice[1],
        test_logit_mean_cell_tsplice
    ]
else:
    x_train_tsplice_input = train_seqs_tsplice
    x_val_tsplice_input = val_seqs_tsplice
    x_test_tsplice_input = test_seqs_tsplice


### Model training

def print_cell_weights(wts):
    string = []
    for i, cell in enumerate(cell_types):
        string.append(f"{cell} {wts[i]}")
    return "\n".join(string) 


tsplice_cell_weights_input = parse_cell_type_weights(tsplice_cell_weights)
cell_weights0 = get_cell_type_weights(
    y_train_logit_tsplice,
    tsplice_cell_weights_input[0]
)
cell_weights1 = get_cell_type_weights(
    y_train_logit_tsplice,
    tsplice_cell_weights_input[1]
)
cell_weights2 = get_cell_type_weights(
    y_train_logit_tsplice,
    tsplice_cell_weights_input[2]
)
cell_weights3 = get_cell_type_weights(
    y_train_logit_tsplice,
    tsplice_cell_weights_input[3]
)

logging.info(
    f"TSplice model 0 cell weights: "
    f"{print_cell_weights(cell_weights0)}")
logging.info(
    f"TSplice model 1 cell weights: "
    f"{print_cell_weights(cell_weights1)}"
)
logging.info(
    f"TSplice model 2 cell weights: "
    f"{print_cell_weights(cell_weights2)}"
)
logging.info(
    f"TSplice model 3 cell weights: "
    f"{print_cell_weights(cell_weights3)}"
)

## first add nan masking to tsplice models
# get the cell type weights
train_size = len(x_train_logit_tsplice[0])
val_size = len(x_val_logit_tsplice[0])
# sqrt to cast inside the mse 
# i.e. 
# (ytrue - ypred)**2 * cell_wt 
# = (ytrue * sqrt(cell_wt) - ypred * sqt(cell_wt))**2
cell_type_wts_train0 = tf.sqrt(
    np.ones((train_size, 1))*cell_weights0
)
cell_type_wts_val0 = tf.sqrt(
    np.ones((val_size, 1))*cell_weights0
)
cell_type_wts_train1 = tf.sqrt(
    np.ones((train_size, 1))*cell_weights1
)
cell_type_wts_val1 = tf.sqrt(
    np.ones((val_size, 1))*cell_weights1
)
cell_type_wts_train2 = tf.sqrt(
    np.ones((train_size, 1))*cell_weights2
)
cell_type_wts_val2 = tf.sqrt(
    np.ones((val_size, 1))*cell_weights2
)
cell_type_wts_train3 = tf.sqrt(
    np.ones((train_size, 1))*cell_weights3
)
cell_type_wts_val3 = tf.sqrt(
    np.ones((val_size, 1))*cell_weights3
)

# get the masking matrices
if nan_mask_strategy == "mask_loss":
    (y_mask_train,
    proportion_not_nan_train,
    y_clean_train) = preprocess_nan_masks(y_train_logit_tsplice)

    (y_mask_val,
    proportion_not_nan_val,
    y_clean_val) = preprocess_nan_masks(y_val_logit_tsplice)
elif nan_mask_strategy == "inpute_zero":
    y_train_logit_tsplice_inputed = np.nan_to_num(y_train_logit_tsplice)
    (y_mask_train,
    proportion_not_nan_train,
    y_clean_train) = preprocess_nan_masks(y_train_logit_tsplice_inputed)

    (y_mask_val,
    proportion_not_nan_val,
    y_clean_val)   = preprocess_nan_masks(y_val_logit_tsplice)
else:
    raise NotImplementedError(
        "nan_mask_strategy '{nan_mask_strategy}' not recognized"
    )


def get_weighted_data(y_clean, cell_wts, proportions):
    """
    Helper function to weigh y data according
    to cell weights and nan proportion.
    (efficiently mean and zero inpute)
    """
    wts = tf.cast(
        cell_wts,
        y_clean.dtype
    )
    y_clean = y_clean*cell_wts
    prop = tf.cast(
        proportions,
        y_clean.dtype
    )
    return y_clean*prop


y_clean_train0 = get_weighted_data(
    y_clean_train,
    cell_type_wts_train0,
    proportion_not_nan_train,
)
y_clean_train1 = get_weighted_data(
    y_clean_train,
    cell_type_wts_train1,
    proportion_not_nan_train,
)
y_clean_train2 = get_weighted_data(
    y_clean_train,
    cell_type_wts_train2,
    proportion_not_nan_train,
)
y_clean_train3 = get_weighted_data(
    y_clean_train,
    cell_type_wts_train3,
    proportion_not_nan_train,
)
y_clean_val0 = get_weighted_data(
    y_clean_val,
    cell_type_wts_val0,
    proportion_not_nan_val,
)
y_clean_val1 = get_weighted_data(
    y_clean_val,
    cell_type_wts_val1,
    proportion_not_nan_val,
)
y_clean_val2 = get_weighted_data(
    y_clean_val,
    cell_type_wts_val2,
    proportion_not_nan_val,
)
y_clean_val3 = get_weighted_data(
    y_clean_val,
    cell_type_wts_val3,
    proportion_not_nan_val,
)

                                                               
# finally add the nan mask
TSplice_Model_0_nan_mask = add_nan_mask_layer(
    TSplice_Model_0,
    y_mask_train.shape,
    proportion_not_nan_train.shape,
    cell_type_wts_train0.shape,
)
TSplice_Model_1_nan_mask = add_nan_mask_layer(
    TSplice_Model_1,
    y_mask_train.shape,
    proportion_not_nan_train.shape,
    cell_type_wts_train1.shape
)
TSplice_Model_2_nan_mask = add_nan_mask_layer(
    TSplice_Model_2,
    y_mask_train.shape,
    proportion_not_nan_train.shape,
    cell_type_wts_train2.shape
)
TSplice_Model_3_nan_mask = add_nan_mask_layer(
    TSplice_Model_3,
    y_mask_train.shape,
    proportion_not_nan_train.shape,
    cell_type_wts_train3.shape
) 

if tsplice_loss == "huber":
    loss=tf.keras.losses.Huber(
        delta=0.5,
        reduction="sum_over_batch_size",
        name="huber_loss"
    )
elif tsplice_loss == "mse":
    loss = "mse"

# TODO: add KL-divergence for PSI-scale training (see MTSplice paper)
TSplice_Model_0_nan_mask.compile(loss = loss, optimizer = adam0)
TSplice_Model_1_nan_mask.compile(loss = loss, optimizer = adam1)
TSplice_Model_2_nan_mask.compile(loss = loss, optimizer = adam2)
TSplice_Model_3_nan_mask.compile(loss = loss, optimizer = adam3)

kwargs = {
    "tsplice_base_scale" : "dlogit",
    "mtsplice_base_scale" : "logit",
    "mmsplice_retrained" : "",
    "tsplice_output_scales" : ["dlogit"],
    "mtsplice_output_scales" : ["logit"],
    "debug" : debug,
}
mtsplice_predictor = MTSplicePredictor(
    tsplice_data,
    mtsplice_data,
    **kwargs,
)
tsplice_training_models = [
    TSplice_Model_0,
    TSplice_Model_1,
    TSplice_Model_2,
    TSplice_Model_3
]
mtsplice_predictor.add_tsplice_all_heads(
    tsplice_training_models,
    version_t="baseline",
    version_mm="baseline",
    version_mt="",
)

## Train each TSplice_Model head
skip_train = False
if debug:
    breakpoint()
    " breakpoint for tsplice train. set skip_train=True to skip "

mt_splice_saved = None

logging.info("Getting baseline TSplice predictions for all output heads")
mtsplice_predictor.predict_data()

table = mtsplice_predictor.make_tsplice_table(
    version_t="baseline",
    include_ensemble=False,
)
logging.info("TSplice R^2 Baseline\n" + table)

if not skip_train:
    logging.info("Organizing nan masked data for TSplice training.")
    x_train_logit_tsplice0 = [
        *x_train_tsplice_input,
        y_mask_train,
        proportion_not_nan_train,
        cell_type_wts_train0
    ]
    x_train_logit_tsplice1 = [
        *x_train_tsplice_input,
        y_mask_train,
        proportion_not_nan_train,
        cell_type_wts_train1
    ]
    x_train_logit_tsplice2 = [
        *x_train_tsplice_input,
        y_mask_train,
        proportion_not_nan_train,
        cell_type_wts_train2
    ]
    x_train_logit_tsplice3 = [
        *x_train_tsplice_input,
        y_mask_train,
        proportion_not_nan_train,
        cell_type_wts_train3
    ]
    data_val_logit_tsplice0 = ([
        *x_val_tsplice_input,
        y_mask_val,
        proportion_not_nan_val,
        cell_type_wts_val0
        ],
        [y_clean_val]
    )
    data_val_logit_tsplice1 = ([
        *x_val_tsplice_input,
        y_mask_val,
        proportion_not_nan_val,
        cell_type_wts_val1
        ],
        [y_clean_val]
    )
    data_val_logit_tsplice2 = ([
        *x_val_tsplice_input,
        y_mask_val,
        proportion_not_nan_val,
        cell_type_wts_val2
        ],
        [y_clean_val]
    )
    data_val_logit_tsplice3 = ([
        *x_val_tsplice_input,
        y_mask_val,
        proportion_not_nan_val,
        cell_type_wts_val3
        ],
        [y_clean_val]
    )
if not skip_train and not hyperparameter_tune:
    logging.info(
        "Start training loop for TSplice Models with no hyperparamter tune"
    )
    logging.info(
        f"Number of epochs: {tsplice_n_epochs}\n"
        f"Train size:       {len(x_train_logit_tsplice0[0])}\n"
        f"Val size:         {len(data_val_logit_tsplice0[0][0])}\n"
    )
    history_tsplice0=TSplice_Model_0_nan_mask.fit(
        x = x_train_logit_tsplice0,
        y = [y_clean_train0],
        epochs = tsplice_n_epochs,
        validation_data = data_val_logit_tsplice0
    ) 
    logging.info("Finished training for Tsplice_model0")
    history_tsplice1=TSplice_Model_1_nan_mask.fit(x
         = x_train_logit_tsplice1,
        y = [y_clean_train1],
        epochs = tsplice_n_epochs,
        validation_data = data_val_logit_tsplice1
    )
    logging.info("Finished training for Tsplice_model1")
    history_tsplice2=TSplice_Model_2_nan_mask.fit(
        x = x_train_logit_tsplice2,
        y = [y_clean_train2],
        epochs = tsplice_n_epochs,
        validation_data = data_val_logit_tsplice2
    ) 
    logging.info("Finished training for Tsplice_model2")
    history_tsplice3=TSplice_Model_3_nan_mask.fit(
        x = x_train_logit_tsplice3,
        y = [y_clean_train3],
        epochs = tsplice_n_epochs,
        validation_data = data_val_logit_tsplice3
    )
    logging.info("Finished training for Tsplice_model3")

elif hyperparameter_tune and not skip_train:
    logging.info(
        "Start training loop for TSplice Models "
        "with no hyperparamter tune"
    )
    logging.info(
        f"Max hyperparamater epochs:     {hyperband_max_n_epochs}\n"
        f"Target hyperparameter epochs:  {hyperband_n_epochs}\n"
        f"with patience:                 {hyperband_patience}\n"
        f"\n"
        f"Retrain tune model epochs:     {hyperband_retrain_n_epochs}\n"
        f"with patience:                 {hyperband_retrain_patience}\n"
        f"\n"
        f"Train size:       {len(x_train_logit_tsplice0[0])}\n"
        f"Val size:         {len(data_val_logit_tsplice0[0][0])}\n"
    )
    mt_splice_saved = mmsplice.mtsplice.MTSplice()

    hyperband_dirs = []
    for n in range(4):
        path = os.path.join(
            tsplice_model_dir_save,
            f"tsplice_{n}_hyperband",
        )
        hyperband_dirs.append(path)

    # TSplice ensemble 0
    TSplice_saved_0 = mt_splice_saved.mtsplice_models[0]
    hyperband_func0 = tsplice_hyperband.get_hyperband_tsplice(
        TSplice_Model_0_nan_mask,
        TSplice_saved_0,
        cell_type_layer_wts=extracted_wts[0],
        lr_max=hyperband_lr_max,
        lr_min=hyperband_lr_min
    ) 
    create_directory(hyperband_dirs[0])
    tuner0, best_hps0 = tsplice_hyperband.run_hyperband_tsplice(
        hyperband_func0,
        hyperband_dirs[0],
        x_train=x_train_logit_tsplice0,
        y_train=[y_clean_train0],
        x_val=data_val_logit_tsplice0[0],
        y_val=data_val_logit_tsplice0[1],
        hyperband_epochs=hyperband_n_epochs,
        hyperband_max_epochs=hyperband_max_n_epochs,
        stop_patience=hyperband_patience
    )
    (tsplice_tuned_model0,
    history_tuner0) = tsplice_hyperband.fit_model_after_tuning(
        tuner0,
        x_train=x_train_logit_tsplice0,
        y_train=[y_clean_train0],
        x_val=data_val_logit_tsplice0[0],
        y_val=data_val_logit_tsplice0[1],
        epochs=hyperband_retrain_n_epochs,
        patience=hyperband_retrain_patience
    )
    logging.info("Finished hyperparameter tuning for Tsplice_model0")
    if debug:
        breakpoint()
        "Trained hyperparameter 0" 

    # TSplice ensemble 1
    TSplice_saved_1 = mt_splice_saved.mtsplice_models[1]
    hyperband_func1 = tsplice_hyperband.get_hyperband_tsplice(
        TSplice_Model_1_nan_mask,
        TSplice_saved_1,
        cell_type_layer_wts=extracted_wts[1],
        lr_max=hyperband_lr_max,
        lr_min=hyperband_lr_min,
    )
    create_directory(hyperband_dirs[1])
    tuner1, best_hps1 = tsplice_hyperband.run_hyperband_tsplice(
        hyperband_func1,
        hyperband_dirs[1],
        x_train=x_train_logit_tsplice1,
        y_train=[y_clean_train1],
        x_val=data_val_logit_tsplice1[0],
        y_val=data_val_logit_tsplice1[1],
        hyperband_epochs=hyperband_n_epochs,
        hyperband_max_epochs=hyperband_max_n_epochs,
        stop_patience=hyperband_patience
    )
    (tsplice_tuned_model1,
    history_tuner1) = tsplice_hyperband.fit_model_after_tuning(
        tuner1,
        x_train=x_train_logit_tsplice1,
        y_train=[y_clean_train1],
        x_val=data_val_logit_tsplice1[0],
        y_val=data_val_logit_tsplice1[1],
        epochs=hyperband_retrain_n_epochs,
        patience=hyperband_retrain_patience
    )
    logging.info("Finished hyperparameter tuning for Tsplice_model1")

    # TSplice ensemble 2
    TSplice_saved_2 = mt_splice_saved.mtsplice_models[2]
    hyperband_func2 = tsplice_hyperband.get_hyperband_tsplice(
        TSplice_Model_2_nan_mask,
        TSplice_saved_2,
        cell_type_layer_wts=extracted_wts[2],
        lr_max=hyperband_lr_max,
        lr_min=hyperband_lr_min,
    ) 
    create_directory(hyperband_dirs[2])
    tuner2, best_hps2 = tsplice_hyperband.run_hyperband_tsplice(
        hyperband_func2,
        hyperband_dirs[2],
        x_train=x_train_logit_tsplice2,
        y_train=[y_clean_train2],
        x_val=data_val_logit_tsplice2[0],
        y_val=data_val_logit_tsplice2[1],
        hyperband_epochs=hyperband_n_epochs,
        hyperband_max_epochs=hyperband_max_n_epochs,
        stop_patience=hyperband_patience,
    )
    (tsplice_tuned_model2,
    history_tuner2) = tsplice_hyperband.fit_model_after_tuning(
        tuner2,
        x_train=x_train_logit_tsplice2,
        y_train=[y_clean_train2],
        x_val=data_val_logit_tsplice2[0],
        y_val=data_val_logit_tsplice2[1],
        epochs=hyperband_retrain_n_epochs,
        patience=hyperband_retrain_patience,
    )
    logging.info("Finished hyperparameter tuning for Tsplice_model2")

    # TSplice ensemble 3
    TSplice_saved_3 = mt_splice_saved.mtsplice_models[3]
    hyperband_func3 = tsplice_hyperband.get_hyperband_tsplice(
        TSplice_Model_3_nan_mask,
        TSplice_saved_3,
        cell_type_layer_wts=extracted_wts[3],
        lr_max=hyperband_lr_max,
        lr_min=hyperband_lr_min,
    ) 
    create_directory(hyperband_dirs[3])
    tuner3, best_hps3 = tsplice_hyperband.run_hyperband_tsplice(
        hyperband_func3,
        hyperband_dirs[3],
        x_train=x_train_logit_tsplice3,
        y_train=[y_clean_train3],
        x_val=data_val_logit_tsplice3[0],
        y_val=data_val_logit_tsplice3[1],
        hyperband_epochs=hyperband_n_epochs,
        hyperband_max_epochs=hyperband_max_n_epochs,
        stop_patience=hyperband_patience,
    )
    (tsplice_tuned_model3,
    history_tuner3) = tsplice_hyperband.fit_model_after_tuning(
        tuner3,
        x_train=x_train_logit_tsplice3,
        y_train=[y_clean_train3],
        x_val=data_val_logit_tsplice3[0],
        y_val=data_val_logit_tsplice3[1],
        epochs=hyperband_retrain_n_epochs,
        patience=hyperband_retrain_patience
    )
    logging.info("Finished hyperparameter tuning for Tsplice_model3")

    # Set final weights to the tuned hyperparameter model weights
    # Should be done automatically by hyperband, but just in case...
    TSplice_Model_0.get_layer(
        "tsplice_delta_logit_model"
    ).set_weights(
        tsplice_tuned_model0.get_layer(
            "tsplice_delta_logit_model"
        ).weights
    )
    TSplice_Model_1.get_layer(
        "tsplice_delta_logit_model"
    ).set_weights(
        tsplice_tuned_model1.get_layer(
            "tsplice_delta_logit_model"
        ).weights
    )
    TSplice_Model_2.get_layer(
        "tsplice_delta_logit_model"
    ).set_weights(
        tsplice_tuned_model2.get_layer(
            "tsplice_delta_logit_model"
        ).weights
    )
    TSplice_Model_3.get_layer(
        "tsplice_delta_logit_model"
    ).set_weights(
        tsplice_tuned_model3.get_layer(
            "tsplice_delta_logit_model"
        ).weights
    )


### TSplice Model Evaluation
if debug:
    breakpoint()
    "Calculating TSplice R values, can load weights by setting test_loaded_weights to true"

if test_loaded_weights:
    if not debug:
        raise ValueError("not allowing test loaded weigths for non-debug")
    for i in range(4):
        path = os.path.join(
            tsplice_model_dir_load,
            f"TSplice_Ensemble_{i}_delta_logit_PSI.hdf5"
        )
        tsplice_load_paths.append(path)
    TSplice_Model_0.load_weights(tsplice_model_dir_load[0])
    TSplice_Model_1.load_weights(tsplice_model_dir_load[1])
    TSplice_Model_2.load_weights(tsplice_model_dir_load[2])
    TSplice_Model_3.load_weights(tsplice_model_dir_load[3])


logging.info("Calculating retrained TSplice predictions for all output heads")
tsplice_training_models = [
    TSplice_Model_0,
    TSplice_Model_1,
    TSplice_Model_2,
    TSplice_Model_3
]
mtsplice_predictor.add_tsplice_all_heads(
    tsplice_training_models,
    version_t="retrained",
    version_mm="baseline",
    version_mt="",
)
mtsplice_predictor.predict_data()
# TODO: change tables to work with mtsplice_predictor
# tables = model_evaluation.output_tsplice_r2_table(tsplice_scaled_predictions)
# logging.info("TSplice R^2 Predictions\n"+"\n\n".join(tables))

table = mtsplice_predictor.make_tsplice_table(
    version_t="retrained",
    include_ensemble=False,
)
logging.info("TSplice R^2 Retrained\n" + table)

### Save trained model
save_tsplice = True
if debug:
    save_tsplice = False
    breakpoint()
    " Breakpoint for saving tsplice model. set save_tsplice = True to save "
if save_tsplice:
    paths = []
    for i in range(4):
        path = os.path.join(
            tsplice_model_dir_save,
            f"TSplice_Ensemble_{i}_delta_logit_PSI.hdf5"
        )
        assert not os.path.exists(path)
        paths.append(path)
    save_model(TSplice_Model_0, paths[0])
    save_model(TSplice_Model_1, paths[1])
    save_model(TSplice_Model_2, paths[2])
    save_model(TSplice_Model_3, paths[3])
    logging.info(f"Saved TSplice_Model_0 to {path[0]}")
    logging.info(f"Saved TSplice_Model_1 to {path[1]}")
    logging.info(f"Saved TSplice_Model_2 to {path[2]}")
    logging.info(f"Saved TSplice_Model_3 to {path[3]}")


### Combined model
trainable = not freeze_mmsplice_layers
mm_splice_saved = get_saved_mmsplice(trainable=False)
mm_splice_baseline = get_saved_mmsplice(trainable=False)
if mt_splice_saved is None:
    mt_splice_saved = mmsplice.mtsplice.MTSplice()
try: # tensorflow 2.17
    tsplice_model_logit_0.name = "tsplice_delta_logit_model_0"
    tsplice_model_logit_1.name = "tsplice_delta_logit_model_1"
    tsplice_model_logit_2.name = "tsplice_delta_logit_model_2"
    tsplice_model_logit_3.name = "tsplice_delta_logit_model_3"
except AttributeError:
    # tensorflow 2.4
    tsplice_model_logit_0._name = "tsplice_delta_logit_model_0"
    tsplice_model_logit_1._name = "tsplice_delta_logit_model_1"
    tsplice_model_logit_2._name = "tsplice_delta_logit_model_2"
    tsplice_model_logit_3._name = "tsplice_delta_logit_model_3"
TSplice_Model_0_saved = get_single_model_custom_loss(
    model_indx=0,
    scale=tsplice_output_cell_scale,
    mt_splice=mt_splice_saved
)
TSplice_Model_1_saved = get_single_model_custom_loss(
    model_indx=1,
    scale=tsplice_output_cell_scale,
    mt_splice=mt_splice_saved
)
TSplice_Model_2_saved = get_single_model_custom_loss(
    model_indx=2,
    scale=tsplice_output_cell_scale,
    mt_splice=mt_splice_saved
)
TSplice_Model_3_saved = get_single_model_custom_loss(
    model_indx=3,
    scale=tsplice_output_cell_scale,
    mt_splice=mt_splice_saved
)

# reload the best hyperparameters from the hyperparameter run 
# and get the weights from the retraining
# the hyperparameters (L1/L2 weights, dropout and lr) probably don't affect the 
# inference (or training if layers are frozen), but it's better to be safe
mmsplice_retrained_path = os.path.join(
    output_models_dir,
    "mmsplice_revised",
    f"mmsplice{mmsplice_model_number}"
)
hyperband_path = os.path.join(
    mmsplice_retrained_path,
    "hyperparam"
)
hyperband_mmsplice_func = get_hyperband_mmsplice(
    mm_splice_saved,
    mm_splice_saved
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


min_l1 = 0.0
max_l1 = 0.03
min_l2 = 0.0
max_l2 = 0.03
min_dropout = 0.1
max_dropout = 0.4
min_lr = 1e-7
max_lr = 1e-5
loss = "mse"
merged_MMSplice_full = get_saved_mmsplice(trainable=trainable)


def freeze_batchnorm_func(model=merged_MMSplice_full):
    """
    check all layers in the MMSplice model
    and freezes all batchnorm layers.
    """
    for layer in model.layers:
        if isinstance(layer, Functional):
            for layer_ in layer.layers:
                if isinstance(layer_, tf.keras.layers.BatchNormalization):
                    layer_.trainable = False
    return


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


mmsplice_hyperband_dummy = os.path.join(
    "mmsplice_revised_debug",
    "hyperparam"
)
tuner = kerastuner.Hyperband(
    hyperband_mmsplice,
    objective="val_loss",
    max_epochs=50,
    factor=3,
    directory=mmsplice_hyperband_dummy
)
tuner._get_checkpoint_fname = load_weights_wrapper(tuner)
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
mm_splice_model = tuner.hypermodel.build(best_hps)
mmsplice_file = (
    f"MMSplice_delta_logit_PSI_model_"
    f"retrained_revised{mmsplice_model_number}.hdf5"
)
mmsplice_retrained_weights = os.path.join(
    mmsplice_retrained_path, mmsplice_file
)

### Define TSplice Ensemble
inL = Input(shape = (400,4),  name = "mt_splice_inL")
inR = Input(shape = (400,4),  name = "mt_splice_inR")
inMean = Input(shape=(1,), name = "mt_splice_in_cell_mean")
tsplice_ensemble_input = [inL, inR]
# functional layer of the TSplice_Model that
# produces the cell delta logits, unscaled.
# defined when setting the baseline dense layer
out0 = tsplice_model_logit_0(tsplice_ensemble_input)
out1 = tsplice_model_logit_1(tsplice_ensemble_input)
out2 = tsplice_model_logit_2(tsplice_ensemble_input)
out3 = tsplice_model_logit_3(tsplice_ensemble_input)
while isinstance(out0, list):
    out0 = out0[0]
    out1 = out1[0]
    out2 = out2[0]
    out3 = out3[0]

out_final_layer = tf.keras.layers.Lambda(
    lambda x : (x[0] + x[1] + x[2] + x[3]) / 4.0,
    name = "ensemble_mean"
)
out_f = out_final_layer.call([out0, out1, out2, out3])
TSplice_Model_ensemble = Model(
    inputs = tsplice_ensemble_input,
    outputs = [out_f]
)
# Repeat for baseline model
out_b0 = TSplice_Model_0_saved([inL, inR])
out_b1 = TSplice_Model_1_saved([inL, inR])
out_b2 = TSplice_Model_2_saved([inL, inR])
out_b3 = TSplice_Model_3_saved([inL, inR])
while isinstance(out_b0, list):
    out_b0 = out_b0[0]
    out_b1 = out_b1[0]
    out_b2 = out_b2[0]
    out_b3 = out_b3[0]
base_final_out = out_final_layer([
    out_b0,
    out_b1,
    out_b2,
    out_b3,
])

TSplice_Model_logit_saved = Model(
    inputs = [inL, inR],
    outputs = [base_final_out]
)

for i in TSplice_Model_0_saved.layers:
    if "model" in i.name or "functional" in i.name:
        i.layers[-1].set_weights(extracted_wts[0])
        break
        
for i in TSplice_Model_1_saved.layers:
    if "model" in i.name or "functional" in i.name:
        i.layers[-1].set_weights(extracted_wts[1])
        break
        
for i in TSplice_Model_2_saved.layers:
    if "model" in i.name or "functional" in i.name:
        i.layers[-1].set_weights(extracted_wts[2])
        break
        
for i in TSplice_Model_3_saved.layers:
    if "model" in i.name or "functional" in i.name:
        i.layers[-1].set_weights(extracted_wts[3])
        break

mtsplice_predictor.add_model(
    TSplice_Model_logit_saved, 
    model_name="tsplice",
    version_t="baseline",
    version_mm="baseline",
    version_mt="",
    head="ensemble",
)
mtsplice_predictor.add_model(
    TSplice_Model_ensemble, 
    model_name="tsplice",
    version_t="retrained",
    version_mm="baseline",
    version_mt="",
    head="ensemble",
)

# mtsplice = combine_models(model, TSplice_Model_logit)
mtsplice = combine_models_scale(
    mm_splice_model,
    TSplice_Model_ensemble
)
mtsplice_predictor.add_model(
    mtsplice, 
    model_name="mtsplice",
    version_t="retrained",
    version_mm="baseline",
    version_mt="",
    head="",
)
mtsplice_baseline = combine_models_scale(
    mm_splice_baseline,
    TSplice_Model_logit_saved
)
mtsplice_predictor.add_model(
    mtsplice, 
    model_name="mtsplice",
    version_t="baseline",
    version_mm="baseline",
    version_mt="",
    head="",
)
logging.info(
    "Getting TSplice and MTSplice predictions "
    "(baseline, retrained TSplice only, "
    " and retrained TSplice + MMSplice)"
)
mtsplice_predictor.predict_data()

table1 = mtsplice_predictor.make_tsplice_table(
    version_t="baseline",
    include_ensemble=True,
)

table2 = mtsplice_predictor.make_tsplice_table(
    version_t="retrained",
    include_ensemble=True,
)

logging.info("TSplice R^2 Baseline\n" + table1 + "\n\nRetrained\n" + table2)


try:
    mm_splice_model.load_weights(mmsplice_retrained_weights)
except ValueError:
    mm_splice_model.load_weights(
        mmsplice_retrained_weights.replace("hdf5", "keras")
    )

mtsplice_predictor.add_model(
    mtsplice, 
    model_name="mtsplice",
    version_t="retrained",
    version_mm="retrained",
    version_mt="",
    head="",
)
mtsplice_predictor.predict_data()


### Freeze MMsplice layers
# TODO: check this worked
for l in mtsplice.layers:
    if "model" in l.name and len(l.inputs) > 2: # i.e. is the mmsplice model
        for j in l.layers:
            j.trainable = False

### Getting nan masked mtsplice data
# define cell type weights for training

    
mtsplice_cell_weight_method = parse_cell_type_weights(mtsplice_cell_weights)

cell_weights_mt = get_cell_type_weights(
    y_train_logit_mtsplice,
    mtsplice_cell_weight_method
)

logging.info(
    f"MTSplice mtsplice cell weigths "
    f"{print_cell_weights(cell_weights_mt)}"
)

## first add nan masking to tsplice models
# get the cell type weights
train_size = len(x_train_logit_mtsplice[0])
val_size = len(x_val_logit_mtsplice[0])
# sqrt to cast inside the mse 
# i.e. 
# (ytrue - ypred)**2 * cell_wt 
# = (ytrue * sqrt(cell_wt) - ypred * sqt(cell_wt))**2


cell_type_wts_train_mt = tf.sqrt(
    np.ones((train_size, 1))*cell_weights_mt
)
cell_type_wts_val_mt = tf.sqrt(
    np.ones((val_size, 1))*cell_weights_mt
)

# get the masking matrices
if nan_mask_strategy == "mask_loss":
    (y_mask_train_mt,
    proportion_not_nan_train_mt,
    y_clean_train_mt) = preprocess_nan_masks(y_train_logit_tsplice)
    (y_mask_val_mt,
    proportion_not_nan_val_mt,
    y_clean_val_mt) = preprocess_nan_masks(y_val_logit_tsplice)
elif nan_mask_strategy == "inpute_zero":
    y_train_logit_mtsplice_inputed = np.nan_to_num(y_train_logit_mtsplice)
    (y_mask_train_mt,
    proportion_not_nan_train_mt,
    y_clean_train_mt) = preprocess_nan_masks(
        y_train_logit_mtsplice_inputed
    )
    (y_mask_val_mt,
    proportion_not_nan_val_mt,
    y_clean_val_mt) = preprocess_nan_masks(y_val_logit_mtsplice)
else:
    raise NotImplementedError(
        "nan_mask_strategy '{nan_mask_strategy}' not recognized"
    )
y_clean_train_mt = get_weighted_data(
    y_clean_train_mt,
    cell_type_wts_train_mt,
    proportion_not_nan_train_mt,
)
y_clean_val_mt = get_weighted_data(
    y_clean_val_mt,
    cell_type_wts_val_mt,
    proportion_not_nan_val_mt,
)


# skip reference input to tsplice since it cancels out in the 
# mtsplice predictions
x_train_logit_mtsplice = (
    x_train_logit_mtsplice[0:10]
    + x_train_logit_mtsplice[12:14] 
    + x_train_logit_mtsplice[14:15]
)
x_val_logit_mtsplice = (
    x_val_logit_mtsplice[0:10]
    + x_val_logit_mtsplice[12:14]
    + x_val_logit_mtsplice[14:15]
)
x_test_logit_mtsplice = (
    x_test_logit_mtsplice[0:10] 
    + x_test_logit_mtsplice[12:14] 
    + x_test_logit_mtsplice[14:15]
)

# finally add the nan mask
mtsplice_nan_mask = add_nan_mask_layer(
    mtsplice,
    y_mask_train_mt.shape,
    proportion_not_nan_train_mt.shape,
    cell_type_wts_train_mt.shape
)


### Retraining mtsplice model
adam = Adam(
    learning_rate=mtsplice_learning_rate,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-05
)  # +- 100x scan for lr, decrease lr
mtsplice_nan_mask.compile(loss=loss, optimizer=adam)
min_delta_se = 1e-5
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=mtsplice_patience,
        min_delta=min_delta_se,
        restore_best_weights=True,
    )
]

skip_train = False
test_loaded_weights = False
if debug:
    breakpoint()
    "breakpoint for training loop of mtsplice model. set skip_train = True to skip. set test_loaded_weights = True to load mtsplice weights"
else:
    logging.info("Fitting mtsplice model")


x_train_logit_mtsplice_masked = (
    x_train_logit_mtsplice 
    + [y_mask_train_mt, proportion_not_nan_train_mt, cell_type_wts_train_mt]
)
x_val_logit_mtsplice_masked  = (
    x_val_logit_mtsplice
    + [y_mask_val_mt, proportion_not_nan_val_mt, cell_type_wts_val_mt]
)

if not skip_train:
    if mtsplice_early_stop:
        history_c = mtsplice_nan_mask.fit(
            x_train_logit_mtsplice_masked,
            y_clean_train_mt,
            validation_data=(x_val_logit_mtsplice_masked, y_clean_val_mt),
            batch_size=mtsplice_batch_size,
            epochs=mtsplice_n_epochs,
            verbose=1,
            callbacks=callbacks
        )
        if debug:
            breakpoint()
            " test early stopping "
    else:
        history_c = mtsplice_nan_mask.fit(
            x_train_logit_mtsplice_masked,
            y_clean_train_mt,
            validation_data=(x_val_logit_mtsplice_masked, y_clean_val_mt),
            batch_size=mtsplice_batch_size,
            epochs=mtsplice_n_epochs,
            verbose=1
        )

if test_loaded_weights:
    directories = [tsplice_model_dir_save, output_models_dir]
    mtsplice_name = (
        f"MTSplice_5_cell_type_"
        f"mtsplice_model.hdf5"
    )
    for d in directories:
        weights_path = os.path.join(d, mtsplice_name)
        if os.path.exists(weights_path):
            mtsplice.load_weights(weights_path)
            logging.info(f"Loaded weights from {weights_path}")
            break

logging.info("Getting retrained mtsplice predictions")
mtsplice_predictor.add_model(
    mtsplice, 
    model_name="mtsplice",
    version_t="retrained",
    version_mm="retrained",
    version_mt="retrained",
    head="",
)
mtsplice_predictor.predict_data()

logging.info(
    "Getting retrained TSplice ensemble and "
    "individual model predictions after training mtsplice"
)
mtsplice_predictor.add_tsplice_all_heads(
    tsplice_training_models,
    version_t="retrained",
    version_mm="retrained",
    version_mt="retrained",
)
mtsplice_predictor.add_model(
    TSplice_Model_ensemble, 
    model_name="tsplice",
    version_t="retrained",
    version_mm="retrained",
    version_mt="retrained",
    head="ensemble",
)
mtsplice_predictor.predict_data()
table_b = mtsplice_predictor.make_mtsplice_table(
    version_t="baseline",
    version_mm="baseline",
    version_mt="",
)
table_rt = mtsplice_predictor.make_mtsplice_table(
    version_t="retrained",
    version_mm="retrained",
    version_mt="retrained",
)
logging.info("MTSplice R^2 Baseline\n" + table_b)
logging.info("MTSplice R^2 Baseline\n" + table_rt)
if debug:
    breakpoint()
    " check MTSplice table "

# logging.info("TSplice R^2 Predictions\n"+"\n\n".join(tables))

### Save MTSplice Model
mtsplice_model_path = os.path.join(
    tsplice_model_dir_save,
    f"MTSplice_5_cell_type_mtsplice_model{mtsplice_model_number}.hdf5"
)
# prevent overwrite
mtsplice_model_path, new_mtsplice_model_number = get_new_dir(
    mtsplice_model_path,
    suffix = ".hdf5"
)
try:
    save_mtsplice = True
    if debug:
        save_mtsplice = False
        breakpoint()
        "breakpoint to save mtsplice model. set save_mtsplice = True to save"
    if save_mtsplice:
        logging.info("saving mtsplice (MTSplice) model")
    # reloading model seems to help it save
    mtsplice = combine_models_scale(mm_splice_model, TSplice_Model_ensemble) 

    # should never happen but just in case
    if os.path.exists(mtsplice_model_path): 
        logging.warning("model name already exists. saving using date")
        now = datetime.datetime.now()
        date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")
        path = (f"model_{date_time_string.replace(" ", "_")}")
    else:
        path = mtsplice_model_path
    if save_mtsplice:
        mtsplice.save(path)
        mtsplice_is_saved = True
    else:
        mtsplice_is_saved = False
except OSError:
    logging.warning("Model layer names are probably not unique")
    mtsplice_is_saved = False

logging.info(f"saved mtsplice model? {mtsplice_is_saved}")
if mtsplice_is_saved:
    logging.info(f"saved mtsplice model to {path}")


### Plotting 
# Parameters for plotting model results
pd.set_option("display.max_colwidth",100)
sns.set(style="ticks", color_codes=True)
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.labelpad"] = 5
plt.rcParams["axes.linewidth"]= 1.5
plt.rcParams["xtick.labelsize"]= 14
plt.rcParams["ytick.labelsize"]= 14
plt.rcParams["xtick.major.size"] = 6
plt.rcParams["ytick.major.size"] = 6
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["xtick.minor.width"] = 1
plt.rcParams["ytick.minor.width"] = 1
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["axes.edgecolor"] = "black"
if debug:
    output_figures_dir = os.path.join(
        PROJECT_ROOT,
        "plots_test"
    )
    breakpoint()
    "Plotting code"
elif ("--output_figures_dir" in opts
     and opts["--output_figures_dir"]): # check for non-empty path
    output_figures_dir = opts["--output_figures_dir"]
else:
    output_figures_dir = os.path.join(tsplice_model_dir_save, "figures")
    output_figures_dir = create_directory(output_figures_dir)
create_directory(output_figures_dir)
logging.info(f"output figures dir: {output_figures_dir}")

## Bar plots
# Removed

## Scatter plot
# Retrained
for scale in output_plot_scale:
    plotting.scatterplot(
        mtsplice_predictor, 
        datasets=["train", "val", "test"],
        scale=scale,
        output_figures_dir=output_figures_dir,
        model="mtsplice",
        version_t="retrained",
        version_mm="retrained",
        version_mt="retrained",
    )
# Baseline
for scale in output_plot_scale:
    plotting.scatterplot(
        mtsplice_predictor, 
        datasets=["train", "val", "test"],
        scale=scale,
        output_figures_dir=output_figures_dir,
        model="mtsplice",
        version_t="baseline",
        version_mm="baseline",
        version_mt="",
    )

## TSplice plots
# Baseline
for scale in output_plot_scale:
    plotting.scatterplot(
        mtsplice_predictor, 
        datasets=["train", "val", "test"],
        scale=scale,
        output_figures_dir=output_figures_dir,
        model="tsplice",
        head="ensemble",
        version_t="baseline",
        version_mm="baseline",
        version_mt="",
    )
# Retrained
for scale in output_plot_scale:
    plotting.scatterplot(
        mtsplice_predictor, 
        datasets=["train", "val", "test"],
        scale=scale,
        output_figures_dir=output_figures_dir,
        model="tsplice",
        head="ensemble",
        version_t="retrained",
        version_mm="retrained",
        version_mt="retrained",
    )

if debug:
    breakpoint()
    " check train val test plots "

## Aggregate data plots
## Scatter plot
# Retrained
for scale in output_plot_scale:
    plotting.scatterplot(
        mtsplice_predictor, 
        scale=scale,
        datasets=["full"],
        output_figures_dir=output_figures_dir,
        model="mtsplice",
        version_t="retrained",
        version_mm="retrained",
        version_mt="retrained",
    )
# Baseline
for scale in output_plot_scale:
    plotting.scatterplot(
        mtsplice_predictor, 
        scale=scale,
        datasets=["full"],
        output_figures_dir=output_figures_dir,
        model="mtsplice",
        version_t="baseline",
        version_mm="baseline",
        version_mt="",
    )

## TSplice plots
# Baseline
for scale in output_plot_scale:
    plotting.scatterplot(
        mtsplice_predictor, 
        scale=scale,
        datasets=["full"],
        output_figures_dir=output_figures_dir,
        model="tsplice",
        head="ensemble",
        version_t="baseline",
        version_mm="baseline",
        version_mt="",
    )
# Retrained
for scale in output_plot_scale:
    plotting.scatterplot(
        mtsplice_predictor, 
        scale=scale,
        datasets=["full"],
        output_figures_dir=output_figures_dir,
        model="tsplice",
        head="ensemble",
        version_t="retrained",
        version_mm="retrained",
        version_mt="retrained",
    )

if debug:
    breakpoint()
    " check full plots "

output_predictions_path = create_directory(
    os.path.join(tsplice_model_dir_save, "predictions")
)

## Save predictions used for plotting
save_model_predictions.save_predictions_as_csv(
    mtsplice_predictor,
    metadata=tsplice_data, # use entire data since it has metadata keys
    model="tsplice",
    output_dir=output_predictions_path,
)
save_model_predictions.save_predictions_as_csv(
    mtsplice_predictor,
    metadata=mtsplice_data,
    model="mtsplice",
    output_dir=output_predictions_path,
)

if debug:
    breakpoint()
    " final breakpoint "

### END
logging.info("done")
