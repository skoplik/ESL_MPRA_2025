import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import numpy as np
from src.utils.check_version import check_version
import sys
import tensorflow as tf


if check_version(tf.__version__, '2.17') == "less":
    logging.info("Tensorflow version < 2.17")
    from tensorflow.keras.layers import Dense, Input, BatchNormalization, Conv1D
    from tensorflow.keras.models import Model
    def save_model(model, path):
        model.save(path)
        return
        
else:
    logging.info("Tensorflow version >= 2.17")
    from keras.src.layers import Dense, Input, BatchNormalization, Conv1D
    from keras.src.models import Model
    from keras.src.saving.saving_api import save_model


def load_mtsplice(MMSplice_MTSplice_path):
    try:
        import mmsplice
    except ModuleNotFoundError:
        sys.path.append(MMSplice_MTSplice_path)
        import mmsplice
    mt_splice = mmsplice.mtsplice.MTSplice()
    return mt_splice.mtsplice_models


# Loading
def get_single_model_custom_loss(model_indx=0, scale = "psi", mt_splice = None, scale_output = False, MMSplice_MTSplice_path=""):
    
    """
    Loads models from the MTSplice Github,
    and redifines them for the specified scale (psi or logit)
    and extracts one of the output heads of the ensemble
    using model_indx
    """
    if mt_splice is None:
        mt_splice = load_mtsplice(MMSplice_MTSplice_path)
    in1 = Input(shape=(400, 4))
    in2 = Input(shape=(400, 4))
    in_mean = Input(shape=(1,))

    model = mt_splice[model_indx]
    dense = model.layers[-1]
    assert "dense" in dense.name
    new_dense = Dense(5, name = "cell_type_dense")
    new_dense_out = new_dense(dense.input)
    new_model = Model(inputs = model.inputs, outputs = [new_dense_out], name="tsplice_delta_logit_model")
    final_out_delta_logit = new_model([in1, in2])
    
    if scale == "psi":
        final_out_cell_logit = final_out_delta_logit + in_mean
        final_out_cell_psi = Expit(name="expit")(final_out_cell_logit)
        final_model = Model(inputs = [in1,in2,in_mean], outputs = [final_out_cell_psi])
    elif scale == "logit":
        if isinstance(final_out_delta_logit, list):
            final_out_delta_logit = final_out_delta_logit[0]
        # final_out_delta_logit_scaled = final_out_delta_logit + in_mean
        if scale_output:
            final_out_delta_logit_scaled = final_out_delta_logit + in_mean
            final_model = Model(inputs=[in1, in2, in_mean],
                                outputs=[final_out_delta_logit_scaled])
        else:
            final_model = Model(inputs=[in1, in2], outputs=[final_out_delta_logit])
        
    return final_model


def get_tissue_index():
    TISSUES = [
        'Retina - Eye', 'RPE/Choroid/Sclera - Eye', 'Subcutaneous - Adipose',
        'Visceral (Omentum) - Adipose', 'Adrenal Gland', 'Aorta - Artery',
        'Coronary - Artery', 'Tibial - Artery', 'Bladder', 'Amygdala - Brain',
        'Anterior cingulate - Brain', 'Caudate nucleus - Brain',
        'Cerebellar Hemisphere - Brain', 'Cerebellum - Brain', 'Cortex - Brain',
        'Frontal Cortex - Brain', 'Hippocampus - Brain', 'Hypothalamus - Brain',
        'Nucleus accumbens - Brain', 'Putamen - Brain',
        'Spinal cord (C1) - Brain', 'Substantia nigra - Brain',
        'Mammary Tissue - Breast', 'EBV-xform lymphocytes - Cells',
        'Leukemia (CML) - Cells', 'Xform. fibroblasts - Cells',
        'Ectocervix - Cervix', 'Endocervix - Cervix', 'Sigmoid - Colon',
        'Transverse - Colon', 'Gastroesoph. Junc. - Esophagus',
        'Mucosa - Esophagus', 'Muscularis - Esophagus', 'Fallopian Tube',
        'Atrial Appendage - Heart', 'Left Ventricle - Heart', 'Cortex - Kidney',
        'Liver', 'Lung', 'Minor Salivary Gland', 'Skeletal - Muscle',
        'Tibial - Nerve', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate',
        'Not Sun Exposed - Skin', 'Sun Exposed (Lower leg) - Skin',
        'Ileum - Small Intestine', 'Spleen', 'Stomach', 'Testis', 'Thyroid',
        'Uterus', 'Vagina', 'Whole Blood'
    ]

    HEK_tissues = ["Cortex - Kidney"]
    HeLa_tissues = ["Uterus", 'Ectocervix - Cervix', 'Endocervix - Cervix']
    K562_tissues = ["Leukemia (CML) - Cells"]
    MCF7_tissues = ["Mammary Tissue - Breast"]
    HMC3_tissues = [tissue for tissue in TISSUES if "Brain" in tissue] # average over all brain tissues

    # Get the weigths by aggregating across the relevant cell types
    # for the 5 cancer cell lines
    tissue_index_map = {}


    def tissue_(keys):
        return [tissue_index_map[i] for i in keys]


    for i in range(len(TISSUES)):
        tissue_index_map[TISSUES[i]] = i
    
    tissue_index = [tissue_(i) for i in [HEK_tissues, HeLa_tissues, K562_tissues, MCF7_tissues, HMC3_tissues]]
    return tissue_index


def get_baseline_wts(dense_wts):
    """
    extracts baseline weights for the final dense
    (weights and bias), according to the tissue
    indexes. The shape changes from (32, 56)
    to (32, 5), where 32 is the dimensions of 
    the previous hidden layer and 56 or 5 are
    the numbers of target cell types.
    """
    tissue_index = get_tissue_index()
    wts = np.zeros((32, 5))
    bias = np.zeros((5,))
    for i in range(len(tissue_index)):
        outputs = []
        for j in tissue_index[i]:
            # get the current vector mapping 32 inputs to 1 tissue type
            wts[:, i] += (dense_wts.weights[0][:,j])
            bias[i] += dense_wts.weights[1][j]
        # average across all the tissues of that cell type
        # (e.g. brain tissues for HMC3)
        wts[:, i] /= len(tissue_index[i])
        bias[i] /= len(tissue_index[i])        
    return wts, bias


def set_baseline_weights(tsplice_models, extracted_weights):
    successfully_set_weight = [False] * len(tsplice_models)
    for i, model in enumerate(tsplice_models):
        for j, layer in enumerate(model.layers):
            if "model" in layer.name or "functional" in layer.name:
                if successfully_set_weight[i]:
                    raise ValueError("Attempted to set the baseline model's last dense weight multiple times")
                # if there is an inner functional
                layer.layers[-1].set_weights(extracted_weights[i])
                successfully_set_weight[i] = True
            elif j == len(model.layers) - 1 and "dense" in layer.name:
                # in case the unprocessed model is passed in
                if successfully_set_weight[i]:
                    raise ValueError("Attempted to set the baseline model's last dense weight multiple times")
                layer.set_weights(extracted_weights[i])
                successfully_set_weight[i] = True
    if sum(successfully_set_weight) != len(tsplice_models):
        breakpoint()
        raise ValueError("Tried to set the baseline weights but could not locate the correct last dense layer")
    return


def rename_layers(tsplice_models):
    for i in range(len(tsplice_models)):
        try:
            tsplice_models[i].name = f"tsplice_delta_logit_model_{i}"
        except AttributeError:
            tsplice_models[i]._name = f"tsplice_delta_logit_model_{i}"
    return


# Combining into ensemble
def build_tsplice_ensemble(tsplice_models):
    
    inL = Input(shape = (400,4),  name = "mt_splice_inL")
    inR = Input(shape = (400,4),  name = "mt_splice_inR")
    inMean = Input(shape=(1,), name = "mt_splice_in_cell_mean")
    tsplice_ensemble_input = [inL, inR]
    rename_layers(tsplice_models)

    # functional layer of the TSplice_Model that
    # produces the cell delta logits, unscaled.
    # defined when setting the baseline dense layer
    ensemble_predictions = [model(tsplice_ensemble_input) for model in tsplice_models]

    while isinstance(ensemble_predictions[0], list):
        for i in range(len(ensemble_predictions)):
            ensemble_predictions[i] = ensemble_predictions[i][0]

    out_final_layer = tf.keras.layers.Lambda(lambda x : (x[0] + x[1] + x[2] + x[3]) / 4.0, name = "ensemble_mean")

    out_f = out_final_layer.call(ensemble_predictions)
    TSplice_Model_ensemble = Model(inputs = tsplice_ensemble_input, outputs = [out_f])
    return TSplice_Model_ensemble



