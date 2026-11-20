"""
Rebuild MMSplice in tensorflow2 framework (i.e. no numpy for top layer and alt+ref module handling)
"""

import os
import tensorflow as tf
import keras_tuner as kerastuner
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    concatenate,
    Subtract,
    Dense,
    Layer
)
from tensorflow.keras import Model, Input
from mmsplice_retraining.mmsplice_model.layers import ConvDNA  # soft linked to MMSplice's layers.py
from mmsplice_retraining.mmsplice_model.global_average_pooling1d_mask0 import GlobalAveragePooling1D_Mask0
from src.utils.check_version import check_version
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


def convert_tf_bool_float(x, suffix=None):
    if suffix is None:
        return tf.where(x, 1.0, 0.0)
    else:
        return tf.where(x, 1.0, 0.0, name="where_"+suffix)


"""
Based on related functions in MMSplice.utils which depends on np.isclose.
Because comparing to 0, (rtol * abs(b)) where b = 0 and therefore the same as 0.
"""
def and_not_close0(x, y, return_bool=False, suffix=None):
    if suffix is None:
        logic_result = tf.math.logical_and(
            x != 0,
            y != 0
        )
    else:
        logic_result = tf.math.logical_and(
            x != 0,
            y != 0,
            name="logical_and_"+suffix
        )
    if return_bool:
        return logic_result
    return convert_tf_bool_float(logic_result, suffix=suffix)


def logit_tensor(x, clip_threshold=0.00001, suffix=None):
    if suffix is None:
        x = tf.clip_by_value(
            x,
            clip_value_min=clip_threshold,
            clip_value_max=1-clip_threshold
        )
        return tf.math.log(x) - tf.math.log(1 - x)
    else:
        x = tf.clip_by_value(
            x,
            clip_value_min=clip_threshold,
            clip_value_max=1-clip_threshold,
            name="clip_by_value_"+suffix
        )
        return (
            tf.math.log(x, name="log_p_"+suffix)
            - tf.math.log(1 - x, name="log_1mp_"+suffix)
        )


class LogitTensorLayer(tf.keras.layers.Layer):
    def __init__(self, clip_threshold=0.00001, suffix=None, **kwargs):
        super(LogitTensorLayer, self).__init__(**kwargs)
        self.clip_threshold = clip_threshold
        self.suffix = suffix
    def call(self, x):
        x = tf.clip_by_value(
            x,
            clip_value_min=self.clip_threshold,
            clip_value_max=1-self.clip_threshold
        )
        return tf.math.log(x) - tf.math.log(1 - x)


class CheckOverlap(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CheckOverlap, self).__init__(**kwargs)
        return
    def call(self, x):
        return 


class Gather(tf.keras.layers.Layer):
    def __init__(self, index, **kwargs):
        super(Gather, self).__init__(**kwargs)
        self.index = index
        return 
    def call(self, x):
        return tf.gather(x, [self.index], axis=1)


class ExonOverlap(tf.keras.layers.Layer):
    def __init__(
            self,
            donor_suffix="exon_overlap_donor",
            acceptor_suffix="exon_overlap_acceptor",
            **kwargs
        ):
        super(ExonOverlap, self).__init__(**kwargs)
        self.donor_suffix = donor_suffix
        self.acceptor_suffix = acceptor_suffix
        return
    def call(
            self,
            acceptor_diff_output,
            exon_diff_output,
            donor_diff_output
        ):
        result=tf.math.logical_or(
            and_not_close0(
                acceptor_diff_output,
                exon_diff_output,
                return_bool=True,
                suffix=self.donor_suffix
            ),
            and_not_close0(
                exon_diff_output,
                donor_diff_output,
                return_bool=True,
                suffix=self.acceptor_suffix
            )
        )
        return result


class IntronOverlap(tf.keras.layers.Layer):
    def __init__(self, suffix, **kwargs):
        super(IntronOverlap, self).__init__(**kwargs)
        self.suffix = suffix
        return
    def call(self, a, b):
        return and_not_close0(a, b, suffix = self.suffix)


class BoolToFloat(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BoolToFloat, self).__init__(**kwargs)
        return
    def call(self,x):
        return convert_tf_bool_float(x)


def rename_pretrained_model2_17(model, suffix, custom_objects = None):
    """
    can't access layers with config in 
    tensorflow 2.17, this is a workaround 
    for cloning a model and renaming the layers
    """
    new_model = tf.keras.models.clone_model(model)
    new_model.name = model.name + suffix
    for layer in new_model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        old_layer = model.get_layer(layer.name)
        layer.set_weights(old_layer.weights)
        layer.name = old_layer.name + suffix
    return new_model


"""
Based on this: https://gist.github.com/nrasadi/60d40447caf5ad1e1b0785c7e1b0234a#file-tfkeras_model_add_prefix-py
"""
def rename_pretrained_model2_4(model, suffix, custom_objects=None):
    config = model.get_config()
    old_to_new = {}
    new_to_old = {}
    
    for layer in config['layers']:
        new_name = layer['name'] + suffix
        old_to_new[layer['name']], new_to_old[new_name] = new_name, layer['name']
        layer['name'] = new_name
        layer['config']['name'] = new_name

        if len(layer['inbound_nodes']) > 0:
            for in_node in layer['inbound_nodes'][0]:
                try:
                    in_node[0] = old_to_new[in_node[0]]
                except KeyError:
                    breakpoint()
    
    for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]
    
    for output_layer in config['output_layers']:
        output_layer[0] = old_to_new[output_layer[0]]
    
    config['name'] = config['name'] + suffix
    new_model = tf.keras.Model().from_config(config, custom_objects)
    
    for layer in new_model.layers:
        layer.set_weights(model.get_layer(new_to_old[layer.name]).get_weights())
    
    return new_model


if check_version(tf.__version__, "2.17") == "less":
    TF_VERSION = "less"
    rename_pretrained_model = rename_pretrained_model2_4
else:
    TF_VERSION = "greater"
    rename_pretrained_model = rename_pretrained_model2_17



def MMSplice_build_subtree(suffix=None, trainable_modules=False, original_MMSplice_models_dir = ""):
    if suffix is None:
        suffix = ""
    custom_objects = {
        'ConvDNA': ConvDNA
    }
    exon_custom_objects = {
        "GlobalAveragePooling1D_Mask0": GlobalAveragePooling1D_Mask0,
        'ConvDNA': ConvDNA
    }
    acceptor_intronM = os.path.join(
        original_MMSplice_models_dir,
        "Intron3.h5"
    )
    acceptorM = os.path.join(
        original_MMSplice_models_dir,
        "Acceptor.h5"
    )
    exonM = os.path.join(
        original_MMSplice_models_dir, 
        "Exon.h5"
    )
    exon3pM = os.path.join(
        original_MMSplice_models_dir,
        "Exon_prime3.h5"
    )
    donorM = os.path.join(
        original_MMSplice_models_dir,
        "Donor.h5"
    )
    donor_intronM = os.path.join(
        original_MMSplice_models_dir,
        "Intron5.h5"
    )
    acceptor_intron_model = load_model(
        acceptor_intronM,
        compile=False,
        custom_objects=custom_objects
    )
    acceptor_model = load_model(
        acceptorM,
        compile=False,
        custom_objects=custom_objects
    )
    exon_model = load_model(
        exonM,
        compile=False,
        custom_objects=exon_custom_objects
    )
    donor_model = load_model(
        donorM,
        compile=False,
        custom_objects=custom_objects
    )
    donor_intron_model = load_model(
        donor_intronM,
        compile=False,
        custom_objects=custom_objects
    )

    # rebuild with renamed internal names
    acceptor_intron_model = rename_pretrained_model(
        acceptor_intron_model,
        "_acceptor_intron_" + suffix,
        custom_objects=custom_objects
    )
    acceptor_model = rename_pretrained_model(
        acceptor_model,
        "_acceptor_" + suffix,
        custom_objects=custom_objects
    )
    exon_model = rename_pretrained_model(
        exon_model,
        "_exon_" + suffix,
        custom_objects=exon_custom_objects
    )
    donor_model = rename_pretrained_model(
        donor_model,
        "_donor_" + suffix,
        custom_objects=custom_objects
    )
    donor_intron_model = rename_pretrained_model(
        donor_intron_model,
        "_donor_intron_" + suffix,
        custom_objects=custom_objects
    )
    
    if not trainable_modules:
        acceptor_intron_model.trainable = False
        acceptor_model.trainable = False
        exon_model.trainable = False
        donor_model.trainable = False
        donor_intron_model.trainable = False

    return (
        acceptor_intron_model,
        acceptor_model,
        exon_model,
        donor_model,
        donor_intron_model
    )


def get_saved_model(saved_model_path = None, trainable = True, clip_threshold=0.00001):
    if saved_model_path is None:
        saved_model_path = os.path.join(
            "/storage",
            "yu",
            "Projects",
            "P000--ESL_continuation",
            "GitHub",
            "MMSplice_MTSplice",
            "mmsplice",
            "models/"
        )

    (acceptor_intron_model_ref,
    acceptor_model_ref,
    exon_model_ref,
    donor_model_ref,
    donor_intron_model_ref) = MMSplice_build_subtree(
        "ref",
        trainable_modules=trainable,
        original_MMSplice_models_dir=saved_model_path
    )

    reference_seqs = [
        Input(shape=(None, 4), name="input_intron3p_ref"),
        Input(shape=(53, 4), name="input_acceptor_ref"),
        Input(shape=(None, 4), name="input_exon_ref"),
        Input(shape=(18, 4), name="input_donor_ref"),
        Input(shape=(None, 4), name="input_intron5p_ref")
    ]
    alternative_seqs = [
        Input(shape=(None, 4), name="input_intron3p_alt"),
        Input(shape=(53, 4), name="input_acceptor_alt"),
        Input(shape=(None, 4), name="input_exon_alt"),
        Input(shape=(18, 4), name="input_donor_alt"),
        Input(shape=(None, 4), name="input_intron5p_alt")
    ]
    print(
        "reference_seqs+alternative_seqs = ",
        reference_seqs+alternative_seqs
    )

    suffix = "ref"
    combined_ref = concatenate(
        [
            acceptor_intron_model_ref(reference_seqs[0]),
            LogitTensorLayer(
                suffix="acceptor_" + suffix,
                name="logit_acceptor_ref",
                clip_threshold=clip_threshold,
            )(acceptor_model_ref(reference_seqs[1])),
            exon_model_ref(reference_seqs[2]),
            LogitTensorLayer(
                suffix="acceptor_" + suffix,
                name="logit_donor_ref",
                clip_threshold=clip_threshold,
            )(donor_model_ref(reference_seqs[3])),
            donor_intron_model_ref(reference_seqs[4])
        ],
        name="concatenate_modules_" + suffix
    )


    suffix = "alt"
    combined_alt = concatenate(
        [
            acceptor_intron_model_ref(alternative_seqs[0]),
            LogitTensorLayer(
                suffix="acceptor_" + suffix,
                name="logit_acceptor_alt",
                clip_threshold=clip_threshold,
            )(acceptor_model_ref(alternative_seqs[1])),
            exon_model_ref(alternative_seqs[2]),
            LogitTensorLayer(
                suffix="acceptor_" + suffix,
                name="logit_donor_alt",
                clip_threshold=clip_threshold,
            )(donor_model_ref(alternative_seqs[3])),
            donor_intron_model_ref(alternative_seqs[4])
        ],
        name="concatenate_modules_" + suffix
    )

    subtract_alt_ref_modules = Subtract(
        name="subtract_alt_ref_modules"
    )([combined_alt, combined_ref])

    print(
        "subtract_alt_ref_modules = ",
        subtract_alt_ref_modules
    )
    print(
        "subtract_alt_ref_modules[:, 2] = ",
        subtract_alt_ref_modules[:, 2]
    )

    # add interaction terms
    acceptor_intron_diff_output = Gather(
        index=0,
        name="gather_acceptor_intron_diff"
    )(subtract_alt_ref_modules)
    acceptor_diff_output = Gather(
        index=1,
        name="gather_acceptor_diff"
    )(subtract_alt_ref_modules)
    exon_diff_output = Gather(
        index=2,
        name="gather_exon_diff"
    )(subtract_alt_ref_modules)
    donor_diff_output = Gather(
        index=3,
        name="gather_donor_diff"
    )(subtract_alt_ref_modules)
    donor_intron_diff_output = Gather(
        index=4,
        name="gather_donor_intron_diff"
    )(subtract_alt_ref_modules)
    exon_overlapper = ExonOverlap(
        donor_suffix="exon_overlap_donor",
        acceptor_suffix="exon_overlap_acceptor",
        name="exon_overlap_logical_or"
    )
    exon_overlap = exon_overlapper(
        acceptor_diff_output=acceptor_diff_output,
        exon_diff_output=exon_diff_output,
        donor_diff_output=donor_diff_output
    )
    donor_intron_overlap = IntronOverlap(
        suffix="donor_ovelap_acceptor",
        name="donor_intron_overlap"
    )(donor_diff_output, donor_intron_diff_output)
    acceptor_intron_overlap_indicator = IntronOverlap(
        suffix="acceptor_overlap_acceptor",
        name="acceptor_intron_overlap"
    )(acceptor_intron_diff_output, acceptor_diff_output)
    exon_overlap_float = BoolToFloat(
        name="bool_to_float_exon"
    )(exon_overlap)

    exon_overlap = tf.keras.layers.Multiply(
        name = "exon_overlap"
    )([exon_diff_output, exon_overlap_float])
    donor_intron_overlap = tf.keras.layers.Multiply(
        name = "donor_intron_multiply"
    )([donor_intron_diff_output, donor_intron_overlap])
    acceptor_intron_overlap = tf.keras.layers.Multiply(
        name = "acceptor_intron_multiply"
    )([acceptor_intron_diff_output, acceptor_intron_overlap_indicator])

    combined_full = concatenate(
        [
            subtract_alt_ref_modules,
            exon_overlap,
            donor_intron_overlap,
            acceptor_intron_overlap
        ],
        name="concatenate_modules_overlaps"
    )
    combined_full = Dense(
        1,
        activation="linear",
        name="mmsplice_dense"
    )(combined_full)

    merged_MMSplice_full = Model(
        inputs=reference_seqs+alternative_seqs,
        outputs=[combined_full]
    )


    """
    Set weights based on MMSplice's linear layer
    Technically transposed matrix, but is equivalent.
    """

    linear_coeff_prior = np.array(
        [
            [0.49685773],
            [0.72322957],
            [1.54760024],
            [0.75011527],
            [2.26187717],
            [-0.69419094],
            [2.40138709],
            [0.88148553]
        ]
    )
    linear_intercept_prior = np.array([0.0006480262366686865])

    merged_MMSplice_full.layers[-1].set_weights(
        [linear_coeff_prior, linear_intercept_prior]
    )
    return merged_MMSplice_full


def get_hyperband_mmsplice(
        mm_splice_saved,
        merged_MMSplice_full,
        min_l1=0.0,
        max_l1=0.03,
        min_l2=0.0,
        max_l2=0.03,
        min_dropout=0.1,
        max_dropout=0.4,
        min_lr=1e-7,
        max_lr=1e-5,
        loss="mse",
    ):
    def hyperband_mmsplice(hp):
        # Reload the pre-trained model for fine-tuning with 
        # new hyperparameters
        for i in merged_MMSplice_full.layers:
            if isinstance(i, Functional):
                for layer in i.layers:
                    if len(layer.weights) > 0:
                        layer.set_weights(
                            mm_splice_saved
                            .get_layer(i.name)
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
            elif (len(i.weights) > 0):
                i.set_weights(
                    mm_splice_saved
                    .get_layer(i.name)
                    .get_weights()
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
            step=((max_lr-min_lr)/10)
        )
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
    return hyperband_mmsplice


