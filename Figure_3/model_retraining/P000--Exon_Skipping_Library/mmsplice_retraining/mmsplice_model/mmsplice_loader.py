"""
Reimplementing MMSplice.layer's GlobalAveragePooling1D_Mask0 class.
This is for saving the full model, and their version doesn't have get_config().
https://stackoverflow.com/questions/58678836/notimplementederror-layers-with-arguments-in-init-must-override-get-conf
"""
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from .layers import normalize_data_format  # soft linked to MMSplice's layers.py


class GlobalAveragePooling1D_Mask0(Layer):
    """
    Global average pooling operation for temporal data.
    Masking out 0-padded input.
    """

    
    def __init__(self, data_format='channels_last', **kwargs):
        super(GlobalAveragePooling1D_Mask0, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)

        
    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1])
        else:
            return (input_shape[0], input_shape[2])

        
    def call(self, inputs):
        inputs, model_inputs = inputs
        steps_axis = 1 if self.data_format == 'channels_last' else 2
        mask = K.max(model_inputs, axis=2, keepdims=True)
        inputs *= mask
        return K.sum(inputs, axis=steps_axis) / K.maximum(
            K.sum(mask, axis=steps_axis), K.epsilon())
    
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'data_format': self.data_format
        })
        return config


"""
Rebuild MMSplice in tensorflow2 framework (i.e. no numpy for top layer and alt+ref module handling)
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import concatenate, Subtract, Dense, Layer
from tensorflow.keras import Model, Input
from .layers import ConvDNA  # soft linked to MMSplice's layers.py


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
        logic_result = tf.math.logical_and(x != 0, y != 0)
    else:
        logic_result = tf.math.logical_and(x != 0, y != 0, name="logical_and_"+suffix)
    if return_bool:
        return logic_result
    return convert_tf_bool_float(logic_result, suffix=suffix)


def logit_tensor(x, clip_threshold=0.00001, suffix=None):
    if suffix is None:
        x = tf.clip_by_value(x, clip_value_min=clip_threshold, clip_value_max=1-clip_threshold)
        return tf.math.log(x) - tf.math.log(1 - x)
    else:
        x = tf.clip_by_value(x, clip_value_min=clip_threshold, clip_value_max=1-clip_threshold, name="clip_by_value_"+suffix)
        return tf.math.log(x, name="log_p_"+suffix) - tf.math.log(1 - x, name="log_1mp_"+suffix)

    
"""
Based on this: https://gist.github.com/nrasadi/60d40447caf5ad1e1b0785c7e1b0234a#file-tfkeras_model_add_prefix-py
"""
def rename_pretrained_model(model, suffix, custom_objects=None):
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
                in_node[0] = old_to_new[in_node[0]]
    
    for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]
    
    for output_layer in config['output_layers']:
        output_layer[0] = old_to_new[output_layer[0]]
    
    config['name'] = config['name'] + suffix
    new_model = tf.keras.Model().from_config(config, custom_objects)
    
    for layer in new_model.layers:
        layer.set_weights(model.get_layer(new_to_old[layer.name]).get_weights())
    
    return new_model


def MMSplice_build_subtree(suffix=None, trainable_modules=False, original_MMSplice_models_dir = ""):
    if suffix is None:
        suffix = ""
    
    custom_objects = {
        'ConvDNA': ConvDNA
    }

    acceptor_intronM = original_MMSplice_models_dir + "Intron3.h5"
    acceptorM = original_MMSplice_models_dir + "Acceptor.h5"
    exonM = original_MMSplice_models_dir + "Exon.h5"
    exon3pM = original_MMSplice_models_dir + "Exon_prime3.h5"
    donorM = original_MMSplice_models_dir + "Donor.h5"
    donor_intronM = original_MMSplice_models_dir + "Intron5.h5"

    acceptor_intron_model = load_model(acceptor_intronM, compile=False, custom_objects=custom_objects)
    acceptor_model = load_model(acceptorM, compile=False, custom_objects=custom_objects)
    exon_model = load_model(exonM, compile=False, custom_objects={
                "GlobalAveragePooling1D_Mask0": GlobalAveragePooling1D_Mask0,
                'ConvDNA': ConvDNA
            })
    donor_model = load_model(donorM, compile=False, custom_objects=custom_objects)
    donor_intron_model = load_model(donor_intronM, compile=False, custom_objects=custom_objects)

    # rebuild with renamed internal names
    acceptor_intron_model = rename_pretrained_model(acceptor_intron_model, "_acceptor_intron_" + suffix, custom_objects=custom_objects)
    acceptor_model = rename_pretrained_model(acceptor_model, "_acceptor_" + suffix, custom_objects=custom_objects)
    exon_model = rename_pretrained_model(exon_model, "_exon_" + suffix, custom_objects={
                "GlobalAveragePooling1D_Mask0": GlobalAveragePooling1D_Mask0,
                'ConvDNA': ConvDNA
            })
    donor_model = rename_pretrained_model(donor_model, "_donor_" + suffix, custom_objects=custom_objects)
    donor_intron_model = rename_pretrained_model(donor_intron_model, "_donor_intron_" + suffix, custom_objects=custom_objects)
    
    if not trainable_modules:
        acceptor_intron_model.trainable = False
        acceptor_model.trainable = False
        exon_model.trainable = False
        donor_model.trainable = False
        donor_intron_model.trainable = False

    return acceptor_intron_model, acceptor_model, exon_model, donor_model, donor_intron_model


def get_saved_model():
    saved_model_path = "/storage/yu/Projects/P000--ESL_continuation/GitHub/MMSplice_MTSplice/mmsplice/models/"

    acceptor_intron_model_ref, acceptor_model_ref, exon_model_ref, donor_model_ref, donor_intron_model_ref = \
        MMSplice_build_subtree("ref", trainable_modules=True, original_MMSplice_models_dir=saved_model_path)

    reference_seqs = [Input(shape=(None, 4), name="input_intron3p_ref"), \
                      Input(shape=(53, 4), name="input_acceptor_ref"), \
                      Input(shape=(None, 4), name="input_exon_ref"), \
                      Input(shape=(18, 4), name="input_donor_ref"), \
                      Input(shape=(None, 4), name="input_intron5p_ref")]
    alternative_seqs = [Input(shape=(None, 4), name="input_intron3p_alt"), \
                      Input(shape=(53, 4), name="input_acceptor_alt"), \
                      Input(shape=(None, 4), name="input_exon_alt"), \
                      Input(shape=(18, 4), name="input_donor_alt"), \
                      Input(shape=(None, 4), name="input_intron5p_alt")]
    print("reference_seqs+alternative_seqs = ", reference_seqs+alternative_seqs)

    suffix = "ref"
    combined_ref = concatenate([acceptor_intron_model_ref(reference_seqs[0]), \
                                logit_tensor(acceptor_model_ref(reference_seqs[1]), suffix="acceptor_"+suffix), \
                                     exon_model_ref(reference_seqs[2]), \
                                logit_tensor(donor_model_ref(reference_seqs[3]), suffix="acceptor_"+suffix), \
                                     donor_intron_model_ref(reference_seqs[4])], name="concatenate_modules_" + suffix)
    suffix = "alt"
    combined_alt = concatenate([acceptor_intron_model_ref(alternative_seqs[0]), \
                                logit_tensor(acceptor_model_ref(alternative_seqs[1]), suffix="acceptor_"+suffix), \
                                     exon_model_ref(alternative_seqs[2]), \
                                logit_tensor(donor_model_ref(alternative_seqs[3]), suffix="acceptor_"+suffix), \
                                     donor_intron_model_ref(alternative_seqs[4])], name="concatenate_modules_" + suffix)

    subtract_alt_ref_modules = Subtract(name = "subtract_alt_ref_modules")( \
        [combined_alt, combined_ref])
    print("subtract_alt_ref_modules = ", subtract_alt_ref_modules)
    print("subtract_alt_ref_modules[:, 2] = ", subtract_alt_ref_modules[:, 2])
    print("tf.gather(subtract_alt_ref_modules, [2], axis=1) = ", tf.gather(subtract_alt_ref_modules, [2], axis=1))

    # add interaction terms
    acceptor_intron_diff_output = tf.gather(subtract_alt_ref_modules, [0], axis=1, name="gather_acceptor_intron_diff")
    acceptor_diff_output = tf.gather(subtract_alt_ref_modules, [1], axis=1, name="gather_acceptor_diff")
    exon_diff_output = tf.gather(subtract_alt_ref_modules, [2], axis=1, name="gather_exon_diff")
    donor_diff_output = tf.gather(subtract_alt_ref_modules, [3], axis=1, name="gather_donor_diff")
    donor_intron_diff_output = tf.gather(subtract_alt_ref_modules, [4], axis=1, name="gather_donor_intron_diff")

    exon_overlap = tf.math.logical_or(and_not_close0(acceptor_diff_output, exon_diff_output, return_bool=True, suffix="exon_overlap_acceptor"), \
                                      and_not_close0(exon_diff_output, donor_diff_output, return_bool=True, suffix="exon_overlap_donor"), \
                                     name="exon_overlap_logical_or")
    donor_intron_overlap = and_not_close0(donor_diff_output, donor_intron_diff_output, suffix="donor_overlap_acceptor")
    acceptor_intron_overlap_indicator = and_not_close0(acceptor_intron_diff_output, acceptor_diff_output, suffix="acceptor_overlap_acceptor")

    exon_overlap = tf.keras.layers.Multiply(name = "exon_overlap")([exon_diff_output, convert_tf_bool_float(exon_overlap)])
    donor_intron_overlap = tf.keras.layers.Multiply(name = "donor_intron_overlap")([donor_intron_diff_output, donor_intron_overlap])
    acceptor_intron_overlap = tf.keras.layers.Multiply(name = "acceptor_intron_overlap")([acceptor_intron_diff_output, acceptor_intron_overlap_indicator])

    combined_full = concatenate([subtract_alt_ref_modules, exon_overlap, donor_intron_overlap, acceptor_intron_overlap], name="concatenate_modules_overlaps")
    combined_full = Dense(1, activation="linear")(combined_full)

    merged_MMSplice_full = Model(inputs=reference_seqs+alternative_seqs, \
                            outputs=[combined_full])

    """
    Set weights based on MMSplice's linear layer
    Technically transposed matrix, but is equivalent.
    """

    linear_coeff_prior = np.array([[0.49685773], [0.72322957], [1.54760024],
                                  [0.75011527], [2.26187717],  [-0.69419094],
                                  [2.40138709],  [0.88148553]])
    linear_intercept_prior = np.array([0.0006480262366686865])
    merged_MMSplice_full.layers[-1].set_weights([linear_coeff_prior, linear_intercept_prior])
    return merged_MMSplice_full
