import tensorflow as tf


"""

logit(phi_e,t^alt) = b_0 + b_tissue + b_alt + b_alt_tissue + eps

b_0 = logit(phi_e,mean^ref)

b_tissue = TSplice(s_ref, tissue)

b_alt = MMSplice(s_ref, s_alt)

b_alt_tissue = TSplice(s_alt, tissue) - TSplice(s_ref tissue)

"""


def combine_models2(mm_splice, t_splice, n_cell_types = 5, scale_by_mean = False):
    s_ref_mm = [tf.keras.layers.Input(shape=(None, 4), name="input_intron3p_ref_mmsplice"), \
                      tf.keras.layers.Input(shape=(53, 4), name="input_acceptor_ref_mmsplice"), \
                      tf.keras.layers.Input(shape=(None, 4), name="input_exon_ref_mmsplice"), \
                      tf.keras.layers.Input(shape=(18, 4), name="input_donor_ref_mmsplice"), \
                      tf.keras.layers.Input(shape=(None, 4), name="input_intron5p_ref_mmsplice")]
    s_var_mm = [tf.keras.layers.Input(shape=(None, 4), name="input_intron3p_alt_mmsplice"), \
                      tf.keras.layers.Input(shape=(53, 4), name="input_acceptor_alt_mmsplice"), \
                      tf.keras.layers.Input(shape=(None, 4), name="input_exon_alt_mmsplice"), \
                      tf.keras.layers.Input(shape=(18, 4), name="input_donor_alt_mmsplice"), \
                      tf.keras.layers.Input(shape=(None, 4), name="input_intron5p_alt_mmsplice")]
    s_ref_mt = [tf.keras.layers.Input(shape=(400, 4), name = "seqL_ref_tsplice"), tf.keras.layers.Input(shape=(400, 4), name = "seqR_ref_tsplice")]
    s_var_mt = [tf.keras.layers.Input(shape=(400, 4), name = "seqL_var_tsplice"), tf.keras.layers.Input(shape=(400, 4), name = "seqR_var_tsplice")]
    mean_ref_logit = tf.keras.layers.Input(shape=(1,), name = "mean_ref_logit")
    # tissue_effect = t_splice(s_ref_mt)
    variant_effect = mm_splice.call(s_ref_mm + s_var_mm)
    variant_tissue_interaction = t_splice.call(s_var_mt) #- t_splice(s_ref_mt) canceling out
    variant_effect_tiled = tf.keras.layers.Lambda(
        lambda x: tf.tile(x, [1, n_cell_types]),
        output_shape = (n_cell_types,),
        name="variant_effect_tiled"
    )(variant_effect)
    if scale_by_mean:
        final = tf.keras.layers.Add()(
            [mean_ref_logit, variant_effect_tiled, variant_tissue_interaction]
        )
        return tf.keras.models.Model(inputs = s_ref_mm + s_var_mm + s_ref_mt + s_var_mt + [mean_ref_logit]
                                 , outputs = [final], name="MTSplice_Combined")
    else:
        final = tf.keras.layers.Add()(
            [variant_effect_tiled, variant_tissue_interaction]
        )
        return tf.keras.models.Model(inputs = s_ref_mm + s_var_mm + s_ref_mt + s_var_mt
                                 , outputs = [final], name="MTSplice_Combined")
        

def combine_models_no_scale(mm_splice, t_splice, n_cell_types = 5):
    s_ref_mm = [tf.keras.layers.Input(shape=(None, 4), name="input_intron3p_ref_mmsplice"), \
                      tf.keras.layers.Input(shape=(53, 4), name="input_acceptor_ref_mmsplice"), \
                      tf.keras.layers.Input(shape=(None, 4), name="input_exon_ref_mmsplice"), \
                      tf.keras.layers.Input(shape=(18, 4), name="input_donor_ref_mmsplice"), \
                      tf.keras.layers.Input(shape=(None, 4), name="input_intron5p_ref_mmsplice")]
    s_var_mm = [tf.keras.layers.Input(shape=(None, 4), name="input_intron3p_alt_mmsplice"), \
                      tf.keras.layers.Input(shape=(53, 4), name="input_acceptor_alt_mmsplice"), \
                      tf.keras.layers.Input(shape=(None, 4), name="input_exon_alt_mmsplice"), \
                      tf.keras.layers.Input(shape=(18, 4), name="input_donor_alt_mmsplice"), \
                      tf.keras.layers.Input(shape=(None, 4), name="input_intron5p_alt_mmsplice")]
    mean_ref_logit = tf.keras.layers.Input(shape=(1,), name = "mean_ref_logit")
    s_ref_mt = [tf.keras.layers.Input(shape=(400, 4), name = "seqL_ref_tsplice"), tf.keras.layers.Input(shape=(400, 4), name = "seqR_ref_tsplice"), mean_ref_logit]
    s_var_mt = [tf.keras.layers.Input(shape=(400, 4), name = "seqL_var_tsplice"), tf.keras.layers.Input(shape=(400, 4), name = "seqR_var_tsplice"), mean_ref_logit]
    # tissue_effect = t_splice(s_ref_mt)
    variant_effect = mm_splice.call(s_ref_mm + s_var_mm)
    variant_tissue_interaction = t_splice.call(s_var_mt) #- t_splice(s_ref_mt) canceling out
    variant_effect_tiled = tf.keras.layers.Lambda(
        lambda x: tf.tile(x, [1, n_cell_types]),
        output_shape = (n_cell_types,),
        name="variant_effect_tiled"
    )(variant_effect)
    final = tf.keras.layers.Add()(
        [variant_effect_tiled, variant_tissue_interaction]
    )
    return tf.keras.models.Model(inputs = s_ref_mm + s_var_mm + s_var_mt, outputs = [final])


def combine_models_scale(mm_splice, t_splice, n_cell_types=5):
    s_ref_mm = [
        tf.keras.layers.Input(
            shape=(None, 4), name="input_intron3p_ref_mmsplice"
        ),
        tf.keras.layers.Input(
            shape=(53, 4),
            name="input_acceptor_ref_mmsplice"
        ),
        tf.keras.layers.Input(
            shape=(None, 4),
            name="input_exon_ref_mmsplice"
        ),
        tf.keras.layers.Input(
            shape=(18, 4),
            name="input_donor_ref_mmsplice"
        ),
        tf.keras.layers.Input(
            shape=(None, 4),
            name="input_intron5p_ref_mmsplice"
        )
    ]
    s_var_mm = [
        tf.keras.layers.Input(
            shape=(None, 4),
            name="input_intron3p_alt_mmsplice"
        ),
        tf.keras.layers.Input(
            shape=(53, 4),
            name="input_acceptor_alt_mmsplice"
        ),
        tf.keras.layers.Input(
            shape=(None, 4),
            name="input_exon_alt_mmsplice"
        ),
        tf.keras.layers.Input(
            shape=(18, 4),
            name="input_donor_alt_mmsplice"
        ),
        tf.keras.layers.Input(
            shape=(None, 4),
            name="input_intron5p_alt_mmsplice"
        )
    ]
    mean_ref_logit = tf.keras.layers.Input(
        shape=(1,),
        name="mean_ref_logit"
    )
    s_ref_mt = [
        tf.keras.layers.Input(shape=(400, 4), name = "seqL_ref_tsplice"),
        tf.keras.layers.Input(shape=(400, 4), name = "seqR_ref_tsplice")
    ]
    s_var_mt = [
        tf.keras.layers.Input(shape=(400, 4), name = "seqL_var_tsplice"),
        tf.keras.layers.Input(shape=(400, 4), name = "seqR_var_tsplice")
    ]
    # tissue_effect = t_splice(s_ref_mt)
    variant_effect = mm_splice.call(s_ref_mm + s_var_mm)
    # canceled out t_splice(s_ref_mt)
    variant_tissue_interaction = t_splice.call(s_var_mt)
    variant_effect_tiled = tf.keras.layers.Lambda(
        lambda x: tf.tile(x, [1, n_cell_types]),
        output_shape = (n_cell_types,),
        name="variant_effect_tiled"
    )(variant_effect)
    final = tf.keras.layers.Add()(
        [variant_effect_tiled, variant_tissue_interaction, mean_ref_logit]
    )
    return tf.keras.models.Model(
                inputs=s_ref_mm + s_var_mm + s_var_mt + [mean_ref_logit],
                outputs=[final]
            )


