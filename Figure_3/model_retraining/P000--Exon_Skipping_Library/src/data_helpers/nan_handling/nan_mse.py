import tensorflow as tf
import numpy as np

def nan_mse(batch_size, use_sample_weights=False, cell_weights = None) :
    if cell_weights is None:
        cell_weights = tf.cast(tf.constant(np.ones((5,))), tf.float32)
    
    
    @tf.function
    def mse(y_true, y_pred) :
        counts = 1
        # sample_weights = tf.ones((batch_size,))
        # if use_sample_weights :
        #     sample_weights = get_sample_weights(counts, batch_size)
        num_not_nan = tf.cast(tf.reduce_sum(1 - tf.cast(tf.math.is_nan(y_true), tf.int32), axis = 1), tf.float32)
        y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
        y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
        L = tf.cast(y_true.shape[1], dtype=tf.float32)
        mse_calc = tf.reduce_mean(((y_true - y_pred)**2) * cell_weights, axis=-1) * L / num_not_nan
        return tf.reduce_mean(mse_calc)

    
    return mse
