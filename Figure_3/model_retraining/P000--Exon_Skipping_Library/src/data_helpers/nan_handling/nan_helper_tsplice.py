import numpy as np
import tensorflow as tf
try:
    from keras.src.layers import Input
    from keras.src.models import Model
except ModuleNotFoundError:
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model

from src.esl import (
    cell_types,
    cell_types2,
)

cell_type_psi = [
    'HEK293_PSI',
    'HeLa_PSI',
    'K562_PSI',
    'MCF7_PSI',
    'HMC3_PSI'
]


def get_nan_size_psi_and_coverage(x):
    """
    get the 'nan size' i.e. number of
    non-nan samples in each family,
    as well as the average psi and coverage
    for running knapsack.
    """
    size = len(x)
    cov = np.sum(x["Avg_coverage"])
    psi = np.mean(x["Avg_PSI"])
    not_nan = np.sum(np.sum(np.isnan(x[cell_type_psi]), axis=1) == 0)
    not_nan_cov = np.sum(
        x["Avg_coverage"][
            np.sum(np.isnan(x[cell_type_psi]), axis=1) == 0
        ]
    )
    hits = 0
    return size, cov, psi, not_nan, not_nan_cov, hits


def get_not_nan_num(x):
    """ returns number of non-nans for each row """
    return np.sum(np.sum(np.isnan(x[cell_type_psi]), axis=1) == 0)



def preprocess_nan_masks(y, n_cell_types=5):
    """
    y is of shape (num_samples, 5)
    returns a 0-1 mask to multiply predicted values by (num_samples, 5)
    also returns the number of NaNs in each row of y (num_samples, 1)
    """
    valid_mask = tf.math.is_finite(y)
    y_ones = np.ones(y.shape)
    y_mask = tf.where(valid_mask, y_ones, 0.0)
    num_not_nan = np.sum(1 - np.isnan(y), axis = 1)
    num_not_nan = np.reshape(num_not_nan, (len(num_not_nan), 1)) 
    num_not_nan = tf.constant(num_not_nan)
    proportion_not_nan = tf.sqrt((n_cell_types / num_not_nan)) 
    y_clean = tf.where(valid_mask, y, 0.0) 
    """ 
    replace nan with 0. while 0 is a possible value,
    this will be masked in y_pred since we already have 
    the mask preprocessed.
    
    it's important to remove the nan so 
    that we can just multiply the mask with y_pred and 
    calculate the MSE, without needing tf.where at training time 
    (0 * nan = nan, but we want 0 * nan = 0)
    """
    return y_mask, proportion_not_nan, y_clean


#   ((y_pred - y_true) ** 2) * (5 / num_not_nan) * cwt [cell weights] 
# = ((y_pred * sqrt(5/num_not_nan) * sqrt(cwt)) 
# - (y_true * sqrt(5/num_not_nan) * sqrt(cwt)) ** 2
class NanMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NanMask, self).__init__(**kwargs)
    def call(self, y_pred, y_mask, proportion_not_nan, cell_wts):
        y_pred = y_pred * y_mask
        y_pred = y_pred * proportion_not_nan
        y_pred = y_pred * cell_wts
        return y_pred
        

def add_nan_mask_layer(
        tsplice,
        y_mask_shape,
        proportion_not_nan_shape,
        cell_wts_shape
    ):
    """
    Add NaN masking for speeding up training
    """
    tsplice_inputs = tsplice.inputs
    y_mask_input  = Input(
        shape=y_mask_shape[1:],
        name=f"{tsplice.name}_y_mask_input"
    )
    not_nan_input = Input(
        shape=proportion_not_nan_shape[1:],
        name=f"{tsplice.name}_not_nan_input"
    )
    cell_wt_input = Input(
        shape=cell_wts_shape[1:],
        name=f"{tsplice.name}_cell_wt"
    )
    y_pred = tsplice.outputs
    y_pred_clean = NanMask()(
        y_pred,
        y_mask_input,
        not_nan_input,
        cell_wt_input
    )
    nan_helper_inputs = [y_mask_input, not_nan_input, cell_wt_input]
    tsplice_nan_inputs = tsplice_inputs + nan_helper_inputs
    tsplice_nan_model = Model(
        inputs=tsplice_nan_inputs,
        outputs=[y_pred_clean]
    )
    return tsplice_nan_model


if __name__ == "__main__":
    print("done")
