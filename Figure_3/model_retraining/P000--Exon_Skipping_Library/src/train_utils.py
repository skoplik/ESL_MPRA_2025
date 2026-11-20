import numpy as np


def weigh_samples(y_train):
    """
    Increase weights for samples
    with high (- or +) dlogit values
   
    encourage the model to pick up meaningful
    splicing changes
    """
    return np.abs(y_train) + 1
