import tensorflow as tf
from tensorflow.python.keras.engine.functional import Functional
import numpy as np

def set_wts_to_zero(layer, noise=0):
    layer_wts = layer.weights
    wts_zero = []
    for wt in layer_wts:
        noise_tensor = np.random.random(size=(wt.shape)) * noise
        wts_zero.append(tf.zeros_like(wt) + noise_tensor)
    layer.set_weights(wts_zero)
    return


def zero_init(model, exceptions = None, noise = 0):
    """
    Sets all weights to zero
    if the model has nested functionals,
    will set weights recursively

    exceptions can be passed in,
    which is a set of layer names.
    the layers in exceptions will not be
    set to zero
    """
    if exceptions is None:
        exceptions = set()
    def set_layer_to_zero(l, nested_model):
        if l is not None and not isinstance(l, Functional):
            wts = nested_model.get_layer(l.name).weights
            if l.name in exceptions:
                return
            set_wts_to_zero(nested_model.get_layer(l.name), noise=noise)
            return
        elif isinstance(l, tf.keras.layers.InputLayer):
            return 
        elif l is None:
            l = nested_model
        for layer in l.layers:
            set_layer_to_zero(layer, l)
        return

    set_layer_to_zero(None, model)
    return


if __name__ == "__main__": 
    import sys
    MMSplice_MTSplice_path = "/storage/yu/Projects/P000--ESL_continuation/GitHub/MMSplice_MTSplice"
    sys.path.append(MMSplice_MTSplice_path)
    import mmsplice
       
    mt_splice = mmsplice.mtsplice.MTSplice()
    model = mt_splice.mtsplice_models[0]
    model.summary()
    print("CALLING ZERO INIT")
    zero_init(model)
    print(model.weights)

    print("DONE")



