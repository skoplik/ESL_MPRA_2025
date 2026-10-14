import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    concatenate,
    Subtract,
    Dense,
    Layer
)
from tensorflow.keras import Model, Input
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.keras.optimizers import Adam
import keras_tuner as kerastuner
import tensorflow as tf
from tensorflow.keras import regularizers


def reinitialize_model(model):
    for i in model._layers:
        if "input" in i.name:
            continue
        if len(i.weights) > 0:
            w = merged_MMSplice_full.get_layer(i.name).get_weights()
            wz = [np.random.normal(size=(wi.shape)) for wi in w]
            merged_MMSplice_full.get_layer(i.name).set_weights(wz)
    return    
            

def add_regularization_L1(model, val=0.01):
    for i in model._layers:
        if "input" in i.name:
            continue
        if len(i.weights) > 0:
            (merged_MMSplice_full
            .get_layer(i.name)
            .kernel_regularizer) = regularizers.L1(val)
    return          
            

def add_regularization_L2(model, val=0.01):
    for i in model._layers:
        if "input" in i.name:
            continue
        if len(i.weights) > 0:
            (merged_MMSplice_full
            .get_layer(i.name)
            .kernel_regularizer) = regularizers.L2(val)
    return  
            

def add_regularization_elastic(model, val1=0.01, val2=0.01):
    for i in model._layers:
        if "input" in i.name:
            continue
        if len(i.weights) > 0:
            (merged_MMSplice_full
            .get_layer(i.name)
            .kernel_regularizer) = regularizers.l1_l2(l1=val1, l2=val2)
    return


def save_weights_wrapper(model):
    """
    janky fix for keras-tuner weird saving error:

    ValueError: The filename must end in `.weights.h5`. 
    Received: filepath=test_tsplice_hyperband_0/
    untitled_project/trial_0002/checkpoint

    normally, models are saved in checkpoint files
    however, Keras 3.11 seems to complain if they do not 
    have the right file extensions
    """
    old_save_func = model.save_weights
    def save(filepath, *args, **kwargs):
        if not filepath.endswith('.weights.h5'):
            filepath = filepath + '.weights.h5'
        old_save_func(filepath, *args, **kwargs)
    return save


def load_weights_wrapper(tuner):
    old_load_func = tuner._get_checkpoint_fname
    def new_load_func(trial_id):
        return old_load_func(trial_id) + ".weights.h5"
    return new_load_func


class HyperbandTemplate():


    def __init__(
            self,
            saved_model,
            model,
            cell_type_layer_wts,
            l1_min=0.0,
            l1_max=0.01,
            l2_min=0.0,
            l2_max=0.01,
            lr_min=1e-7,
            lr_max=1e-5,
            n_steps=10,
            dropout_min=0.1,
            dropout_max=0.4,
            model_name="functional"
        ):
        self.model=model
        self.saved_model=saved_model
        self.cell_type_wts=cell_type_layer_wts
        self.l1_min=l1_min
        self.l1_max=l1_max
        self.l2_min=l2_min
        self.l2_max=l2_max
        self.lr_min=lr_min
        self.lr_max=lr_max
        self.n_steps=n_steps
        self.dropout_min=dropout_min
        self.dropout_max=dropout_max
        self.saved_model_named_layers=set([
            i.name for i in self.saved_model.layers
        ])
        self.model.save_weights = save_weights_wrapper(self.model)
        self.model_name = model_name


    def init_func(self):
        def hyperband(hp):
            # Reload the pre-trained model for fine-tuning with 
            # new hyperparameters
            for i in self.model.layers:
                if isinstance(i, Functional) or self.model_name in i.name:
                    if isinstance(i, tf.keras.layers.InputLayer):
                        continue
                    for layer in i.layers:
                        if len(layer.weights) > 0:
                            if layer.name in self.saved_model_named_layers:
                                layer.set_weights(self.saved_model.get_layer(layer.name).get_weights())
                            elif layer.name=="cell_type_dense":
                                layer.set_weights(self.cell_type_wts)
                            hp_l1 = hp.Float("l1_" + layer.name, min_value = self.l1_min, max_value = self.l1_max, step = (self.l1_max - self.l1_min) / self.n_steps)
                            hp_l2 = hp.Float("l2_" + layer.name, min_value = self.l2_min, max_value = self.l2_max, step = (self.l2_max - self.l2_min) / self.n_steps)
                            layer.kernel_regularizer = regularizers.l1_l2(l1=hp_l1, l2=hp_l2)
                        if "dropout" in layer.name:
                            hp_drop = hp.Float("drop_rate_" + layer.name, min_value = self.dropout_min, max_value = self.dropout_max, step = (self.dropout_max - self.dropout_min) / self.n_steps)
                            layer.rate = hp_drop
            hp_lr = hp.Float('learning_rate', min_value=self.lr_min, max_value=self.lr_max, step=((self.lr_max-self.lr_min)/self.n_steps))
            # use epsilon = 1e-07 due to numerical stability issues

            
            # "lr" is not an allowed argument in tensorflow2.17, change to "learning_rate" (which is also in tf2.4)
            adam = Adam(learning_rate=hp_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
            # MSE seems to perform better with the new dataset
            self.model.compile(loss="mse", optimizer=adam)
            return self.model
        return hyperband


def get_hyperband_tsplice(tsplice, tsplice_saved, cell_type_layer_wts, lr_min=1e-8, lr_max=1e-2):
    """
    Initialize function to reload 
    pretrained tsplice weights and add the hyperparameters

    tsplice points to the model which is to be trained
    tsplice_saved just has the model weights

    It's important to create a copy so the initial 
    weights are not overwritten during training
    """
    hyperband_tsplice = HyperbandTemplate(model=tsplice,
                                          saved_model=tsplice_saved,
                                          cell_type_layer_wts=cell_type_layer_wts,
                                          l1_max=0.1,
                                          l2_max=0.1,
                                          lr_min=lr_min,
                                          lr_max=lr_max)
    hyperband_func = hyperband_tsplice.init_func()
    return hyperband_func


def run_hyperband_tsplice(hyperband_func, \
                          directory, \
                          x_train, \
                          y_train, \
                          x_val, \
                          y_val, \
                          hyperband_epochs=15, \
                          hyperband_max_epochs=20,\
                          stop_patience=5,\
                          **kwargs):
    tuner = kerastuner.Hyperband(hyperband_func,
                                 objective='val_loss',
                                 max_epochs=hyperband_max_epochs,
                                 factor=3,
                                 directory=directory,
                                 **kwargs)
    tuner._get_checkpoint_fname = load_weights_wrapper(tuner)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0.001, patience=stop_patience)
    tuner.search(x_train, y_train, epochs=hyperband_epochs, \
             validation_data=(x_val, y_val), \
            callbacks=[stop_early])
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    for value in best_hps.values:
        print(value, best_hps.get(value)) 
    return tuner, best_hps


def fit_model_after_tuning(tuner, x_train, y_train, x_val, y_val, epochs=50, patience=20):
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    # Build the model with the optimal hyperparameters and train until it overfits then revert to best validation set epoch
    model = tuner.hypermodel.build(best_hps)
    #monitor to restore best epoch 
    stop_early_save_best = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0.001, patience=patience, restore_best_weights=True) 
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), callbacks = [stop_early_save_best])
    val_acc_per_epoch = history.history['val_loss']
    best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,), min(val_acc_per_epoch))
    return model, history




