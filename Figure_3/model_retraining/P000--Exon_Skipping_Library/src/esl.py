import numpy as np
import tensorflow as tf
import pickle
import os
from collections import defaultdict

cell_types = ["HEK293", "HeLa", "K562", "MCF7", "HMC3"]
cell_types2 = [i if i != "HEK293" else "HEK" for i in cell_types]
cell_type_psi = ['HEK293_PSI', 'HeLa_PSI', 'K562_PSI', 'MCF7_PSI', 'HMC3_PSI']

CITRINE_EXON1 = "ATGGTGTCCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTCAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAACTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGATGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAAGTGAAGTTCGAGGGCGACACCCTCGTGAACCGCATCGAGCTAAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAAGTGAACTTCAAGATCCGCCACAACATCGAG"
CITRINE_EXON2 = "GACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAG"
SMN2_INTRON6  = "GTAAGTAATCACTCAGCATCTTTTCCTGACAATTTTTTTGTAGTTATGTGACTTTGTTTTGTAAATTTATAAAATACTACTTGCTTCTCTCTTTATATTACTAAAAAATAAAAATAAAAAAATACAACTGTCTGAGGCTTAAATTACTCTCAACTTAATTTCTGATCATATTTTGTTGAATAAAATAAGTAAAATGTCTTGTGAAACAAAATGCTTTTTAACATCCATATAAAGCTATCTATATATAGCTATCTATATCTA"
SMN2_INTRON7  = "AAAGTGAATCTTACTTTTGTAAAACTTTATGGTTTGTGGAAAACAAATGTTTTTGAACATTTAAAAAGTTCAGATGTTAGAAAGTTGAAAGGTTAATGTAAAACAATCAATATTAAAGAATTTTGATGCCAAAACTATTAGATAAAAGGTTAATCTACATCCCTACTAGAATTCTCATACTTAACTGGTTGGTTGTGTGGAAGAAACATACTTTCACAATAAAGAGCTTTAGGATATGATGCCATTTTATATCACTAGTAGGCAGACCAGCAGACTTTTTTTTATTGTGATATGGGATAACCTAGGCATACTGCACTGTACACTCTGACATATGAAGTGCTCTAGTCAAGTTTAACTGGTGTCCACAGAGGACATGGTTTAACTGGAATTCGTCAAGCCTCTGGTTCTAATTTCTCATTTGCAG"


def clip(x, clip_threshold=0.01):
    return np.clip(x, clip_threshold, 1 - clip_threshold)


def logit(x, clip_threshold=0.01):
    x = clip(x, clip_threshold=clip_threshold)
    return np.log(x) - np.log(1 - x)
    
    
class Logit(tf.keras.layers.Layer):
    def __init__(self, clip_threshold=0.01, **kwargs):
        super(Logit, self).__init__(**kwargs)
        self.clip_threshold = clip_threshold
        return
    def call(self, x):
        x = tf.clip_by_value(
            x,
            clip_value_min=self.clip_threshold,
            clip_value_max=1-self.clip_threshold
        )
        return tf.math.log(x) - tf.math.log(1 - x)


def expit(x):
    return 1. / (1. + np.exp(-x))


class Expit(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Expit, self).__init__(**kwargs)
        return 
    def call(self, x):
        return 1. / (1. + tf.math.exp(-x))


def get_loaded_pickle(
        data_dir,
        point_file = "x_train.p",
        extension=None
    ):
    """
    Helper function to load saved data.
    loads in every file from a directory,
    unless the specific extension is specified.

    Checks if path exists, and keeps track
    of wether the reading was successful
    """
    dict_ = {}
    if not os.path.exists(os.path.join(data_dir, point_file)):
        print(os.path.join(data_dir, point_file))
        return False, {}
    else:
        for name in os.listdir(data_dir):
            if extension is not None and name.split(".")[-1] != extension:
                continue
            path = os.path.join(data_dir, name)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    dict_[name.split(".")[0]] = pickle.load(f)
            else:
                print("path", path, "does not exist")
    return True, dict_


def multi_scale_dlogit(ref_psi, var_psi):
    out = {}
    out["clip_mean_logit"] = (
        logit(mean(clip(var_psi)))
        - logit(mean(clip(ref_psi)))
    )
    out["mean_clip_logit"] = (
        logit(clip(mean(var_psi))) 
        - logit(clip(mean(ref_psi)))
    )
    out["clip_logit_mean"] = (
        mean(logit(clip(var_psi)))
        - mean(logit(clip(ref_psi)))
    )
    # MMSplice scaling:
    out["mean_dlogit"] = (
        np.nanmean(logit(clip(var_psi)).to_numpy()
        - logit(clip(ref_psi)).to_numpy())
    )
    return out


def aggregate_values_by_family(
        values,
        metadata,
        metadata_key = "event_id_161",
        method = "mean"
    ):
    eids = metadata[metadata_key].tolist()
    eids_rev = {}
    eids_computed = defaultdict(float)
    eids_size = defaultdict(int)
    index = 0
    for i, val in enumerate(eids):
        if val not in eids_computed:
            eids_rev[val] = index
            eids_computed[val] = 0.0
            eids_size[val] = 0
            index += 1
        eids_computed[val] += values[i]
        eids_size[val] += 1
    values_out = np.zeros((len(eids_computed),))
    for eid in eids_rev:
        i = eids_rev[eid]
        val = eids_computed[eid]
        if method == "mean":
            val = val / eids_size[eid]
        values_out[i] = val
    return values_out
         
    

