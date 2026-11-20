         
if __name__ == "__main__":
    import sys
    import os
    PROJECT_NAME = "P000--Exon_Skipping_Library"
    node = os.getcwd()
    for i in range(5): # max depth
        tree, node = os.path.split(node)
        if node == PROJECT_NAME:
            break 
        else:
            node = tree
    PROJECT_ROOT = os.path.join(tree, node)
    if node != PROJECT_NAME:
        raise ModuleNotFoundError(
            f"Could not find project directory {PROJECT_NAME}"
        )
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)


from src.esl import (
    logit,
    expit
)


def mtsplice_logit_scale(predictions, ref_cell_psi):
    return predictions


def mtsplice_dlogit_scale(predictions, ref_cell_psi):
    return predictions - logit(ref_cell_psi)


def mtsplice_psi_scale(predictions, ref_cell_psi):
    return expit(predictions)


def mtsplice_dpsi_scale(predictions, ref_cell_psi):
    return expit(predictions) - ref_cell_psi


def tsplice_natural_scale(prediction, cell_mean):
    return prediction + cell_mean


def tsplice_psi_scale(prediction, cell_mean):
    return expit(prediction + cell_mean) 


def tsplice_dpsi_scale(prediction, cell_mean):
    return tsplice_psi_scale(prediction, cell_mean) - expit(cell_mean)


def tsplice_dlogit_scale(prediction, cell_mean):
    return prediction


def scale_mtsplice(
        predictions,
        ref_cell_psi,
        curr_scale,
        target_scale
    ):
    """
    Scales predictions from 
    delta logit, variant logit, variant psi
    or delta psi into any other scale.

    ref_cell_psi is required to convert when
    predictions is in a delta scale.
    """
    # scale to default variant logit scale
    if curr_scale == "dlogit":
        predictions += logit(ref_cell_psi)
    elif curr_scale == "dpsi":
        predictions += ref_cell_psi
        predictions = logit(predictions)
    elif curr_scale == "psi":
        predictions = logit(predictions)
    else:
        pass # predictions in logit scale already
    # use functions to convert from logit to target scale
    if target_scale == "dlogit":
        return mtsplice_dlogit_scale(predictions, ref_cell_psi)
    elif target_scale == "dpsi":
        return mtsplice_dpsi_scale(predictions, ref_cell_psi)
    elif target_scale == "psi":
        return mtsplice_psi_scale(predictions, ref_cell_psi)
    return predictions # if logit is target


def scale_tsplice(
        predictions,
        cell_mean_logit,
        curr_scale,
        target_scale
    ):
    """
    Scales predictions from 
    delta logit, logit, psi
    or psi into any other scale.

    cell_mean_logit is required to convert when
    predictions is in a delta scale.
    """
    # scale to default variant logit scale
    if curr_scale == "logit":
        predictions += cell_mean_logit
    elif curr_scale == "dpsi":
        predictions += expit(cell_mean_logit)
        predictions = logit(predictions) - cell_mean_logit
    elif curr_scale == "psi":
        predictions = logit(predictions) - cell_mean_logit
    else:
        pass # predictions in dlogit scale already
    # use functions to convert from logit to target scale
    if target_scale == "logit":
        return tsplice_natural_scale(predictions, cell_mean_logit)
    elif target_scale == "dpsi":
        return tsplice_dpsi_scale(predictions, cell_mean_logit)
    elif target_scale == "psi":
        return tsplice_psi_scale(predictions, cell_mean_logit)
    return predictions # if dlogit is target


if __name__ == "__main__":
    import numpy as np
    from src.esl import clip
    psi = clip(np.random.random((100, 5)))
    ref_psi =  clip(np.random.random((100, 5)))
    dlogit = scale_mtsplice(
        psi, ref_psi,
        curr_scale="psi", target_scale="dlogit"
    )
    psi_rev = scale_mtsplice(
        dlogit, ref_psi,
        curr_scale="dlogit", target_scale="psi"
    )
    assert np.allclose(psi, rev_psi)
    
