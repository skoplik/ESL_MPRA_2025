import numpy as np
from mtsplice_retraining.model_evaluation.r2_calc import (
    calc_r2_with_nan_tsplice
)
from mtsplice_retraining.model_evaluation.print_r2 import print_table

cell_types = ["HEK293", "HeLa", "K562", "MCF7", "HMC3"]


def clip(x, clip_threshold=0.01):
    return np.clip(x, clip_threshold, 1 - clip_threshold)


def logit(x, clip_threshold=0.01):
    x = clip(x, clip_threshold=clip_threshold)
    return np.log(x) - np.log(1 - x)
    
    
def expit(x):
    return 1. / (1. + np.exp(-x))


def tsplice_natural_scale(prediction, cell_mean):
    return prediction + cell_mean


def tsplice_psi_scale(prediction, cell_mean):
    return expit(prediction + cell_mean) 


def tsplice_dpsi_scale(prediction, cell_mean):
    return tsplice_psi_scale(prediction, cell_mean) - expit(cell_mean)


def tsplice_dlogit_scale(prediction, cell_mean):
    return prediction


def output_spliceai_r2_table(scaled_data):
    tables = {}
    for dataset in scaled_data:
        scaled_data_i = scaled_data[dataset]
        scale_mapping = scaled_data_i.todict()
        for scale in scale_mapping:
            if scale in tables:
                tables[scale][dataset] = scale_mapping[scale]
            else:
                tables[scale] = {dataset : scale_mapping[scale]}
    display = []
    for scale in tables:
        display.append(
            print_table(
                cell_types,
                tables[scale],
                title=f"R^2 Delta Logit {scale}",
                print_string=False
            )
        )
    return display



def output_tsplice_r2_table(tsplice_predictions, splice="tsplice"):
    assert splice == "tsplice" or splice == "mtsplice", (
        f"Got unexpected splice argument '{splice}'."
        f"expected either 'tsplice' or 'mtsplice'"
    )
    dlogit_r2 = {}
    logit_r2 = {}
    psi_r2 = {}
    dpsi_r2 = {}
    models = set()
    datasets = set()
    tasks = set()
    scaling = set()
    # reordering keys so baseline appears next to retrained
    for key in tsplice_predictions:
        if key[0:2] != "r2":
            continue
        if splice == "tsplice":
            _, dataset, _, model_head, task, scale = key.split("_")
        elif splice == "mtsplice":
            _, dataset, _, task, scale = key.split("_")
            model_head = " "
        models.add(model_head)
        datasets.add(dataset)
        tasks.add(task)
        scaling.add(scale)
    models = sorted(models)
    if len(datasets) == 3:
        datasets = ["train", "val", "test"]
    else:
        datasets = ["train", "val", "test", "full"]
    scaling = ["dlogit", "logit", "dpsi", "psi"]
    keys = []
    for data in datasets:
        for model in models:
            for scale in scaling:
                for task in tasks:
                    if splice == "tsplice":
                        key = f"r2_{data}_{splice}_{model}_{task}_{scale}"
                    elif splice == "mtsplice":
                        key = f"r2_{data}_{splice}_{task}_{scale}"
                    if key in tsplice_predictions:
                        keys.append(key)
    for key in keys:
        if key[0:2] != "r2":
            continue
        if splice == "tsplice":
            _, dataset, _, model_head, task, scale = key.split("_")
            model_ = f" model {model_head} "
        elif splice == "mtsplice":
            _, dataset, _, task, scale = key.split("_")
            model_ = " "
        val_key = f"{dataset}{model_}({task})"
        match scale:
            case "logit":
                logit_r2[val_key] = tsplice_predictions[key]
            case "dlogit":
                dlogit_r2[val_key] = tsplice_predictions[key]
            case "psi":
                psi_r2[val_key] = tsplice_predictions[key]
            case "dpsi":
                dpsi_r2[val_key] = tsplice_predictions[key]

    if splice == "tsplice":
        model_title = "TSplice"
    elif splice == "mtsplice":
        model_title = "MTSplice"
    dlogit_out = print_table(
        cell_types,
        dlogit_r2,
        title=f"{model_title} R^2 Delta Logit (cell - mean cell)",
        print_string=False
    )
    logit_out = print_table(
        cell_types,
        logit_r2,
        title=f"{model_title} R^2 Cell Logit",
        print_string=False
    )
    psi_out = print_table(
        cell_types,
        psi_r2,
        title=f"{model_title} R^2 Cell PSI",
        print_string=False
    )
    dpsi_out = print_table(
        cell_types,
        dpsi_r2,
        title=f"{model_title} R^2 Delta PSI (PSI cell - PSI mean cell)",
        print_string=False
    )
    dlogit_out = "\n".join(dlogit_out)
    logit_out = "\n".join(logit_out)
    psi_out = "\n".join(psi_out)
    dpsi_out = "\n".join(dpsi_out)
    return dlogit_out, logit_out, psi_out, dpsi_out
                      

def scale_tsplice_predictions(
        predictions,
        y_true,
        cell_mean,
        data_key=None,
        model_key=None
    ):
    """
    Scales Delta Logit TSplice outputs
    Returns scaled predictions and r^2 values
    """ 
    if data_key is None:
        data_key = ["train", "val", "test", "full"]
    if model_key is None:
        model_key = ["0", "1", "2", "3", "ensemble"]
    results = {}
    for task in predictions:
        assert len(predictions[task]) in (1,4,5), "Num heads not 1, 4 or 5"
        for model_head, datasets in enumerate(predictions[task]):
            assert len(datasets) in (3,4), "Num datasets not 3 or 4"
            if len(datasets) == 4:
                size_equal = (
                    sum([len(i) for i in datasets][:-1])
                    == len(datasets[-1])
                )
                assert size_equal, (
                    "full size not equal to sum of train val and test"
                )
            elif len(datasets) == 3:
                assert len(datasets[0]) > len(datasets[1]), (
                    "train not bigger than val"
                )
                assert len(datasets[0]) > len(datasets[2]), (
                    "train not bigger than test"
                )
            for data_i, data in enumerate(datasets):
                y_psi = tsplice_psi_scale(
                    y_true[data_i],
                    cell_mean[data_i]
                )
                y_logit = tsplice_natural_scale(
                    y_true[data_i],
                    cell_mean[data_i]
                )
                y_dpsi = tsplice_dpsi_scale(
                    y_true[data_i],
                    cell_mean[data_i]
                )
                pr_psi = tsplice_psi_scale(data, cell_mean[data_i])
                pr_logit = tsplice_natural_scale(data, cell_mean[data_i])
                pr_dpsi = tsplice_dpsi_scale(data, cell_mean[data_i])
                r2_psi = calc_r2_with_nan_tsplice(pr_psi, y_psi)
                r2_logit = calc_r2_with_nan_tsplice(pr_logit, y_logit)
                r2_dpsi = calc_r2_with_nan_tsplice(pr_dpsi, y_dpsi)
                r2_dlogit = calc_r2_with_nan_tsplice(data, y_true[data_i])
                pr_prefix = (
                    f"{data_key[data_i]}_tsplice_"
                    f"{model_key[model_head]}_{task}_"
                )
                r2_prefix = (
                    f"r2_{data_key[data_i]}_tsplice_"
                    f"{model_key[model_head]}_{task}_"
                )
                true_prefix = f"{data_key[data_i]}_true_"
                results[pr_prefix + "dlogit"] = data
                results[pr_prefix + "dpsi"] = pr_dpsi
                results[pr_prefix + "psi"] = pr_psi
                results[pr_prefix + "logit"] = pr_logit
                results[r2_prefix + "dlogit"] = r2_dlogit
                results[r2_prefix + "dpsi"] = r2_dpsi
                results[r2_prefix + "psi"] = r2_psi
                results[r2_prefix + "logit"] = r2_logit
                results[true_prefix + "dlogit"] = y_true[data_i]
                results[true_prefix + "dpsi"] = y_dpsi
                results[true_prefix + "psi"] = y_psi
                results[true_prefix + "logit"] = y_logit
    return results


def mtsplice_logit_scale(predictions, ref_cell_psi):
    return predictions


def mtsplice_dlogit_scale(predictions, ref_cell_psi):
    return predictions - logit(ref_cell_psi)


def mtsplice_psi_scale(predictions, ref_cell_psi):
    return expit(predictions)


def mtsplice_dpsi_scale(predictions, ref_cell_psi):
    return expit(predictions) - ref_cell_psi


def scale_mtsplice_predictions(mtsplice_predictions, y_true, ref_cell_psi):
    """
    Change predictions from logit_var_cell scale
    And get r^2 values
    """
    data_key = ["train", "val", "test", "full"]
    results = {}
    for task in mtsplice_predictions:
        datasets = mtsplice_predictions[task]
        assert len(datasets) in (3,4), "Num datasets not 3 or 4"
        if len(datasets) == 4:
            size_equal = (
                sum([len(i) for i in datasets][:-1])
                == len(datasets[-1])
            )
            assert size_equal, (
                "full size not equal to sum of train val and test"
            )
        elif len(datasets) == 3:
            assert len(datasets[0]) > len(datasets[1]), (
                "train not bigger than val"
            )
            assert len(datasets[0]) > len(datasets[2]), (
                "train not bigger than test"
            )
        for data_i, data in enumerate(datasets):
            y_psi = mtsplice_psi_scale(
                y_true[data_i],
                ref_cell_psi[data_i]
            )
            y_dlogit = mtsplice_dlogit_scale(
                y_true[data_i],
                ref_cell_psi[data_i]
            )
            y_dpsi = mtsplice_dpsi_scale(
                y_true[data_i],
                ref_cell_psi[data_i]
            )
            pr_psi = mtsplice_psi_scale(
                data,
                ref_cell_psi[data_i]
            )
            pr_dlogit = mtsplice_dlogit_scale(
                data,
                ref_cell_psi[data_i]
            )
            pr_dpsi = mtsplice_dpsi_scale(
                data,
                ref_cell_psi[data_i]
            )
            r2_psi = calc_r2_with_nan_tsplice(pr_psi, y_psi)
            r2_logit = calc_r2_with_nan_tsplice(data, y_true[data_i])
            r2_dpsi = calc_r2_with_nan_tsplice(pr_dpsi, y_dpsi)
            r2_dlogit = calc_r2_with_nan_tsplice(pr_dlogit, y_dlogit)
            pr_prefix = f"{data_key[data_i]}_mtsplice_{task}_" 
            r2_prefix = f"r2_{data_key[data_i]}_mtsplice_{task}_"
            true_prefix = f"{data_key[data_i]}_true_"
            results[pr_prefix + "dlogit"] = pr_dlogit
            results[pr_prefix + "dpsi"] = pr_dpsi
            results[pr_prefix + "psi"] = pr_psi
            results[pr_prefix + "logit"] = data
            results[r2_prefix + "dlogit"] = r2_dlogit
            results[r2_prefix + "dpsi"] = r2_dpsi
            results[r2_prefix + "psi"] = r2_psi
            results[r2_prefix + "logit"] = r2_logit
            results[true_prefix + "dlogit"] = y_dlogit
            results[true_prefix + "dpsi"] = y_dpsi
            results[true_prefix + "psi"] = y_psi
            results[true_prefix + "logit"] = y_true[data_i]
    return results






