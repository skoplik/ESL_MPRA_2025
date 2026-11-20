import pickle
import numpy as np
from src.esl import (
    cell_types, 
    cell_types2,
)
import pandas as pd
import os
import logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_pickle(data, path):
    """helper function to save data"""
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return


def save_tsplice_predictions(
        tsplice_predictions,
        output_dir
    ):
    # save initial retraining and baseline
    for model_head in range(4):
        for data in ("train", "val", "test"):
            for task in ("retrained", "baseline"):
                key = f"{data}_tsplice_{model_head}_{task}_dlogit"
                if key not in tsplice_predictions:
                    print("Model predictions not saved:", key)
                    continue 
                path = os.path.join(
                    output_dir,
                    f"{key}_predictions.p"
                )
                save_pickle(
                    data=tsplice_predictions[key],
                    path=path
                )
    # save mtsplice retraining
    for model_head in ("0", "1", "2", "3", "ensemble"):
        for data in ("train", "val", "test"):
            for task in ("mtsplice retrained", "mtsplice baseline"):
                key = f"{data}_tsplice_{model_head}_{task}_dlogit"
                if key not in tsplice_predictions:
                    print("Model predictions not saved:", key)
                    continue 
                path = os.path.join(
                    output_dir,
                    f"{key.replace(' ', '_')}_predictions.p"
                )
                save_pickle(
                    data=tsplice_predictions[key],
                    path=path
                )
    return 


def save_mtsplice_predictions(
        mtsplice_predictions,
        ref_psi,
        output_dir
    ):
    datasets = ("train", "val", "test")
    for data in ("train", "val", "test"):
        for task in ("retrained", "baseline"):
            key = f"{data}_mtsplice_{task}_logit"
            if key not in mtsplice_predictions:
                print("Model predictions not saved:", key)
                continue 
            path = os.path.join(
                output_dir,
                f"{key}_variant_predictions.p"
            )
            if np.sum(np.isnan(mtsplice_predictions[key])):
                breakpoint()
                " nan found "
            save_pickle(
                data=mtsplice_predictions[key],
                path=path
            )
    for i in range(len(datasets)):
        path = os.path.join(
            output_dir,
            f"{datasets[i]}_ref_cell_psi.p"
        )
        save_pickle(
            data=ref_psi[i],
            path=path
        )
    return


def check_sizes(df):
    """
    verify all sizes in dictionary
    are the same before trying 
    to convert to pandas dataframe.
    output all sizes if any are incorrect
    (pandas doesn't say by default)
    """
    mismatch = False
    for split in df:
        named_sizes = [(i, len(df[split][i])) for i in df[split]]
        sizes = [i[1] for i in named_sizes]
        if len(set(sizes)) != 1:
            mismatch = True
            print("\nSIZE MISMATCH IN", split)
            for i in named_sizes:
                print(i[0], i[1])
    if mismatch:
        raise ValueError("Check df sizes for mismatch")
    return
        

def add_metadata(df, metadata, split_name):
    """
    Adds metadata columns 
    to the predictions dataframe.
    Pass in split name (train, val or test)
    as a column to easily access 
    a specific split with pandas indexing.
    """
    if "var_id" in metadata:
        var_ids = metadata["var_id"].tolist()
        ref_ids = metadata["ref_id"].tolist()
        df["var_id"] = var_ids
        df["ref_id"] = ref_ids
    else:
        refs = metadata["Reference"].tolist()
        df["Reference"] = refs
    intron1_sizes = metadata["intron1_len"].tolist()
    gene_exon = metadata["gene_exon"].tolist()
    if "event_id_161" in metadata:
        event_ids = metadata["event_id_161"].tolist()
    else:
        event_ids = metadata["event_id"].tolist()
    df["intron1_size"] = intron1_sizes
    df["event_id_161"] = event_ids
    df["gene_exon"] = gene_exon
    df["split"] = [split_name] * len(metadata)
    return df


def save_tsplice_predictions_csv(
        tsplice_predictions,
        output_dir,
        metadata,
        scale="dlogit",
        df=None,
        save=True,
    ):
    """
    Saves scaled predictions to pandas dataframe,
    maintaining split information and other metadata
    (event id, references, etc.)
    """
    if df is None:
        df = {}
    data_splits = ("train", "val", "test")
    # save initial retraining and baseline
    for model_head in range(4):
        for i, data in enumerate(data_splits):
            if data not in df:
                df[data] = {}
                df[data] = add_metadata(
                    df[data],
                    metadata[f"metadata_{data}"],
                    split_name=data
                )
                for c, cell in enumerate(cell_types):
                    key = f"{data}_true_{scale}"
                    cell_key = f"{cell}_true_{scale}"
                    try:
                        df[data][cell_key] = tsplice_predictions[key][:,c]
                    except TypeError:
                        breakpoint()
                        " debug tsplice type "
            for task in ("retrained", "baseline"):
                key = f"{data}_tsplice_{model_head}_{task}_{scale}"
                key_no_data = f"tsplice_{model_head}_{task}_{scale}"
                if key not in tsplice_predictions:
                    print("Model predictions not saved:", key)
                    continue 
                for c, cell in enumerate(cell_types):
                    cell_key = f"{cell}_" + key_no_data
                    col = tsplice_predictions[key][:,c]
                    df[data][cell_key] = col

    # save mtsplice retraining
    # this are the predictions of just TSplice
    # after training again with MMSplice added
    for model_head in ("0", "1", "2", "3", "ensemble"):
        for i, data in enumerate(data_splits):
            for task in ("mtsplice retrained", "mtsplice baseline"):
                key = f"{data}_tsplice_{model_head}_{task}_{scale}"
                key_no_data = (
                    f"tsplice_{model_head}_"
                    f"{task.replace(' ', '_')}_{scale}"
                )
                if key not in tsplice_predictions:
                    print("Model predictions not saved:", key)
                    continue 
                for c, cell in enumerate(cell_types):
                    cell_key = f"{cell}_" + key_no_data
                    col = tsplice_predictions[key][:,c]
                    df[data][cell_key] = col
    check_sizes(df)
    df = pd.concat([
        pd.DataFrame(df["train"]),
        pd.DataFrame(df["val"]),
        pd.DataFrame(df["test"]),
    ])
    save_path = os.path.join(
        output_dir,
        f"tsplice_predictions_{scale}.csv"
    )
    if save:
        df.to_csv(save_path, index=False) 
    return df


def save_mtsplice_predictions_csv(
        mtsplice_predictions,
        output_dir,
        metadata,
        scale="dlogit",
        df=None,
        save=True,
    ):
    """
    Saves scaled predictions to pandas dataframe,
    maintaining split information and other metadata
    (event id, var and ref ids, etc.)
    """
    if df is None:
        df = {}
    datasets = ("train", "val", "test")
    for i, data in enumerate(datasets):
        if data not in df:
            df[data] = {}
            df[data] = add_metadata(
                df[data],
                metadata[f"metadata_{data}"],
                split_name=data,
            )
            for c, cell in enumerate(cell_types):
                key = f"{data}_true_{scale}"
                cell_key = f"{cell}_true_{scale}"
                df[data][cell_key] = mtsplice_predictions[key][:,c]
        for task in ("retrained", "baseline"):
            key = f"{data}_mtsplice_{task}_{scale}"
            key_no_data = f"mtsplice_{task}_{scale}"
            if key not in mtsplice_predictions:
                print("Model predictions not saved:", key)
                continue 
            for c, cell in enumerate(cell_types):
                cell_key = f"{cell}_" + key_no_data
                col = mtsplice_predictions[key][:,c]
                df[data][cell_key] = col
    df = pd.concat([
        pd.DataFrame(df["train"]),
        pd.DataFrame(df["val"]),
        pd.DataFrame(df["test"]),
    ])
    save_path = os.path.join(
        output_dir,
        f"mtsplice_predictions_{scale}.csv"
    )
    if save:
        df.to_csv(save_path, index=False) 
    return df


def save_predictions_as_csv(
        mtsplice_predictor,
        output_dir,
        metadata,
        scale="dlogit",
        model="tsplice",
        df=None,
        save=True,
    ):
    """
    Saves scaled predictions to pandas dataframe,
    maintaining split information and other metadata
    (event id, var and ref ids, etc.)
    """
    if df is None:
        df = {}
    datasets = mtsplice_predictor.datasets
    head = "ensemble" if model == "tsplice" else ""
    for i, data in enumerate(datasets):
        if data not in df:
            df[data] = {}
            df[data] = add_metadata(
                df[data],
                metadata[f"metadata_{data}"],
                split_name=data,
            )
            for c, cell in enumerate(cell_types):
                true = mtsplice_predictor.predictions.get_true_data(
                    data=data,
                    scale=scale,
                    model=model,
                )
                cell_key = f"{cell}_true_{scale}"
                df[data][cell_key] = true[:,c]
        for task in ("retrained", "baseline"):
            version_mt = "" if task == "baseline" else task
            prediction = mtsplice_predictor.get_prediction(
                data=data,
                scale=scale,
                model=model,
                head=head,
                version_t=task,
                version_mm=task,
                version_mt=version_mt,
                output="prediction",
            )
            key_no_data = f"{model}_{task}_{scale}"
            for c, cell in enumerate(cell_types):
                cell_key = f"{cell}_" + key_no_data
                col = prediction[:,c]
                df[data][cell_key] = col
    df = pd.concat([
        pd.DataFrame(df["train"]),
        pd.DataFrame(df["val"]),
        pd.DataFrame(df["test"]),
    ])
    save_path = os.path.join(
        output_dir,
        f"{model}_predictions_{scale}.csv"
    )
    if save:
        df.to_csv(save_path, index=False) 
        logging.info(f"Saving {model} predictions to {save_path}")
    return df
