import os
import getopt
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
import re


def rename_id_col(df, final_id):
    id_col = check_id_col(df)
    df = df.rename(
        columns={
            id_col : final_id
        },
    )
    return df


def check_id_col(df):
    if "Reference" in df:
        return "Reference"
    if "var_id" in df:
        return "var_id"
    if "Variant ID" in df:
        return "Variant ID"
    if "Variant_ID" in df:
        return "Variant_ID"
    raise ValueError("FILTER does not have known variant ID column.")


def print_first_key(df_name, df):
    for i in df:
        print(df_name, i)
        return


def print_delta_logit_cols(df):
    for i in df:
        if i.endswith("delta_logit"):
            print(i)
    return


def read_prediction(path):
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    else:
        return pd.read_csv(path)


def read_split_file(path):
    dir_, file_ = os.path.split(path)
    predictions_df = []
    for i in range(4):
        file_split = re.sub(
            r"_split_\d+\.tsv",
            f"_split_{i}.tsv",
            file_,
            count=0,
            flags=0
        )
        path = os.path.join(dir_, file_split)
        predictions_df.append(pd.read_csv(path, sep="\t"))
    ag_predictions = pd.concat(predictions_df, ignore_index=True) 
    return ag_predictions
        


help_="""
Merges predictions from HAL and Pangolin
into unfiltered alphagenome + spliceai + mmsplice
into merged (ref_id, var_id) format.

(16kb alphagenome)
(1Mb alphagenome)
(SpliceAI)
(Baseline MMSplice)
(Retrained MMSplice)
"""
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:],"", [
        "alphagenome_16kb_path=",
        "alphagenome_1Mb_path=",
        "spliceai_path=",
        "mmsplice_path=",
        "full_data_path=",
        "output_path=",
        "debug",
        "help",
    ])
    opts = dict(opts)
    if "--help" in opts:
        print(help_)
        sys.exit(0)

    debug = "--debug" in opts
    if debug:
        alphagenome_16kb_path = "/storage/yu/Projects/P000--ESL_continuation/GitHub/P000--Exon_Skipping_Library/spliceai_and_ag_testing/alphagenome_predictions/predictions_16kb/alphagenome_esl_predictions_16kb_split_0.tsv"
        alphagenome_1Mb_path = "/storage/yu/Projects/P000--ESL_continuation/GitHub/P000--Exon_Skipping_Library/spliceai_and_ag_testing/alphagenome_predictions/predictions_1Mb/alphagenome_esl_predictions_1Mb_split_0.tsv"
        spliceai_path = "/storage/yu/Projects/P000--ESL_continuation/GitHub/P000--Exon_Skipping_Library/spliceai_and_ag_testing/spliceai_predictions/spliceai_predictions_compass.csv"
        mmsplice_path = "/storage/yu/Projects/P000--ESL_continuation/GitHub/P000--Exon_Skipping_Library/models/mmsplice_revised/mmsplice139/predictions.csv"
        full_data_path = "/storage/yu/Projects/P000--ESL_continuation/Analysis/model_comparison/model_predictions_MAY_v2.csv"
        output_path = "/storage/yu/Projects/P000--ESL_continuation/Data/model_comparison/model_predictions_GCF.csv"
    else:
        alphagenome_16kb_path = opts["--alphagenome_16kb_path"]
        alphagenome_1Mb_path = opts["--alphagenome_1Mb_path"]
        spliceai_path = opts["--spliceai_path"]
        mmsplice_path = opts["--mmsplice_path"]
        full_data_path = opts["--full_data_path"]
        output_path = opts["--output_path"]
        print(opts, flush=True)

    if "split" in alphagenome_16kb_path:
        ag_16kb_predictions = read_split_file(alphagenome_16kb_path)
    else:
        ag_16kb_predictions = read_prediction(alphagenome_16kb_path)
    if "split" in alphagenome_1Mb_path:
        ag_1Mb_predictions = read_split_file(alphagenome_1Mb_path)
    else:
        ag_1Mb_predictions = read_prediction(alphagenome_1Mb_path)
    ag_16kb_predictions = ag_16kb_predictions.rename(
        columns={
            "alphagenome_dlogit": "alphagenome_16kb_dlogit"
        }
    )
    ag_1Mb_predictions = ag_1Mb_predictions.rename(
        columns={
            "alphagenome_dlogit": "alphagenome_1Mb_dlogit"
        }
    )
    spliceai_predictions = read_prediction(spliceai_path)
    spliceai_predictions = spliceai_predictions.rename(
        columns={
            "predicted_dlogit": "spliceai_dlogit"
        }
    )
    mmsplice_predictions = read_prediction(mmsplice_path)
    full_data_predictions = read_prediction(full_data_path)
    full_data_predictions = full_data_predictions[
        ~np.isnan(full_data_predictions["alphagenome_delta_logit"])
    ]
    full_data_predictions = full_data_predictions[
        full_data_predictions["snp"] != "none"
    ]


    print("alphagenome (16kb) len".ljust(25), len(ag_16kb_predictions))
    print("alphagenome (1Mb) len".ljust(25), len(ag_1Mb_predictions))
    print("spliceai len".ljust(25), len(spliceai_predictions))
    print("mmsplice len".ljust(25), len(mmsplice_predictions))
    print("full data len".ljust(25), len(full_data_predictions))
    print_first_key("ag_16kb_predictions", ag_16kb_predictions)
    print_first_key("ag_1Mb_predictions", ag_1Mb_predictions)
    print_first_key("spliceai_predictions", spliceai_predictions)
    print_first_key("mmsplice_predictions", mmsplice_predictions)
    print_first_key("full_data_predictions", full_data_predictions)
        
    if len(ag_16kb_predictions) != len(
            set(
                ag_16kb_predictions[check_id_col(ag_16kb_predictions)].tolist()
            )):
        raise ValueError("Alphagenome doesn't have unique variants")
    id_col = check_id_col(ag_16kb_predictions)
    ag_1Mb_predictions = rename_id_col(ag_1Mb_predictions, id_col)
    spliceai_predictions = rename_id_col(spliceai_predictions, id_col)
    mmsplice_predictions = rename_id_col(mmsplice_predictions, id_col)
    full_data_predictions = rename_id_col(full_data_predictions, id_col)

    if "var_id" not in full_data_predictions:
        breakpoint()

    df_merged = (
        ag_16kb_predictions.set_index(check_id_col(ag_16kb_predictions))
        .combine_first(ag_1Mb_predictions.set_index(check_id_col(ag_1Mb_predictions)))
        .combine_first(spliceai_predictions.set_index(check_id_col(spliceai_predictions)))
        .combine_first(mmsplice_predictions.set_index(check_id_col(mmsplice_predictions)))
        .combine_first(full_data_predictions.set_index(check_id_col(full_data_predictions)))
        .reset_index()
    )


    missing_keys = []
    try:
        df_merged["baseline_mmsplice_delta_logit"] = df_merged["Baseline_MMSplice_Predicted_Delta_Logit"]
    except KeyError as e:
        missing_keys.append("Baseline_MMSplice_Predicted_Delta_Logit")
        print(f"missing key {e}")
    try:
        df_merged["retrained_mmsplice_delta_logit"] = df_merged["Retrained_MMSplice_Predicted_Delta_Logit"]
    except KeyError as e:
        missing_keys.append("Retrained_MMSplice_Predicted_Delta_Logit")
        print(f"missing key {e}")
    try:
        df_merged["spliceai_delta_logit"] = df_merged["spliceai_dlogit"]
    except KeyError as e:
        missing_keys.append("spliceai_dlogit")
        print(f"missing key {e}")
    try:
        df_merged["alphagenome_delta_logit"] = df_merged["alphagenome_16kb_dlogit"]
    except KeyError as e:
        missing_keys.append("alphagenome_16kb_dlogit")
        print(f"missing key {e}")
    try:
        df_merged["alphagenome_16kb_delta_logit"] = df_merged["alphagenome_16kb_dlogit"]
    except KeyError as e:
        missing_keys.append("alphagenome_16kb_dlogit")
        print(f"missing key {e}")
    try:
        df_merged["alphagenome_1Mb_delta_logit"] = df_merged["alphagenome_1Mb_dlogit"]
    except KeyError as e:
        missing_keys.append("alphagenome_1Mb_dlogit")
        print(f"missing key {e}")

    if missing_keys:
        raise KeyError(repr(missing_keys))

    print_delta_logit_cols(df_merged)
    df_merged.to_csv(output_path, index=False)
    print("SAVED TO:", output_path)
    
    logging.info("done")

