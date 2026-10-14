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


from tensorflow.keras.models import load_model
import scipy.stats as stats

project_name = "P000--Exon_Skipping_Library"
node = os.getcwd()
for i in range(5): # max depth
    tree, node = os.path.split(node)
    if node == project_name:
        break
    else:
        node = tree
PROJECT_ROOT = os.path.join(tree, node)
if node != project_name:
    # couldn't find project folder 
    raise ModuleNotFoundError(
        f"Could not find project directory {project_name}"
    )
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.esl import (
    logit,
    clip, 
    expit,
    CITRINE_EXON1,
    CITRINE_EXON2,
    SMN2_INTRON6,
    SMN2_INTRON7,
    cell_types,
    cell_types2
)
from src.load_data import load_splits_dir


def go_up_n_dir(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path


def load_spliceai(pre_path):
    model1 = load_model(os.path.join(pre_path, 'models', 'spliceai1.h5'))
    model2 = load_model(os.path.join(pre_path, 'models', 'spliceai2.h5'))
    model3 = load_model(os.path.join(pre_path, 'models', 'spliceai3.h5'))
    model4 = load_model(os.path.join(pre_path, 'models', 'spliceai4.h5'))
    model5 = load_model(os.path.join(pre_path, 'models', 'spliceai5.h5'))
    return [model1, model2, model3, model4, model5]


def get_seq_combiner():
    context = "N" * 5000
    left_context = [
        context,
        CITRINE_EXON1,
        SMN2_INTRON6,
    ]
    right_context = [
        SMN2_INTRON7,
        CITRINE_EXON2,
        context,
    ]
    def combine_seq(row):
        full = left_context + [
            row["intron1"],
            row["exon"],
            row["intron2"],
        ] + right_context
        return "".join(full)
    def combine_ref_seq(row):
        full = left_context + [
            row["intron1_ref"],
            row["exon_ref"],
            row["intron2_ref"],
        ] + right_context
        return "".join(full)
    return {"var" : combine_seq, "ref" : combine_ref_seq}


def get_sequences(merged_df):
    """
    get sequences from merged data
    by combining intron1, exon and
    intron2 with SMN2 introns and
    Citrine exons.
    """
    seq_combiner = get_seq_combiner()
    merged_df["full"] = merged_df.apply(seq_combiner["var"], axis=1)
    merged_df["full_ref"] = merged_df.apply(seq_combiner["ref"], axis=1)
    var_lens = {}
    ref_lens = {}
    for i in ("intron1", "exon", "intron2"):
        var_lens[i] = merged_df[i].apply(len).tolist()
        ref_lens[i] = merged_df[i + "_ref"].apply(len).tolist()
    return (
        merged_df["full"].tolist(),
        merged_df["full_ref"].tolist(),
        var_lens,
        ref_lens
    )

            
def predict_sequences(
        var_seqs,
        ref_seqs,
        var_lens,
        ref_lens,
        splcieai_models,
    ):

    # var_seqs = ["N" * 5000 + CITRINE_EXON1 + SMN2_INTRON6 + i + SMN2_INTRON7 + CITRINE_EXON2 + "N" * 5000 for i in var_seqs]
    # ref_seqs = ["N" * 5000 + CITRINE_EXON1 + SMN2_INTRON6 + i + SMN2_INTRON7 + CITRINE_EXON2 + "N" * 5000 for i in ref_seqs]

    spl_pr = []
    var_one_hots = []
    ref_one_hots = []


    # one hot encode sequences
    for seq in var_seqs:
        var_one_hots.append(one_hot_encode(seq)[None, :])
    for seq in ref_seqs:
        ref_one_hots.append(one_hot_encode(seq)[None, :])
    var_one_hots = np.concatenate(var_one_hots)
    ref_one_hots = np.concatenate(ref_one_hots)

    # get spliceai predictions
    logging.info("SpliceAI variant predictions")
    y_var = np.mean(
        [
            spliceai_models[m].predict(var_one_hots, batch_size=96, verbose=2)
            for m in range(5)
        ],
        axis=0
    )
    logging.info("SpliceAI reference predictions")
    y_ref = np.mean(
        [
            spliceai_models[m].predict(ref_one_hots, batch_size=96, verbose=2)
            for m in range(5)
        ],
        axis=0
    )


    # get splice site specific predictions for donor and acceptor
    df = {"predicted_var_sa":[], "predicted_var_sd":[],
                "predicted_ref_sa":[], "predicted_ref_sd":[],
                "predicted_dlogit":[]
    }
                
    pr_var_logit = np.zeros((len(y_var),))
    for i, val in enumerate(y_var):
        SA = len(CITRINE_EXON1) + len(SMN2_INTRON6) + var_lens["intron1"][i]
        SD = SA + var_lens["exon"][i] - 1
        acc = y_var[i, SA, 1]
        donor = y_var[i, SD, 2]
        assert not np.isnan(acc), i
        assert not np.isnan(donor), i
        pr_var_logit[i] = logit(acc * donor, clip_threshold=0.01)
        df["predicted_var_sa"].append(acc)
        df["predicted_var_sd"].append(donor)
    pr_ref_logit = np.zeros((len(y_ref),))
    for i, val in enumerate(y_ref):
        SA = len(CITRINE_EXON1) + len(SMN2_INTRON6) + ref_lens["intron1"][i]
        SD = SA + ref_lens["exon"][i] - 1
        acc = y_ref[i, SA, 1]
        donor = y_ref[i, SD, 2]
        assert not np.isnan(acc), i
        assert not np.isnan(donor), i
        pr_ref_logit[i] = logit(acc * donor, clip_threshold=0.01)
        df["predicted_ref_sa"].append(acc)
        df["predicted_ref_sd"].append(donor)

    # transform to dlogit predictions
    pr_dlogit = pr_var_logit - pr_ref_logit
    df["predicted_dlogit"] = pr_dlogit
    try:
        df = pd.DataFrame(df)
    except ValueError as e:
        print(e)
        for i in df:
            print(i, len(df[i]))
        breakpoint()
    return df


def copy_metadata(df, merged_df,
                  skip_keys=None):
    """
    copies information from merged_df
    to df, ignoring certain columns.
    """
    if skip_keys is None:
        skip_keys = set(
            ["full", "full_ref"]
        )
    elif isinstance(skip_keys, list):
        skip_keys = set(skip_keys)
    cell_types_ = set(cell_types)
    for i in merged_df:
        if i in skip_keys:
            continue
        if i.split("_")[0] in cell_types_:
            continue
        df[i] = merged_df[i]
    return df
        
        
def plot_predictions(df, output_dir, save=True, split="", debug=False):
    x = df["predicted_dlogit"]
    y = df["experimental_dlogit"]
    x_r = x[np.isfinite(y)]
    y_r = y[np.isfinite(y)]
    try:
        r = stats.linregress(x_r, y_r).rvalue
        sr = stats.spearmanr(x_r, y_r)[0]
    except ValueError:
        r = 0.0
        sr = 0.0
    plt.scatter(x = x_r, y = y_r, alpha = 0.2)
    plt.xlabel(
        "SpliceAI Predictions ΔLogit (SA×SD)\n"
        "(1Mb context)"
    )
    plt.ylabel("Oberserved Mean Cell ΔLogit")
    if split:
        split_title = f" {split.title()} Set"
        split_save = f"_{split}"
    else:
        split_title = ""
        split_save = ""
    plt.title(
        f"SpliceAI Variant Effect Prediction on COMPASS{split_title}\n"
        f"n={len(x_r)}, r={round(r,3)}, spearman={round(sr,3)}"
    )
    lim = max(abs(x_r).max(), abs(y_r).max()) * 1.1
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.axhline(0, color='gray', lw=0.5); plt.axvline(0, color='gray', lw=0.5)
    plt.tight_layout()
    path = f"spliceai_predicted_dlogit{split_save}.pdf"

    if save:
        path = os.path.join(output_dir, path)
        print(f"Saved to: {path}")
        plt.savefig(f"{path}.pdf")
        plt.savefig(f"{path}.png")
        
    plt.show()
    plt.close()
    return


def filter_to_metadata(df, m, split):
    """
    Find ref-var pairs in df
    that are in the metadata with a 
    given split. 
    """
    m_ids = set(zip(m["ref_id"], m["var_id"]))
    indexes = []
    for i, row in df.iterrows():
        sample = (row["ref_id"], row["var_id"]) 
        if sample in m_ids:
            indexes.append(i) 
    return df.iloc[indexes]


help_="""
Predictions for SpliceAI

spliceai_path : 
    path containing spliceai models and one hot
    encoding code
merged_df_path : 
    path for the merged dataframe containing
    samples with matching ref var pairs
output_dir : 
    output directory to save predictions csv file
    and plots for each split.
mmsplice_data_dir : 
    optional argument, but necessary for 
    predicting splicing outcomes on a specific
    train/val/test split
"""
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:],"", [
        "spliceai_path=",
        "merged_df_path=",
        "output_dir=",
        "mmsplice_data_dir=",
        "help",
        "debug",
    ])
    opts = dict(opts)
    if "--help" in opts:
        print(help_)
        sys.exit(0)
    debug = "--debug" in opts
    if debug:
        projects = os.path.join(
            "/storage",
            "yu",
            "Projects",
        )
        spliceai_path = os.path.join(
            projects,
            "model_repository",
            "SpliceAI",
            "spliceai",
        )
        merged_df_path = os.path.join(
            projects,
            "P000--ESL_continuation",
            "Figures_SK",
            "General_preprocessing",
            "output_5_23_2026",
            "merged_05_23_2026.csv",
        )
        output_dir = os.path.join(
            PROJECT_ROOT,
            "spliceai_and_ag_testing",
            "spliceai_predictions",
            "debug",
        )
        mmsplice_data_dir = os.path.join(
            PROJECT_ROOT,
            "models",
            "data",
            "mmsplice14",
        )
    else:
        spliceai_path = opts["--spliceai_path"]
        merged_df_path = opts["--merged_df_path"]
        output_dir = opts["--output_dir"]
        mmsplice_data_dir = opts.get("--mmsplice_data_dir", "")

    print(opts)


    merged_df = pd.read_csv(merged_df_path)
    if debug:
        merged_df = merged_df.iloc[range(5)]
    logging.info(f"read merged_df from {merged_df_path}")
    
    sys.path.append(spliceai_path)
    from utils import one_hot_encode
    logging.info(f"loading spliceai models from {spliceai_path}")
    spliceai_models = load_spliceai(spliceai_path)

    logging.info("building sequences")
    var_seqs, ref_seqs, var_lens, ref_lens = get_sequences(merged_df)
    logging.info("predicting sequences")
    df = predict_sequences(
        var_seqs, ref_seqs,
        var_lens, ref_lens,
        spliceai_models
    )
    df = copy_metadata(df, merged_df,
        skip_keys = ["full", "full_ref"])

    df = df.rename(columns={"mean_dlogit" : "experimental_dlogit"})
    df.to_csv(os.path.join(
        output_dir,
        "spliceai_predictions_compass.csv",
    ), index=False) 
    plot_predictions(
        df,
        output_dir,
        save=True,
    )
    if not mmsplice_data_dir:
        logging.warning(
            "Skipping plotting for train/val/test "
            "due to empty or missing mmsplice_data_dir input"
        )
        sys.exit(0)

    if debug:
        df = pd.read_csv("/storage/yu/Projects/P000--ESL_continuation/GitHub/P000--Exon_Skipping_Library/spliceai_and_ag_testing/spliceai_predictions/spliceai_predictions_compass.csv")
        
    # load metadata to get splits
    loaded, metadata = load_splits_dir(mmsplice_data_dir, extension=".csv")
    
    for dataset in ("train", "val", "test"):
        m = metadata[f"metadata_{dataset}"]
        df_split = filter_to_metadata(df, m, dataset)
        plot_predictions(
            df_split,
            output_dir, 
            save=True,
            split=dataset,
        )
    
    logging.info("done")

