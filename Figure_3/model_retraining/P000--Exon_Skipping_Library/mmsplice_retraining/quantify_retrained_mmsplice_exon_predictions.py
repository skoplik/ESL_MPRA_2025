import os
import kerastuner
import pickle
import getopt
import sys
import json
from collections import defaultdict

if __name__ == "__main__":
    MTSPLICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if MTSPLICE_ROOT not in sys.path:
        sys.path.append(MTSPLICE_ROOT)
    PROJECT_ROOT = os.path.split(MTSPLICE_ROOT)[0]
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    PROJECT_DIR = PROJECT_ROOT

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mmsplice_retraining.mmsplice_model.mmsplice_build import (
    get_saved_model as get_saved_mm_splice
)
import scipy.stats as stats
from src.load_data import read_csvs, process_psi_csv, load_splits_dir
import mmsplice_retraining.data_pipeline.make_mmsplice_data
from src.utils.parsing import ErrorWrapper, bool_map


fig_exons = {
"fig3_d":set([
"MYH11 exon 42"
]),

"fig3_e":set([
"LMNA exon 10"
]),

"fig3_f":set([
"SCN5A exon 24",
]),

"fig3_h":set([
"ADA exon 7",
"AFF2 exon 5",
"AKAP9 exon 4",
"AMPD1 exon 2",
"ANK2 exon 36",
"ANK3 exon 23",
"CACNA1C exon 10",
"CACNA1C exon 21",
"CACNA1C exon 30",
"CACNA1C exon 31",
"CASK exon 19",
"DEPDC5 exon 33",
"DEPDC5 exon 6",
"DMD exon 71",
"DRD2 exon 6",
"GNAS exon 3",
"KCNQ2 exon 9",
"MAPT exon 9",
"MEF2C exon 5",
"MEF2C exon 8",
"NF1 exon 31",
"NRXN1 exon 3",
"NRXN1 exon 8",
"PAX5 exon 9",
"PAX6 exon 6",
"PCCA exon 2",
"PCCB exon 3",
"PCCB exon 4",
"SYNGAP1 exon 14",
"TSC2 exon 32",
"TTN exon 224",
]),

"fig4_b":set([
"BIN1 exon 16",
]),

"fig5_c_high_cluster_79":set([
"TIMM44 exon 11",
"CLN3 exon 5",
"DEPDC5 exon 33",
]),

"fig5c_low_cluster_79":set([
"IMPDH1 exon 4",
"COL11A2 exon 6",
"MEF2C exon 5",
"GPI exon 5",
"IL2RA exon 5",
"IVD exon 2",
"HMBS exon 2",
"ELN exon 24",
"NDUFAF5 exon 9",
]),

"fig5c_low_cluster_23":set([
"NPC2 exon 4",
"ASAH1 exon 4",
"DYSF exon 17",
"BCKDHA exon 3",
"BIN1 exon 16",
]),

"fig5d_high_cluser_50":set([
"CFTR exon 5",
"ACADVL exon 3",
"COL11A2 exon 6",
"GFM2 exon 2",
]),

"fig5d_high_cluster_54":set([
"NDUFAF5 exon 9",
"ERCC2 exon 3",
"HMBS exon 6",
"MYBPC1 exon 4",
"COL11A2 exon 7",
"SHOX2 exon 2",
"NEK1 exon 20",
"MUTYH exon 6",
]),

"fig_5d_low_cluster_54" : set([
"L1CAM exon 3",
"MEF2C exon 5",
"HMBS exon 10",
]),

"fig_5d_low_cluster_23" : set([
"NPC2 exon 4",
"DYSF exon 17",
"ASAH1 exon 4",
"BCKDHA exon 3",
"BIN1 exon 16",
]),

"fig_5d_lo2_cluster_61" : set([
"ECEL1 exon 15",
"GH1 exon 3",
"SNCA exon 3",
"DRD2 exon 6",
"ADA exon 7",
"GCDH exon 5",
"TNNT2 exon 5",
]),

"fig_5e_high_cluster_26" : set([
"NEK1 exon 20",
"SCN5A exon 24",
"TNNT2 exon 5",
]),

"fig_5e_low_cluster_26": set([
"ART4 exon 2",
"TTC8 exon 7",
]),

"fig_5e_high_cluster_143" : set([
"SLC26A5 exon 12",
"CACNA2D4 exon 28",
"IMPDH1 exon 7",
"SCN5A exon 24",
"MEFV exon 8",
"MYBPC1 exon 4",
"COLQ exon 5",
"RYR1 exon 94",
"MFAP5 exon 3",
"DEPDC5 exon 33",
"CACNA1C exon 30",
"ECEL1 exon 15",
"COL11A2 exon 7",
]),

"fig_5e_low_cluster_75" : set([
"PAX6 exon 6",
"CHRND exon 3",
"SOS1 exon 21",
]),

"fig5e_low_cluster_118" : set([
"SCN5A exon 24",
"ASAH1 exon 4",
"TIMM44 exon 11",
]),

"fig6b": set([
"MEF2C exon 8",
]),

"fig6c": set([
"MEF2C exon 8",
]),

"fig6d": set([
"DIAPH1 exon 2",
"USP28 exon 6",
"TNNT2 exon 5",
"OCA2 exon 10",
]),

"fig6f": set([
"DMD exon 71",
]),

"fig6h": set([
"UROS exon 3",
"ABCB7 exon 2",
]),
}


def read_model_run_info(model_number, alt_path = ""):
    if alt_path:
        mmsplice_dir = alt_path
    else:
        mmsplice_dir = os.path.join(
            PROJECT_DIR,
            "models",
            "mmsplice_revised",
            "mmsplice" + model_number
        )
    metadata_path = os.path.join(mmsplice_dir, "run_metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        run_metadata = json.load(f)
    run_metadata["mmsplice_dir"] = mmsplice_dir
    return run_metadata


def get_aggregate_data(x_train, x_val, x_test):
    x_full = [None] * len(x_train)
    for i in range(len(x_train)):
        x_full[i] = np.concatenate(
            [x_train[i], x_val[i], x_test[i]],
            axis=0
        )
    return x_full


def quantify_predictions_mse(predictions, baseline, true, threshold):
    """
    quantify exons that have better performance
    above a threshold (%)
    """
    baseline_mse = (baseline - true)**2
    predictions_mse = (predictions - true)**2
    percent_change_error = predictions_mse / baseline_mse
    print(
        f"Number of predictions with MSE less than "
        f"{threshold * 100}% of baseline:",
        np.sum(percent_change_error < threshold)
    )
    print(
        f"{'Percentage of these predictions out of the total:'.ljust(59)} "
        f"{100*np.sum(percent_change_error < threshold)/len(true)}%"
        )
    return
        

def write_ref_nums_to_file(path, data):
    train_df = {}
    val_df = {}
    test_df = {}
    for k in data:
        if "train" in k:
            train_df[(k.split("train")[1])] = data[k]
        elif "val" in k:
            val_df[(k.split("val")[1])] = data[k]
        elif "test" in k:
            test_df[(k.split("test")[1])] = data[k]
        else:
            raise KeyError("Unrecognized dataset:" + k)
    train_df = pd.DataFrame(train_df)
    val_df = pd.DataFrame(val_df)
    test_df = pd.DataFrame(test_df)
    train_df.to_csv(os.path.join(path, "train_refs.csv"), index=False)
    val_df.to_csv(os.path.join(path, "val_refs.csv"), index=False)
    test_df.to_csv(os.path.join(path, "test_refs.csv"), index=False)
    return 


def filter_magnitude(true, threshold):
    """
    filter for predictions with some 
    dlogit variant effect,
    agreeing with the true values
    """
    significant = np.abs(true) > threshold
    # significant = significant * (np.abs(predictions) > threshold)
    insig = np.logical_not(significant)
    return significant, insig


def filter_magnitude_and_predictions(predicted, true, threshold):
    """
    filter for predictions with some 
    dlogit variant effect,
    agreeing with the true values
    """
    significant = np.abs(true) > threshold
    predicted_significant = np.abs(predicted) > threshold
    predicted_insignificant = np.abs(predicted) <= threshold
    # significant = significant * (np.abs(predictions) > threshold)
    insig = np.logical_not(significant)
    return significant, insig, predicted_significant, predicted_insignificant


def quant_plot_thresholded(
        retrained,
        baseline,
        true,
        thresholds,
        plot_dirs=("threshold_mmsplice",),
        plot_name=None,
        plot_metadata=None,
        plot_title=None
    ):
    """
    Makes a plot indicating which predictions
    fall at a certain threshold and have the 
    correct directions
    """
    baseline_correct = (
        (baseline < 0)*(true < 0) 
        + (baseline > 0)*(true > 0) 
        + (baseline == 0)*(true == 0)
    )
    predictions_correct = (
        (retrained < 0)*(true < 0)
        + (retrained > 0)*(true > 0)
        + (retrained == 0)*(true == 0)
    )
    baseline_fp = (baseline > 0) * (true <= 0)
    baseline_fn = (baseline < 0) * (true >= 0)
    retrained_fp = (retrained > 0) * (true <= 0)
    retrained_fn = (retrained < 0) * (true >= 0)
    baseline_tp = (baseline >= 0) * (true >= 0)

    # will override true_positive, 
    # but only at == 0 which will become isignificant
    baseline_tn = (baseline <= 0) * (true <= 0) 
    retrained_tp = (retrained >= 0) * (true >= 0)
    retrained_tn = (retrained <= 0) * (true <= 0)

    out_tsv = {
        "baseline":np.squeeze(baseline), 
        "true":np.squeeze(true),
        "retrained":np.squeeze(retrained),
        "baseline_correct" : np.squeeze(baseline_correct),
        "retrained_correct" : np.squeeze(predictions_correct)
    }
    plot_colors = {
        "insig"   : "tab:grey",
        "reverse" : "tab:red",
        "true"    : "tab:green",
        "false"   : "tab:orange"
    }

    if plot_metadata is not None:
        for key in plot_metadata:
            out_tsv[key] = np.array(plot_metadata[key])

    if plot_title is None:
        plot_title = (
            "MMSplice Test Set Direction "
            "Concordance with Thresholded Values"
        )

    fig, ax = plt.subplots(len(thresholds), 2, figsize = (15,15))
    def get_ax(i):
        if len(thresholds) > 1:
            return ax[i]
        else:
            return ax
    
    for i, threshold in enumerate(thresholds):
        
        (sgnf,
        insig,
        retrained_sg,
        retrained_isg) = filter_magnitude_and_predictions(
            retrained,
            true,
            threshold
        )

        (sgnf,
        insig,
        baseline_sg,
        baseline_isg)  = filter_magnitude_and_predictions(
            baseline,
            true,
            threshold
        )
        
        # TODO: Implement false positive and false negative
        false_pr = (
            (sgnf * retrained_fp)
            + (sgnf * retrained_fn)
            + (insig * retrained_sg)
            + (sgnf * retrained_isg)
        )
        false_base = (
            (sgnf * baseline_fp)
            + (sgnf * baseline_fn)
            + (insig * baseline_sg)
            + (sgnf * baseline_isg)
        )
        true_pr = np.logical_and(
            (sgnf * retrained_tp) + (sgnf * retrained_tn), 
            np.logical_not(false_pr)
        )
        true_base = np.logical_and(
            (sgnf * baseline_tp)  + (sgnf * baseline_tn),
            np.logical_not(false_base)
        )

        insig_base = np.logical_and(
            np.logical_and(insig, baseline),
            np.logical_not(false_base)
        )
        insig_pr = np.logical_and(
            np.logical_and(insig, retrained),
            np.logical_not(false_pr)
        )

        total_pr = (
            np.sum(false_pr)
            + np.sum(true_pr)
            + np.sum(insig_pr)
        )
        total_base = (
            np.sum(false_base)
            + np.sum(true_base)
            + np.sum(insig_base)
        )
        assert total_pr == len(retrained), "total_pr != len(retrained)"
        assert total_base == len(baseline), "total_base != len(baseline)"

        suffix = f"_threshold_{threshold}"
        out_tsv[f"correct_baseline{suffix}"]  = np.squeeze(
            true_base
        )
        out_tsv[f"correct_retrained{suffix}"] = np.squeeze(
            true_pr
        )
        out_tsv[f"wrong_baseline{suffix}"]  = np.squeeze(
            false_base
        )
        out_tsv[f"wrong_retrained{suffix}"] = np.squeeze(
            false_pr
        )
        out_tsv[f"insignificant_baseline{suffix}"] = np.squeeze(
            insig_base
        )
        out_tsv[f"insignificant_retrained{suffix}"] = np.squeeze(
            insig_pr
        )
        
        improved = np.logical_and(true_pr, np.logical_not(true_base))
        worsened = np.logical_and(true_base, np.logical_not(true_pr))
        out_tsv[f"improved_threshold_{threshold}"] = np.squeeze(improved)
        out_tsv[f"worse_threshold_{threshold}"] = np.squeeze(worsened)
        
        improved_percent = np.round(100*np.mean(improved), 3)
        worsened_percent = np.round(100*np.mean(worsened), 3)
        retrained_percent = np.round(100*np.mean(true_pr), 3)
        baseline_percent = np.round(100*np.mean(true_base), 3)
        baseline_incorrect = np.round(100*np.mean(false_base), 3)
        baseline_insignificant = np.round(100*np.mean(insig_base), 3)
        retrained_incorrect = np.round(100*np.mean(false_pr), 3)
        retrained_insignificant = np.round(100*np.mean(insig_pr), 3)

        # make baseline plot
        get_ax(i)[0].scatter(
            x=baseline[true_base],
            y=true[true_base],
            c="tab:green",
            alpha=0.35,
            label=f"correct {baseline_percent}%"
        ) 
        get_ax(i)[0].scatter(
            x=baseline[false_base],
            y=true[false_base],
            c="tab:red",
            alpha=0.35,
            label=f"incorrect {baseline_incorrect}%"
        ) 
        get_ax(i)[0].scatter(
            x=baseline[insig_base],
            y=true[insig_base],
            c="tab:grey",
            alpha=0.35,
            label=f"insignificant {baseline_insignificant}%"
        ) 
        line_kwargs = {
            "color" : "r",
            "linestyle" : "--",
            "label" : "",
        }
        get_ax(i)[0].axvline(x=-threshold, **line_kwargs)
        get_ax(i)[0].axvline(x=threshold, **line_kwargs)
        get_ax(i)[0].axhline(y=-threshold, **line_kwargs)
        get_ax(i)[0].axhline(y=threshold, **line_kwargs)
        get_ax(i)[0].set_xlabel("Baseline Predicted ∆logit")
        get_ax(i)[0].set_ylabel("Observed ∆logit")
        get_ax(i)[0].set_title(
            f"Baseline Correct with {threshold} "
            f"∆logit threshold "
            f"({baseline_percent}%)\n"
            f"({worsened_percent}% correct that is incorrect in retrained)"
        )
        get_ax(i)[0].legend(loc="upper left")

        retrained_green = np.logical_and(true_pr, np.logical_not(improved))
        
        # now do retrained
        get_ax(i)[1].scatter(
            x=retrained[retrained_green],
            y=true[retrained_green],
            c="tab:green",
            alpha=0.35,
            label=f"correct {retrained_percent}%"
        ) 
        get_ax(i)[1].scatter(
            x=retrained[improved],
            y=true[improved],
            c="tab:purple",
            alpha=0.35,
            label=f"improved"
        ) 
        get_ax(i)[1].scatter(
            x=retrained[false_pr],
            y=true[false_pr],
            c="tab:red",
            alpha=0.35,
            label=f"incorrect {retrained_incorrect}%"
        ) 
        get_ax(i)[1].scatter(
            x=retrained[insig_pr],
            y=true[insig_pr],
            c="tab:grey",
            alpha=0.35,
            label=f"insignificant {retrained_insignificant}%"
        )
        get_ax(i)[1].axvline(x=-threshold, **line_kwargs)
        get_ax(i)[1].axvline(x=threshold, **line_kwargs)
        get_ax(i)[1].axhline(y=-threshold, **line_kwargs)
        get_ax(i)[1].axhline(y=threshold, **line_kwargs)
        get_ax(i)[1].set_xlabel("Retrained Predicted ∆logit")
        get_ax(i)[1].set_ylabel("Observed ∆logit")
        get_ax(i)[1].set_title(
            f"Retrained Correct with {threshold} "
            f"∆logit threshold "
            f"({retrained_percent}%)\n"
            f"({improved_percent}% correct that is incorrect in baseline)"
        )
        get_ax(i)[1].legend(loc="upper left")

    fig.suptitle(plot_title, y = 0.99) 
    plt.tight_layout()
    if plot_name is None:
        plot_name = ""
    out_tsv = pd.DataFrame(out_tsv)
    for plot_dir in plot_dirs:
        plot_path = os.path.join(
            plot_dir,
            plot_name
        )
        plt.savefig(plot_path + ".pdf")
        plt.savefig(plot_path + ".png")
        out_tsv.to_csv(
            plot_path + ".tsv",
            sep="\t",
            index=False
        )
    return out_tsv


def quantify_predictions_direction(predictions, baseline, true, threshold):
    baseline_correct = (baseline < 0) * (true < 0) + (baseline > 0) * (true > 0) + (baseline == 0) * (true == 0)
    predictions_correct = (predictions < 0) * (true < 0) + (predictions > 0) * (true > 0) + (predictions == 0) * (true == 0)
    N = len(true)
    predictions_correct = np.logical_and(predictions_correct, filter_magnitude(true, threshold)[0])
    baseline_correct = np.logical_and(baseline_correct, filter_magnitude(true, threshold)[0])
    num_baseline_correct = np.sum(baseline_correct)
    num_predictions_correct = np.sum(predictions_correct)
    percent_baseline_correct = round(100 * (num_baseline_correct / N), 4)
    percent_predictions_correct = round(100 * (num_predictions_correct / N), 4)
    print(f"Threshold ∆logit = {threshold}")
    print(f"number of correct directions (baseline): {num_baseline_correct} ({percent_baseline_correct})%")
    print(f"number of correct directions (retrained): {num_predictions_correct} ({percent_predictions_correct})%")
    corrected = np.logical_and(predictions_correct, np.logical_not(baseline_correct))
    worse = np.logical_and(baseline_correct, np.logical_not(predictions_correct))
    num_worse = np.sum(worse)
    percent_worse = round(100 * num_worse / N, 4)
    same = np.logical_not(np.logical_xor(predictions_correct, baseline_correct))
    num_same = np.sum(same)
    percent_same = round(100 * num_same / N, 4)
    num_corrected = np.sum(corrected)
    percent_corrected = round(100 * num_corrected / N, 4)
    print(f"number corrected (wrong in baseline, correct retrained): {num_corrected} ({percent_corrected})%")
    print(f"number worse (wrong in retrained, correct baseline): {num_worse} ({percent_worse})%")
    print(f"number same (either correct or wrong in both): {num_same} ({percent_same})%")
    return corrected, worse, same


def map_variants_info(merged_psi, ref_nums):
    ref_info = {}
    for i in range(len(merged_psi)):
        row = merged_psi.iloc[i]
        ref = int(row["ref_id"])
        ref_info[ref] = {}

        ref_info[ref]["gene_exon"]  = row["gene_exon"]
        ref_info[ref]["snp"] = row["snp"]
        ref_info[ref]["event_id"] = row["event_id"]
    return ref_info


def search_gene_exon(gene_exon):
    """
    annotates exon if it is part of 
    a figure in the paper
    """
    hits = set()
    for fig in fig_exons:
        if gene_exon in fig_exons[fig]:
            hits.add(fig)
    return hits


def get_loaded_pickle(data_dir, point_file = "x_train.p", extension=None):
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


def is_double(x):
    return ";" in x["snp"]


help_="""
Script to quantify which exons
have better predictions with the 
retrained MMSplice model in the 
held-out dataset

'Better' is defined as having 
an improved MSE with the true
delta logit (variant logit - ref logit) 

"""
inputs_ = [
"model_number=",
"data_dir=",
"write_ref_nums=",
"ref_num_output_dir=",
"write_quant_file=",
"output_quant_file=",
"old_predictions_path=",
"hyperparameter_run=",
"plot_dir=",
"plot_name=",
"thresholds=",
"split=",
"drop_double_vars=",
"help",
"debug"
]
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:],"", inputs_)
    opts = dict(opts)

    if "--help" in opts:
        print(help_)
        sys.exit(1)
    if "--debug" in opts:
        data_dir = os.path.join(
            PROJECT_DIR,
            "models",
            "data",
            "mmsplice14"
        )
        mmsplice_dir = os.path.join(
            PROJECT_DIR,
            "models",
            "mmsplice_revised",
            "mmsplice139",
        )
        write_ref_nums = False
        write_quant_output = False
        output_quant_file = "corrected_mmsplice_exons.txt"
        plot_dir = os.path.join(
            PROJECT_DIR,
            "models",
            "figures",
            "quantify_outputs_mmsplice"
        )
        plot_dir2 = os.path.join(
            PROJECT_DIR,
            "models",
            "mmsplice_revised",
            "mmsplice139",
            "figures"
        )
        plot_name = "threshold_mmsplice_model_139"
        thresholds = [0.2, 1.0]
        split = "Test"
        drop_double_vars = True
        debug = True
    else:
        opts = ErrorWrapper(opts, max_printout_size=300) 
        model_number = opts["--model_number"]
        mmsplice_dir = opts.get("--mmsplice_dir", "")
        run_info = read_model_run_info(model_number, mmsplice_dir)
        data_dir = run_info["data_dir"]
        figures_dir = run_info["figures_dir"]
        mmsplice_dir = run_info["mmsplice_dir"]
        plot_dir = opts.get("--plot_dir", None)
        split = opts.get("--split", "Test")
        drop_double_vars = bool_map(opts.get("--drop_double_vars", False))

        if "--data_dir" in opts:
            data_dir = opts["--data_dir"]
        if plot_dir is not None:
            plot_dirs = [plot_dir, figures_dir]
        else:
            plot_dir = os.path.join(
                f"{PROJECT_DIR}",
                "models",
                "figures",
                "quantify_outputs_mmsplice"
            )
            plot_dirs = [plot_dir, figures_dir]

        plot_name = opts.get("--plot_name", "")
        if not plot_name:
            plot_name = f"threshold_mmsplice_model_{model_number}_{split}"
    
        thresholds = [float(i) for i in opts["--thresholds"].split(",")]
        debug = False
        opts.check_inputs()

    print(opts)


    # if plot_dir is None or not plot_dir:
        # plot_dir = os.path.join(mmsplice_dir, "figures")
    # if plot_name is None or not plot_name:
        # model_number = run_metadata["model_number"]
        # plot_name = f"threshold_mmsplice_model_{model_number}"

    # data_dir = run_metadata["data_dir"]
    metadata_path = os.path.join(data_dir, "run_metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data_info = json.load(f)

    merged_df_path = data_info["merged_df_path"]
    merged_df = pd.read_csv(merged_df_path)
    predictions = pd.read_csv(os.path.join(mmsplice_dir, "predictions.csv"))
    if splitlower() != "full":
        test_set = predictions[predictions["Split"] == split]
    else:
        test_set = predictions

    _, metadata = load_splits_dir(data_dir, extension=".csv")
    try:
        metadata_test = metadata["metadata_" + split.lower()]
    except KeyError as e:
        if split == "full":
            metadata_test = pd.concat([
                metadata["metadata_train"],
                metadata["metadata_val"],
                metadata["metadata_test"]
            ], ignore_index=True)
        else:
            raise FileNotFoundError(f"Metadata does not exist for split '{split}'")

    if drop_double_vars:
        is_double_vec = metadata_test.apply(is_double, axis=1)
        is_single_vec = np.logical_not(is_double_vec)
        print("metadata len before filter:", len(metadata_test))
        print("test_set len before filter:", len(test_set))
        
        metadata_test = metadata_test[is_single_vec]
        test_set = test_set.reset_index()[is_single_vec]
            
        print("metadata len after filter:", len(metadata_test))
        print("test_set len after filter:", len(test_set))
        assert (metadata_test["var_id"].values == test_set["Variant_ID"].values).mean() == 1.0

    test_predictions = test_set["Retrained_MMSplice_Predicted_Delta_Logit"]
    y_test_logit_baseline = test_set["Baseline_MMSplice_Predicted_Delta_Logit"]
    y_test_logit_mmsplice = test_set["True_Delta_Logit"]

    if debug:
        breakpoint()

    baseline_r_value = stats.linregress(
        y_test_logit_baseline,
        y_test_logit_mmsplice
    ).rvalue

    retrained_r_value = stats.linregress(
        test_predictions,
        y_test_logit_mmsplice
    ).rvalue

    baseline_mse = np.mean(
        (y_test_logit_baseline - y_test_logit_mmsplice)**2
    ) 
    retrained_mse = np.mean(
        (test_predictions - y_test_logit_mmsplice)**2
    ) 
    
    print(
    f"""
Baseline {split} R:    {baseline_r_value}
Retrained {split} R:   {retrained_r_value}

Baseline {split} MSE:  {baseline_mse}
Retrained {split} MSE: {retrained_mse}
    """
    )

    corrected = {}
    worse = {}
    same = {}
    # quantify_predictions_mse(test_predictions, y_test_logit_baseline, y_test_logit_mmsplice, 0.9)
    # quantify_predictions_mse(test_predictions, y_test_logit_baseline, y_test_logit_mmsplice, 0.1)
    # for threshold in thresholds:
        # c, w, s = quantify_predictions_direction(test_predictions, y_test_logit_baseline, y_test_logit_mmsplice, threshold)
        # print()
        # corrected[threshold] = c
        # worse[threshold] = w
        # same[threshold] = s

    # test_var_ids = metadata_test["var_id"].astype(int).tolist()
    test_ref_ids = metadata_test["ref_id"].astype(int).tolist()
    test_event_ids = metadata_test["event_id"].tolist()
    # ref_num_info = map_variants_info(merged_df, test_var_ids)

    plot_metadata = { "refs"       : test_ref_ids,
                      "event_ids"  : test_event_ids,}

    if debug:
        print(plot_dir)
        print(plot_dir2)
        breakpoint()
        " making quant plot "
        plot_dirs = (plot_dir, plot_dir2)

    plot_title = f"MMSplice {split.title()} Set Direction Concordance with Thresholded Values"

    out_tsv = quant_plot_thresholded(
        test_predictions,
        y_test_logit_baseline,
        y_test_logit_mmsplice, [0.2,1.0],
        plot_dirs=plot_dirs,
        plot_name = plot_name,
        plot_metadata = plot_metadata,
        plot_title = plot_title)

    improved_exons_tsv = {"variant_id":[],
                          "event_id":[],
                          "snp":[],
                          "gene_exon":[],
                          "figure":[],
                            }

    if debug:
        breakpoint()
        " debug per exon info "

    for threshold in thresholds:
        corrected[threshold] = out_tsv[f"improved_threshold_{threshold}"].tolist()
        worse[threshold] = out_tsv[f"worse_threshold_{threshold}"].tolist()
        same[threshold] = np.logical_not(np.logical_xor(corrected[threshold], worse[threshold]))

    eid_col = "event_id_161" if "event_id_161" in metadata_test else "event_id"
    percent_by_exon = {}
    gene_exon_to_eid = defaultdict(set)
    total_variants_by_exon = {}
    metadata_improved = defaultdict(list)
    for i, row in metadata_test.iterrows():
        event_id = row[eid_col]
        snp = row["snp"]
        gene_exon = row["gene_exon"]
        var_id = row["var_id"]
        gene_exon_to_eid[gene_exon].add(event_id)

        if gene_exon in total_variants_by_exon:
            total_variants_by_exon[gene_exon] += 1
        else:
            total_variants_by_exon[gene_exon] = 1
            percent_by_exon[gene_exon] = {}
            for threshold in thresholds:
                percent_by_exon[gene_exon][threshold] = {"corrected":0, "worse":0, "same":0}
        for threshold in thresholds:
            if corrected[threshold][i]:
                percent_by_exon[gene_exon][threshold]["corrected"] += 1
                metadata_improved[f"threshold_{threshold}_status"].append("corrected")
            elif worse[threshold][i]:
                percent_by_exon[gene_exon][threshold]["worse"] += 1
                metadata_improved[f"threshold_{threshold}_status"].append("worse")
            else:
                percent_by_exon[gene_exon][threshold]["same"] += 1
                metadata_improved[f"threshold_{threshold}_status"].append("same")
    
    for i in metadata_improved:
        metadata_test[i] = metadata_improved[i]

    for i in test_set:
        if i not in metadata_test:
            metadata_test[i] = test_set.reset_index()[i]

    for i in range(len(plot_dirs)):
        metadata_test.to_csv(
            f"{plot_dirs[0]}/{plot_name}_{split.lower()}_all_predictions.tsv",
            sep='\t',
            index=False
        )
        print(
            "saved metadata + improved + predictions:",
            f"{plot_dirs[0]}/{plot_name}_{split.lower()}_all_predictions.tsv"
        )

    improved_exons_tsv = defaultdict(list)
    print("\nNumber of corrected variants by exon (test dataset)")
    for exon in percent_by_exon:
        improved_exons_tsv["gene_exon"].append(exon)
        improved_exons_tsv["num_variants"].append(total_variants_by_exon[exon])
        for threshold in thresholds:
            corrected_ = percent_by_exon[exon][threshold]["corrected"]
            worse_ = percent_by_exon[exon][threshold]["worse"]
            same_ = percent_by_exon[exon][threshold]["same"]
            total_ = corrected_ + worse_ + same_
            improved_exons_tsv[f"corrected_threshold_{threshold}"].append(corrected_)
            improved_exons_tsv[f"worse_threshold_{threshold}"].append(worse_)
            improved_exons_tsv[f"same_threshold_{threshold}"].append(same_)

            improved_exons_tsv[f"%corrected_threshold_{threshold}"].append(np.round(100 * corrected_ / total_, 3))
            improved_exons_tsv[f"%worse_threshold_{threshold}"].append(np.round(100 * worse_ / total_, 3))
            improved_exons_tsv[f"%same_threshold_{threshold}"].append(np.round(100 *same_ / total_, 3))

        improved_exons_tsv[eid_col].append(",".join(list(gene_exon_to_eid[exon])))

        hits = search_gene_exon(exon)
        improved_exons_tsv["figure"].append(",".join(list(hits)))

        print(f"{exon} ({total_variants_by_exon[exon]} total variants):")
        for threshold in percent_by_exon[exon]:
            line = f"threshold {threshold}: "
            for result in percent_by_exon[exon][threshold]:
                n = percent_by_exon[exon][threshold][result]
                percent = np.round(100 * n / total_variants_by_exon[exon], 3)
                line += f"{n} {result} ({percent}%) ".ljust(20)
            print(line)
        if hits:
            print("found in the following figures:")
            for hit in hits:
                print(hit) 
        print()
 

    if debug:
        breakpoint()


    improved_exons_tsv = pd.DataFrame(improved_exons_tsv)
    for i in range(len(plot_dirs)):
        improved_exons_tsv.to_csv(f"{plot_dirs[0]}/{plot_name}_{split.lower()}_improved_exon_info.tsv", sep='\t', index=False)
        improved_exons_tsv.to_csv(f"{plot_dirs[1]}/{plot_name}_{split.lower()}_improved_exon_info.tsv", sep='\t', index=False)
    print("done")
             


