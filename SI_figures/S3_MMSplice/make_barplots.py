import os
import getopt
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

psi_class = {
    "0-0.2\n(near-excluded)" : (0, 0.2),
    "0.2-0.8\n(intermediate)" : (0.2, 0.8),
    "0.8-1.0\n(near-included)" : (0.8, float("inf")),
}


dlogit_class = {
    "<0.5\n(sub-threshold)" : (0, 0.5),
    "0.5-1.0\n(moderate)" : (0.5, 1.0),
    "1.0-2.0\n(SDV)" : (1.0, 2.0),
    ">2\n(strong SDV)": (2.0, float("inf")),
}


positional_class = {
    "0-4 nt\n(core)" : (0, 4),
    "5-20 nt\n(proximal)" : (5, 20),
    "21-80 nt\n(distal)" : (21, 80),
    "81+ nt\n(distal)" : (81, float("inf")),
    "Double variant" : (100,100),
}


COLOR = {
    "Baseline":  "#7d51f9",  # bright purple (lum≈0.45)
    "Retrained": "#377EB8",  # blue (fixed)
}


def clip(x, clip_threshold=0.01):
    return np.clip(x, clip_threshold, 1 - clip_threshold)


def add_deltas(ax, rt_bars, bl_bars, good="low"):
    for rt_bar, bl_bar in zip(rt_bars, bl_bars):
        rt = rt_bar.get_height()
        bl = bl_bar.get_height()
        delta = rt - bl
        if good == "high":  
            color = "green" if delta > 0 else "red"
        else:
            assert good == "low"
            color = "red" if delta > 0 else "green"
        # place label above the taller bar
        y = max(rt, bl, 0)
        ax.text(
            (rt_bar.get_x() + rt_bar.get_width() / 2 +
             bl_bar.get_x() + bl_bar.get_width() / 2) / 2,
            y,
            f"Δ={delta:.3f}",
            color=color,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    return


def calc_r_with_nan(x, y, model=None, stat="pearson"):
    """
    since we are using the MMSplice data, 
    there shouldnt be any NaNs. However,
    check for those just in case
    """
    if model is None:
        predictions = x
    else:
        predictions = model.predict(x)
    if len(predictions.shape) == 1:
        predictions = np.reshape(predictions, (predictions.shape[0], 1))
    if isinstance(y, pd.core.series.Series):
        y = y.to_numpy()
    if isinstance(x, pd.core.series.Series):
        x = x.to_numpy()
    if isinstance(y, list):
        y = np.array(y).reshape(len(y), 1)
    if len(y.shape) == 1:
        y = np.reshape(y, (y.shape[0], 1))
    if isinstance(x, list):
        x = np.array(x).reshape(len(x), 1)
    if len(x.shape) == 1:
        x = np.reshape(x, (x.shape[0], 1))
    scores = []
    for i in range(y.shape[1]):
        finite = np.logical_and(
            np.isfinite(y[:,i]),
            np.isfinite(x[:,i])
        )
        x_r = x[:,i][finite]
        y_r = y[:,i][finite]
        if stat.lower() == "pearson":
            r_value = stats.linregress(x_r, y_r).rvalue
        elif stat.lower() == "spearman":
            r_value = stats.spearmanr(x_r, y_r).statistic
        else:
            raise ValueError(
                f"Expected stat to be 'pearson' "
                f"or 'spearman' but got '{stat}'"
            )
        scores.append(r_value)
    return np.stack(scores)


def mse(x, y):
    return np.mean((x - y) ** 2)


def distance_from_site(row):
    SA = row["intron1_len"]
    SD = SA + row["exon_len"]
    if ";" in row["snp"]:
        return "Double variant"
    var_loc = int(row["snp"].split(":")[0])
    var_dist = min(
        np.abs(var_loc - SA),
        np.abs(var_loc - SD)
    )
    for p in positional_class:
        l, r = positional_class[p]
        if l <= var_dist <= r:
            return p
    return "NONE"


def transcript_class(row):
    if ";" in row["snp"]:
        return "Double variant"
    var_loc = int(row["snp"].split(":")[0])
    if var_loc < row["intron1_len"]:
        return "intron1"
    elif var_loc < row["intron1_len"] + row["exon_len"]:
        return "exon"
    else:
        return "intron2"


def measurement_class(key):
    if "psi" in key.lower():
        map_ = psi_class
    elif "logit" in key.lower():
        map_ = dlogit_class
    def get_class_row(row):
        value = row[key] 
        for p in map_:
            if map_[p][0] <= np.abs(value) < map_[p][1]:
                return p
        return "NONE"
    return get_class_row
        


def make_barplot(
        predictions,
        task,
        class_map,
        split,
        output_dir=".",
        file_name="test",
        width=0.6,
        baseline_col="Baseline_MMSplice_Predicted_Delta_Logit",
        retrained_col='Retrained_MMSplice_Predicted_Delta_Logit',
        true_col="True_Delta_Logit",
        ylabel="",
        title="",
    ):
    predictions = predictions[predictions["Split"] == split]
    base = predictions[baseline_col]
    retr = predictions[retrained_col]
    true = predictions[true_col]
    bars_bl_mse = []
    bars_bl_r = []
    bars_rt_mse = []
    bars_rt_r = []
    keys = []
    bar_dx = [-width/4, width/4]
    for p in class_map:
        data = predictions[predictions[task] == p]
        n = len(data)
        base = data[baseline_col]
        retr = data[retrained_col]
        true = data[true_col]
        base_r = calc_r_with_nan(base, true)[0]
        base_mse = mse(base, true)
        retr_r = calc_r_with_nan(retr, true)[0]
        retr_mse = mse(retr, true)
        bars_bl_mse.append(base_mse)
        bars_bl_r.append(base_r)
        bars_rt_mse.append(retr_mse)
        bars_rt_r.append(retr_r)
        keys.append(f"{p}\n(n={n})")
    fig, ax = plt.subplots(1, 2, figsize=(13.5,3.3), constrained_layout=True)
    plot_bar_bl_mse = ax[0].bar(
        np.arange(len(bars_bl_mse)) + bar_dx[0],
        bars_bl_mse,
        color=COLOR["Baseline"],
        edgecolor="black",
        width=width/2,
        label="Baseline",
    )
    ax[0].set_ylabel("MSE")
    plot_bar_rt_mse = ax[0].bar(
        np.arange(len(bars_rt_mse)) + bar_dx[1],
        bars_rt_mse,
        color=COLOR["Retrained"],
        edgecolor="black",
        width=width/2,
        label="Retrained",
    )
    plot_bar_bl_r = ax[1].bar(
        np.arange(len(bars_bl_r)) + bar_dx[0],
        bars_bl_r,
        color=COLOR["Baseline"],
        edgecolor="black",
        width=width/2,
        label="Baseline",
    )
    ax[1].set_ylabel("Pearson r")
    plot_bar_rt_r = ax[1].bar(
        np.arange(len(bars_rt_r)) + bar_dx[1],
        bars_rt_r,
        color=COLOR["Retrained"],
        edgecolor="black",
        width=width/2,
        label="Retrained",
    )
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xticks(np.arange(len(bars_rt_r)))
    ax[0].set_xticklabels(keys, fontsize=10, rotation=0)
    ax[1].set_xticks(np.arange(len(bars_rt_r)))
    ax[1].set_xticklabels(keys, fontsize=10, rotation=0)
    ax[0].set_title(f"MSE by {task.replace('_', ' ')}")
    ax[1].set_title(f"Pearson r by {task.replace('_', ' ')}")
    add_deltas(ax[0], plot_bar_rt_mse, plot_bar_bl_mse, good="low")
    add_deltas(ax[1], plot_bar_rt_r, plot_bar_bl_r, good="high")
    # plt.tight_layout()
    output_path = os.path.join(
        output_dir,
        file_name
    )
    plt.savefig(output_path+".png", bbox_inches="tight", dpi=300)
    print("saved:", output_path+".png")
    plt.savefig(output_path+".pdf", bbox_inches="tight", dpi=300)
    print("saved:", output_path+".pdf")
    plt.savefig(output_path+".tif", bbox_inches="tight", dpi=300)
    print("saved:", output_path+".tif")
    plt.close(fig)
    return


help_="""
Makes barplots
panels C, D, E, F
"""
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:],"", [
        "predictions_path=",
        "output_dir=",
        "debug",
        "help",
    ])
    opts = dict(opts)
    if "--help" in opts:
        print(help_)
        sys.exit(0)
    debug = "--debug" in opts
    if debug:
        predictions_path = "/storage/yu/Projects/P000--ESL_continuation/Analysis/model_comparison/model_predictions_GCF.csv"
        output_dir = "/storage/yu/Projects/P000--ESL_continuation/Analysis/Figure_S3"
    else:
        predictions_path = opts["--predictions_path"]
        output_dir = opts["--output_dir"]
        print(opts, flush=True)

    predictions = pd.read_csv(predictions_path)
    split = "Test"
    dlogit_key = "True_Delta_Logit"
    psi_key = "avg_psi_wt"


    predictions["avg_psi_wt"] = predictions[
        [
            "HEK_wt_pooled_psi_raw",
            "HMC3_wt_pooled_psi_raw",
            "HeLa_wt_pooled_psi_raw",
            "K562_wt_pooled_psi_raw",
            "MCF7_wt_pooled_psi_raw",
        ]
    ].apply(clip).mean(axis=1)
    predictions["variant_location"] = predictions.apply(distance_from_site, axis=1)
    predictions["reference_PSI_bin"] = predictions.apply(
        measurement_class(psi_key),
        axis=1
    )
    predictions["SDV_strength"] = predictions.apply(
        measurement_class(dlogit_key),
        axis=1
    )
    predictions["transcript_SNV_location"] = predictions.apply(
        transcript_class,
        axis=1
    )
    class_map = positional_class
    make_barplot(
        predictions,
        "variant_location",
        positional_class,
        split,
        file_name="baseline_vs_retrained_bar_position",
        output_dir=output_dir,
        width=0.6,
        baseline_col="Baseline_MMSplice_Predicted_Delta_Logit",
        retrained_col='Retrained_MMSplice_Predicted_Delta_Logit',
        true_col="True_Delta_Logit",
    )
    make_barplot(
        predictions,
        "SDV_strength",
        dlogit_class,
        split,
        file_name="baseline_vs_retrained_bar_sdv",
        output_dir=output_dir,
        width=0.6,
        baseline_col="Baseline_MMSplice_Predicted_Delta_Logit",
        retrained_col='Retrained_MMSplice_Predicted_Delta_Logit',
        true_col="True_Delta_Logit",
    )
    make_barplot(
        predictions,
        "reference_PSI_bin",
        psi_class,
        split,
        file_name="baseline_vs_retrained_bar_psi",
        output_dir=output_dir,
        width=0.6,
        baseline_col="Baseline_MMSplice_Predicted_Delta_Logit",
        retrained_col='Retrained_MMSplice_Predicted_Delta_Logit',
        true_col="True_Delta_Logit",
    )
    make_barplot(
        predictions,
        "transcript_SNV_location",
        ["intron1", "exon", "intron2"],
        split,
        file_name="baseline_vs_retrained_transcript",
        output_dir=output_dir,
        width=0.6,
        baseline_col="Baseline_MMSplice_Predicted_Delta_Logit",
        retrained_col='Retrained_MMSplice_Predicted_Delta_Logit',
        true_col="True_Delta_Logit",
    )
    logging.info("done")

