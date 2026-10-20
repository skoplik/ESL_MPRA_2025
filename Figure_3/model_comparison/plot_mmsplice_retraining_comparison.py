import os
import getopt
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import datetime
import seaborn as sns
import scipy.stats as stats
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def calc_r(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value

def comma_separate(num):
    num_str = str(num)
    comma_str = []
    n = 1
    for i in range(len(num_str) -1, 0, -1):
        comma_str.append(num_str[i])
        if i != len(num_str) - 1 and n % 3 == 0:
            comma_str.append(",")
        n += 1
    comma_str.append(num_str[0])
    return "".join(comma_str[::-1])




COLOR = {
    "Baseline":  "#7d51f9",  # bright purple (lum≈0.45)
    "Retrained": "#377EB8",  # blue (fixed)
}
dlogit_key = 'Δlogit(PSI)'
alpha=0.2
s=3


help_="""
Makes panel D, E
comparison of the MMSplice retrained model
vs baseline on held-out test set.
"""
inputs = [
    "predictions_path=",
    "output_dir=",
    "alpha=",
    "s=",
    "debug",
    "help",
]
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "", inputs)
    opts = dict(opts)
    if "--help" in opts:
        print(help_)
        sys.exit(0)
    debug = "--debug" in opts
    if debug:
        predictions_path = "/storage/yu/Projects/P000--ESL_continuation/GitHub/P000--Exon_Skipping_Library/models/mmsplice_revised/mmsplice139/predictions.csv"
        output_dir = "/storage/yu/Projects/P000--ESL_continuation/Analysis/model_comparison/mmsplice_retrained"
    else:
        predictions_path = opts["--predictions_path"]
        output_dir = opts["--output_dir"]
        alpha_input = opts.get("--alpha", "")
        if alpha_input:
            alpha = float(alpha_input)
        s_input = opts.get("--s", "")
        if s_input:
            s = int(s_input)
        print(opts, flush = True)

    def save_plot(plot_name):
        plot_path = os.path.join(
            output_dir,
            plot_name
        )
        plt.savefig(
            plot_path + ".pdf",
            bbox_inches="tight",
            pad_inches="layout"
        )
        print("Saved:", plot_path + ".pdf")
        plt.savefig(
            plot_path + ".png",
            dpi=300,
            bbox_inches="tight",
            pad_inches="layout"
        )
        print("Saved:", plot_path + ".png")
        plt.savefig(
            plot_path + ".tif",
            dpi=300,
            bbox_inches="tight",
            pad_inches="layout"
        )
        print("Saved:", plot_path + ".tif")
        return

    date = (
        f"{datetime.datetime.now().month}_"
        f"{datetime.datetime.now().day}"
    )
    predictions = pd.read_csv(predictions_path)

    base = predictions['Baseline_MMSplice_Predicted_Delta_Logit']
    true = predictions["True_Delta_Logit"]

    n = comma_separate(len(true))
    r_val = calc_r(base, true)
    fig, ax = plt.subplots(figsize=(3.3,3.3), constrained_layout=True)
    sns.scatterplot(x = base, y = true, alpha=alpha, ax=ax, s=s, color=COLOR["Baseline"])
    ax.set_xlabel('Predicted Δlogit(PSI)')
    ax.set_ylabel('Measured Δlogit(PSI)')
    ax.set_title(
        f"Baseline MMSplice {dlogit_key}\n"
        f"predictions on aggregate data\nn={n}, r={round(r_val, 3)}"
    )
    ax.plot((-11, 11), (-11, 11), "--", color="gray", lw=0.5, zorder=0)
    ax.set_xlim((-11, 11))
    ax.set_ylim((-11, 11))
    save_plot(f"mmsplice_baseline_full_remade_{date}")


    retr = predictions['Retrained_MMSplice_Predicted_Delta_Logit']
    true = predictions["True_Delta_Logit"]

    r_val = calc_r(retr, true)
    fig, ax = plt.subplots(figsize=(3.3,3.3), constrained_layout=True)
    sns.scatterplot(x = retr, y = true, alpha=alpha, ax=ax, s=s, color=COLOR["Retrained"])
    ax.set_xlabel('Predicted Δlogit(PSI)')
    ax.set_ylabel('Measured Δlogit(PSI)')
    ax.set_title(f"Retrained MMSplice {dlogit_key}\npredictions on aggregate data\nn={n}, r={round(r_val, 3)}")
    ax.plot((-11, 11), (-11, 11), "--", color="gray", lw=0.5, zorder=0)
    ax.set_xlim((-11, 11))
    ax.set_ylim((-11, 11))
    save_plot(f"mmsplice_retrained_full_remade_{date}")


    

    for split in ["Train", "Val", "Test"]:
        base = predictions[predictions["Split"] == split]['Baseline_MMSplice_Predicted_Delta_Logit']
        true = predictions[predictions["Split"] == split]["True_Delta_Logit"]
        n = comma_separate(len(true))
        r_val = calc_r(base, true)
        fig, ax = plt.subplots(figsize=(3.3,3.3), constrained_layout=True)
        sns.scatterplot(x = base, y = true, alpha=alpha, ax=ax, s=s, color=COLOR["Baseline"])
        ax.set_xlabel('Predicted Δlogit(PSI)')
        ax.set_ylabel('Measured Δlogit(PSI)')
        ax.set_title(
            f"Baseline MMSplice {dlogit_key} on {split.lower()} split\n"
            f"n={n}, r={round(r_val, 3)}"
        )
        # plt.axhline(0, color='gray', lw=0.5); plt.axvline(0, color='gray', lw=0.5)
        ax.plot((-11, 11), (-11, 11), "--", color="gray", lw=0.5, zorder=0)
        ax.set_xlim((-11, 11))
        ax.set_ylim((-11, 11))
        save_plot(f"mmsplice_baseline_{split.lower()}_remade_{date}.pdf")


    

    for split in ["Train", "Val", "Test"]:
        retr = predictions[predictions["Split"] == split]['Retrained_MMSplice_Predicted_Delta_Logit']
        true = predictions[predictions["Split"] == split]["True_Delta_Logit"]
        n = comma_separate(len(true))
        r_val = calc_r(retr, true)
        fig, ax = plt.subplots(figsize=(3.3,3.3), constrained_layout=True)
        sns.scatterplot(x = retr, y = true, alpha=alpha, ax=ax, s=s)
        ax.plot((-11, 11), (-11, 11), "--", color="gray", lw=0.5, zorder=0)
        ax.set_xlim((-11, 11))
        ax.set_ylim((-11, 11))
        ax.set_xlabel('Predicted Δlogit(PSI)')
        ax.set_ylabel('Measured Δlogit(PSI)')
        ax.set_title(
            f"Retrained MMSplice {dlogit_key}\n"
            f"predictions on {split.lower()} split\n"
            f"n={n}, r={round(r_val, 3)}"
        )
        save_plot(f"mmsplice_retrained_{split.lower()}_remade_{date}")



    
    fig, ax = plt.subplots(3, 2, figsize=(12.4/2.54,12.4/2.54), constrained_layout=True)
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    for i, split in enumerate(["Train", "Val", "Test"]):
        for j, task in enumerate(["Baseline", "Retrained"]):
            retr = predictions[predictions["Split"] == split][f'{task}_MMSplice_Predicted_Delta_Logit']
            true = predictions[predictions["Split"] == split]["True_Delta_Logit"]
            r_val = calc_r(retr, true)
            sns.scatterplot(x = retr, y = true, alpha=alpha, ax=ax[i][j], s=s, color=COLOR[task])
            n = comma_separate(len(true))
            if i == 0:        
                ax[i][j].set_title(f"{task} MMSplice \nn={n}, r={round(r_val, 3)}")
            else:
                ax[i][j].set_title(f"n={n}, r={round(r_val, 3)}")
            if j == 0:
                ax[i][j].set_ylabel(split)
            else:
                ax[i][j].set_ylabel("")
            ax[i][j].set_xlabel('')
            # ax[i][j].set_ylabel('Mean Δlogit PSI Across Cell Types')
            # ax[i][j].axhline(0, color='gray', lw=0.5)
            # ax[i][j].axvline(0, color='gray', lw=0.5)
            min_x = min(min_x, np.min(retr))
            max_x = max(max_x, np.max(retr))
            min_y = min(min_y, np.min(true))
            max_y = max(max_y, np.max(true))

    for i in range(3):
        for j in range(2):
            ax[i][j].plot((-11, 11), (-11, 11), "--", color="gray", lw=0.5, zorder=0)
            ax[i][j].set_xlim((-11, 11))
            ax[i][j].set_ylim((-11, 11))
            # ax[i][j].set_xlim(min_x, max_x)
            # ax[i][j].set_ylim(min_y, max_y)
            # min_all = min(min_x, min_y)
            # max_all = max(max_x, max_y)
            # ax[i][j].plot((min_all, max_all), (min_all, max_all), "--", color="gray", lw=0.5, zorder=0)
            
    fig.supxlabel('Predicted Δlogit(PSI)')
    fig.supylabel('Measured Δlogit(PSI)')
    save_plot(f"mmsplice_vs_retrained_remade_{date}")


    base = predictions['Baseline_MMSplice_Predicted_Delta_Logit']
    true = predictions["True_Delta_Logit"]

    # plot aggregate
    n_ag = comma_separate(len(true))
    r_val = calc_r(base, true)
    fig, ax = plt.subplots(figsize=(3.3,3.3), constrained_layout=True)
    sns.scatterplot(x = base, y = true, alpha=alpha, ax=ax, s=s, color=COLOR["Baseline"])
    ax.set_xlabel('Predicted Δlogit(PSI)')
    ax.set_ylabel('Measured Δlogit(PSI)')

    # plot test on top 
    base = predictions[predictions["Split"] == "Test"]['Baseline_MMSplice_Predicted_Delta_Logit']
    true = predictions[predictions["Split"] == "Test"]["True_Delta_Logit"]
    n_test = comma_separate(len(true))
    r_val_test = round(calc_r(base, true), 3)
    sns.scatterplot(x=base, y=true, alpha=alpha, ax=ax, s=s, color="black")

    legend = ax.legend(["Aggregate", "Test"], loc="upper left", fontsize=9)
    for handle in legend.legend_handles:
        handle.set_sizes([30])
        handle.set_alpha([1])
    # plt.axhline(0, color='gray', lw=0.5); plt.axvline(0, color='gray', lw=0.5)
    ax.set_title(
        f"Retrained MMSplice {dlogit_key}\n"
        f"predictions (aggregate vs test)\n"
        f"n={n_ag} (aggregate) and {n_test} (test),\n"
        f"r={round(r_val, 3)} (aggregate) vs {r_val_test} (test)"
    )

    ax.plot((-11, 11), (-11, 11), "--", color="gray", lw=0.5, zorder=0)
    ax.set_xlim((-11, 11))
    ax.set_ylim((-11, 11))

    # plt.tight_layout()
    save_plot(f"mmsplice_baseline_full_and_test_remade_{date}")


    retr = predictions['Retrained_MMSplice_Predicted_Delta_Logit']
    true = predictions["True_Delta_Logit"]

    # plot aggregate
    n_ag = comma_separate(len(true))
    r_val = calc_r(retr, true)
    fig, ax = plt.subplots(figsize=(3.3,3.3), constrained_layout=True)
    sns.scatterplot(x = retr, y = true, alpha=alpha, ax=ax, s=s, color=COLOR["Retrained"])
    ax.set_xlabel('Predicted Δlogit(PSI)')
    ax.set_ylabel('Measured Δlogit(PSI)')

    # plot test on top 
    retr = predictions[predictions["Split"] == "Test"]['Retrained_MMSplice_Predicted_Delta_Logit']
    true = predictions[predictions["Split"] == "Test"]["True_Delta_Logit"]
    n_test = comma_separate(len(true))
    r_val_test = round(calc_r(retr, true), 3)
    sns.scatterplot(x=retr, y=true, alpha=alpha, ax=ax, s=s, color="black")

    legend = ax.legend(["Aggregate", "Test"], loc="upper left", fontsize=9)
    for handle in legend.legend_handles:
        handle.set_sizes([30])
        handle.set_alpha([1])
    # plt.axhline(0, color='gray', lw=0.5); plt.axvline(0, color='gray', lw=0.5)
    ax.set_title(
        f"Retrained MMSplice {dlogit_key}\n"
        f"predictions (aggregate vs test)\n"
        f"n={n_ag} (aggregate) and {n_test} (test),\n"
        f"r={round(r_val, 3)} (aggregate) vs {r_val_test} (test)"
    )

    ax.plot((-11, 11), (-11, 11), "--", color="gray", lw=0.5, zorder=0)
    ax.set_xlim((-11, 11))
    ax.set_ylim((-11, 11))


    save_plot(f"mmsplice_retrained_full_and_test_remade_{date}")





    splits = ["Train", "Val"]
    fig, ax = plt.subplots(len(splits), 2, figsize=(12.4/2.54,12.4/2.54), constrained_layout=True)
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    for i, split in enumerate(["Train", "Val"]):
        for j, task in enumerate(["Baseline", "Retrained"]):
            retr = predictions[predictions["Split"] == split][f'{task}_MMSplice_Predicted_Delta_Logit']
            true = predictions[predictions["Split"] == split]["True_Delta_Logit"]
            r_val = calc_r(retr, true)
            sns.scatterplot(x = retr, y = true, alpha=alpha, ax=ax[i][j], s=s, color=COLOR[task])
            n = comma_separate(len(true))
            if i == 0:        
                ax[i][j].set_title(f"{task} MMSplice \nn={n}, r={round(r_val, 3)}")
            else:
                ax[i][j].set_title(f"n={n}, r={round(r_val, 3)}")
            if j == 0:
                ax[i][j].set_ylabel(split)
            else:
                ax[i][j].set_ylabel("")
            ax[i][j].set_xlabel('')
            # ax[i][j].set_ylabel('Mean Δlogit PSI Across Cell Types')
            # ax[i][j].axhline(0, color='gray', lw=0.5)
            # ax[i][j].axvline(0, color='gray', lw=0.5)
            min_x = min(min_x, np.min(retr))
            max_x = max(max_x, np.max(retr))
            min_y = min(min_y, np.min(true))
            max_y = max(max_y, np.max(true))

    for i in range(len(splits)):
        for j in range(2):
            ax[i][j].plot((-11, 11), (-11, 11), "--", color="gray", lw=0.5, zorder=0)
            ax[i][j].set_xlim((-11, 11))
            ax[i][j].set_ylim((-11, 11))
            # ax[i][j].set_xlim(min_x, max_x)
            # ax[i][j].set_ylim(min_y, max_y)
            # min_all = min(min_x, min_y)
            # max_all = max(max_x, max_y)
            # ax[i][j].plot((min_all, max_all), (min_all, max_all), "--", color="gray", lw=0.5, zorder=0)
            
    fig.supxlabel('Predicted Δlogit(PSI)')
    fig.supylabel('Measured Δlogit(PSI)')
    save_plot(f"mmsplice_vs_retrained_train_val_remade_{date}")


    splits = ["Train"]
    fig, ax = plt.subplots(len(splits), 2, figsize=(6.6,3.3), constrained_layout=True)

    def get_axis(M, N, ax):
        def axis_getter(i, j):
            if M == 1 and N == 1:
                return ax
            elif M == 1 and N > 1:
                return ax[j]
            elif M > 1 and N == 1:
                return ax[i]
            elif M > 1 and N > 1:
                return ax[i][j]
            else:
                raise IndexError(f"Invalid index ({i}, {j}) for plot ({M}, {N})")
        return axis_getter


    axis = get_axis(len(splits), 2, ax)

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    for i, split in enumerate(splits):
        for j, task in enumerate(["Baseline", "Retrained"]):
            retr = predictions[predictions["Split"] == split][f'{task}_MMSplice_Predicted_Delta_Logit']
            true = predictions[predictions["Split"] == split]["True_Delta_Logit"]
            r_val = calc_r(retr, true)
            sns.scatterplot(x = retr, y = true, alpha=alpha, ax=axis(i, j), s=s, color=COLOR[task])
            n = comma_separate(len(true))
            if i == 0:        
                axis(i, j).set_title(f"{task} MMSplice \nn={n}, r={round(r_val, 3)}")
            else:
                axis(i, j).set_title(f"n={n}, r={round(r_val, 3)}")
            if j == 0:
                axis(i, j).set_ylabel(split)
            else:
                axis(i, j).set_ylabel("")
            axis(i, j).set_xlabel('')
            # ax[i][j].set_ylabel('Mean Δlogit PSI Across Cell Types')
            # ax[i][j].axhline(0, color='gray', lw=0.5)
            # ax[i][j].axvline(0, color='gray', lw=0.5)
            min_x = min(min_x, np.min(retr))
            max_x = max(max_x, np.max(retr))
            min_y = min(min_y, np.min(true))
            max_y = max(max_y, np.max(true))

    for i in range(len(splits)):
        for j in range(2):
            axis(i, j).plot((-11, 11), (-11, 11), "--", color="gray", lw=0.5, zorder=0)
            axis(i, j).set_xlim((-11, 11))
            axis(i, j).set_ylim((-11, 11))
            # ax[i][j].set_xlim(min_x, max_x)
            # ax[i][j].set_ylim(min_y, max_y)
            # min_all = min(min_x, min_y)
            # max_all = max(max_x, max_y)
            # ax[i][j].plot((min_all, max_all), (min_all, max_all), "--", color="gray", lw=0.5, zorder=0)
            
    fig.supxlabel('Predicted Δlogit(PSI)')
    fig.supylabel('Measured Δlogit(PSI)')
    save_plot(f"mmsplice_vs_retrained_train_remade_{date}")


    splits = ["Val"]
    fig, ax = plt.subplots(len(splits), 2, figsize=(6.6,3.3), constrained_layout=True)

    def get_axis(M, N, ax):
        def axis_getter(i, j):
            if M == 1 and N == 1:
                return ax
            elif M == 1 and N > 1:
                return ax[j]
            elif M > 1 and N == 1:
                return ax[i]
            elif M > 1 and N > 1:
                return ax[i][j]
            else:
                raise IndexError(f"Invalid index ({i}, {j}) for plot ({M}, {N})")
        return axis_getter


    axis = get_axis(len(splits), 2, ax)

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    for i, split in enumerate(splits):
        for j, task in enumerate(["Baseline", "Retrained"]):
            retr = predictions[predictions["Split"] == split][f'{task}_MMSplice_Predicted_Delta_Logit']
            true = predictions[predictions["Split"] == split]["True_Delta_Logit"]
            r_val = calc_r(retr, true)
            sns.scatterplot(x = retr, y = true, alpha=alpha, ax=axis(i, j), s=s, color=COLOR[task])
            n = comma_separate(len(true))
            if i == 0:        
                axis(i, j).set_title(f"{task} MMSplice \nn={n}, r={round(r_val, 3)}")
            else:
                axis(i, j).set_title(f"n={n}, r={round(r_val, 3)}")
            if j == 0:
                axis(i, j).set_ylabel(split)
            else:
                axis(i, j).set_ylabel("")
            axis(i, j).set_xlabel('')
            # ax[i][j].set_ylabel('Mean Δlogit PSI Across Cell Types')
            # ax[i][j].axhline(0, color='gray', lw=0.5)
            # ax[i][j].axvline(0, color='gray', lw=0.5)
            min_x = min(min_x, np.min(retr))
            max_x = max(max_x, np.max(retr))
            min_y = min(min_y, np.min(true))
            max_y = max(max_y, np.max(true))

    for i in range(len(splits)):
        for j in range(2):
            axis(i, j).plot((-11, 11), (-11, 11), "--", color="gray", lw=0.5, zorder=0)
            axis(i, j).set_xlim((-11, 11))
            axis(i, j).set_ylim((-11, 11))
            # ax[i][j].set_xlim(min_x, max_x)
            # ax[i][j].set_ylim(min_y, max_y)
            # min_all = min(min_x, min_y)
            # max_all = max(max_x, max_y)
            # ax[i][j].plot((min_all, max_all), (min_all, max_all), "--", color="gray", lw=0.5, zorder=0)
            
    fig.supxlabel('Predicted Δlogit(PSI)')
    fig.supylabel('Measured Δlogit(PSI)')
    save_plot(f"mmsplice_vs_retrained_val_remade_{date}")


    logging.info("done")

