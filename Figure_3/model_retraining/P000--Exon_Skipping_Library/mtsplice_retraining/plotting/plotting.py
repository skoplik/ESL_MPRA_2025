import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mtsplice_retraining.model_evaluation import model_evaluation
from src.esl import cell_types

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_axis(m, n, ax = None):
    """ 
    necessary when plt.subplots has variable 
    dimensions 
    """
    if m == 1 and n == 1:
        def axis(i, j):
            return ax
    elif m == 1:
        def axis(i, j):
            return ax[j]
    elif n == 1:
        def axis(i, j):
            return ax[i]
    else:
        def axis(i, j):
            return ax[i][j]
    return axis


def scatterplot(
        predictions,
        datasets,
        output_figures_dir,
        **kwargs,
    ):
    """
    uses scaled values to make tsplice scatter plots
    for each dataset
    """
    r_vals = [None] * len(datasets)
    fig, ax = plt.subplots(len(datasets), 5, figsize=(20, 6*len(datasets)))
    axis = get_axis(len(datasets), 5, ax)

    for i, data in enumerate(datasets):
        r2_values = predictions.get_prediction(
            data=data,
            **kwargs,
            output="r2"
        )
        x = predictions.get_prediction(
            data=data,
            **kwargs,
            output="prediction"
        )
        y = predictions.predictions.get_true_data(
            data=data,
            **kwargs,
        )
        r_vals[i] = np.sqrt(r2_values)
        for cell in range(5):
            sns.scatterplot(
                x=x[:,cell],
                y=y[:,cell],
                ax=axis(i, cell),
                alpha=0.1
            )
            subplot_title = predictions.make_subplot_title(
                cell=cell,
                data=data,
                **kwargs,
            )
            axis(i, cell).set_title(
                subplot_title
                + f"\nr={round((r_vals[i][cell]),3)}"
            )
    
    xlabel = predictions.make_axis_label("Predicted", **kwargs)
    ylabel = predictions.make_axis_label("Observed", **kwargs)
    title = predictions.make_plot_title(**kwargs)
    fig.supxlabel(xlabel, y=-0.05)
    fig.supylabel(ylabel, x=-0.03)
    fig.suptitle(title, y=1.05)
    plt.tight_layout()


    def save_plot(name):
        path = os.path.join(output_figures_dir, name)
        plt.savefig(path, bbox_inches="tight")
        logging.info("Saving TSplice scatter plot to " + path)
        return


    model_save_name = predictions.make_plot_save_name(
        datasets,
        **kwargs
    )
    save_plot(model_save_name + ".png")
    save_plot(model_save_name + ".pdf")
    plt.close()
    return 


def barplot(
    predictions, 
    datasets,
    tasks,
    **kwargs,
    ):
    """
    bar plot of R^2 values
    """
    fig, ax = plt.subplots(
        len(datasets),
        len(tasks),
        figsize=(15, 15)
    )
    axis = get_axis(len(datasets), len(tasks), ax)
    for i, data in enumerate(datasets):
        for j, task in enumerate(tasks):
            r2key = f"r2_{data}_mtsplice_{task}_{scale}" 
            r2 = predictions[r2key]
            axis(i, j).bar(
                cell_types,
                r2,
                align='center',
                color="tab:purple",
                edgecolor='tab:blue',
                linewidth=1.5
            )
            axis(i, j).set_xlim((-1,5))
            axis(i, j).set_ylim((0,1.0))
            title = f"{task.title()} {data.title()} {scale} by Cell Type"
            axis(i, j).set_title(title)
            axis(i, j).set_ylabel("R^2")
    
    fig.suptitle(
        f"MTSplice R^2 Predicted vs Observed {scale} by Cell Type",
        x=0.5,
        y=1.0
    )

    if not save_plot:
        plt.show()
        plt.close()
        return

    task_title = "_".join(tasks)
    datasets_save = "_".join(datasets)


    def save_plot(name):
        path = os.path.join(output_figures_dir, name)
        plt.savefig(path, bbox_inches="tight")
        logging.info("Saving MTSplice bar plot to " + path)
        return


    save_plot(f"mtsplice_r2_{task_title}_bar_{datasets_save}.png")
    save_plot(f"mtsplice_r2_{task_title}_bar_{datasets_save}.pdf")
    plt.close()
    return 


def mtsplice_barplot(
        predictions, 
        y, 
        ref_psi, 
        scale, 
        datasets, 
        tasks, 
        output_figures_dir,
        save_plot = True,
    ):
    """
    bar plot of R^2 values
    """
    fig, ax = plt.subplots(len(datasets), len(tasks), figsize = (15,15))
    axis = get_axis(len(datasets), len(tasks), ax)
    for i, data in enumerate(datasets):
        for j, task in enumerate(tasks):
            r2key = f"r2_{data}_mtsplice_{task}_{scale}" 
            r2 = predictions[r2key]
            axis(i, j).bar(
                cell_types,
                r2,
                align='center',
                color="tab:purple",
                edgecolor='tab:blue',
                linewidth=1.5
            )
            axis(i, j).set_xlim((-1,5))
            axis(i, j).set_ylim((0,1.0))
            title = f"{task.title()} {data.title()} {scale} by Cell Type"
            axis(i, j).set_title(title)
            axis(i, j).set_ylabel("R^2")
    
    fig.suptitle(
        f"MTSplice R^2 Predicted vs Observed {scale} by Cell Type",
        x=0.5,
        y=1.0
    )

    if not save_plot:
        plt.show()
        plt.close()
        return

    task_title = "_".join(tasks)
    datasets_save = "_".join(datasets)


    def save_plot(name):
        path = os.path.join(output_figures_dir, name)
        plt.savefig(path, bbox_inches="tight")
        logging.info("Saving MTSplice bar plot to " + path)
        return


    save_plot(f"mtsplice_r2_{task_title}_bar_{datasets_save}.png")
    save_plot(f"mtsplice_r2_{task_title}_bar_{datasets_save}.pdf")
    plt.close()
    return 


def get_scaling_params_tsplice(scale):
    match scale:
        case "logit":
            func = model_evaluation.tsplice_natural_scale
            title = "Cell Logit PSI"
        case "dlogit":
            func = model_evaluation.tsplice_dlogit_scale
            title = "\nΔLogit (Cell - Mean Cell)"
        case "psi":
            func = model_evaluation.tsplice_psi_scale
            title = "Cell PSI"
        case "dpsi":
            func = model_evaluation.tsplice_dpsi_scale
            title = "\nΔPSI (Cell - Mean Cell)"
    return func, title


def tsplice_barplot(
        predictions,
        y,
        mean_cell_logit,
        scale,
        datasets,
        condition,
        condition_keys,
        output_figures_dir,
        task = "baseline",
        save_plot = True,
    ):
    """
    condition lets the function plot
    either against different model heads (0-3, ensemble)
    or against different tasks (baseline, retrained)
    """
    fig, ax = plt.subplots(
        len(datasets),
        len(condition_keys),
        figsize = (15,15)
    )
    axis = get_axis(len(datasets), len(condition_keys), ax)
    for i, data in enumerate(datasets):
        for j, curr_condition in enumerate(condition_keys):
            if condition == "task":
                r2key = f"r2_{data}_tsplice_ensemble_{curr_condition}_{scale}" 
            elif condition == "model":
                r2key = f"r2_{data}_tsplice_{curr_condition}_{task}_{scale}"
                curr_condition = "model " + curr_condition
            r2 = predictions[r2key]
            axis(i, j).bar(
                cell_types,
                r2,
                align='center',
                color="tab:purple",
                edgecolor='tab:blue',
                linewidth=1.5
            )
            axis(i, j).set_xlim((-1,5))
            axis(i, j).set_ylim((0,1.0))
            title = (
                f"{curr_condition.title()} "
                f"{data.title()} {scale} "
                f"by Cell Type"
            )
            axis(i, j).set_title(title)
            axis(i, j).set_ylabel("R^2")

    if condition == "model":
        task_title = "By Model in Ensemble"
    else:
        task_title = " vs ".join(condition_keys).replace("_", " ").title()

    task_save = "_".join(condition_keys).replace(" ", "_")
    datasets_save = "_".join(datasets)
    fig.suptitle(
        f"TSplice R^2 Predicted vs "
        f"Observed {scale} by Cell Type {task_title}",
        x=0.5,
        y=1.0)

    if not save_plot:
        plt.show()
        plt.close()
        return


    def save_plot(name):
        path = os.path.join(output_figures_dir, name)
        plt.savefig(path, bbox_inches="tight")
        logging.info("Saving TSplice bar plot to " + path)
        return


    save_plot(f"tsplice_r2_{task_save}_bar_{datasets_save}.png")
    save_plot(f"tsplice_r2_{task_save}_bar_{datasets_save}.pdf")
    plt.close()
    return 


def get_scaling_params_mtsplice(scale):
    match scale:
        case "logit":
            func = model_evaluation.mtsplice_logit_scale
            title = "Variant Logit"
        case "dlogit":
            func = model_evaluation.mtsplice_dlogit_scale
            title = "ΔLogit (Variant - Reference)"
        case "psi":
            func = model_evaluation.mtsplice_psi_scale
            title = "Variant PSI"
        case "dpsi":
            func = model_evaluation.mtsplice_dpsi_scale
            title = "ΔPSI (Variant - Reference)"
    return func, title


def mtsplice_scatterplot(
        mt_splice_scaled,
        y_values,
        ref_cell_psi,
        scale,
        datasets,
        task,
        output_figures_dir
    ):
    """
    uses scaled values to make mtsplice scatter plots
    for each dataset
    """
    r_vals = [None] * len(datasets)
    fig, ax = plt.subplots(
        len(datasets),
        5,
        figsize = (20, 6 * len(datasets))
    )
    axis = get_axis(len(datasets), 5, ax)
    scaling_func, scale_title = get_scaling_params_mtsplice(scale)

    for i, data in enumerate(datasets):
        r2key = f"r2_{data}_mtsplice_{task}_{scale}"
        prkey = f"{data}_mtsplice_{task}_{scale}"
        r_vals[i] = np.sqrt(mt_splice_scaled[r2key])
        x = mt_splice_scaled[prkey]
        y = scaling_func(y_values[i], ref_cell_psi[i])
        for cell in range(5):
            try:
                sns.scatterplot(
                    x=x[:,cell],
                    y=y[:,cell],
                    ax=axis(i, cell),
                    alpha=0.1
                )
            except IndexError:
                print("Sns line error")
                print("x shape:", x.shape)
                print("y shape:", y.shape)
                print("axis i cell", axis(i, cell))
                return
            axis(i, cell).set_title(
                f"{cell_types[cell]} {data.title()} "
                f"{scale_title}\nr={round((r_vals[i][cell]),3)}"
            )

    scale_title_flat = scale_title.replace("\n", "")
    fig.supxlabel(f"MTSplice Predicted {scale_title_flat}", y = -0.05)
    fig.supylabel(f"Observed {scale_title_flat}", x = -0.03)

    fig.suptitle(
        f"{task.title()} MTSplice Predicted {scale_title_flat}",
        y = 1.05
    )
    
    plt.tight_layout()
        
    datasets_save = "_".join(datasets)


    def save_plot(name):
        path = os.path.join(output_figures_dir, name)
        plt.savefig(path, bbox_inches="tight")
        logging.info("Saving MTSplice scatter plot to " + path)
        return


    save_plot(
        f"mtsplice_{task}_{scale}_prediction_"
        f"observation_scatter_{datasets_save}.png"
    )
    save_plot(
        f"mtsplice_{task}_{scale}_prediction_"
        f"observation_scatter_{datasets_save}.pdf"
    )
    plt.close()
    return


