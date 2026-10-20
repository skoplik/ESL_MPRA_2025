import os
import getopt
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from matplotlib import colormaps
from matplotlib import cm
import matplotlib.patches as patches
from collections import defaultdict
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

import scipy.stats as stats
import ast

CLINVAR_COLORS = {
    "Pathogenic": "red",
    "Likely_pathogenic": "orange",
    "Likely_benign": "#9ACD32",
    "Benign": "#258d4c",
    "Uncertain_significance": "slateblue",
    "Conflicting": "orchid",
}
nt_list = ["A", "C", "G", "T"]
var_locs = {nt : i for i, nt in enumerate(nt_list)}



def get_snp_loc(snp):
    if snp.lower() == "none":
        return -1
    loc, change = snp.split(":")
    ref, var = change.split(">")
    var_loc = var_locs[var]
    ref_loc = var_locs[ref]
    return int(loc), ref_loc, var_loc


def nan_add_x(ax, matrix):
    nan_indices = np.argwhere(np.isnan(matrix))
    for row, col in nan_indices:
        ax.text(
            col,
            row,
            "X",
            color="black",
            ha="center",
            va="center",
            fontsize=4,
            fontweight="bold",
        )
    return


def get_nts(df):
    full = (
        df.iloc[0]["ref_intron1"] 
        + df.iloc[0]["ref_exon"] 
        + df.iloc[0]["ref_intron2"] 
    )
    return [i for i in full]


def add_colorbar(fig, im, position, font_size=6, label=None):
    cax = fig.add_axes(position)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=font_size, width=0.5)
    cbar.outline.set_linewidth(0.5)
    if label:
        cbar.set_label(label, fontsize=font_size, labelpad=2)




def parse_clinvar_significance(clinvar_item):
    sigs = set()

    if isinstance(clinvar_item, str):
        clinvar_item = ast.literal_eval(clinvar_item)

    if isinstance(clinvar_item, dict) and "ClinVar_info" in clinvar_item:
        clinvar_item = clinvar_item["ClinVar_info"]

    if isinstance(clinvar_item, list):
        for entry in clinvar_item:
            if isinstance(entry, tuple) and len(entry) == 4:
                meta = entry[3]
                if "CLNSIG=" in meta:
                    for part in meta.split(";"):
                        if part.startswith("CLNSIG="):
                            raw_sigs = part.split("=")[1].replace(" ", "_").split("|")
                            sigs.update(raw_sigs)
    elif isinstance(clinvar_item, str):
        for part in clinvar_item.split(";"):
            if part.startswith("CLNSIG="):
                raw_sigs = part.split("=")[1].replace(" ", "_").split("|")
                sigs.update(raw_sigs)

    normalized = set()
    for s in sigs:
        s_lower = s.strip().lower()
        if s_lower in {"conflicting_classifications_of_pathogenicity", "conflicting"}:
            normalized.add("Conflicting")
        elif s_lower == "not_provided":
            normalized.add("Not_provided")
        else:
            normalized.add(s.replace(" ", "_"))

    if not normalized:
        return None
    elif len(normalized) > 1:
        return "Conflicting"
    else:
        return list(normalized)[0]


def get_all_annotations(clinvar_df):
    clinvar_labels = []
    if "annotation" not in clinvar_df:
        clinvar_df["annotation"] = clinvar_df["ClinVar_item"].apply(parse_clinvar_significance)
    for i, row in clinvar_df.iterrows():
        mut_positions = ast.literal_eval(row["mut_ref_mut_positions"])
        sig = row.get("annotation", "")
        if sig in CLINVAR_COLORS:
            for clin_pos, (clin_ref, clin_alt) in mut_positions:
                label = f"{clin_ref}>{','.join(clin_alt)}"
                color = CLINVAR_COLORS[sig]
                clinvar_labels.append((clin_pos, label, color))
    return clinvar_labels


def check_missing(true_matrix, label, pos):
    ref_nt, var_nt = label.split(">")
    nt_index = var_locs[var_nt]
    nt_index = 3 - nt_index
    if np.isnan(true_matrix[nt_index][pos]):
        return True
    return False


def filter_clinvar_labels(clinvar_labels, true_matrix):
    """
    checks label is not missing
    and filters multiple
    """
    
    def add_snp(annotation_tuple, label):
        clin_pos, label_old, color = annotation_tuple
        ref_nt, var_nt = label.split(">")
        label_new = label_old + "," + var_nt
        return (clin_pos, label_new, color)

    by_position = {}
    for annotation_tuple in clinvar_labels:
        clin_pos, label, color = annotation_tuple
        is_missing = check_missing(true_matrix, label, clin_pos)
        if is_missing:
            continue
        if clin_pos in by_position:
            by_position[clin_pos] = add_snp(by_position[clin_pos], label)
        else:
            by_position[clin_pos] = annotation_tuple
    return list(by_position.values())


def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0


def mse(x, y):
    return np.mean((x - y) ** 2)


def mae(x, y):
    return np.mean(np.abs(x - y))


def perc_diff(x, y, threshold=0.05):
    if x == 0:
        return False
    return ((y - x) / x) < threshold


def thresholded_predictions(threshold, small=True):
    outcomes_df = {
    "Predicted large positive" : [0, 0],
    "Predicted large negative" : [0, 0],
    "Predicted small negative" : [0, 0],
    "Predicted small positive" : [0, 0],
    }
    outcomes_df = pd.DataFrame(outcomes_df)
    outcomes_df.index = ["Baseline MMSplice", "Retrained MMSplice"]
    if small:
        if threshold < 0:
            filtered_samples = all_samples[
                (all_samples[true_col] < 0)
                & (all_samples[true_col] > threshold)
            ]
            correct_key = "Predicted small negative"
        elif threshold > 0:
            filtered_samples = all_samples[
                (0 < all_samples[true_col])
                & (all_samples[true_col] < threshold)
            ]
            correct_key = "Predicted small positive"
    elif not small:
        if threshold < 0:
            filtered_samples = all_samples[
                (all_samples[true_col] < threshold)
            ]
            correct_key = "Predicted large negative"
        elif threshold > 0:
            filtered_samples = all_samples[
                (all_samples[true_col] > threshold)
            ]
            correct_key = "Predicted large positive"
            
    total = 0
    for i, row in filtered_samples.iterrows():
        rt = row[retrained_col]
        bl = row[baseline_col]
        tr = row[true_col]

        left_threshold = -np.abs(threshold)
        right_threshold = np.abs(threshold)
    
        if bl < left_threshold:
            outcomes_df.loc["Baseline MMSplice", "Predicted large negative"] += 1
        elif left_threshold < bl < 0:
            outcomes_df.loc["Baseline MMSplice", "Predicted small negative"] += 1
        elif 0 < bl < right_threshold:
            outcomes_df.loc["Baseline MMSplice", "Predicted small positive"] += 1
        elif right_threshold < bl:
            outcomes_df.loc["Baseline MMSplice", "Predicted large positive"] += 1
        else:
            raise ValueError(f"Baseline prediction is exactly at threshold {bl}")
            
        if rt < left_threshold:
            outcomes_df.loc["Retrained MMSplice", "Predicted large negative"] += 1
        elif left_threshold < rt < 0:
            outcomes_df.loc["Retrained MMSplice", "Predicted small negative"] += 1
        elif 0 < rt < right_threshold:
            outcomes_df.loc["Retrained MMSplice", "Predicted small positive"] += 1
        elif right_threshold < rt:
            outcomes_df.loc["Retrained MMSplice", "Predicted large positive"] += 1
        else:
            raise ValueError(f"Retrained prediction is exactly at threshold {bl}")
            
        total += 1
        
    total_incorrect_bl = 0
    total_incorrect_rt = 0
    for i in outcomes_df:
        if i == correct_key:
            continue
        total_incorrect_bl += outcomes_df.loc["Baseline MMSplice", i]
        total_incorrect_rt += outcomes_df.loc["Retrained MMSplice", i]
    
    outcomes_df["Total correct"] = [
        outcomes_df.loc["Baseline MMSplice", correct_key],
        outcomes_df.loc["Retrained MMSplice", correct_key]
    ]
    outcomes_df["Total incorrect"] = [
        total_incorrect_bl,
        total_incorrect_rt
    ]
    return outcomes_df, total


def mse_predictions(threshold, small=True):
    outcomes_df = {
    "MSE" : [0, 0],
    }
    outcomes_df = pd.DataFrame(outcomes_df)
    outcomes_df.index = ["Baseline MMSplice", "Retrained MMSplice"]
    if small:
        if threshold < 0:
            filtered_samples = all_samples[
                (all_samples[true_col] < 0)
                & (all_samples[true_col] > threshold)
            ]
            correct_key = "Predicted small negative"
        elif threshold > 0:
            filtered_samples = all_samples[
                (0 < all_samples[true_col])
                & (all_samples[true_col] < threshold)
            ]
            correct_key = "Predicted small positive"
    elif not small:
        if threshold < 0:
            filtered_samples = all_samples[
                (all_samples[true_col] < threshold)
            ]
            correct_key = "Predicted large negative"
        elif threshold > 0:
            filtered_samples = all_samples[
                (all_samples[true_col] > threshold)
            ]
            correct_key = "Predicted large positive"

    rt_mse = mse(filtered_samples[retrained_col], filtered_samples[true_col])
    bl_mse = mse(filtered_samples[baseline_col], filtered_samples[true_col])
    rt_r = calc_r_with_nan(filtered_samples[retrained_col], filtered_samples[true_col])
    bl_r = calc_r_with_nan(filtered_samples[baseline_col], filtered_samples[true_col])

    outcomes_df["MSE"] = [
        bl_mse,
        rt_mse
    ]
    outcomes_df["R"] = [
        bl_r, 
        rt_r
    ]
    return outcomes_df


def threshold_str(threshold_tuple):
    l, r = threshold_tuple
    l = str(round(l, 1))
    r = str(round(r, 1))
    return f"{l} to {r}"


def two_threshold_predictions(left_threshold, right_threshold, **kwargs):
    outcomes_df = {
    "Predicted within threshold" : [0, 0],
    "Predicted above threshold" : [0, 0],
    "Predicted below threshold" : [0, 0],
    }
    outcomes_df = pd.DataFrame(outcomes_df)
    outcomes_df.index = ["Baseline MMSplice", "Retrained MMSplice"]

    filtered_samples = all_samples[
        (all_samples[true_col] > left_threshold) 
        & (all_samples[true_col] < right_threshold)
    ]
            
    total = 0
    for i, row in filtered_samples.iterrows():
        rt = row[retrained_col]
        bl = row[baseline_col]
        tr = row[true_col]
    
        if bl < left_threshold:
            outcomes_df.loc["Baseline MMSplice", "Predicted below threshold"] += 1
        elif left_threshold < bl < right_threshold:
            outcomes_df.loc["Baseline MMSplice", "Predicted within threshold"] += 1
        elif right_threshold < bl:
            outcomes_df.loc["Baseline MMSplice", "Predicted above threshold"] += 1
        else:
            raise ValueError(f"Baseline prediction is exactly at threshold {bl}")
    
        if rt < left_threshold:
            outcomes_df.loc["Retrained MMSplice", "Predicted below threshold"] += 1
        elif left_threshold < rt < right_threshold:
            outcomes_df.loc["Retrained MMSplice", "Predicted within threshold"] += 1
        elif right_threshold < rt:
            outcomes_df.loc["Retrained MMSplice", "Predicted above threshold"] += 1
        else:
            raise ValueError(f"Retrained prediction is exactly at threshold {bl}")
            
        total += 1
        
    total_incorrect_bl = 0
    total_incorrect_rt = 0
    for i in outcomes_df:
        if i == "Predicted within threshold":
            continue
        total_incorrect_bl += outcomes_df.loc["Baseline MMSplice", i]
        total_incorrect_rt += outcomes_df.loc["Retrained MMSplice", i]
    
    outcomes_df["Total correct"] = [
        outcomes_df.loc["Baseline MMSplice", "Predicted within threshold"],
        outcomes_df.loc["Retrained MMSplice", "Predicted within threshold"]
    ]
    outcomes_df["Total incorrect"] = [
        total_incorrect_bl,
        total_incorrect_rt
    ]
    return outcomes_df, total
            

def soft_two_threshold_predictions(
        left_threshold,
        right_threshold,
        alpha=0.01,
        errf=mse
    ):
    outcomes_df = {
    "Predicted within threshold" : [0, 0],
    "Predicted above threshold" : [0, 0],
    "Predicted below threshold" : [0, 0],
    }
    outcomes_df = pd.DataFrame(outcomes_df)
    outcomes_df.index = ["Baseline MMSplice", "Retrained MMSplice"]

    filtered_samples = all_samples[
        (all_samples[true_col] > left_threshold) 
        & (all_samples[true_col] < right_threshold)
    ]
            
    total = 0
    for i, row in filtered_samples.iterrows():
        rt = row[retrained_col]
        bl = row[baseline_col]
        tr = row[true_col]
    
        if bl < left_threshold and errf(bl, tr) > alpha:
            outcomes_df.loc["Baseline MMSplice", "Predicted below threshold"] += 1
        elif right_threshold < bl and errf(bl, tr) > alpha:
            outcomes_df.loc["Baseline MMSplice", "Predicted above threshold"] += 1
        else:
            outcomes_df.loc["Baseline MMSplice", "Predicted within threshold"] += 1
    
        if rt < left_threshold and errf(rt, tr) > alpha:
            outcomes_df.loc["Retrained MMSplice", "Predicted below threshold"] += 1
        elif right_threshold < rt and errf(rt, tr) > alpha:
            outcomes_df.loc["Retrained MMSplice", "Predicted above threshold"] += 1
        else:
            outcomes_df.loc["Retrained MMSplice", "Predicted within threshold"] += 1
            
        total += 1
        
    total_incorrect_bl = 0
    total_incorrect_rt = 0
    for i in outcomes_df:
        if i == "Predicted within threshold":
            continue
        total_incorrect_bl += outcomes_df.loc["Baseline MMSplice", i]
        total_incorrect_rt += outcomes_df.loc["Retrained MMSplice", i]
    
    outcomes_df["Total correct"] = [
        outcomes_df.loc["Baseline MMSplice", "Predicted within threshold"],
        outcomes_df.loc["Retrained MMSplice", "Predicted within threshold"]
    ]
    outcomes_df["Total incorrect"] = [
        total_incorrect_bl,
        total_incorrect_rt
    ]
    return outcomes_df, total


def mse_two_threshold_predictions(
        left_threshold,
        right_threshold,
        alpha=0.01,
        errf=mse
    ):
    outcomes_df = {
    "Predicted within threshold" : [0.0, 0.0],
    "Predicted above threshold" : [0.0, 0.0],
    "Predicted below threshold" : [0.0, 0.0],
    }
    outcomes_df = pd.DataFrame(outcomes_df)
    outcomes_df.index = ["Baseline MMSplice", "Retrained MMSplice"]

    filtered_samples = all_samples[
        (all_samples[true_col] > left_threshold) 
        & (all_samples[true_col] < right_threshold)
    ]
            
    total = 1
    for i, row in filtered_samples.iterrows():
        rt = row[retrained_col]
        bl = row[baseline_col]
        tr = row[true_col]
        outcomes_df.loc["Baseline MMSplice"] += errf(bl, tr)
        outcomes_df.loc["Retrained MMSplice"] += errf(rt, tr)
        total += 1
        
    total_incorrect_bl = 0
    total_incorrect_rt = 0
    for i in outcomes_df:
        if i == "Predicted within threshold":
            continue
        total_incorrect_bl += outcomes_df.loc["Baseline MMSplice", i]
        total_incorrect_rt += outcomes_df.loc["Retrained MMSplice", i]
    
    outcomes_df["Total correct"] = [
        outcomes_df.loc["Baseline MMSplice", "Predicted within threshold"],
        outcomes_df.loc["Retrained MMSplice", "Predicted within threshold"]
    ]
    outcomes_df["Total incorrect"] = [
        total_incorrect_bl,
        total_incorrect_rt
    ]
    return outcomes_df, total


def darken_color(color, amount=0.7):
    c = mcolors.to_rgb(color)
    return tuple(max(0, min(1, channel * amount)) for channel in c)


def plot_nssm(
        fig,
        ax,
        matrix,
        cmap,
        wt_positions,
        splice_junctions=None,
        vmin=-6,
        vmax=6,
        clinvar_labels=None
    ):
    matrix = np.asarray(matrix, dtype='float64')
    masked = np.ma.masked_invalid(matrix)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    num_rows, num_cols = masked.shape

    wt_pos_set = set((x, y) for x, y in wt_positions)

    # Create colormap-mapped heat array
    for row in range(num_rows):
        for col in range(num_cols):
            value = masked[row, col]
            is_masked = masked.mask[row, col]
            color = 'white' if is_masked else cmap(norm(value))
            rect = plt.Rectangle(
                (col, num_rows - 1 - row), 1, 1,
                facecolor=color,
                edgecolor=None,  # Remove grid lines
                zorder=1
            )
            ax.add_patch(rect)

            # Draw 'X' if NaN and not a WT position
            if is_masked and (col, row) not in wt_pos_set:
                ax.text(
                    col + 0.5, num_rows - row-0.4, 'X',
                    ha='center', va='center',
                    fontsize=2, color='black',
                    zorder=5
                )

    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # Draw WT squares fully
    for x, y in wt_positions:
        if 0 <= x < num_cols and 0 <= y < num_rows:
            ax.add_patch(
                plt.Rectangle((x, num_rows - 1 - y), 1, 1,
                              facecolor='black', edgecolor=None, zorder=10)
            )

    # Plot ClinVar labels
    if clinvar_labels:
        for pos, label, color in clinvar_labels:
            edge_color = darken_color(color)
            ax.plot(pos + 0.5, -1, marker='o',
                    transform=ax.transData, clip_on=False,
                    color=color,alpha=0.6, markersize=1.8,
                    markeredgecolor=edge_color, markeredgewidth=0.3, zorder=12)
            ax.text(pos + 0.5, -1.8, label,
                    transform=ax.transData, clip_on=False,
                    ha='center', va='bottom', fontsize=2.5,
                    color='black', rotation=90, zorder=13)

        handles = [
            plt.Line2D([0], [0], marker='o', linestyle='None',
                       color='w', markerfacecolor=v,
                       markeredgecolor=darken_color(v),
                       markeredgewidth=0.3, markersize=4,
                       label=k.replace("_", " "))
            for k, v in CLINVAR_COLORS.items()
        ]
        leg = fig.legend(
            handles=handles,
            loc='center left',           # use middle-left of legend
            bbox_to_anchor=(0.01, 0.6),  # halfway up the figure
            fontsize=3,
            frameon=False,
            borderpad=0,
            handletextpad=0.5
        )
        
        # shrink markers only in legend (not in the plot)
        try:
            for lh in leg.legendHandles:
                lh.set_markersize(2)  # smaller legend dots
        except AttributeError as e:
            for lh in leg.legend_handles:
                lh.set_markersize(2)  # smaller legend dots


    # Dummy image for colorbar
    dummy = ax.imshow(
        np.array([[vmin, vmax]]),
        cmap=cmap,
        visible=False,
        aspect='auto',
        extent=[0, 0, 0, 0]
    )
    return dummy


def setup_axes(ax, title, splice_junctions=None):
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0.5, 161.5, 10))
    ax.set_xticklabels(np.arange(0, 161, 10), fontsize=4)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(['A', 'C', 'G', 'T'], fontsize=2.3)
    ax.set_title(title, fontsize=6)
    ax.tick_params(axis='both', width=0.3, length=1,pad=0.5)
    if splice_junctions:
        sj1 = splice_junctions[0]
        sj2 = splice_junctions[0] + splice_junctions[1]
        ax.axvline(sj1, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(sj2, color='black', linestyle='--', linewidth=0.5)
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks([sj1, sj2])
        ax_top.set_xticklabels(["SA", "SD"], fontsize=5)
        ax_top.tick_params(width=0.5, length=10, pad=0.8)
        ax_top.spines["top"].set_visible(False)
    return


def calc_r_with_nan(x, y, stat="pearson", model=None):
    """
    since we are using the MMSplice data, 
    there shouldnt be any NaNs. However,
    check for those just in case
    """
    if model is None:
        pass
    else:
        x = model.predict(x)
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


def integral(values, w=0.5):
    if len(values) < 2:
        return 0
    lh = values[-2]
    rh = values[-1]
    h = np.abs(max(lh, rh)) - np.abs(min(lh, rh))
    area = 0.5 * w * h
    rec_h = max(np.abs(min(lh, rh)), 0)
    rec_area = w * rec_h
    return area + rec_area


def get_plot_bounds(smn):
    bl_l = 0
    bl_r = len(smn["bl"]) - 1
    rt_l = 0
    rt_r = len(smn["rt"]) - 1

    for i, val in enumerate(smn["bl"]):
        if val != 0:
            bl_l = i
            break
            
    for i, val in enumerate(smn["rt"]):
        if val != 0:
            rt_l = i
            break

    for i in range(len(smn["bl"])-1, -1, -1):
        val = smn["bl"][i]
        if val != 0:
            bl_r = i
            break

    for i in range(len(smn["rt"])-1, -1, -1):
        val = smn["rt"][i]
        if val != 0:
            rt_r = i
            break    

    plot_l = max(min(bl_l, rt_l) - 1, 0)
    plot_r = min(max(bl_r, rt_r) + 1, len(smn["bl"]) - 1)
    return plot_l, plot_r


def get_thresholds(min_, max_, bin_):
    double_thresholds = []
    while min_ < max_:
        double_thresholds.append(
            (min_,
            min_ + bin_)
        )
        min_ = min_ + bin_
    
    db_threshold_keys = [threshold_str(d) for d in double_thresholds]
    return double_thresholds, db_threshold_keys
    

def generate_density(double_thresholds, threshold_func=two_threshold_predictions, errf=mse):
    smn = {"bl" : [], "rt" : []}
    totals = []
    area_bl = 0
    area_rt = 0
    for threshold in double_thresholds:
        l, r = threshold
        outcomes, total = threshold_func(l, r, errf=errf)
        totals.append(total)
        if total == 0:
            total = 1
        smn["bl"].append(outcomes.loc["Baseline MMSplice", "Total correct"] / total)
        smn["rt"].append(outcomes.loc["Retrained MMSplice", "Total correct"] / total)
        area_bl += integral(smn["bl"])
        area_rt += integral(smn["rt"])
    return smn, {"bl": area_bl, "rt": area_rt}, totals


def density_threshold_plot(
        min_,
        max_,
        bin_,
        threshold_func=two_threshold_predictions,
        errf=mse,
        title=None,
        xlabel=None,
        ylabel=None,
    ):
    fig, ax = plt.subplots(1)
    if title is None:
        ax.set_title(
            "Baseline vs retrained MMSplice\n"
            "Prediction concordance with observed values,\n"
            "separated by observed ∆logit range"
        )
    else:
        ax.set_title(title)
    if xlabel is None:
        ax.set_xlabel("Observed ∆logit value range")
    else:
        ax.set_xlabel(xlabel)

    double_thresholds, db_threshold_keys = get_thresholds(min_, max_, bin_)
    smn, area, _ = generate_density(double_thresholds, threshold_func=threshold_func, errf=errf)
    area_bl = area["bl"]
    area_rt = area["rt"]
    plot_l, plot_r = get_plot_bounds(smn)
    
    smn["bl"] = smn["bl"][plot_l:plot_r+1]
    smn["rt"] = smn["rt"][plot_l:plot_r+1]
    db_threshold_keys_copy = db_threshold_keys[plot_l:plot_r+1]
    
    area_bl = str(round(area_bl, 3))
    area_rt = str(round(area_rt, 3))
    ax.plot(db_threshold_keys_copy, smn["bl"])
    ax.plot(db_threshold_keys_copy, smn["rt"])
    
    ax.legend([f"Baseline (AUC={area_bl})", f"Retrained (AUC={area_rt})"], loc="best")
    ax.tick_params(axis='x', rotation=90)

    if ylabel is None:
        fig.supylabel(
            "Proportion of variants correctly\n"
            "predicted to fall in ∆logit range"
        )
    else:
        fig.supylabel(ylabel)
        
    plt.tight_layout()
    return
    

def float_print(*args):
    string = ""
    for i in args:
        s = str(round(i, 3)).ljust(5)
        string += s + " "
    print(string)


def read_integer(x, default):
    if isinstance(x, int):
        return x
    elif isinstance(x, str):
        if not x:
            return default
        return int(x)
    else:
        type_ = type(x)
        raise TypeError("input integer must be 'str' or 'int', got '{type_}'")


def read_df(path):
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(
            f"Expected file to be .csv or .tsv, got path '{path}'"
        )
    return


def stacked_barplot(
        improved,
        threshold,
        scale="dlogit",
        splits="Test",
        save=True,
        plot_dirs=None,
        plot_name=None,
        y_scale="linear",
        err_func="MSE",
        split_legend=False,
    ):


    if plot_dirs is None and save:
        raise ValueError("plot_dirs not provided but save is True")

    sorted_indx = {}
    improved_ = {}
    for split in improved:
        df = improved[split]
        keys = [
            i for i in df
            if "corrected" in i
            and str(threshold) in i
            and "%" not in i
        ]
        df = df[keys + ["gene_exon"]]
        matrix = df.set_index("gene_exon").to_numpy()
        sum_rows = np.sum(matrix, axis=1)
        indexes = np.argsort(sum_rows)
        sorted_indx[split] = indexes
        improved_[split] = df

    fig, ax = plt.subplots(figsize=(3.3, 2.3), constrained_layout=True)
    bottom = [0] * len(splits)
    N = 0
    for i in improved_:
        N = max(N, len(improved_[i]))
    bars_df = defaultdict(list)

    for i in range(N):
        for split in splits:
            indexes = sorted_indx[split]
            back_index = len(indexes) - 1 - i
            if back_index < 0:
                bars_df[split + "_n"].append(0)
                bars_df[split + "_exon"].append("")
                continue
            indx = indexes[len(indexes) - 1 - i]
            row = improved_[split].iloc[indx]
            exon = row["gene_exon"]
            value = row[f"corrected_threshold_{threshold}"]
            bars_df[split + "_n"].append(value)
            if False:
                bars_df[split + "_exon"].append("other")
            else:
                bars_df[split + "_exon"].append(exon)

    color_key = [
        "tab:blue",
        "tab:orange",
        "tab:red",
        "tab:green",
        "tab:brown",
        "tab:purple",
        "tab:pink",
        "tab:cyan",
        "tab:olive",
        "blue",
        "tab:gray",
    ]
    try:
        bars_df = pd.DataFrame(bars_df)
    except ValueError:
        return bars_df


    textures_ref = ['', '\\\\\\', '***', '||', '']
    textures = {}
    for i in range(len(splits)):
        textures[splits[i]] = textures_ref[min(i, len(textures_ref) - 1)]

    split_legend_handle = defaultdict(list)
    split_legend_label = defaultdict(list)
    legend_handles = []
    legend_labels = []
    gray_values = []
    for i, row in bars_df.iterrows():
        if 10 < i < len(bars_df) - 1:
            gray_values = bars_df[[s + "_n" for s in splits]].iloc[i]
            values += gray_values
            continue
        elif i < 10: 
            values = bars_df[[s + "_n" for s in splits]].iloc[i]
            exons = bars_df[[s + "_exon" for s in splits]].iloc[i]
        elif i == 10:
            values = bars_df[[s + "_n" for s in splits]].iloc[i]
            continue
        elif i == len(bars_df) - 1:
            exons = {s + "_exon" : f"Other ({s})" for s in splits} 
        
        colors = ["white"] * len(splits)
        colors[0] = color_key[min(i, len(color_key) - 1)]
        lw = [1] * len(splits)
        lw[0] = 0

        bars = ax.bar(
            splits,
            values,
            bottom=bottom,
            label="",
            color=colors,
            linewidth=lw,
        )
        for j, s in enumerate(splits):
            # rgba_color = bars[j].get_facecolor()
            # rgba_darker = [i * 0.6 for i in rgba_color]
            # bars[j].set_edgecolor(mcolors.to_hex(rgba_darker))
            if j > 0:
                bars[j].set_edgecolor(color_key[min(i, len(color_key) - 1)])
            else:
                bars[j].set_edgecolor("black")
            bars[j].set_hatch(textures[s])
            
            split_legend_handle[s].append(bars[j])
            split_legend_label[s].append(exons[s + "_exon"])
            legend_handles.append(bars[j])
            legend_labels.append(exons[s + "_exon"])
        bottom = [b + v for b, v in zip(bottom, values)]

    # for j, s in enumerate(splits):
    #     split_legend_handle[s].append(bars[j])
    #     split_legend_label[s].append(f"Other ({s})")

    scale_title = scale
    scale_title = scale.replace("d", "Δ")
    scale_title = scale_title.replace("logit", "logit(PSI)")
    scale_title = scale_title.replace("psi", "PSI")

    ax.set_xlabel("Split", fontsize=6)
    ax.set_ylabel(f"Number of variants with\nlower {err_func} in retrained MMSplice", fontsize=6)
    ax.tick_params(axis='both', labelsize=7)
    # if y_scale == "log":
    #     def power_of_ten(x, pos):
    #         return f"$10^{{{int(x)}}}$"
    #     ax.yaxis.set_major_formatter(ticker.FuncFormatter(power_of_ten))
    fig.suptitle(
        f"MMSplice retrained variants with lower {err_func}\nthan "
        f"baseline MMSplice per exon family,\n"
        f"measured |{scale_title}| > {threshold}", fontsize=8,
        # y=0.98,
        # x=0.72/2,
    )
    if y_scale == "log":
        plt.yscale("log")

    if not split_legend:
        ax.legend(
            legend_handles,
            legend_labels,
            title=f"Exon Family",
            loc="best",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=2,
            title_fontsize=4,
        )
    else:
        y = 1.0
        legends = []
        n = 0
        for s in split_legend_handle:
            if n == len(split_legend_handle) - 1:
                final = s
                break
            legend = ax.legend(
                split_legend_handle[s],
                split_legend_label[s],
                title=f"{s.title()} exon family",
                bbox_to_anchor=(1.02, y),
                fontsize=3,
                title_fontsize=4,
            )
            y -= 0.55
            legends.append(legend)
            ax.add_artist(legends[-1])
            n += 1

        ax.legend(
            split_legend_handle[final],
            split_legend_label[final],
            title=f"{final.title()} exon family",
            bbox_to_anchor=(1.02, y),
            fontsize=3,
            title_fontsize=4,
        )
        
    if plot_name is None:
        plot_name = f"top_exons_mmsplice_{threshold}_{scale}"
    # plt.tight_layout(rect=[0, 0, 0.72, 1])
    # plt.tight_layout()
    if not save:
        return split_legend_handle
    for plot_dir in plot_dirs:
        plot_path = os.path.join(
            plot_dir,
            plot_name
        )
        plt.savefig(plot_path + ".pdf", bbox_inches="tight", dpi=300)
        print("Saved:", plot_path + ".pdf")
        plt.savefig(plot_path + ".tif", bbox_inches="tight", dpi=300)
        print("Saved:", plot_path + ".tif")
        plt.savefig(plot_path + ".png", bbox_inches="tight", dpi=300)
        print("Saved:", plot_path + ".png")
    plt.show()
    plt.close()
    return


help_="""
Make NSSMs, binned dlogit analyses
and tables.
"""
inputs = [
    "output_dir=",
    "all_predictions_path=",
    "mse_improvements_dir=",
    "clinvar_annotations=",
    "nssm_exons=",
    "min_family_size=",
    "nssm_vmin=",
    "nssm_vmax=",
    "debug",
    "help",
]
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:],"", inputs)
    opts = dict(opts)
    if "--help" in opts:
        print(help_)
        sys.exit(0)
    debug = "--debug" in opts
    if debug:
        output_dir = (
            "/storage/yu/Projects/P000--ESL_continuation/Analysis/exon_analysis_10_14" 
        )
        all_samples_path = (
            "/storage/yu/Projects/P000--ESL_continuation/GitHub/"
            "P000--Exon_Skipping_Library/models/mmsplice_revised/mmsplice139/"
            "figures/mmsplice_exons_test_test_all_predictions.tsv"
        )

        clinvar_annotations_path = (
            "/storage/yu/Projects/P000--ESL_continuation/"
            "Data/supertable_ClinVar_matched_in_supertable.tsv"
        )
        mse_improvements_dir = os.path.join(
            "/storage",
            "yu",
            "Projects",
            "P000--ESL_continuation",
            "GitHub",
            "P000--Exon_Skipping_Library",
            "models",
            "mmsplice_revised",
            "mmsplice139",
            "improved_mse_mae_exons",
        )
        exons = "SCN5A exon 24,SHOX2 exon 2"
        min_family_size = 20
        vmin = -6
        vmax = 6
    else:
        output_dir = opts["--output_dir"]
        all_samples_path = opts["--all_predictions_path"]
        clinvar_annotations_path = opts["--clinvar_annotations"]
        mse_improvements_dir = opts["--mse_improvements_dir"]
        exons = opts.get("--nssm_exons", "ALL")
        min_family_size = read_integer(opts.get("--min_family_size", "20"), 20)
        vmin = read_integer(opts.get("--nssm_vmin", "-6"), -6)
        vmax = read_integer(opts.get("--nssm_vmax", "6"), 6)
        print(opts, flush=True)
        
    all_samples = read_df(all_samples_path)
    clinvar_annotations = read_df(clinvar_annotations_path)
    
    sub_dirs = [
        "SSMs",
        "Count_aggregate",
        "Small_vs_large",
        "density_threshold_binned",
        "density_mse_mae_binned",
        "heatmap_all_bins",
        "metrics_positional_info",
        "stacked_barplots",
    ]

    os.makedirs(output_dir, exist_ok=True)
    for i in sub_dirs:
        os.makedirs(os.path.join(output_dir, i), exist_ok=True)
    single_vars = all_samples[~all_samples["snp"].str.contains(";")]
    scn5a = single_vars[single_vars["gene_exon"] == "SCN5A exon 24"]
    shox2 = single_vars[single_vars["gene_exon"] == "SHOX2 exon 2"]
   
    if exons and exons != "ALL":
        exons = exons.replace("_", " ")
        exons = exons.split(",") 
    else:
        exons = list(set(single_vars["gene_exon"].tolist()))
    

    if debug:
        breakpoint()
    # make NSSM plot for all exon queries
    for i, e in enumerate(exons):
        nts_true = np.ones((4, 161)) * np.nan
        nts_baseline = np.ones((4, 161)) * np.nan
        nts_retrained = np.ones((4, 161)) * np.nan
        gene = e
        df = single_vars[single_vars["gene_exon"] == e]
        splice_junctions = (
            (df["intron1_len"].iloc[0]),
            (df["exon_len"].iloc[0]),
            (161 - df["intron1_len"].iloc[0] - df["exon_len"].iloc[0])
        )
        if len(df) < min_family_size:
            continue
        ref_locs = []
        ref_nts = [""] * 161
        for i, row in df.iterrows():
            y = row['True_Delta_Logit']
            b = row['Baseline_MMSplice_Predicted_Delta_Logit']
            rt = row['Retrained_MMSplice_Predicted_Delta_Logit']
            loc, ref, var = get_snp_loc(row["snp"])
            if loc < 0:
                continue
            nts_true[3-var][loc] = y
            nts_baseline[3-var][loc] = b
            nts_retrained[3-var][loc] = rt
            ref_locs.append((loc, 3-ref))
            ref_nts[loc] = nt_list[ref]

        mse_baseline = np.mean(
            (
                df['Baseline_MMSplice_Predicted_Delta_Logit']
                - df['True_Delta_Logit'])
            ** 2)
        mae_baseline = np.mean(
            np.abs(df['Baseline_MMSplice_Predicted_Delta_Logit']
            - df['True_Delta_Logit'])
        )
        mse_baseline = str(round(mse_baseline, 3))
        mae_baseline = str(round(mae_baseline, 3))
        rBL = calc_r_with_nan(
            df['Baseline_MMSplice_Predicted_Delta_Logit'],
            df['True_Delta_Logit']
        )
        rBL = str(round(rBL[0], 3))
        mse_retrained = np.mean(
            (
                df['Retrained_MMSplice_Predicted_Delta_Logit']
                - df['True_Delta_Logit'])
            ** 2)
        mae_retrained = np.mean(
            np.abs(df['Retrained_MMSplice_Predicted_Delta_Logit']
            - df['True_Delta_Logit'])
        )
        rRT = calc_r_with_nan(
            df['Retrained_MMSplice_Predicted_Delta_Logit'],
            df['True_Delta_Logit']
        )
        rRT = str(round(rRT[0], 3))
        mse_retrained = str(round(mse_retrained, 3))
        mae_retrained = str(round(mae_retrained, 3))

        clinvar_df = clinvar_annotations[clinvar_annotations["gene"].apply(lambda x : x in e)].copy()
        clinvar_labels = get_all_annotations(clinvar_df)
        clinvar_labels = filter_clinvar_labels(clinvar_labels, nts_true)
            
        fig, ax = plt.subplots(3, figsize=(6,2.5))
        cmap = plt.colormaps.get_cmap('RdBu').copy()
        cmap.set_bad(color='white')

        true = plot_nssm(
            fig,
            ax[0],
            nts_true,
            cmap,
            ref_locs,
            splice_junctions=splice_junctions,
            vmin=vmin,
            vmax=vmax,
            clinvar_labels=clinvar_labels
        )
        baseline = plot_nssm(
            fig,
            ax[1],
            nts_baseline,
            cmap,
            ref_locs,
            splice_junctions=[],
            vmin=vmin,
            vmax=vmax,
            clinvar_labels=clinvar_labels
        )
        retrained = plot_nssm(
            fig,
            ax[2],
            nts_retrained,
            cmap,
            ref_locs,
            splice_junctions=[],
            vmin=vmin,
            vmax=vmax,
            clinvar_labels=clinvar_labels
        )
        setup_axes(ax[0], f"{e} COMPASS Measurements", splice_junctions)
        setup_axes(ax[1],
            f"{e} MMSplice Baseline MAE = {mae_baseline}, "
            f"MSE = {mse_baseline}, r = {rBL}",
            splice_junctions
        )
        setup_axes(ax[2],
            f"{e} MMSplice Retrained MAE = {mae_retrained}, "
            f"MSE = {mse_retrained}, r = {rRT}",
            splice_junctions
        )
        fig.subplots_adjust(right=0.85)
        add_colorbar(fig, true, [0.87, 0.25, 0.01, 0.5], label="Δlogit(PSI)")
        ssm_plots_dir = os.path.join(
            output_dir,
            "SSMs",
            "plots"
        )
        os.makedirs(ssm_plots_dir, exist_ok=True)
        plt.savefig(os.path.join(ssm_plots_dir, f"{e.replace(' ', '_')}.tif"), dpi=300)
        plt.savefig(os.path.join(ssm_plots_dir, f"{e.replace(' ', '_')}.png"), dpi=300)
        plt.close()
    
    
    df = single_vars[single_vars["gene_exon"] == "SCN5A exon 24"]
    df[df['threshold_1.0_status'] == "corrected"][
    [
        "gene_exon",
        "snp",
        "Baseline_MMSplice_Predicted_Delta_Logit",
        "Retrained_MMSplice_Predicted_Delta_Logit",
        "True_Delta_Logit",
    ]
    ]
    df.to_csv(
        os.path.join(
            output_dir,
            "SSMs",
            "SCN5A_corrected_threshold_1.csv"
        ),
        index=False
    )

    df = single_vars[single_vars["gene_exon"] == "SHOX2 exon 2"]
    shox2_02 = df[df['threshold_0.2_status'] == "corrected"][
    [
        "gene_exon",
        "snp",
        "Baseline_MMSplice_Predicted_Delta_Logit",
        "Retrained_MMSplice_Predicted_Delta_Logit",
        "True_Delta_Logit",
    ]
    ]
    shox2_02.to_csv(
        os.path.join(
            output_dir,
            "SSMs",
            "SHOX2_corrected_threshold_02.csv"
        ),
        index=False
    )

    df = single_vars[single_vars["gene_exon"] == "SHOX2 exon 2"]
    df[df['threshold_1.0_status'] == "corrected"][
    [
        "gene_exon",
        "snp",
        "Baseline_MMSplice_Predicted_Delta_Logit",
        "Retrained_MMSplice_Predicted_Delta_Logit",
        "True_Delta_Logit",
    ]
    ]
    shox2_02.to_csv(
        os.path.join(
            output_dir,
            "SSMs",
            "SHOX2_corrected_threshold_1.csv"
        ),
        index=False
    )

    baseline_col = [i for i in all_samples if "Baseline" in i][0]
    true_col = [i for i in all_samples if "True" in i][0]
    retrained_col = [i for i in all_samples if "Retrained" in i][0]

    print(baseline_col, true_col, retrained_col)


    mse_col = "Num Exon Families Improved MSE"
    r_col = "Num Exon Families Improved R"

    outcomes_df = {
        mse_col : [0, 0],
        r_col : [0, 0],
    }
    outcomes_df = pd.DataFrame(outcomes_df)
    outcomes_df.index = ["Baseline Better", "Retrained Better"]

    for i, e in enumerate(exons):
        df = all_samples[all_samples["gene_exon"] == e]
        mse_baseline = np.mean(
            (
                df['Baseline_MMSplice_Predicted_Delta_Logit']
                - df['True_Delta_Logit'])
            ** 2)
        rBL = calc_r_with_nan(
            df['Baseline_MMSplice_Predicted_Delta_Logit'],
            df['True_Delta_Logit']
        )[0]
        mse_retrained = np.mean(
            (
                df['Retrained_MMSplice_Predicted_Delta_Logit']
                - df['True_Delta_Logit'])
            ** 2)
        rRT = calc_r_with_nan(
            df['Retrained_MMSplice_Predicted_Delta_Logit'],
            df['True_Delta_Logit']
        )[0]

        if mse_baseline < mse_retrained:
            outcomes_df.loc["Baseline Better", mse_col] += 1
        elif mse_retrained < mse_baseline:    
            outcomes_df.loc["Retrained Better", mse_col] += 1

        if rBL > rRT:
            outcomes_df.loc["Baseline Better", r_col] += 1
        elif rRT > rBL:
            outcomes_df.loc["Retrained Better", r_col] += 1
    

    outcomes_df.to_csv(
        os.path.join(
            output_dir,
            "Count_aggregate",
            "mse_and_r_aggregate_by_exon.csv"
    ))


    threshold = 1.0
    outcomes_df = {
        "Predicted small" : [0, 0],
        "Predicted large positive" : [0, 0],
        "Predicted large negative" : [0, 0],
    }
    outcomes_df = pd.DataFrame(outcomes_df)
    outcomes_df.index = ["Baseline MMSplice", "Retrained MMSplice"]

    total = 0
    for i, row in all_samples.iterrows():
        rt = row[retrained_col]
        bl = row[baseline_col]
        tr = row[true_col]
        mse_bl = mse(bl, tr)
        mse_rt = mse(rt, tr)
        
        if np.abs(tr) > threshold:
            continue

        if bl > threshold:
            outcomes_df.loc["Baseline MMSplice", "Predicted large positive"] += 1
        elif bl < -threshold:
            outcomes_df.loc["Baseline MMSplice", "Predicted large negative"] += 1
        else:
            outcomes_df.loc["Baseline MMSplice", "Predicted small"] += 1
            
        if rt > threshold:
            outcomes_df.loc["Retrained MMSplice", "Predicted large positive"] += 1
        elif rt < -threshold:
            outcomes_df.loc["Retrained MMSplice", "Predicted large negative"] += 1
        else:
            outcomes_df.loc["Retrained MMSplice", "Predicted small"] += 1
        total += 1
    

    outcomes_df.to_csv(
        os.path.join(
            output_dir,
            "Small_vs_large",
            "above_magnitude_1_predictions.csv"
    ))


    threshold = 1.0
    outcomes_df = {
        "Predicted small" : [0, 0],
        "Predicted large positive" : [0, 0],
        "Predicted large negative" : [0, 0],
    }
    outcomes_df = pd.DataFrame(outcomes_df)
    outcomes_df.index = ["Baseline MMSplice", "Retrained MMSplice"]

    total = 0
    for i, row in all_samples.iterrows():
        rt = row[retrained_col]
        bl = row[baseline_col]
        tr = row[true_col]
        mse_bl = mse(bl, tr)
        mse_rt = mse(rt, tr)
        
        if not (tr > threshold):
            continue

        if bl > threshold:
            outcomes_df.loc["Baseline MMSplice", "Predicted large positive"] += 1
        elif bl < -threshold:
            outcomes_df.loc["Baseline MMSplice", "Predicted large negative"] += 1
        else:
            outcomes_df.loc["Baseline MMSplice", "Predicted small"] += 1
        if rt > threshold:
            outcomes_df.loc["Retrained MMSplice", "Predicted large positive"] += 1
        elif rt < -threshold:
            outcomes_df.loc["Retrained MMSplice", "Predicted large negative"] += 1
        else:
            outcomes_df.loc["Retrained MMSplice", "Predicted small"] += 1
            
        total += 1
        
    outcomes_df.to_csv(
        os.path.join(
            output_dir,
            "Small_vs_large",
            "above_threshold_1_predictions.csv"
    ))

        
        
    threshold = -1.0


    outcomes_df = {
        "Predicted small" : [0, 0],
        "Predicted large positive" : [0, 0],
        "Predicted large negative" : [0, 0],
    }
    outcomes_df = pd.DataFrame(outcomes_df)
    outcomes_df.index = ["Baseline MMSplice", "Retrained MMSplice"]

    total = 0
    for i, row in all_samples.iterrows():
        rt = row[retrained_col]
        bl = row[baseline_col]
        tr = row[true_col]
        mse_bl = mse(bl, tr)
        mse_rt = mse(rt, tr)
        
        if not (tr < threshold):
            continue

        if bl > np.abs(threshold):
            outcomes_df.loc["Baseline MMSplice", "Predicted large positive"] += 1
        elif bl < threshold:
            outcomes_df.loc["Baseline MMSplice", "Predicted large negative"] += 1
        else:
            outcomes_df.loc["Baseline MMSplice", "Predicted small"] += 1
        if rt > np.abs(threshold):
            outcomes_df.loc["Retrained MMSplice", "Predicted large positive"] += 1
        elif rt < threshold:
            outcomes_df.loc["Retrained MMSplice", "Predicted large negative"] += 1
        else:
            outcomes_df.loc["Retrained MMSplice", "Predicted small"] += 1
            
        total += 1

    outcomes_df.to_csv(
        os.path.join(
            output_dir,
            "Small_vs_large",
            "below_threshold_-1_predictions.csv"
    ))
        

    df,_ = thresholded_predictions(threshold=0.1, small=True)
    df.to_csv(os.path.join(
        output_dir,
        "Small_vs_large",
        "below_threshold_01_small_predictions.csv"
    ))

    df,_ = thresholded_predictions(threshold=-0.1, small=True)
    df.to_csv(os.path.join(
        output_dir,
        "Small_vs_large",
        "above_threshold_-01_small_predictions.csv"
    ))

    df,_ =thresholded_predictions(threshold=0.1, small=False)
    df.to_csv(os.path.join(
        output_dir,
        "Small_vs_large",
        "above_threshold_01_large_predictions.csv"
    ))
    df,_ = thresholded_predictions(threshold=-0.1, small=False)
    df.to_csv(os.path.join(
        output_dir,
        "Small_vs_large",
        "below_threshold_-01_large_predictions.csv"
    ))
    thresholds = np.array([i/10 for i in range(1, 21)])

    fig, ax = plt.subplots(2,2)
    ax[0][0].set_title("Small negative variant prediction")
    ax[0][1].set_title("Large negative variant prediction")
    ax[1][0].set_title("Small positive variant prediction")
    ax[1][1].set_title("Large positive variant prediction")
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xlabel("threshold")

    smn = {"bl" : [], "rt" : []}
    lrn = {"bl" : [], "rt" : []}
    smp = {"bl" : [], "rt" : []}
    lrp = {"bl" : [], "rt" : []}


    for threshold in thresholds:
        outcomes, total = thresholded_predictions(threshold=-threshold, small=True)
        smn["bl"].append(outcomes.loc["Baseline MMSplice", "Total correct"] / total)
        smn["rt"].append(outcomes.loc["Retrained MMSplice", "Total correct"] / total)
    ax[0][0].plot(-thresholds, smn["bl"])
    ax[0][0].plot(-thresholds, smn["rt"])

    for threshold in thresholds:
        outcomes, total = thresholded_predictions(threshold=-threshold, small=False)
        lrn["bl"].append(outcomes.loc["Baseline MMSplice", "Total correct"] / total)
        lrn["rt"].append(outcomes.loc["Retrained MMSplice", "Total correct"] / total)
    ax[0][1].plot(-thresholds, lrn["bl"])
    ax[0][1].plot(-thresholds, lrn["rt"])

    for threshold in thresholds:
        outcomes, total = thresholded_predictions(threshold=threshold, small=True)
        smp["bl"].append(outcomes.loc["Baseline MMSplice", "Total correct"] / total)
        smp["rt"].append(outcomes.loc["Retrained MMSplice", "Total correct"] / total)
    ax[1][0].plot(thresholds, smp["bl"])
    ax[1][0].plot(thresholds, smp["rt"])

    for threshold in thresholds:
        outcomes, total = thresholded_predictions(threshold=threshold, small=False)
        lrp["bl"].append(outcomes.loc["Baseline MMSplice", "Total correct"] / total)
        lrp["rt"].append(outcomes.loc["Retrained MMSplice", "Total correct"] / total)
    ax[1][1].plot(thresholds, lrp["bl"])
    ax[1][1].plot(thresholds, lrp["rt"])

    ax[0][1].legend(["Baseline", "Retrained"], loc="best")


    # fig.legend(["baseline", "retrained"], loc=(0.78,0.82))
    fig.supylabel("Proportion correct variants")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            "Small_vs_large",
            "proportion_variant_size_prediction_single_threshold_binned_05.pdf"
        )
    )
    
    fig, ax = plt.subplots(2,2)
    ax[0][0].set_title("Small negative variant prediction")
    ax[0][1].set_title("Large negative variant prediction")
    ax[1][0].set_title("Small positive variant prediction")
    ax[1][1].set_title("Large positive variant prediction")
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xlabel("threshold")

    smn = {"bl" : [], "rt" : []}
    lrn = {"bl" : [], "rt" : []}
    smp = {"bl" : [], "rt" : []}
    lrp = {"bl" : [], "rt" : []}

    for threshold in thresholds:
        outcomes = mse_predictions(threshold=-threshold, small=True)
        smn["bl"].append(outcomes.loc["Baseline MMSplice", "MSE"])
        smn["rt"].append(outcomes.loc["Retrained MMSplice", "MSE"] )
    ax[0][0].plot(-thresholds, smn["bl"])
    ax[0][0].plot(-thresholds, smn["rt"])

    for threshold in thresholds:
        outcomes = mse_predictions(threshold=-threshold, small=False)
        lrn["bl"].append(outcomes.loc["Baseline MMSplice", "MSE"] )
        lrn["rt"].append(outcomes.loc["Retrained MMSplice", "MSE"] )
    ax[0][1].plot(-thresholds, lrn["bl"])
    ax[0][1].plot(-thresholds, lrn["rt"])

    for threshold in thresholds:
        outcomes = mse_predictions(threshold=threshold, small=True)
        smp["bl"].append(outcomes.loc["Baseline MMSplice", "MSE"])
        smp["rt"].append(outcomes.loc["Retrained MMSplice", "MSE"])
    ax[1][0].plot(thresholds, smp["bl"])
    ax[1][0].plot(thresholds, smp["rt"])

    for threshold in thresholds:
        outcomes = mse_predictions(threshold=threshold, small=False)
        lrp["bl"].append(outcomes.loc["Baseline MMSplice", "MSE"] )
        lrp["rt"].append(outcomes.loc["Retrained MMSplice", "MSE"] )
    ax[1][1].plot(thresholds, lrp["bl"])
    ax[1][1].plot(thresholds, lrp["rt"])

    ax[0][1].legend(["Baseline", "Retrained"], loc="best")

    # fig.legend(["baseline", "retrained"], loc=(0.78,0.62))
    fig.supylabel("MSE")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            "Small_vs_large",
            "mse_variant_size_prediction_single_threshold_binned_05.pdf"
        )
    )


    fig, ax = plt.subplots(2,2)
    ax[0][0].set_title("Small negative variant prediction")
    ax[0][1].set_title("Large negative variant prediction")
    ax[1][0].set_title("Small positive variant prediction")
    ax[1][1].set_title("Large positive variant prediction")
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xlabel("dlogit threshold")
            # ax[i][j].set_ylim(0, 1)

    smn = {"bl" : [], "rt" : []}
    lrn = {"bl" : [], "rt" : []}
    smp = {"bl" : [], "rt" : []}
    lrp = {"bl" : [], "rt" : []}


    for threshold in thresholds:
        outcomes = mse_predictions(threshold=-threshold, small=True)
        smn["bl"].append(outcomes.loc["Baseline MMSplice", "R"])
        smn["rt"].append(outcomes.loc["Retrained MMSplice", "R"] )
    ax[0][0].plot(-thresholds, smn["bl"])
    ax[0][0].plot(-thresholds, smn["rt"])

    for threshold in thresholds:
        outcomes = mse_predictions(threshold=-threshold, small=False)
        lrn["bl"].append(outcomes.loc["Baseline MMSplice", "R"] )
        lrn["rt"].append(outcomes.loc["Retrained MMSplice", "R"] )
    ax[0][1].plot(-thresholds, lrn["bl"])
    ax[0][1].plot(-thresholds, lrn["rt"])

    for threshold in thresholds:
        outcomes = mse_predictions(threshold=threshold, small=True)
        smp["bl"].append(outcomes.loc["Baseline MMSplice", "R"])
        smp["rt"].append(outcomes.loc["Retrained MMSplice", "R"])
    ax[1][0].plot(thresholds, smp["bl"])
    ax[1][0].plot(thresholds, smp["rt"])
    for threshold in thresholds:
        outcomes = mse_predictions(threshold=threshold, small=False)
        lrp["bl"].append(outcomes.loc["Baseline MMSplice", "R"] )
        lrp["rt"].append(outcomes.loc["Retrained MMSplice", "R"] )
    ax[1][1].plot(thresholds, lrp["bl"])
    ax[1][1].plot(thresholds, lrp["rt"])

    ax[0][1].legend(["Baseline", "Retrained"], loc="best")

    # fig.legend(["baseline", "retrained"], loc=(0.77,0.825))
    fig.supylabel("Pearson R")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            "Small_vs_large",
            "r_variant_size_prediction_single_threshold_binned_05.pdf"
        )
    )

    min_ = -11.5
    max_ = 11.5
    bin_ = 0.2

    double_thresholds = []
    while min_ < max_:
        double_thresholds.append(
            (min_,
            min_ + bin_)
        )
        min_ = min_ + bin_

    db_threshold_keys = [threshold_str(d) for d in double_thresholds]


    fig, ax = plt.subplots(1)
    ax.set_title(
        "Baseline vs retrained MMSplice\n"
        "Prediction concordance with observed values,\n"
        "separated by observed ∆logit range"
    )
    for i in range(2):
        for j in range(2):
            ax.set_xlabel("Observed ∆logit value range")

    smn = {"bl" : [], "rt" : []}
    area_bl = 0
    area_rt = 0

    for threshold in double_thresholds:
        l, r = threshold
        outcomes, total = two_threshold_predictions(l, r)
        if total == 0:
            total = 1
        smn["bl"].append(outcomes.loc["Baseline MMSplice", "Total correct"] / total)
        smn["rt"].append(outcomes.loc["Retrained MMSplice", "Total correct"] / total)
        area_bl += integral(smn["bl"])
        area_rt += integral(smn["rt"])

    plot_l, plot_r = get_plot_bounds(smn)
    smn["bl"] = smn["bl"][plot_l:plot_r+1]
    smn["rt"] = smn["rt"][plot_l:plot_r+1]
    db_threshold_keys_copy = db_threshold_keys[plot_l:plot_r+1]

    area_bl = str(round(area_bl, 3))
    area_rt = str(round(area_rt, 3))
    ax.plot(db_threshold_keys_copy, smn["bl"])
    ax.plot(db_threshold_keys_copy, smn["rt"])

    ax.legend([f"Baseline (AUC={area_bl})", f"Retrained (AUC={area_rt})"], loc="best")
    ax.tick_params(axis='x', rotation=90)


    # fig.legend(["baseline", "retrained"], loc=(0.78,0.82))
    fig.supylabel("Proportion of variants correctly\npredicted to fall in ∆logit range")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            "density_threshold_binned",
            "proportion_per_logit_range_double_threshold_binned_02.pdf"
        )
    )

    bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for bin_ in bins:
        db, _ = get_thresholds(-11.5, 11.5, bin_)
        smn, area, totals = generate_density(db)
        total_correct_bl = np.sum(np.array(smn["bl"] * np.array(totals)))
        total_correct_rt = np.sum(np.array(smn["rt"] * np.array(totals)))
        float_print(
            bin_,
            area["bl"],
            area["rt"],
            int(total_correct_bl),
            int(total_correct_rt)
        )


    bin_analysis_df = ({
        "Bin size" : [],
        "Baseline AUC" : [],
        "Retrained AUC" : [],
        "Baseline correct" : [],
        "Retrained correct" : [],
    })

    for bin_ in bins:
        db, _ = get_thresholds(-11.5, 11.5, bin_)
        smn, area, totals = generate_density(db)
        total_correct_bl = np.sum(np.array(smn["bl"] * np.array(totals)))
        total_correct_rt = np.sum(np.array(smn["rt"] * np.array(totals)))
        bin_analysis_df["Bin size"].append(bin_)
        bin_analysis_df["Baseline AUC"].append(area["bl"])
        bin_analysis_df["Retrained AUC"].append(area["rt"])
        bin_analysis_df["Baseline correct"].append(int(total_correct_bl))
        bin_analysis_df["Retrained correct"].append(int(total_correct_rt))

    bin_analysis_df = pd.DataFrame(bin_analysis_df)
    bin_analysis_df.to_csv(
        os.path.join(
            output_dir,
            "density_threshold_binned",
            "bin_size_analysis_auc_and_count.csv",
        ),
        index = False
    )

    density_threshold_plot(
        -5.5,
        5.5,
        0.5,
        threshold_func=mse_two_threshold_predictions,
        errf=mse,
        ylabel=(
            "MSE of variants with\nobserved ∆logit in range\n"
            "vs predicted ∆logit"
        ),
        title= (
            "Baseline vs retrained MMSplice\n"
            "per ∆logit range MSE"
        )
    )
    plt.savefig(
        os.path.join(
            output_dir,
            'density_mse_mae_binned',
            "mse_per_logit_range_double_threshold_binned_05.pdf"
        )
    )
    density_threshold_plot(
        -5.5,
        5.5,
        0.2,
        threshold_func=mse_two_threshold_predictions,
        errf=mse,
        ylabel=(
            "MSE of variants with\nobserved ∆logit in range\n"
            "vs predicted ∆logit"
        ),
        title= (
            "Baseline vs retrained MMSplice\n"
            "per ∆logit range MSE"
        )
    )
    plt.savefig(
        os.path.join(
            output_dir,
            'density_mse_mae_binned',
            "mse_per_logit_range_double_threshold_binned_02.pdf"
        )
    )

    density_threshold_plot(
        -5.5,
        5.5,
        0.8,
        threshold_func=mse_two_threshold_predictions,
        errf=mse,
        ylabel=(
            "MSE of variants with\nobserved ∆logit in range\n"
            "vs predicted ∆logit"
        ),
        title= (
            "Baseline vs retrained MMSplice\n"
            "per ∆logit range MSE"
        )
    )
    plt.savefig(
        os.path.join(
            output_dir,
            'density_mse_mae_binned',
            "mse_per_logit_range_double_threshold_binned_08.pdf"
        )
    )

    # All delta logit windows heatmap

    offset = 0.0
    matrix = np.zeros((22, 20))
    matrix_ylabels = np.zeros((22,))
    matrix_xlabels = np.zeros((20,))
    row, col = 0, 0
    for i in range(-11, 11):
        col = 0
        matrix_ylabels[row] = i
        for j in range(1, 21):
            l = i + offset - (j / 20)
            r = i + offset + (j / 20)
            outcomes, total = mse_two_threshold_predictions(l, r, errf=mse)
            rt = outcomes.loc["Retrained MMSplice", "Total correct"] / total
            bl = outcomes.loc["Baseline MMSplice", "Total correct"] / total
            matrix[row][col] = rt - bl
            matrix_xlabels[col] = j / 20
            col += 1
        row += 1

    vmin = matrix.min()
    vmax = matrix.max()
    cap = max(np.abs(vmin), np.abs(vmax))
    norm = mcolors.TwoSlopeNorm(vmin=-cap, vcenter=0, vmax=cap)

    matrix_ylabels = matrix_ylabels[::-1]
    matrix = matrix[::-1]

    fig, ax = plt.subplots(1, figsize=(3.3,3.3), constrained_layout=True)

    def every_N(x, i, L, N, omit, start):
        if i in omit:
            return ""
        if (start + i) % N == 0 or i == start or i == L:
            return x
        return ""

    ax.set_yticks(
    ticks=np.arange(matrix.shape[0]), # 0..21
    labels=matrix_ylabels.astype(int), # -11..10
        fontsize=7
    )
    ax.set_xticks(
    ticks=np.arange(matrix.shape[1]), # 0..21
    labels=[
        every_N(
            matrix_xlabels[i],
            i,
            len(matrix_xlabels)-1,
            2,
            [0],
            1
        )
            for i in range(len(matrix_xlabels))], # -11..10
        fontsize=7
    )
    im = ax.imshow(matrix, cmap='RdBu', norm=norm)
    ax.set_xlabel("Distance from center (±∆logit)", fontsize=9)
    ax.set_ylabel("Window center (∆logit)", fontsize=9)
    fig.suptitle(
        "MMSplice retrained MSE - MMSplice baseline MSE\n"
        "For variants with observed ∆logit in window",
        fontsize=9
    )
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label("Model ∆MSE", fontsize=9)
    cbar.set_ticks([vmin, 0, vmax])
    ticks = [
        (-cap),
        (-cap*3/4),
        (-cap/2),
        (-cap/4),
        (0),
        (cap/4),
        (cap/2),
        (cap*3/4),
        (cap)
    ]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([round(ticks[i], 1) for i in range(len(ticks))])
    plt.savefig(os.path.join(
        output_dir,
        'heatmap_all_bins',
        "delta_mse_per_dlogit_windows_heatmap.pdf")
    )
    plt.savefig(os.path.join(
        output_dir,
        'heatmap_all_bins',
        "delta_mse_per_dlogit_windows_heatmap.tif"), dpi=300
    )
     

    matrix = np.zeros((22, 20))
    matrix_ylabels = np.zeros((22,))
    matrix_xlabels = np.zeros((20,))
    row, col = 0, 0
    for i in range(-11, 11):
        col = 0
        matrix_ylabels[row] = i
        for j in range(1, 21):
            l = i - (j / 20)
            r = i + (j / 20)
            outcomes, total = mse_two_threshold_predictions(l, r, errf=mae)
            rt = outcomes.loc["Retrained MMSplice", "Total correct"] / total
            bl = outcomes.loc["Baseline MMSplice", "Total correct"] / total
            matrix[row][col] = rt - bl
            matrix_xlabels[col] = j / 20
            col += 1
        row += 1

    vmin = matrix.min()
    vmax = matrix.max()
    cap = max(np.abs(vmin), np.abs(vmax))
    norm = mcolors.TwoSlopeNorm(vmin=-cap, vcenter=0, vmax=cap)

    matrix_ylabels = matrix_ylabels[::-1]
    matrix = matrix[::-1]

    fig, ax = plt.subplots(1, figsize=(3.3,3.3), constrained_layout=True)

    ax.set_yticks(
    ticks=np.arange(matrix.shape[0]), # 0..21
    labels=matrix_ylabels.astype(int), # -11..10
        fontsize=7
    )
    ax.set_xticks(
    ticks=np.arange(matrix.shape[1]), # 0..21
    labels=[
        every_N(
            matrix_xlabels[i],
            i,
            len(matrix_xlabels)-1,
            2,
            [0],
            1
        )
            for i in range(len(matrix_xlabels))], # -11..10
        fontsize=7
    )
    im = ax.imshow(matrix, cmap='RdBu', norm=norm)
    ax.set_xlabel("Distance from center (±∆logit)", fontsize=9)
    ax.set_ylabel("Window center (∆logit)", fontsize=9)
    fig.suptitle(
        "MMSplice retrained MAE - MMSplice baseline MAE\n"
        "For variants with observed ∆logit in window",
        fontsize=9
    )
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label("Model ∆MAE", fontsize=9)
    cbar.set_ticks([vmin, 0, vmax])
    ticks = [(-cap), (-cap*3/4), (-cap/2), (-cap/4), (0), (cap/4), (cap/2), (cap*3/4), (cap)]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([round(ticks[i], 1) for i in range(len(ticks))])
    heatmap_file = os.path.join(
        output_dir,
        "heatmap_all_bins",
        "delta_mae_per_dlogit_windows_heatmap"
    )
    plt.savefig(f"{heatmap_file}.pdf")
    plt.savefig(f"{heatmap_file}.tif", dpi=300)


    starts = [i for i in range(-11, 12)]
    sizes = [i/20 for i in range(1, 21)]
    fig, ax = plt.subplots(1)
    cmap = colormaps.get_cmap("RdBu")
    v = np.nanmax(np.abs(matrix))
    norm = mcolors.TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)

    for i, start in enumerate(starts):
        for j, size in enumerate(sizes):
            box_x = start
            box_width = size * 2
            box_y = (j + 1) * 2
            box_height = 0.5
            rect = patches.Rectangle(
                (box_x, box_y), box_width, box_height,
                facecolor='skyblue',
                edgecolor='navy',
                linewidth=1,
                alpha = 0.1
            )
            circ = patches.Circle(
                (box_x, box_y),
                max(size, 0.2),
                facecolor=cmap(norm(matrix[-(i - 1)][j - 1])),
                linewidth=0.5,
                edgecolor="purple",
                alpha=0.7)
            ax.add_patch(circ)

    ax.set_xlabel("Measured ∆logit(PSI) window center")
    ax.set_xlim(-12, 12)
    ax.set_ylim(1, 45)
    ax.set_ylabel("Measured ∆logit(PSI) window half-width\n(window = center ± half width)")
    ticks = ax.get_yticks()

    ticks = [(j + 1) * 2 for j in range(20)]
    tick_labels = [str((i / 40 )) for i in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_title(
        "MMSplice MAE difference after retraining,\n"
        "across measured Δlogit(PSI) windows of varying size"
    )

    circ = patches.Circle(
        (9, 43),
        max(size, 0.2),
        facecolor="tab:green",
        linewidth=0.5,
        edgecolor="purple",
        alpha=0.7
    )
    # ax.add_patch(circ)
    # ax.text(-7.6, 42.5, "Measured ∆logit(PSI) between 8.0 and 10.0 →", fontsize=8)

    ax.text(-11, 42, "|", fontsize=8)
    ax.text(-10, 42, "|", fontsize=8)
    ax.text(-11, 43.3, "____", fontsize=7)
    ax.text(-10.5, 43.3, "|", fontsize=6)
    ax.text(-10.5, 44.5, "____", fontsize=7)
    ax.text(-9, 43.5, "1.0 Δlogit(PSI)", fontsize=7)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Model ∆MAE (retrained - baseline)\n(Negative favors retrained)")

    dotplot_file = os.path.join(
        output_dir,
        "heatmap_all_bins",
        "delta_mae_per_dlogit_windows_dotplot"
    )
    plt.savefig(f"{dotplot_file}.pdf")
    plt.savefig(f"{dotplot_file}.tif", dpi=300)
    plt.savefig(f"{dotplot_file}.png", dpi=300)




    improved_test = pd.read_csv(
        os.path.join(
            mse_improvements_dir,
            "exon_specific_mse_improvements_test.tsv",
        ),
        sep="\t",
    )

    improved_val = pd.read_csv(
        os.path.join(
            mse_improvements_dir,
            "exon_specific_mse_improvements_val.tsv",
        ),
        sep="\t",
    )

    improved_train = pd.read_csv(
        os.path.join(
            mse_improvements_dir,
            "exon_specific_mse_improvements_train.tsv",
        ),
        sep="\t",
    )


    improved = {
        "train": improved_train,
        "val": improved_val,
        "test": improved_test,
    }

    barplot_dir = os.path.join(output_dir, "stacked_barplots")
    result = stacked_barplot(
        improved,
        "0.2",
        scale="dlogit",
        splits=["val", "test"],
        save=True,
        plot_name="mmsplice_mse_improved_exons_bar_02_val_test",
        plot_dirs=[barplot_dir],
        y_scale="linear",
        err_func="MSE",
        split_legend=True,
    )

    result = stacked_barplot(
        improved,
        "1.0",
        scale="dlogit",
        splits=["val", "test"],
        save=True,
        plot_name="mmsplice_mse_improved_exons_bar_1_val_test",
        plot_dirs=[barplot_dir],
        y_scale="linear",
        err_func="MSE",
        split_legend=True,
    )

    logging.info("done")
