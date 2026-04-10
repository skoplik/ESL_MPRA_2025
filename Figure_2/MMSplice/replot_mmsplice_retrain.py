#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple, List
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

# Keep text editable in Illustrator while allowing rasterized markers
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# === User paths (kept exactly as provided) ===
RETRAINED_TEST = "/ESL/ESL_MPRA/Figure_2/MMSplice/model_output_csv/retrained_mmsplice_model_predictions_test_dataset7_model10_04_09_26_updated.csv"
RETRAINED_FULL = "/ESL/ESL_MPRA/Figure_2/MMSplice/model_output_csv/retrained_mmsplice_model_predictions_dataset7_model10_04_09_26_updated.csv"

BASELINE_TEST = "/ESL/ESL_MPRA/Figure_2/MMSplice/model_output_csv/retrained_mmsplice_baseline_test_dataset7_model10_04_09_26_updated.csv"
BASELINE_FULL = "/ESL/ESL_MPRA/Figure_2/MMSplice/model_output_csv/retrained_mmsplice_baseline_aggregate_dataset7_model10_04_09_26_updated.csv"

OUTDIR = "/ESL/ESL_MPRA/Figure_2/plots/MMSplice"
os.makedirs(OUTDIR, exist_ok=True)

RETRAINED_OUT = os.path.join(OUTDIR, "retrained_overlay_pred_vs_true_delta_logit.pdf")
BASELINE_OUT = os.path.join(OUTDIR, "baseline_overlay_pred_vs_true_delta_logit.pdf")

# === Column name candidates ===
PRED_COL_CANDIDATES: List[str] = [
    "Retrained_MMSplice_Predicted_Delta_Logit",
    "Predicted_Delta_Logit",
    "predicted_delta_logit",
    "pred_delta_logit",
    "delta_logit_pred",
    "y_pred",
    "pred",
    "model_predicted_delta_logit",
    "mmsplice_predicted_delta_logit"
]

TRUE_COL_CANDIDATES: List[str] = [
    "True_Delta_Logit",
    "true_delta_logit",
    "delta_logit_true",
    "y_true",
    "true",
    "target_delta_logit",
    "experimental_delta_logit",
    "measured_delta_logit"
]

ALT_PAIR_CANDIDATES = [
    ("delta_logit_pred_hela", "delta_logit_true_hela"),
    ("delta_logit_pred", "delta_logit_true"),
    ("delta_logodds_pred", "delta_logodds_true"),
    ("predicted", "observed"),
]

def infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = set(df.columns.str.strip())
    pred = next((c for c in PRED_COL_CANDIDATES if c in cols), None)
    true = next((c for c in TRUE_COL_CANDIDATES if c in cols), None)
    if pred and true:
        return pred, true
    for p, t in ALT_PAIR_CANDIDATES:
        if p in cols and t in cols:
            return p, t
    pred_like = [c for c in df.columns if "pred" in c.lower() and "logit" in c.lower()]
    true_like = [c for c in df.columns if ("true" in c.lower() or "obs" in c.lower() or "meas" in c.lower()) and "logit" in c.lower()]
    if pred_like and true_like:
        return pred_like[0], true_like[0]
    generic_pred = [c for c in df.columns if "delta" in c.lower() and "logit" in c.lower() and ("pred" in c.lower() or "model" in c.lower())]
    generic_true = [c for c in df.columns if "delta" in c.lower() and "logit" in c.lower() and ("true" in c.lower() or "obs" in c.lower() or "meas" in c.lower() or "target" in c.lower())]
    if generic_pred and generic_true:
        return generic_pred[0], generic_true[0]
    raise ValueError("Could not infer predicted and true delta-logit columns from file with columns: " + ", ".join(df.columns))

def load_xy(path: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    df = pd.read_csv(path)
    pred_col, true_col = infer_columns(df)
    x = pd.to_numeric(df[pred_col], errors="coerce")
    y = pd.to_numeric(df[true_col], errors="coerce")
    valid = x.notna() & y.notna()
    x = x[valid].values
    y = y[valid].values
    return x, y, pred_col, true_col

def compute_limits(x1, y1, x2, y2, pad=0.05) -> Tuple[float, float]:
    all_vals = np.concatenate([x1, y1, x2, y2]) if (len(x2) and len(y2)) else np.concatenate([x1, y1])
    if all_vals.size == 0:
        return -1.0, 1.0
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
    span = vmax - vmin
    if span == 0:
        vmin -= 0.5
        vmax += 0.5
    else:
        vmin -= pad * span
        vmax += pad * span
    m = max(abs(vmin), abs(vmax))
    return -m, m

def corr_text(x, y) -> str:
    if len(x) == 0 or len(y) == 0:
        return "r=NA\nn=0"
    r, _ = pearsonr(x, y)
    return f"r={r:.2f}\nn={len(x)}"

def scatter_overlay(
    file_test: str,
    file_full: str,
    out_pdf: str,
    title: str,
    test_color: str,
    full_color: str
):
    x_test, y_test, _, _ = load_xy(file_test)
    x_full, y_full, _, _ = load_xy(file_full)

    xminmax, xmaxmax = compute_limits(x_test, y_test, x_full, y_full)

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)

    # y=x diagonal reference line
    ax.plot([xminmax, xmaxmax], [xminmax, xmaxmax],
            color='grey', linestyle='--', linewidth=1, zorder=1)

    # Plot full first (behind), test second (on top)
    ax.scatter(x_full, y_full, s=10, color=full_color, alpha=0.1,
               rasterized=True, zorder=2)
    ax.scatter(x_test, y_test, s=10, color=test_color, alpha=0.1,
               rasterized=True, zorder=3)

    ax.set_xlim(xminmax, xmaxmax)
    ax.set_ylim(xminmax, xmaxmax)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("Predicted Δlogit(PSI)")
    ax.set_ylabel("Measured Δlogit(PSI)")
    ax.set_title(title, pad=8)

    # Legend with opaque markers
    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               label=f"Full (aggregate)\n{corr_text(x_full,y_full)}",
               markerfacecolor=full_color, markersize=6),
        Line2D([0], [0], marker='o', color='w',
               label=f"Test set\n{corr_text(x_test,y_test)}",
               markerfacecolor=test_color, markersize=6),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left")

    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def main():
    try:
        # Baseline plot: full = orange, test = black
        scatter_overlay(
            file_test=BASELINE_TEST,
            file_full=BASELINE_FULL,
            out_pdf=BASELINE_OUT,
            title="Baseline MMSplice vs Measured Δlogit(PSI)",
            test_color="black",
            full_color="#e4a44e"
        )
    except Exception as e:
        sys.stderr.write(f"[Error] Baseline plot failed: {e}\n")

    try:
        # Retrained plot: full = blue, test = black
        scatter_overlay(
            file_test=RETRAINED_TEST,
            file_full=RETRAINED_FULL,
            out_pdf=RETRAINED_OUT,
            title="Retrained MMSplice vs Measured Δlogit(PSI)",
            test_color="black",
            full_color="#377eb8"
        )
    except Exception as e:
        sys.stderr.write(f"[Error] Retrained plot failed: {e}\n")

if __name__ == "__main__":
    main()
