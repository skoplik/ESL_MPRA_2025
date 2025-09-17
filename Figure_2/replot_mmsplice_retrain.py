#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple, List
from scipy.stats import pearsonr

# Keep text editable in Illustrator while allowing rasterized markers
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# === User paths (kept exactly as provided) ===
RETRAINED_TEST = "/ESL/ESL_MPRA/Figure_2/model_output_csv/retrained_mmsplice_model_predictions_test_dataset7_model10.csv"
RETRAINED_FULL = "/ESL/ESL_MPRA/Figure_2/model_output_csv/retrained_mmsplice_model_predictions_dataset7_model10.csv"

BASELINE_TEST = "/ESL/ESL_MPRA/Figure_2/model_output_csv/retrained_mmsplice_baseline_test_dataset7_model10.csv"
BASELINE_FULL = "/ESL/ESL_MPRA/Figure_2/model_output_csv/retrained_mmsplice_baseline_aggregate_dataset7_model10.csv"

OUTDIR = "/ESL/ESL_MPRA/Figure_2/mmsplice_retrain_plots"
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
        return "n=0"
    r, _ = pearsonr(x, y)
    return f"r={r:.2f}, n={len(x)}"

def scatter_overlay(
    file_orange: str,
    file_blue: str,
    out_pdf: str,
    title: str,
    label_orange: str = "Retrained (test set)",
    label_blue: str = "Full (aggregate)"
):
    x1, y1, _, _ = load_xy(file_orange)
    x2, y2, _, _ = load_xy(file_blue)

    xminmax, xmaxmax = compute_limits(x1, y1, x2, y2)

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)

    # Scatter points: plot blue first, orange second so orange is on top
    ax.scatter(x2, y2, s=10, color="#1f77b4", alpha=0.1,
               label=f"{label_blue} ({corr_text(x2,y2)})", rasterized=True, zorder=2)
    ax.scatter(x1, y1, s=10, color="#e68613", alpha=0.1,
               label=f"{label_orange} ({corr_text(x1,y1)})", rasterized=True, zorder=3)

    ax.set_xlim(xminmax, xmaxmax)
    ax.set_ylim(xminmax, xmaxmax)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("Predicted Δlogit(PSI)")
    ax.set_ylabel("True Δlogit(PSI)")
    ax.set_title(title, pad=8)

    ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def main():
    try:
        scatter_overlay(
            file_orange=BASELINE_TEST,
            file_blue=BASELINE_FULL,
            out_pdf=BASELINE_OUT,
            title="Baseline MMSplice: Predicted vs True Δlogit(PSI)",
            label_orange="Retrained (test set)",
            label_blue="Full (aggregate)"
        )
    except Exception as e:
        sys.stderr.write(f"[Error] Baseline plot failed: {e}\n")
    try:
        scatter_overlay(
            file_orange=RETRAINED_TEST,
            file_blue=RETRAINED_FULL,
            out_pdf=RETRAINED_OUT,
            title="Retrained MMSplice: Predicted vs True Δlogit(PSI)",
            label_orange="Retrained (test set)",
            label_blue="Full (aggregate)"
        )
    except Exception as e:
        sys.stderr.write(f"[Error] Retrained plot failed: {e}\n")

if __name__ == "__main__":
    main()
