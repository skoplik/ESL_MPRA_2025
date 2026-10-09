#!/usr/bin/env python3
help_="""
Plot scatter + bar chart comparing all splicing model predictions
against COMPASS experimental avg_delta_logit_pooled.

Models: SpliceAI, AlphaGenome, Pangolin, HAL,
        Baseline MMSplice, Retrained MMSplice

All comparisons use the full aggregate (no test-set split).
Outputs saved to: /ESL/ESL_MPRA/Figure_3/plots_MAY_v2/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import pearsonr
import getopt
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_proper_model_name(model_key):
    model_key = model_key.replace("alphagenome", "AlphaGenome")
    model_key = model_key.replace("spliceai", "SpliceAI")
    model_key = model_key.replace("mmsplice", "MMSplice")
    model_key = model_key.replace("baseline", "Baseline")
    model_key = model_key.replace("retrained", "Retrained")
    model_key = model_key.replace("hal", "HAL")
    model_key = model_key.replace("pangolin", "Pangolin")
    model_key = model_key.replace("_", " ")
    return model_key


def compute_offset(w, n_filters):
    bar = w/n_filters
    pos = 0 - w/2 + bar/2
    offsets = []
    for i in range(n_filters):
        offsets.append(pos)
        pos += bar
    return offsets


def align_text(text, max_line = 10):
    """
    split text into multiple lines
    if the number of characters 
    exceed max_line, without splitting
    words. (greedy version)
    """
    lines = []
    words = text.split(" ")
    curr_line = []
    curr_len = 0
    for i in range(len(words)):
        if len(words[i]) + curr_len > max_line:
            lines.append(" ".join(curr_line))
            curr_line = [words[i]]
            curr_len = 0
        else:
            curr_line.append(words[i])
            curr_len += len(words[i]) + 1
    if curr_line:
        lines.append(" ".join(curr_line))
    return "\n".join(lines)


def path_to_file(path):
    path, file_ = os.path.split(path)
    file_ = file_.split(".")[0]
    return file_


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

def pearson_str(x, y):
    if len(x) < 2:
        return "r=NA\nn=0"
    r, _ = pearsonr(x, y)
    return f"r={r:.2f}\nn={len(x):,}"


def scatter_ax(ax, x, y, color, label, alpha=0.10, s=8):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    lim = max(np.abs(np.concatenate([x, y])).max() * 1.05, 1.0)
    ax.plot([-lim, lim], [-lim, lim], color="grey", linestyle="--",
            linewidth=1, zorder=1)
    ax.scatter(x, y, s=s, color=color, alpha=alpha, linewidths=0,
               rasterized=True, zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted Δlogit(PSI)", fontsize=9)
    ax.set_ylabel("Measured Δlogit(PSI)", fontsize=9)
    ax.set_title(label, fontsize=9, pad=4)
    ax.text(0.05, 0.95, pearson_str(x, y),
            transform=ax.transAxes, va="top", ha="left", fontsize=8)


def save_fig(fig, dirpath, stem):
    for ext in ("pdf", "png"):
        path = os.path.join(dirpath, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")


def save_single(x, y, color, title, fname, label):
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
    scatter_ax(ax, x, y, color, title)
    plt.tight_layout()
    stem = fname.replace(".pdf", "")
    save_fig(fig, MODEL_OUTDIR[label], stem)
    plt.close(fig)


def r_n(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return float("nan"), 0
    r, _ = pearsonr(x, y)
    return round(float(r), 4), int(len(x))

# ── Load arguments ─────────────────────────────────────────────────────────────
opts, args = getopt.getopt(sys.argv[1:],"", [
    "MEGA_FILE=",
    "PLOTDIR=",
    "FILTERS=",
    "FILTER_LABELS=",
    "MODEL_KEYS=",
    "TRUE_COL=",
    "DEBUG",
    "HELP",
])
opts = dict(opts)

if "--HELP" in opts:
    print(help_)
    sys.exit(0)

debug = "--DEBUG" in opts
if debug:
    MEGA_FILE = "/ESL/ESL_MPRA/Figure_3/model_comparison/mega_pred_file_MAY_v2.csv"
    PLOTDIR   = "/ESL/ESL_MPRA/Figure_3/plots_MAY_v2"
    MODEL_KEYS = "spliceai,alphagenome,pangolin,hal,baseline_mmsplice"
    FILTERS = ""
    FILTER_LABELS = ""
    TRUE_COL = "avg_delta_logit_pooled"
else:
    MEGA_FILE = opts["--MEGA_FILE"]
    PLOTDIR = opts["--PLOTDIR"]
    FILTERS = opts["--FILTERS"]
    FILTER_LABELS = opts.get("--FILTER_LABELS", "")
    MODEL_KEYS = opts["--MODEL_KEYS"]
    TRUE_COL = opts["--TRUE_COL"]
    print(opts, flush=True)

MODEL_KEYS = MODEL_KEYS.split(",")

PROPER_MODELS = []
proper_model_map = {}
for m in MODEL_KEYS:
    proper = get_proper_model_name(m)
    PROPER_MODELS.append(proper)
    proper_model_map[m] = proper

OUTDIR = {
    "merged_output": os.path.join(PLOTDIR, "merged_output"),
}
MODEL_OUTDIR = {}
for model in PROPER_MODELS:
    if "MMSplice" in model:
        OUTDIR[model] = os.path.join(PLOTDIR, "MMSplice")
        OUTDIR["MMSplice"] = os.path.join(PLOTDIR, "MMSplice")
    else:
        OUTDIR[model] = os.path.join(PLOTDIR, model)
    MODEL_OUTDIR[model] = OUTDIR[model]

for d in OUTDIR.values():
    os.makedirs(d, exist_ok=True)

# Colors: Plasma colormap shades; Retrained MMSplice stays blue
# plasma positions: Baseline=0.05, SpliceAI=0.25, AlphaGenome=0.45, Pangolin=0.65, HAL=0.85
COLOR = {
    "Baseline MMSplice":  "#7d51f9",  # bright purple (lum≈0.45)
    "Retrained MMSplice": "#377EB8",  # blue (fixed)
    "SpliceAI":           "#a219d1",  # mid magenta (lum≈0.34)
    "AlphaGenome":        "#ed7a52",  # plasma 0.67 — orange-red
    "AlphaGenome 16kb":   "#ed7a52",  # plasma 0.67 — orange-red
    "AlphaGenome 1Mb":    "tab:green",
    "Pangolin":           "#fa9c3c",  # plasma 0.77 — amber-orange
    "HAL":                "#fdc627",  # plasma 0.88 — yellow-amber
}

# 5 models for general benchmarking (no retrained MMSplice — that has its own script)
# Each model uses its own full coverage (n varies)
MODELS = [
    (f"{model}_delta_logit", proper_model_map[model], COLOR[proper_model_map[model]])
    for model in MODEL_KEYS
]



# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading mega pred file...")
mega = pd.read_csv(MEGA_FILE, low_memory=False)
print(f"  {len(mega):,} rows")

var_filters = {}
filter_labels = {}
filter_file_names = {}
if FILTERS:
    filters = FILTERS.split(",")
    if FILTER_LABELS:
        filter_labels_overwrite = FILTER_LABELS.split(",")
        assert len(filter_labels_overwrite) == len(filters)
    else:
        filter_labels_overwrite = None
    for i, f in enumerate(filters):
        filter_file_names[f] = path_to_file(f)
        if filter_labels_overwrite is None:
            filter_labels[f] = get_proper_model_name(filter_file_names[f])
        else:
            filter_labels[f] = filter_labels_overwrite[i]
        var_filter_df = pd.read_csv(f)
        id_col = check_id_col(var_filter_df)
        var_filter_df = var_filter_df.rename(columns={
            id_col : "var_id"
        })
        id_col = "var_id"
        var_filter = set(var_filter_df[id_col].to_list())
        var_filters[f] = var_filter
else:
    var_filters["None"] = set(mega.var_id.tolist())


mega_filtered = {}
for f in var_filters:
    mega_filtered[f] = mega[mega[id_col].isin(var_filters[f])]
    print(f"  {len(mega_filtered[f]):,} rows (after {f} filter)")


# ── Individual scatter plots ───────────────────────────────────────────────────
print("\n── Individual scatter plots ──")
stats = {}
for filter_ in mega_filtered:
    mega_ = mega_filtered[filter_]
    filter_file_name = filter_file_names[filter_]
    filter_plot_name = filter_labels[filter_]
    for pred_col, label, color in MODELS:
        sub = mega_[[pred_col, TRUE_COL]].dropna()
        x = sub[pred_col].values
        y = sub[TRUE_COL].values
        r, n = r_n(x, y)
        stats[(label,)] = (r, n)
        print(f"  {label:25s}  r={r:.4f}  n={n:,}")
        safe_name = label.lower().replace(" ", "_")
        save_single(
            x,
            y,
            color,
            f"{label} — {filter_plot_name}",
            f"general_bench_{safe_name}_{filter_file_name}.pdf",
            label
        )

    # ── 5-panel scatter (1 row × 5) ───────────────────────────────────────────────
    print("\n── 5-panel scatter ──")
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.2), dpi=300)
    for ax, (pred_col, label, color) in zip(axes, MODELS):
        sub = mega_[[pred_col, TRUE_COL]].dropna()
        scatter_ax(ax, sub[pred_col].values, sub[TRUE_COL].values, color, label)

    plt.tight_layout()
    save_fig(
        fig,
        OUTDIR["merged_output"],
        f"general_bench_5panel_scatter_{filter_file_name}"
    )
    plt.close(fig)

# ── Bar plot 1: Full model comparison (no retrained MMSplice), sorted low→high ──
print("\n── Bar plot: full model comparison ──")

# Fixed display order (low→high by aggregate r): Baseline MMSplice, HAL, Pangolin, SpliceAI, AlphaGenome
BAR1_MODELS = [
    ("baseline_mmsplice_delta_logit",  "Baseline\nMMSplice", COLOR["Baseline MMSplice"]),
    ("hal_delta_logit",                "HAL",                COLOR["HAL"]),
    ("pangolin_delta_logit",           "Pangolin",           COLOR["Pangolin"]),
    ("spliceai_delta_logit",           "SpliceAI",           COLOR["SpliceAI"]),
    ("alphagenome_delta_logit",        "AlphaGenome",        COLOR["AlphaGenome"]),
    ("alphagenome_16kb_delta_logit",   "AlphaGenome\n(16kb)",COLOR["AlphaGenome"]),
    ("alphagenome_1Mb_delta_logit",    "AlphaGenome\n(1Mb)", COLOR["AlphaGenome 1Mb"]),
]
BAR1_MODELS_ = []
for bar in BAR1_MODELS:
    if bar[0].replace("_delta_logit", "") in proper_model_map:
        BAR1_MODELS_.append(bar)
BAR1_MODELS = BAR1_MODELS_
    

bar_labels = [m[1] for m in BAR1_MODELS]
bar_colors = [m[2] for m in BAR1_MODELS]
bar_rs = {}
bar_ns = {}


num_filters = len(mega_filtered)
hatches = ["", "///", "...", "xxx", "***"]
width = 0.6
bars_filter = {}
n_filter = {}
r_filter = {}
legend_handles = []

x = np.arange(len(BAR1_MODELS))
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
offsets = compute_offset(width, num_filters)
filter_num = 0
for filter_ in mega_filtered:
    r_filter[filter_] = []
    n_filter[filter_] = []
    mega_ = mega_filtered[filter_]
    for pred_col, lbl, _ in BAR1_MODELS:
        sub = mega_[[pred_col, TRUE_COL]].dropna()
        r, n = r_n(sub[pred_col].values, sub[TRUE_COL].values)
        r_filter[filter_].append(r)
        n_filter[filter_].append(n)

    bars_filter[filter_] = ax.bar(
        x + offsets[filter_num],
        r_filter[filter_],
        color=bar_colors,
        edgecolor="black",
        width=width/num_filters,
        hatch=hatches[filter_num],
        label=filter_labels[filter_],
    )
    for bar, r_val, n_val in zip(
            bars_filter[filter_],
            r_filter[filter_],
            n_filter[filter_]
        ):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            r_val + 0.012,
            f"r={r_val:.2f}\nn={n_val:,}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=0 if num_filters < 2 else 90,
        )
    legend_handles.append(
        Patch(
            facecolor="lightgray",
            edgecolor="black",
            label=align_text(filter_labels[filter_], 15),
            hatch=hatches[filter_num],
        )
    )
    filter_num += 1


ax.set_ylim(0, 1)
ax.set_ylabel("Pearson r", fontsize=12)
ax.set_title("Splicing Model Performance\n(avg Δlogit(PSI) across cell lines)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(bar_labels, fontsize=10, rotation=45, ha="right")
ax.axhline(0, color="black", linewidth=0.5)
if len(legend_handles) > 1:
    ax.legend(
        handles=legend_handles,
        title="Subset",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )

plt.tight_layout()
save_fig(fig, OUTDIR["merged_output"], "bar_full_model_comparison")
plt.close(fig)

print("\nDone.")
