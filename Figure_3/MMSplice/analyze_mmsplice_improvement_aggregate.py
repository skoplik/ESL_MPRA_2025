#!/usr/bin/env python3
"""
Characterize which variants benefit from MMSplice retraining.
Runs on three subsets:
  - test set only  (mmsplice_is_test == True,  n~3,526)
  - train set only (mmsplice_is_test == False, n~67,018)
  - aggregate      (all rows with predictions, n~70,843)

Outputs saved to plots/MMSplice/aggregate_analysis/
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

MEGA_FILE = "/ESL/ESL_MPRA/Figure_3/model_comparison/mega_pred_file_filtered.csv"
DATA_PATH = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz"
OUTDIR = "/ESL/ESL_MPRA/Figure_3/plots/MMSplice/aggregate_analysis"
os.makedirs(OUTDIR, exist_ok=True)


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
mega = pd.read_csv(MEGA_FILE, low_memory=False)
seqs = pd.read_csv(DATA_PATH, low_memory=False,
                   usecols=["Reference", "intron1", "exon", "intron2"])
seqs["intron1_len"] = seqs["intron1"].str.len()
seqs["exon_len"]    = seqs["exon"].str.len()
seqs["intron2_len"] = seqs["intron2"].str.len()

# Drop rows missing predictions
PRED_COLS = ["baseline_mmsplice_delta_logit", "retrained_mmsplice_delta_logit",
             "avg_delta_logit_pooled"]
mega = mega.dropna(subset=PRED_COLS)
mega = mega.merge(seqs[["Reference", "intron1_len", "exon_len", "intron2_len"]],
                  on="Reference", how="left")

# ── Define subsets ─────────────────────────────────────────────────────────────
subsets = {
    "test":      mega[mega["mmsplice_is_test"] == True].copy(),
    "train":     mega[mega["mmsplice_is_test"] == False].copy(),
    "aggregate": mega.copy(),
}
for name, df in subsets.items():
    print(f"  {name}: {len(df):,} variants")


# ── Feature engineering (shared) ──────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    df["var_pos"] = df["snp"].str.extract(r"^(\d+)").astype(float)

    df["region"] = "intron1"
    df.loc[df["var_pos"] >= df["intron1_len"], "region"] = "exon"
    df.loc[df["var_pos"] >= df["intron1_len"] + df["exon_len"], "region"] = "intron2"

    df["pos_rel_5ss"] = df["var_pos"] - df["intron1_len"]
    df["pos_rel_3ss"] = df["var_pos"] - (df["intron1_len"] + df["exon_len"])
    df["dist_nearest_ss"] = df[["pos_rel_5ss", "pos_rel_3ss"]].abs().min(axis=1)

    def pos_bin(d):
        if d <= 4:   return "0–4 nt\n(core)"
        if d <= 20:  return "5–20 nt\n(proximal)"
        if d <= 80:  return "21–80 nt\n(intermediate)"
        return "81+ nt\n(distal)"

    df["pos_bin"] = df["dist_nearest_ss"].apply(pos_bin)

    psi = df["HEK_wt_pooled_psi_raw"]
    df["psi_bin"] = pd.cut(
        psi, bins=[0, 0.2, 0.8, 1.0],
        labels=["Near-excluded\n(0–0.2)", "Intermediate\n(0.2–0.8)", "Near-included\n(0.8–1.0)"],
        include_lowest=True
    )

    df["err_baseline"]  = (df["baseline_mmsplice_delta_logit"]  - df["avg_delta_logit_pooled"]).abs()
    df["err_retrained"] = (df["retrained_mmsplice_delta_logit"] - df["avg_delta_logit_pooled"]).abs()
    df["improvement"]   = df["err_baseline"] - df["err_retrained"]

    df["outcome"] = "unchanged"
    df.loc[df["improvement"] >  0.05, "outcome"] = "improved"
    df.loc[df["improvement"] < -0.05, "outcome"] = "degraded"
    return df

subsets = {k: add_features(v) for k, v in subsets.items()}

POS_ORDER  = ["0–4 nt\n(core)", "5–20 nt\n(proximal)",
              "21–80 nt\n(intermediate)", "81+ nt\n(distal)"]
REGION_COLOR = {"intron1": "#e07b39", "exon": "#377EB8", "intron2": "#4daf4a"}
PSI_COLOR = {
    "Near-excluded\n(0–0.2)":   "#e41a1c",
    "Intermediate\n(0.2–0.8)":  "#984ea3",
    "Near-included\n(0.8–1.0)": "#377EB8",
}


# ── Plot helper: improvement by position × region (bar chart) ─────────────────
def plot_improvement_bars(df, label, outdir):
    regions   = ["intron1", "exon", "intron2"]
    n_regions = len(regions)
    n_bins    = len(POS_ORDER)
    bar_width = 0.22
    x = np.arange(n_bins)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    for i, region in enumerate(regions):
        sub = df[df["region"] == region]
        grp = sub.groupby("pos_bin")["improvement"]
        means = [grp.get_group(b).mean() if b in grp.groups else np.nan for b in POS_ORDER]
        sems  = [grp.get_group(b).sem()  if b in grp.groups else np.nan for b in POS_ORDER]
        ns    = [grp.get_group(b).count() if b in grp.groups else 0      for b in POS_ORDER]

        offset = (i - n_regions / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width,
               color=REGION_COLOR[region], edgecolor="black", linewidth=0.5,
               label=region, yerr=sems, capsize=3, error_kw={"linewidth": 0.8})
        for xi, (m, s, n) in enumerate(zip(means, sems, ns)):
            if n > 0 and not np.isnan(m):
                top = (m + s) if not np.isnan(s) else m
                ax.text(x[xi] + offset, max(top, 0) + 0.005, f"n={n}",
                        ha="center", va="bottom", fontsize=5.5, rotation=90)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(POS_ORDER, fontsize=9)
    ax.set_ylabel("Mean improvement in absolute error\n(baseline − retrained, Δlogit units)", fontsize=9)
    ax.set_title(f"MMSplice retraining benefit by position and region\n({label}, n={len(df):,})", fontsize=10)
    ax.legend(title="Region", frameon=False, fontsize=8)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"improvement_by_position_region_{label}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Plot helper: baseline vs retrained error scatter ──────────────────────────
def plot_error_scatter(df, label, outdir):
    PSI_LABEL = {k: k.replace("\n", " ") for k in PSI_COLOR}
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    lim = max(df["err_baseline"].max(), df["err_retrained"].max()) * 1.05

    for psi_bin, color in PSI_COLOR.items():
        sub = df[df["psi_bin"] == psi_bin]
        ax.scatter(sub["err_baseline"], sub["err_retrained"],
                   s=6, color=color, alpha=0.15, linewidths=0,
                   rasterized=True, label=PSI_LABEL[psi_bin])

    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=1, zorder=5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Baseline MMSplice |error| (Δlogit)", fontsize=10)
    ax.set_ylabel("Retrained MMSplice |error| (Δlogit)", fontsize=10)
    ax.set_title(f"Baseline vs. retrained prediction error\n({label}; below diagonal = retrained better)", fontsize=9)

    frac_imp = (df["outcome"] == "improved").mean()
    frac_deg = (df["outcome"] == "degraded").mean()
    ax.text(0.97, 0.05,
            f"Improved: {frac_imp:.1%}\nDegraded: {frac_deg:.1%}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    ax.legend(title="Reference PSI (HEK)", frameon=False, fontsize=8, loc="upper left")
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"baseline_vs_retrained_error_scatter_{label}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Plot: Pearson r comparison across subsets ──────────────────────────────────
def plot_r_comparison(subsets, outdir, mega_full):
    """Bar chart comparing Pearson r of baseline vs retrained across test/train/aggregate.
    Adds reference lines for SpliceAI and Pangolin on each subset."""
    rows = []
    for label, df in subsets.items():
        r_base, _ = stats.pearsonr(df["baseline_mmsplice_delta_logit"],  df["avg_delta_logit_pooled"])
        r_ret,  _ = stats.pearsonr(df["retrained_mmsplice_delta_logit"], df["avg_delta_logit_pooled"])
        rows.append({"subset": label, "model": "Baseline MMSplice", "r": r_base, "n": len(df)})
        rows.append({"subset": label, "model": "Retrained MMSplice", "r": r_ret,  "n": len(df)})

    rdf = pd.DataFrame(rows)
    print("\n── Pearson r by subset ──")
    print(rdf.to_string(index=False))

    # Reference model r per subset
    ref_models = {
        "SpliceAI":  "spliceai_delta_logit",
        "Pangolin":  "pangolin_delta_logit",
    }
    ref_r = {}  # (subset, model) -> r
    subsets_order = ["test", "train", "aggregate"]
    subset_dfs = {
        "test":      mega_full[mega_full["mmsplice_is_test"] == True].dropna(subset=["avg_delta_logit_pooled"]),
        "train":     mega_full[mega_full["mmsplice_is_test"] == False].dropna(subset=["avg_delta_logit_pooled"]),
        "aggregate": mega_full.dropna(subset=["avg_delta_logit_pooled"]),
    }
    for sname, sdf in subset_dfs.items():
        for mname, col in ref_models.items():
            sub = sdf.dropna(subset=[col])
            r, _ = stats.pearsonr(sub[col], sub["avg_delta_logit_pooled"])
            ref_r[(sname, mname)] = r
            print(f"  {mname} {sname}: r={r:.3f} (n={len(sub):,})")

    models = ["Baseline MMSplice", "Retrained MMSplice"]
    colors = {"Baseline MMSplice": "#aec6e8", "Retrained MMSplice": "#1f77b4"}
    ref_styles = {
        "SpliceAI": {"color": "#d62728", "linestyle": "--", "linewidth": 1.5},
        "Pangolin": {"color": "#ff7f0e", "linestyle": ":",  "linewidth": 1.5},
    }
    x = np.arange(len(subsets_order))
    bar_width = 0.35
    group_half = (len(models) * bar_width) / 2 + 0.05  # half-width of bar group

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    for i, model in enumerate(models):
        vals = [rdf[(rdf["subset"] == s) & (rdf["model"] == model)]["r"].values[0]
                for s in subsets_order]
        ns   = [rdf[(rdf["subset"] == s) & (rdf["model"] == model)]["n"].values[0]
                for s in subsets_order]
        offset = (i - len(models)/2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width,
               color=colors[model], edgecolor="black", linewidth=0.6, label=model)
        for xi, (v, n) in enumerate(zip(vals, ns)):
            ax.text(x[xi] + offset, v + 0.004, f"r={v:.3f}\n(n={n:,})",
                    ha="center", va="bottom", fontsize=6.5)

    # Draw per-subset reference lines spanning only that group's bars
    for xi, sname in enumerate(subsets_order):
        for mname, style in ref_styles.items():
            r_val = ref_r[(sname, mname)]
            ax.plot([x[xi] - group_half, x[xi] + group_half], [r_val, r_val],
                    label=mname if xi == 0 else "_nolegend_", **style)
            ax.text(x[xi] + group_half + 0.02, r_val, f"{mname}\nr={r_val:.3f}",
                    va="center", fontsize=6, color=style["color"])

    ax.set_xticks(x)
    ax.set_xticklabels(["Held-out\ntest set", "Training\nset", "Aggregate\n(all)"], fontsize=9)
    ax.set_ylabel("Pearson r (predicted vs. measured Δlogit)", fontsize=9)
    ax.set_title("MMSplice baseline vs. retrained: Pearson r across data splits\n"
                 "(SpliceAI/Pangolin shown as reference — pretrained on hg38)", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"pearson_r_by_split.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Plot: improvement by effect size bin (test set only) ──────────────────────
def plot_improvement_by_effect_size(df, label, outdir):
    df = df.copy()
    df["effect_bin"] = pd.cut(
        df["avg_delta_logit_pooled"].abs(),
        bins=[0, 0.5, 1, 2, np.inf],
        labels=["<0.5\n(sub-threshold)", "0.5–1\n(moderate)", "1–2\n(SDV)", ">2\n(strong SDV)"]
    )
    BIN_ORDER  = ["<0.5\n(sub-threshold)", "0.5–1\n(moderate)", "1–2\n(SDV)", ">2\n(strong SDV)"]
    BIN_COLORS = ["#d9d9d9", "#fdae6b", "#f16913", "#7f2704"]

    grp   = df.groupby("effect_bin")["improvement"]
    means = [grp.get_group(b).mean()  if b in grp.groups else np.nan for b in BIN_ORDER]
    sems  = [grp.get_group(b).sem()   if b in grp.groups else np.nan for b in BIN_ORDER]
    ns    = [grp.get_group(b).count() if b in grp.groups else 0      for b in BIN_ORDER]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    x = np.arange(len(BIN_ORDER))
    bars = ax.bar(x, means, color=BIN_COLORS, edgecolor="black", linewidth=0.6,
                  yerr=sems, capsize=4, error_kw={"linewidth": 0.8})
    for xi, (m, s, n) in enumerate(zip(means, sems, ns)):
        if not np.isnan(m):
            top = (m + s) if not np.isnan(s) else m
            ax.text(xi, max(top, 0) + 0.005, f"n={n:,}",
                    ha="center", va="bottom", fontsize=7.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_ORDER, fontsize=9)
    ax.set_xlabel("|Measured Δlogit(PSI)|", fontsize=10)
    ax.set_ylabel("Mean improvement in absolute error\n(baseline − retrained, Δlogit units)", fontsize=9)
    ax.set_title(f"Retraining benefit by variant effect size\n({label}, n={len(df):,})", fontsize=10)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"improvement_by_effect_size_{label}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Plot: improvement by reference PSI bin (test set only) ────────────────────
def plot_improvement_by_psi_bin(df, label, outdir):
    df = df.copy()
    PSI_BINS   = [0, 0.1, 0.3, 0.7, 0.9, 1.0]
    PSI_LABELS = ["0–0.1\n(near-excl.)", "0.1–0.3", "0.3–0.7\n(interm.)", "0.7–0.9", "0.9–1.0\n(near-incl.)"]
    PSI_COLORS = ["#e41a1c", "#fc8d59", "#ffffbf", "#91bfdb", "#4575b4"]

    df["psi_bin"] = pd.cut(df["HEK_wt_pooled_psi_raw"], bins=PSI_BINS,
                            labels=PSI_LABELS, include_lowest=True)

    grp   = df.groupby("psi_bin")["improvement"]
    means = [grp.get_group(b).mean()  if b in grp.groups else np.nan for b in PSI_LABELS]
    sems  = [grp.get_group(b).sem()   if b in grp.groups else np.nan for b in PSI_LABELS]
    ns    = [grp.get_group(b).count() if b in grp.groups else 0      for b in PSI_LABELS]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    x = np.arange(len(PSI_LABELS))
    ax.bar(x, means, color=PSI_COLORS, edgecolor="black", linewidth=0.6,
           yerr=sems, capsize=4, error_kw={"linewidth": 0.8})
    for xi, (m, s, n) in enumerate(zip(means, sems, ns)):
        if not np.isnan(m):
            top = (m + s) if (not np.isnan(s) and m >= 0) else (m - s) if (not np.isnan(s) and m < 0) else m
            ax.text(xi, max(top, 0) + 0.005 if m >= 0 else min(top, 0) - 0.005,
                    f"n={n:,}", ha="center",
                    va="bottom" if m >= 0 else "top", fontsize=7.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(PSI_LABELS, fontsize=9)
    ax.set_xlabel("Reference exon PSI (HEK293)", fontsize=10)
    ax.set_ylabel("Mean improvement in absolute error\n(baseline − retrained, Δlogit units)", fontsize=9)
    ax.set_title(f"Retraining benefit by reference PSI\n({label}, n={len(df):,})", fontsize=10)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"improvement_by_psi_bin_{label}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Plot: improvement by seq_type (single vs double) × effect size ────────────
def plot_improvement_by_seqtype(df, label, outdir):
    df = df.copy()
    df["effect_bin"] = pd.cut(
        df["avg_delta_logit_pooled"].abs(),
        bins=[0, 0.5, 1, 2, np.inf],
        labels=["<0.5\n(sub-thresh.)", "0.5–1\n(moderate)", "1–2\n(SDV)", ">2\n(strong SDV)"]
    )
    BIN_ORDER  = ["<0.5\n(sub-thresh.)", "0.5–1\n(moderate)", "1–2\n(SDV)", ">2\n(strong SDV)"]
    TYPE_COLOR = {"single": "#2ca02c", "double": "#9467bd"}

    seq_types = [t for t in ["single", "double"] if t in df["seq_type"].values]
    n_types   = len(seq_types)
    bar_width = 0.35
    x = np.arange(len(BIN_ORDER))

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    for i, stype in enumerate(seq_types):
        sub = df[df["seq_type"] == stype]
        grp = sub.groupby("effect_bin")["improvement"]
        r_base, _ = stats.pearsonr(sub["baseline_mmsplice_delta_logit"],  sub["avg_delta_logit_pooled"])
        r_ret,  _ = stats.pearsonr(sub["retrained_mmsplice_delta_logit"], sub["avg_delta_logit_pooled"])
        means = [grp.get_group(b).mean()  if b in grp.groups else np.nan for b in BIN_ORDER]
        sems  = [grp.get_group(b).sem()   if b in grp.groups else np.nan for b in BIN_ORDER]
        ns    = [grp.get_group(b).count() if b in grp.groups else 0      for b in BIN_ORDER]

        offset = (i - n_types / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width,
               color=TYPE_COLOR[stype], edgecolor="black", linewidth=0.5,
               yerr=sems, capsize=3, error_kw={"linewidth": 0.8},
               label=f"{stype} (r: {r_base:.3f}→{r_ret:.3f})")
        for xi, (m, s, n) in enumerate(zip(means, sems, ns)):
            if n > 0 and not np.isnan(m):
                top = (m + s) if not np.isnan(s) else m
                ax.text(x[xi] + offset, max(top, 0) + 0.005, f"n={n:,}",
                        ha="center", va="bottom", fontsize=5.5, rotation=90)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_ORDER, fontsize=9)
    ax.set_xlabel("|Measured Δlogit(PSI)|", fontsize=10)
    ax.set_ylabel("Mean improvement in absolute error\n(baseline − retrained, Δlogit units)", fontsize=9)
    ax.set_title(f"Retraining benefit: single vs. double variants by effect size\n({label}, n={len(df):,})", fontsize=10)
    ax.legend(title="Variant type (Pearson r: base→retrained)", frameon=False, fontsize=8)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"improvement_by_seqtype_{label}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Plot: SpliceAI vs MMSplice (baseline & retrained) scatter ─────────────────
def plot_spliceai_vs_mmsplice(df, outdir):
    sub = df.dropna(subset=["spliceai_delta_logit", "baseline_mmsplice_delta_logit",
                             "retrained_mmsplice_delta_logit", "avg_delta_logit_pooled"])

    r_base, _ = stats.pearsonr(sub["spliceai_delta_logit"], sub["baseline_mmsplice_delta_logit"])
    r_ret,  _ = stats.pearsonr(sub["spliceai_delta_logit"], sub["retrained_mmsplice_delta_logit"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    for ax, col, r, title in [
        (axes[0], "baseline_mmsplice_delta_logit",  r_base, f"Baseline MMSplice vs SpliceAI\nr={r_base:.3f}, n={len(sub):,}"),
        (axes[1], "retrained_mmsplice_delta_logit", r_ret,  f"Retrained MMSplice vs SpliceAI\nr={r_ret:.3f}, n={len(sub):,}"),
    ]:
        ax.scatter(sub["spliceai_delta_logit"], sub[col],
                   s=3, alpha=0.2, color="#1f77b4",
                   linewidths=0, rasterized=True)
        lim = max(sub["spliceai_delta_logit"].abs().max(), sub[col].abs().max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="grey", linestyle="--", linewidth=0.8, zorder=5)
        ax.axhline(0, color="black", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("SpliceAI Δlogit", fontsize=10)
        ax.set_ylabel(col.replace("_delta_logit", "").replace("_", " ").title() + " Δlogit", fontsize=10)
        ax.set_title(title, fontsize=9)
        ax.grid(False)

    fig.suptitle("SpliceAI vs MMSplice predictions (aggregate, n={:,})".format(len(sub)),
                 fontsize=10, y=1.01)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"spliceai_vs_mmsplice_scatter.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Run all plots ──────────────────────────────────────────────────────────────
for label, df in subsets.items():
    print(f"\n── Plotting: {label} (n={len(df):,}) ──")
    plot_improvement_bars(df, label, OUTDIR)
    plot_error_scatter(df, label, OUTDIR)
    plot_improvement_by_effect_size(df, label, OUTDIR)
    plot_improvement_by_psi_bin(df, label, OUTDIR)
    plot_improvement_by_seqtype(df, label, OUTDIR)

plot_r_comparison(subsets, OUTDIR, mega)
plot_spliceai_vs_mmsplice(mega, OUTDIR)

print("\nDone.")
