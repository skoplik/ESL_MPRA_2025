#!/usr/bin/env python3
"""
Family specificity stripplots.
Coloring rule: a variant point is colored (cell-line color) ONLY if it passes
BOTH criteria from the mingap pipeline:
    1. mingap >= MINGAP_THRESHOLD (0.25)
    2. ANOVA BH-adjusted p < 0.05 (cell_type_specific == True)
All other variant points are plotted in light grey.

Plots individual PDFs for all TARGET_FAMILIES, plus a 1x4 subplot panel
(shared y-axis) for SUBPLOT_FAMILIES.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import numpy as np

# Fonts: keep text editable in Illustrator
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# === Config ===================================================================
supertable_file = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/ALL_WITH_WT.csv"
mingap_file     = "/ESL/ESL_MPRA/Figure_5/outputs/mingap/mingap_variants_all_filtered.tsv"
output_dir      = "/ESL/ESL_MPRA/Figure_5/outputs/family_specificity_all"
os.makedirs(output_dir, exist_ok=True)

# Gene-exon families to plot (case-insensitive match against gene_exon column)
TARGET_FAMILIES = [
    "mef2c exon 8",
    "diaph1 exon 2",
    "usp28 exon 6",
    "tnnt2 exon 5",
    "oca2 exon 10",
    "dmd exon 71",
    "uros exon 3",
    "abcb7 exon 2",
    "colq exon 5",
]

# Subset to render as a 1x4 shared-y subplot panel
SUBPLOT_FAMILIES = [
    "diaph1 exon 2",
    "usp28 exon 6",
    "tnnt2 exon 5",
    "oca2 exon 10",
]

# Subset to render as a 1x2 shared-y subplot panel
SUBPLOT_FAMILIES_2 = [
    "uros exon 3",
    "abcb7 exon 2",
]

cell_to_logit  = {"HEK": "HEK_pooled_logit",  "HeLa": "HeLa_pooled_logit",
                  "K562": "K562_pooled_logit", "MCF7": "MCF7_pooled_logit",
                  "HMC3": "HMC3_pooled_logit"}
cell_to_psi    = {"HEK": "HEK_pooled_psi_raw",  "HeLa": "HeLa_pooled_psi_raw",
                  "K562": "K562_pooled_psi_raw", "MCF7": "MCF7_pooled_psi_raw",
                  "HMC3": "HMC3_pooled_psi_raw"}
cell_to_dlogit = {"HEK": "HEK_delta_logit_pooled",  "HeLa": "HeLa_delta_logit_pooled",
                  "K562": "K562_delta_logit_pooled", "MCF7": "MCF7_delta_logit_pooled",
                  "HMC3": "HMC3_delta_logit_pooled"}

cell_lines  = list(cell_to_logit.keys())
plot_labels = {"HEK": "HEK293", "HeLa": "HeLa", "K562": "K562",
               "MCF7": "MCF7",  "HMC3": "HMC3"}
categories  = [plot_labels[cl] for cl in cell_lines]

custom_rgb_colors = {
    "HEK293": "#E984B6",
    "HeLa":   "#7FBE7E",
    "MCF7":   "#807CB9",
    "HMC3":   "#EF4025",
    "K562":   "#F9AE33",
}

# === Load mingap file =========================================================
mingap = pd.read_csv(mingap_file, sep="\t")
mingap["Reference"] = mingap["Reference"].astype(str)
mingap["event_id"]  = mingap["event_id"].astype(str)

sig_mingap_refs = set(mingap["Reference"].unique())
sig_mingap_eids = set(mingap["event_id"].unique())

print(f"[INFO] sig+mingap: {len(sig_mingap_refs)} variants across "
      f"{len(sig_mingap_eids)} exon families")

# === Load supertable ==========================================================
df = pd.read_csv(supertable_file)
df["Reference"]    = df["Reference"].astype(str)
df["event_id"]     = df["event_id"].astype(str)
df["gene_exon_lc"] = df["gene_exon"].str.lower().str.strip()

target_lc = [f.lower().strip() for f in TARGET_FAMILIES]
df = df[df["gene_exon_lc"].isin(target_lc)]
df = df[df["event_id"].isin(sig_mingap_eids)]

required_dlogit_cols = list(cell_to_dlogit.values())
df = df[df[required_dlogit_cols].notnull().all(axis=1)]

found_families = df["gene_exon"].unique().tolist()
print(f"[INFO] Retained {len(df)} rows across {len(found_families)} target families: "
      f"{found_families}")

found_lc = set(df["gene_exon_lc"].unique())
for fam in target_lc:
    if fam not in found_lc:
        print(f"[WARN] Family not found / no sig+mingap rows: '{fam}'")

highlighted_variants = []


# =============================================================================
# Helper: build long-form plot_df
# =============================================================================
def build_plot_df(sub, ydict):
    rows = sub[sub["snp"] != "none"].copy()
    vals, labs, snps, refs = [], [], [], []
    for cl in cell_lines:
        col     = ydict[cl]
        cl_rows = rows[rows[col].notna()]
        vals.extend(cl_rows[col].values)
        labs.extend([plot_labels[cl]] * len(cl_rows))
        snps.extend(cl_rows["snp"].values)
        refs.extend(cl_rows["Reference"].values)
    plot_df = pd.DataFrame({"cell_line": labs, "value": vals,
                             "snp": snps, "Reference": refs})
    plot_df["is_sig_mingap"] = plot_df["Reference"].isin(sig_mingap_refs)
    return plot_df


# =============================================================================
# Helper: draw one panel onto an Axes
# =============================================================================
def draw_panel(ax, plot_df, sub, suffix, gene_exon, eid,
               ylabel, show_ylabel=True, show_xlabel=True):
    sig_rows   = plot_df[plot_df["is_sig_mingap"]].copy()
    other_rows = plot_df[~plot_df["is_sig_mingap"]].copy()
    n_variants = plot_df["Reference"].nunique()
    n_colored  = sig_rows["Reference"].nunique()

    # Grey (non-sig) points -- behind everything
    if not other_rows.empty:
        sns.stripplot(
            data=other_rows, x="cell_line", y="value",
            order=categories,
            color="lightgrey", size=3, jitter=True,
            zorder=3, ax=ax,
        )

    # Boxplot -- in front of grey dots
    sns.boxplot(
        data=plot_df, x="cell_line", y="value",
        order=categories,
        width=0.4, showcaps=True, showfliers=False,
        boxprops={"facecolor": "none", "edgecolor": "dimgrey"},
        whiskerprops={"color": "dimgrey"},
        medianprops={"color": "dimgrey"},
        capprops={"color": "dimgrey"},
        zorder=10, ax=ax,
    )
    for artist in ax.get_children():
        if isinstance(artist, (mpl.lines.Line2D, mpl.patches.PathPatch,
                                mpl.patches.FancyArrow)):
            artist.set_zorder(10)

    # Colored (sig + mingap) points -- in front of box/whisker
    if not sig_rows.empty:
        highlighted_variants.extend(
            sig_rows[["Reference", "snp"]]
            .assign(gene_exon=gene_exon, event_id=eid)
            .drop_duplicates()
            .to_dict("records")
        )
        sns.stripplot(
            data=sig_rows, x="cell_line", y="value",
            order=categories,
            palette=custom_rgb_colors, hue="cell_line",
            hue_order=categories,
            size=4, jitter=True, alpha=0.8,
            edgecolor="black", linewidth=0.8,
            zorder=20, ax=ax,
        )
        if ax.legend_:
            ax.legend_.remove()

    # WT PSI dots (PSI plots only)
    if suffix == "psi":
        x_index = {cat: pos for pos, cat in enumerate(categories)}
        for cl in cell_lines:
            wt_col = f"{cl}_wt_pooled_psi_raw"
            if wt_col in sub.columns:
                wt_val = sub[wt_col].iloc[0]
                if pd.notna(wt_val):
                    ax.scatter(
                        x_index[plot_labels[cl]], wt_val,
                        facecolors="black", edgecolors="black",
                        s=12, zorder=20, marker="o", linewidth=1,
                    )

    ax.set_title(f"{gene_exon} (n={n_variants})", fontsize=11)
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=10)
    ax.set_xlabel("Cell line" if show_xlabel else "", fontsize=10)
    ax.tick_params(axis='x', labelsize=9, rotation=0)
    ax.tick_params(axis='y', labelsize=9)


# =============================================================================
# Individual plots
# =============================================================================
def plot_family(sub, eid, gene_exon, ydict, ylabel, suffix):
    plot_df = build_plot_df(sub, ydict)
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(3, 2.6))
    draw_panel(ax, plot_df, sub, suffix, gene_exon, eid, ylabel)
    plt.tight_layout()
    safe_name = gene_exon.replace(' ', '_').replace('/', '_')
    out_pdf = os.path.join(output_dir, f"{safe_name}_{suffix}.pdf")
    plt.savefig(out_pdf, dpi=300)
    plt.close()
    print(f"[PLOT] {out_pdf}")


# =============================================================================
# 1x4 subplot panel (shared y-axis)
# =============================================================================
def plot_subplot_panel(df_all, ydict, ylabel, suffix, family_list, name_prefix):
    subplot_lc = [f.lower().strip() for f in family_list]

    entries = []
    for fam_lc in subplot_lc:
        sub_fam = df_all[df_all["gene_exon_lc"] == fam_lc]
        if sub_fam.empty:
            print(f"[WARN] Subplot family not found: '{fam_lc}' -- skipping panel slot")
            continue
        eid       = sub_fam["event_id"].iloc[0]
        gene_exon = sub_fam["gene_exon"].iloc[0]
        plot_df   = build_plot_df(sub_fam, ydict)
        entries.append((sub_fam, eid, gene_exon, plot_df))

    if not entries:
        print(f"[WARN] No subplot families found for {suffix}, skipping panel.")
        return

    n = len(entries)

    # Global y limits across all panels for shared axis alignment
    all_vals = pd.concat([e[3]["value"] for e in entries], ignore_index=True).dropna()
    y_pad    = (all_vals.max() - all_vals.min()) * 0.05
    y_min    = all_vals.min() - y_pad
    y_max    = all_vals.max() + y_pad

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 2.6),
                              sharey=True)
    fig.subplots_adjust(wspace=0.12)
    if n == 1:
        axes = [axes]

    for i, (sub_fam, eid, gene_exon, plot_df) in enumerate(entries):
        ax = axes[i]
        draw_panel(
            ax, plot_df, sub_fam, suffix, gene_exon, eid, ylabel,
            show_ylabel=(i == 0),
            show_xlabel=True,
        )
        ax.set_ylim(y_min, y_max)
        # Hide redundant left spine/ticks on non-first panels but keep border
        if i > 0:
            ax.tick_params(axis='y', left=False, labelleft=False)
            ax.spines["left"].set_visible(True)

    safe_suffix = suffix.replace(" ", "_")
    out_pdf = os.path.join(output_dir, f"{name_prefix}_{safe_suffix}.pdf")
    out_svg = out_pdf.replace(".pdf", ".svg")
    fig.savefig(out_pdf, dpi=300, format="pdf")
    fig.savefig(out_svg, dpi=300, format="svg")
    plt.close()
    print(f"[PANEL] {out_pdf}")


# === Run =====================================================================
for eid, sub in df.groupby("event_id"):
    gene_exon = sub["gene_exon"].iloc[0]
    plot_family(sub, eid, gene_exon, cell_to_psi, "PSI", "psi")

print("[INFO] Generating 1x4 subplot panel...")
plot_subplot_panel(df, cell_to_psi, "PSI", "psi", SUBPLOT_FAMILIES,  "subplot_panel_1x4")
print("[INFO] Generating 1x2 subplot panel...")
plot_subplot_panel(df, cell_to_psi, "PSI", "psi", SUBPLOT_FAMILIES_2, "subplot_panel_1x2")

if highlighted_variants:
    out_tsv = os.path.join(output_dir, "highlighted_variants.tsv")
    pd.DataFrame(highlighted_variants).drop_duplicates().to_csv(out_tsv, sep="\t", index=False)
    print(f"[INFO] Exported highlighted variants to {out_tsv}")

print("[DONE]")