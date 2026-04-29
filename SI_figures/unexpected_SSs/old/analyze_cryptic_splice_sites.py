"""
Cryptic Splice Site Analysis
============================
Identifies cryptic (unexpected) splice junctions per sequence, compares variants to WT,
and generates publication-quality figures.

Outputs (in same directory as this script):
  - cryptic_junctions_per_sequence.csv
  - variant_vs_wt_cryptic_comparison.csv
  - cryptic_splice_site_analysis.pdf
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from pathlib import Path

# Make PDF/SVG text editable
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).parent

# Splicing count file locations
BASE_CELL_LINES = Path(
    "/home/ec2-user/environment/ESL/Analysis/PSI_from_STAR/Cell_lines_04_29_2024"
)
BASE_ORIG = Path(
    "/home/ec2-user/environment/ESL/Analysis/STAR_alignment/"
    "separate_concat_2023_09_19_d1c_ms75_from_s3_2023_11_27/original_SJ_pipeline"
)

# Map: cell_line_label -> list of (sample_name, base_dir)
CELLLINE_REPS = {
    "HEK": [
        ("HEK293_Rep1", BASE_ORIG),
        ("HEK293_Rep2", BASE_ORIG),
        ("HEK293_Rep3", BASE_ORIG),
    ],
    "HeLa": [
        ("HeLa_Rep1", BASE_ORIG),
        ("HeLa_Rep3_20231031", BASE_ORIG),
    ],
    "K562": [
        ("K562_Rep1", BASE_ORIG),
        ("K562_Rep3_20231031", BASE_ORIG),
    ],
    "MCF7": [
        ("MCF7_Rep1", BASE_CELL_LINES),
        ("MCF7_Rep2", BASE_CELL_LINES),
    ],
    "HMC3": [
        ("HMC3_Rep1", BASE_CELL_LINES),
        ("HMC3_Rep2", BASE_CELL_LINES),
    ],
}

SUPERTABLE = Path(
    "/home/ec2-user/environment/ESL/ESL_MPRA/Data_Pre-Processing/"
    "st_final_with_snp_and_coords_05_30_25.csv.gz"
)

PSI_DIR = Path("/ESL/Figures_SK/General_preprocessing/output_03_16_2026")
PSI_PREFIX = "03_16_2026_1e-2"

MINCOV = 10  # minimum total reads to report cryptic fraction

OUTPUT_DIR = SCRIPT_DIR  # write outputs alongside the script

# =============================================================================
# STEP 1: Load and tag cryptic junctions
# =============================================================================

def get_splicing_file(sample_name, base_dir):
    """Return path to _separate_all_splicing_counts.txt for a sample."""
    return (
        base_dir
        / f"{sample_name}_separate_splicing_profiles"
        / f"{sample_name}_separate_all_splicing_counts.txt"
    )


def load_splicing_counts(path):
    """Load a splicing counts file and return a DataFrame."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype={
            "Parent_ref": int,
            "SJ_5": int,
            "SJ_3": int,
            "Count": int,
            "All_freqs": float,
            "i1_jxn": str,
            "i2_jxn": str,
            "e_jxn": str,
        },
        na_values=["N/A", ""],
        keep_default_na=True,
    )
    return df


def tag_cryptic(df):
    """Add is_cryptic column: True when none of the canonical flags is set."""
    df = df.copy()
    df["i1_jxn"] = df["i1_jxn"].str.strip()
    df["i2_jxn"] = df["i2_jxn"].str.strip()
    df["e_jxn"] = df["e_jxn"].str.strip()
    df["is_cryptic"] = (
        (df["i1_jxn"] != "True")
        & (df["i2_jxn"] != "True")
        & (df["e_jxn"] != "True")
    )
    return df


def load_all_junctions():
    """
    Load splicing counts for all cell lines and replicates.
    Returns a DataFrame with columns:
      Parent_ref, SJ_5, SJ_3, Count, All_freqs, i1_jxn, i2_jxn, e_jxn,
      is_cryptic, cell_line, sample
    """
    all_dfs = []
    for cell_line, reps in CELLLINE_REPS.items():
        for sample_name, base_dir in reps:
            path = get_splicing_file(sample_name, base_dir)
            if not path.exists():
                print(f"  WARNING: missing {path}", file=sys.stderr)
                continue
            print(f"  Loading {cell_line} / {sample_name} ...")
            df = load_splicing_counts(path)
            df = tag_cryptic(df)
            df["cell_line"] = cell_line
            df["sample"] = sample_name
            all_dfs.append(
                df[
                    [
                        "Parent_ref",
                        "SJ_5",
                        "SJ_3",
                        "Count",
                        "is_cryptic",
                        "i1_jxn",
                        "i2_jxn",
                        "e_jxn",
                        "cell_line",
                        "sample",
                    ]
                ]
            )
    return pd.concat(all_dfs, ignore_index=True)


def pool_junctions_per_cellline(jxn_df):
    """
    Pool replicates within each cell line: sum Counts, keep is_cryptic flag.
    Returns DataFrame with columns:
      Parent_ref, SJ_5, SJ_3, Count_pooled, All_freqs_pooled, is_cryptic, cell_line
    """
    pooled = (
        jxn_df.groupby(["cell_line", "Parent_ref", "SJ_5", "SJ_3", "is_cryptic"])["Count"]
        .sum()
        .reset_index()
        .rename(columns={"Count": "Count_pooled"})
    )
    # Compute pooled All_freqs = Count_pooled / total_pooled per (cell_line, Parent_ref)
    total = pooled.groupby(["cell_line", "Parent_ref"])["Count_pooled"].transform("sum")
    pooled["All_freqs_pooled"] = np.where(total > 0, pooled["Count_pooled"] / total, np.nan)
    return pooled


# =============================================================================
# STEP 2: Per-sequence cryptic summary table
# =============================================================================

def compute_per_sequence_summary(pooled_jxn, supertable):
    """
    For each (Parent_ref, cell_line), compute:
      - total_reads, cryptic_reads, cryptic_fraction
      - n_distinct_cryptic_junctions
      - dominant_cryptic_junction (SJ_5, SJ_3 as string)
    Returns wide-format DataFrame merged with supertable metadata.
    """
    records = []
    for (cell_line, parent_ref), grp in pooled_jxn.groupby(["cell_line", "Parent_ref"]):
        total_reads = grp["Count_pooled"].sum()
        cryptic_grp = grp[grp["is_cryptic"]]
        cryptic_reads = cryptic_grp["Count_pooled"].sum()
        if total_reads >= MINCOV:
            cryptic_fraction = cryptic_reads / total_reads
        else:
            cryptic_fraction = np.nan
        n_distinct = len(cryptic_grp)
        if len(cryptic_grp) > 0:
            dom_idx = cryptic_grp["Count_pooled"].idxmax()
            dom_jxn = f"{cryptic_grp.loc[dom_idx, 'SJ_5']}:{cryptic_grp.loc[dom_idx, 'SJ_3']}"
        else:
            dom_jxn = np.nan
        records.append(
            {
                "Parent_ref": parent_ref,
                "cell_line": cell_line,
                "total_reads": total_reads,
                "cryptic_reads": cryptic_reads,
                "cryptic_fraction": cryptic_fraction,
                "n_distinct_cryptic_junctions": n_distinct,
                "dominant_cryptic_junction": dom_jxn,
            }
        )
    summary = pd.DataFrame(records)

    # Pivot to wide format
    wide_dfs = {}
    for col in ["total_reads", "cryptic_reads", "cryptic_fraction",
                 "n_distinct_cryptic_junctions", "dominant_cryptic_junction"]:
        wide_dfs[col] = summary.pivot(
            index="Parent_ref", columns="cell_line", values=col
        ).add_suffix(f"_{col}")
    wide = pd.concat(wide_dfs.values(), axis=1).reset_index()

    # Merge with supertable
    st_cols = ["Reference", "event_id", "gene_name", "seq_type", "source", "snp",
               "exon", "intron1", "intron2"]
    st_use = supertable[st_cols].rename(columns={"Reference": "Parent_ref"})
    wide = wide.merge(st_use, on="Parent_ref", how="left")

    return wide, summary


# =============================================================================
# STEP 3: Variant vs. WT cryptic comparison
# =============================================================================

def load_psi_delta_logit():
    """
    Load delta_logit_pooled and dpsi_pooled from the WITH_WT merged files.
    Returns a DataFrame with Reference + per-cell-line delta_logit and dpsi columns,
    plus seq_type and source.
    """
    cell_lines = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]
    meta_cols = ["Reference", "event_id", "seq_type", "source", "snp", "gene_exon"]
    all_dfs = []
    first = True
    base_df = None
    for cl in cell_lines:
        path = PSI_DIR / f"{PSI_PREFIX}_{cl}_WITH_WT.csv"
        if not path.exists():
            print(f"  WARNING: PSI file missing: {path}", file=sys.stderr)
            continue
        print(f"  Loading PSI data for {cl} ...")
        needed = meta_cols + [
            f"{cl}_delta_logit_pooled",
            f"{cl}_dpsi_pooled",
            f"{cl}_pooled_psi_raw",
        ]
        # Load only needed columns (check which exist)
        header = pd.read_csv(path, nrows=0)
        existing = [c for c in needed if c in header.columns]
        df = pd.read_csv(path, usecols=existing)
        if first:
            base_df = df[meta_cols].drop_duplicates("Reference")
            first = False
        cl_df = df[["Reference"] + [c for c in existing if c not in meta_cols]]
        all_dfs.append(cl_df)
    if base_df is None:
        return pd.DataFrame()
    merged = base_df
    for df in all_dfs:
        merged = merged.merge(df, on="Reference", how="outer")
    return merged


def make_variant_comparison(summary_long, psi_df, supertable):
    """
    For each variant (seq_type != 'reference'), find WT counterpart via event_id.
    Compute delta_cryptic_fraction per cell line and merge with delta_logit.
    """
    cell_lines = list(CELLLINE_REPS.keys())
    # Get per-seq-per-CL cryptic fraction from summary_long
    cf = summary_long[["Parent_ref", "cell_line", "cryptic_fraction"]].copy()

    # Merge with supertable to get event_id and seq_type
    st_cols = ["Reference", "event_id", "seq_type", "source", "snp", "gene_name"]
    st_use = supertable[st_cols].rename(columns={"Reference": "Parent_ref"})
    cf = cf.merge(st_use, on="Parent_ref", how="left")

    # Separate WT sequences (seq_type='none' means no variant = reference/WT)
    wt = cf[cf["seq_type"] == "none"][
        ["event_id", "cell_line", "cryptic_fraction"]
    ].rename(columns={"cryptic_fraction": "wt_cryptic_fraction"})

    # Variants (all non-WT sequences)
    var = cf[cf["seq_type"] != "none"].copy()
    var = var.merge(wt, on=["event_id", "cell_line"], how="left")
    var["delta_cryptic_fraction"] = var["cryptic_fraction"] - var["wt_cryptic_fraction"]

    # Merge with delta_logit
    if psi_df is not None and len(psi_df) > 0:
        psi_use = psi_df.rename(columns={"Reference": "Parent_ref"})
        # Select relevant delta_logit columns
        dl_cols = ["Parent_ref"] + [c for c in psi_df.columns if "delta_logit_pooled" in c]
        psi_use = psi_use[dl_cols]
        var = var.merge(psi_use, on="Parent_ref", how="left")

    return var


# =============================================================================
# STEP 4: Canonical junction positions per Parent_ref (for Figure B)
# =============================================================================

def extract_canonical_positions(jxn_df):
    """
    For each Parent_ref, extract:
      - upstream_5ss: SJ_5 of e_jxn or i1_jxn row
      - exon_3ss:    SJ_3 of i1_jxn row (3' end of intron1 = exon start)
      - exon_5ss:    SJ_5 of i2_jxn row (5' end of intron2 = exon end)
      - downstream_3ss: SJ_3 of e_jxn or i2_jxn row
    Returns DataFrame indexed by Parent_ref.
    """
    # Use pooled data from one representative cell line (or all combined)
    canon = jxn_df.copy()

    i1 = canon[canon["i1_jxn"] == "True"][["Parent_ref", "SJ_5", "SJ_3"]].rename(
        columns={"SJ_5": "upstream_5ss", "SJ_3": "exon_3ss"}
    )
    i2 = canon[canon["i2_jxn"] == "True"][["Parent_ref", "SJ_5", "SJ_3"]].rename(
        columns={"SJ_5": "exon_5ss", "SJ_3": "downstream_3ss"}
    )
    e = canon[canon["e_jxn"] == "True"][["Parent_ref", "SJ_5", "SJ_3"]].rename(
        columns={"SJ_5": "e_upstream_5ss", "SJ_3": "e_downstream_3ss"}
    )

    # Keep one row per Parent_ref (take first occurrence across all samples)
    i1 = i1.drop_duplicates("Parent_ref")
    i2 = i2.drop_duplicates("Parent_ref")
    e = e.drop_duplicates("Parent_ref")

    canon_pos = i1.merge(i2, on="Parent_ref", how="outer").merge(e, on="Parent_ref", how="outer")
    # Fill upstream_5ss from e_jxn if i1 missing
    canon_pos["upstream_5ss"] = canon_pos["upstream_5ss"].fillna(canon_pos["e_upstream_5ss"])
    canon_pos["downstream_3ss"] = canon_pos["downstream_3ss"].fillna(canon_pos["e_downstream_3ss"])
    canon_pos = canon_pos[["Parent_ref", "upstream_5ss", "exon_3ss", "exon_5ss", "downstream_3ss"]]
    return canon_pos


def compute_cryptic_relative_positions(pooled_jxn, jxn_raw):
    """
    For each cryptic junction, compute position of SJ_5 and SJ_3 relative to
    upstream_5ss (the constitutive 5'SS), so position=0 is the start of the
    variable region and ~161 is the end.
    Returns a DataFrame with relative positions and which part they fall in.
    """
    canon_pos = extract_canonical_positions(jxn_raw)
    cryptic = pooled_jxn[pooled_jxn["is_cryptic"]].copy()
    cryptic = cryptic.merge(canon_pos, on="Parent_ref", how="left")

    cryptic["rel_5ss"] = cryptic["SJ_5"] - cryptic["upstream_5ss"]
    cryptic["rel_3ss"] = cryptic["SJ_3"] - cryptic["upstream_5ss"]

    # Tag region
    def region_5ss(row):
        if pd.isna(row["exon_3ss"]):
            return "unknown"
        if row["SJ_5"] < row["exon_3ss"]:
            return "intron1"
        elif row["SJ_5"] <= row["exon_5ss"]:
            return "exon"
        else:
            return "intron2"

    def region_3ss(row):
        if pd.isna(row["exon_3ss"]):
            return "unknown"
        if row["SJ_3"] < row["exon_3ss"]:
            return "intron1"
        elif row["SJ_3"] <= row["exon_5ss"]:
            return "exon"
        else:
            return "intron2"

    cryptic["region_5ss"] = cryptic.apply(region_5ss, axis=1)
    cryptic["region_3ss"] = cryptic.apply(region_3ss, axis=1)
    return cryptic


# =============================================================================
# STEP 5: Figures
# =============================================================================

CELL_LINES = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]

# Color palette matching paper style
SEQ_TYPE_ORDER = ["none", "single", "double"]
# source values: clinvar_single, clinvar_double, exac, geuvadis
SOURCE_COLORS = {
    "clinvar_single": "#d62728",
    "clinvar_double": "#9467bd",
    "exac": "#1f77b4",
    "geuvadis": "#ff7f0e",
    "none": "#7f7f7f",   # WT/reference
}


def _get_source_color(source):
    if pd.isna(source):
        return "#bcbd22"
    src = str(source).lower()
    for k, v in SOURCE_COLORS.items():
        if k in src:
            return v
    return "#bcbd22"


def figure_A_cryptic_by_source(summary_wide):
    """
    Figure A: Average cryptic fraction by source/seq_type (violin + strip).
    """
    fig, axes = plt.subplots(1, len(CELL_LINES), figsize=(14, 4), sharey=True)

    # Average cryptic fraction across cell lines per sequence
    cf_cols = [f"{cl}_cryptic_fraction" for cl in CELL_LINES if f"{cl}_cryptic_fraction" in summary_wide.columns]
    summary_wide = summary_wide.copy()
    summary_wide["avg_cryptic_fraction"] = summary_wide[cf_cols].mean(axis=1)

    # Build category label from actual seq_type/source values
    # seq_type: none=WT, single=single variant, double=double variant
    # source: clinvar_single, clinvar_double, exac, geuvadis
    def get_category(row):
        stype = str(row.get("seq_type", "")).strip()
        src = str(row.get("source", "")).strip().lower()
        if stype == "none":
            return "WT"
        if src == "clinvar_single":
            return "ClinVar\nSingle"
        if src == "clinvar_double":
            return "ClinVar\nDouble"
        if src == "exac":
            return "ExAC"
        if src == "geuvadis":
            return "Geuvadis"
        if stype == "single":
            return "Single\nVariant"
        if stype == "double":
            return "Double\nVariant"
        return "Other"

    summary_wide["category"] = summary_wide.apply(get_category, axis=1)
    cat_order = ["WT", "ClinVar\nSingle", "ClinVar\nDouble", "ExAC", "Geuvadis",
                 "Single\nVariant", "Double\nVariant", "Other"]
    cat_order = [c for c in cat_order if c in summary_wide["category"].unique()]

    for i, (cl, ax) in enumerate(zip(CELL_LINES, axes)):
        col = f"{cl}_cryptic_fraction"
        if col not in summary_wide.columns:
            ax.set_visible(False)
            continue
        plot_df = summary_wide[["category", col]].dropna()
        cat_present = [c for c in cat_order if c in plot_df["category"].unique()]
        sns.violinplot(
            data=plot_df,
            x="category",
            y=col,
            order=cat_present,
            cut=0,
            inner=None,
            color="#aec6cf",
            linewidth=0.8,
            ax=ax,
        )
        strip = sns.stripplot(
            data=plot_df,
            x="category",
            y=col,
            order=cat_present,
            size=1.2,
            alpha=0.3,
            color="#444444",
            jitter=True,
            ax=ax,
        )
        # Rasterize individual points so the PDF stays small
        for coll in ax.collections:
            coll.set_rasterized(True)
        ax.set_title(cl, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Cryptic fraction" if i == 0 else "")
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle("Cryptic splice site usage by variant class", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def figure_B_position_map(cryptic_pos_df):
    """
    Figure B: Histogram of cryptic 5'SS and 3'SS positions relative to upstream 5'SS.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    # Filter to positions within [0, 200] (variable region + small buffer)
    for ax, (col, label) in zip(
        axes, [("rel_5ss", "Cryptic 5' splice site"), ("rel_3ss", "Cryptic 3' splice site")]
    ):
        pos = cryptic_pos_df[col].dropna()
        pos = pos[(pos >= -20) & (pos <= 220)]
        weights = cryptic_pos_df.loc[pos.index, "Count_pooled"].fillna(1)
        ax.hist(pos, bins=50, weights=weights, color="#4878cf", edgecolor="white", linewidth=0.3, alpha=0.85)
        ax.set_xlabel(f"Position relative to constitutive 5'SS (nt)")
        ax.set_ylabel("Read count")
        ax.set_title(label)
        # Mark approximate variable region boundaries
        # Most constructs have upstream 5'SS at 0, variable region ~0-161
        ax.axvline(0, color="#d62728", lw=1.2, linestyle="--", label="Constitutive 5'SS")
        ax.axvline(161, color="#2ca02c", lw=1.2, linestyle="--", label="~161 nt boundary")
        ax.legend(fontsize=8)

    fig.suptitle("Distribution of cryptic splice site positions", fontsize=12)
    plt.tight_layout()
    return fig


def figure_C_scatter_delta(var_df):
    """
    Figure C: Scatter delta_cryptic_fraction vs. delta_logit_pooled per cell line.
    var_df has one row per (Parent_ref, cell_line) — 'delta_cryptic_fraction' and
    '{CL}_delta_logit_pooled' columns (wide, from PSI merge).
    """
    # Build per-cell-line sub-DataFrames
    panel_data = []
    for cl in CELL_LINES:
        dl_col = f"{cl}_delta_logit_pooled"
        if dl_col not in var_df.columns:
            continue
        sub = var_df[var_df["cell_line"] == cl][
            ["Parent_ref", "delta_cryptic_fraction", dl_col, "source", "gene_name"]
        ].dropna(subset=["delta_cryptic_fraction", dl_col])
        if len(sub) > 0:
            panel_data.append((cl, dl_col, sub))

    n = len(panel_data)
    if n == 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "No delta_logit data available", ha="center", va="center")
        return fig

    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    def get_color(src):
        src = str(src).lower()
        if "clinvar_single" in src:
            return "#d62728"
        if "clinvar_double" in src:
            return "#9467bd"
        if "exac" in src:
            return "#1f77b4"
        return "#7f7f7f"

    for ax, (cl, dl_col, plot_df) in zip(axes, panel_data):
        colors = plot_df["source"].apply(get_color)
        ax.scatter(
            plot_df[dl_col],
            plot_df["delta_cryptic_fraction"],
            c=colors,
            s=8,
            alpha=0.4,
            linewidths=0,
            rasterized=True,
        )
        # Pearson r
        r, pval = stats.pearsonr(plot_df[dl_col], plot_df["delta_cryptic_fraction"])
        ax.text(
            0.05, 0.95,
            f"r = {r:.3f}\np = {pval:.2e}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
        )
        ax.axhline(0, color="black", lw=0.5, linestyle="--")
        ax.axvline(0, color="black", lw=0.5, linestyle="--")
        ax.set_xlabel(f"Δlogit(PSI) [{cl}]", fontsize=9)
        ax.set_ylabel("Δ cryptic fraction", fontsize=9)
        ax.set_title(cl, fontsize=10)

        # Highlight top variants (large |delta_cryptic| AND large |delta_logit|)
        top = plot_df[
            (plot_df["delta_cryptic_fraction"].abs() > 0.1) &
            (plot_df[dl_col].abs() > 1.0)
        ].head(20)
        for _, row in top.iterrows():
            ax.annotate(
                str(row.get("gene_name", "")),
                (row[dl_col], row["delta_cryptic_fraction"]),
                fontsize=5,
                alpha=0.7,
            )

    # Remove unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#d62728", markersize=6, label="ClinVar single"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#9467bd", markersize=6, label="ClinVar double"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#1f77b4", markersize=6, label="ExAC"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#7f7f7f", markersize=6, label="Other"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Δcryptic fraction vs. Δlogit(PSI) per variant", fontsize=12)
    plt.tight_layout()
    return fig


def figure_D_top_genes(summary_wide, supertable):
    """
    Figure D: Top 20 genes by average cryptic fraction in WT/reference sequences.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    cf_cols = [f"{cl}_cryptic_fraction" for cl in CELL_LINES if f"{cl}_cryptic_fraction" in summary_wide.columns]
    wt_mask = summary_wide["seq_type"] == "none"
    wt_df = summary_wide[wt_mask].copy()
    wt_df["avg_cryptic"] = wt_df[cf_cols].mean(axis=1)
    gene_avg = (
        wt_df.groupby("gene_name")["avg_cryptic"]
        .mean()
        .sort_values(ascending=False)
        .head(20)
    )

    ax.barh(gene_avg.index[::-1], gene_avg.values[::-1], color="#4878cf", edgecolor="white")
    ax.set_xlabel("Mean cryptic fraction (WT sequences)")
    ax.set_title("Top 20 genes by cryptic splice site usage (WT)", fontsize=11)
    ax.axvline(0.1, color="#d62728", lw=1, linestyle="--", label="10% threshold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=== Cryptic Splice Site Analysis ===")

    # Load supertable
    print("\n[1/6] Loading supertable...")
    supertable = pd.read_csv(SUPERTABLE)
    print(f"  Supertable: {len(supertable)} rows, columns: {list(supertable.columns)}")

    # Step 1: Load all junctions
    print("\n[2/6] Loading splicing junction counts...")
    jxn_raw = load_all_junctions()
    print(f"  Total junction rows: {len(jxn_raw):,}")
    n_cryptic = jxn_raw["is_cryptic"].sum()
    print(f"  Cryptic junctions: {n_cryptic:,} ({100*n_cryptic/len(jxn_raw):.1f}%)")

    # Pool replicates per cell line
    print("\n[3/6] Pooling junctions per cell line...")
    pooled_jxn = pool_junctions_per_cellline(jxn_raw)

    # Step 2: Per-sequence summary
    print("\n[4/6] Computing per-sequence cryptic summary...")
    summary_wide, summary_long = compute_per_sequence_summary(pooled_jxn, supertable)
    out_seq = OUTPUT_DIR / "cryptic_junctions_per_sequence.csv"
    summary_wide.to_csv(out_seq, index=False)
    print(f"  Saved: {out_seq}")

    # Sanity check
    cf_cols = [c for c in summary_wide.columns if "cryptic_fraction" in c and not c.startswith("n_")]
    if cf_cols:
        ref_mask = summary_wide.get("seq_type", pd.Series(dtype=str)) == "none"
        ref_median = summary_wide.loc[ref_mask, cf_cols].median().mean()
        print(f"  Sanity check — median cryptic fraction (WT sequences): {ref_median:.4f}")

    # Step 3: Variant vs. WT comparison
    print("\n[5/6] Computing variant vs. WT cryptic comparison...")
    psi_df = load_psi_delta_logit()
    var_comparison = make_variant_comparison(summary_long, psi_df, supertable)
    out_var = OUTPUT_DIR / "variant_vs_wt_cryptic_comparison.csv"
    var_comparison.to_csv(out_var, index=False)
    print(f"  Saved: {out_var}")

    # Compute positions for Figure B
    print("  Computing cryptic junction positions...")
    cryptic_pos = compute_cryptic_relative_positions(pooled_jxn, jxn_raw)

    # Step 4: Figures
    print("\n[6/6] Generating figures...")
    out_pdf = OUTPUT_DIR / "cryptic_splice_site_analysis.pdf"

    def save_fig(fig, name, pdf):
        """Save to both the multi-page PDF and an individual PNG."""
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        fig.savefig(OUTPUT_DIR / f"{name}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out_pdf) as pdf:
        fig_a = figure_A_cryptic_by_source(summary_wide)
        save_fig(fig_a, "figA_cryptic_by_source", pdf)
        print("  Figure A saved (cryptic usage by variant class)")

        fig_b = figure_B_position_map(cryptic_pos)
        save_fig(fig_b, "figB_position_map", pdf)
        print("  Figure B saved (position map)")

        if len(var_comparison) > 0:
            fig_c = figure_C_scatter_delta(var_comparison)
            save_fig(fig_c, "figC_delta_scatter", pdf)
            print("  Figure C saved (delta_cryptic vs delta_logit scatter)")
        else:
            print("  Figure C skipped (no variant comparison data)")

        fig_d = figure_D_top_genes(summary_wide, supertable)
        save_fig(fig_d, "figD_top_genes", pdf)
        print("  Figure D saved (top genes by cryptic usage)")

    print(f"\nDone. PDF: {out_pdf}")


if __name__ == "__main__":
    main()
