"""
Outlier Exon Family Analysis
=============================
Focuses on genes where variants strongly activate cryptic splice sites.
Reads pre-computed CSVs from analyze_cryptic_splice_sites.py.

Outputs:
  - outlier_gene_summary.csv
  - outlier_exon_families.pdf          (summary heatmap + one page per gene)
  - outlier_fig0_summary_heatmap.png
  - outlier_fig{N}_{GENE}.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.spines.top"]   = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["font.size"] = 9

# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).parent

VAR_CSV    = SCRIPT_DIR / "variant_vs_wt_cryptic_comparison.csv"
WIDE_CSV   = SCRIPT_DIR / "cryptic_junctions_per_sequence.csv"
SUPERTABLE = Path(
    "/home/ec2-user/environment/ESL/ESL_MPRA/Data_Pre-Processing/"
    "st_final_with_snp_and_coords_05_30_25.csv.gz"
)

CELL_LINES = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]
CL_COLORS  = {
    "HEK":  "#e41a1c",
    "HeLa": "#377eb8",
    "K562": "#4daf4a",
    "MCF7": "#984ea3",
    "HMC3": "#ff7f00",
}
SOURCE_COLORS = {
    "clinvar_single": "#d62728",
    "clinvar_double": "#9467bd",
    "exac":           "#1f77b4",
    "geuvadis":       "#ff7f0e",
}

# Region colors for position scan
REGION_COLORS = {
    "intron1": "#a8c8e8",   # light blue
    "exon":    "#f4a582",   # salmon
    "intron2": "#a1d99b",   # light green
}

MIN_ABS_DELTA_CRYPTIC = 0.15
MIN_OUTLIER_VARIANTS  = 3
TOP_N_GENES           = 15
UPSTREAM_5SS          = 26   # constitutive upstream 5'SS position in minigene coords

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    print("Loading variant comparison data...")
    var = pd.read_csv(VAR_CSV)
    # Add matched delta_logit (this row's own cell line)
    def matched_dl(row):
        return row.get(f"{row['cell_line']}_delta_logit_pooled", np.nan)
    var["delta_logit"] = var.apply(matched_dl, axis=1)

    print("Loading per-sequence summary data...")
    wide = pd.read_csv(WIDE_CSV)

    print("Loading supertable for canonical splice site positions...")
    st = pd.read_csv(
        SUPERTABLE,
        usecols=["Reference", "event_id", "gene_name", "intron1", "exon", "intron2", "seq_type"],
    )
    # Canonical boundaries per event (from WT sequences)
    wt = st[st["seq_type"] == "none"].copy()
    wt["exon_start"] = wt["intron1"].str.len()          # canonical 3'SS = exon start
    wt["exon_end"]   = wt["exon_start"] + wt["exon"].str.len()  # canonical 5'SS = exon end
    canon = (
        wt.groupby(["gene_name", "event_id"])[["exon_start", "exon_end"]]
        .first()
        .reset_index()
    )
    return var, wide, canon


# =============================================================================
# IDENTIFY OUTLIER GENES
# =============================================================================

def identify_outlier_genes(var):
    var = var.copy()
    var["abs_delta"] = var["delta_cryptic_fraction"].abs()

    records = []
    for gene, grp in var.groupby("gene_name"):
        n_vars = grp["Parent_ref"].nunique()
        max_abs = grp["abs_delta"].max()
        n_outlier = grp[grp["abs_delta"] >= MIN_ABS_DELTA_CRYPTIC]["Parent_ref"].nunique()
        cl_maxes = {}
        for cl in CELL_LINES:
            sub = grp[grp["cell_line"] == cl]["delta_cryptic_fraction"]
            cl_maxes[f"{cl}_max_delta_cryptic"] = sub.max() if len(sub) > 0 else np.nan
        records.append({
            "gene_name": gene,
            "n_variants": n_vars,
            "max_abs_delta_cryptic": max_abs,
            "n_outlier_variants": n_outlier,
            "frac_outlier_variants": n_outlier / n_vars if n_vars > 0 else 0,
            **cl_maxes,
        })

    summary = pd.DataFrame(records)
    summary = summary[summary["n_outlier_variants"] >= MIN_OUTLIER_VARIANTS]
    summary = summary.sort_values("max_abs_delta_cryptic", ascending=False)
    return summary


# =============================================================================
# SNP PARSING
# =============================================================================

def parse_snp_position(snp_str):
    """'12:A>G' → (12, 'A', 'G'), NaN on failure."""
    if pd.isna(snp_str) or str(snp_str) == "none":
        return None, None, None
    first = str(snp_str).split(";")[0].strip()
    try:
        pos_part, change = first.split(":")
        ref, alt = change.split(">")
        return int(pos_part), ref.strip(), alt.strip()
    except Exception:
        return None, None, None


# =============================================================================
# FIGURE: SUMMARY HEATMAP
# =============================================================================

def figure_summary_heatmap(summary, var, top_n=TOP_N_GENES):
    top_genes = summary.head(top_n)["gene_name"].tolist()
    matrix = np.full((len(top_genes), len(CELL_LINES)), np.nan)
    for i, gene in enumerate(top_genes):
        grp = var[var["gene_name"] == gene]
        for j, cl in enumerate(CELL_LINES):
            vals = grp[grp["cell_line"] == cl]["delta_cryptic_fraction"].dropna()
            if len(vals) > 0:
                matrix[i, j] = np.percentile(vals, 95)

    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(top_genes) + 1.5)))
    vmax = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="95th-pct Δcryptic fraction", shrink=0.6)
    ax.set_xticks(range(len(CELL_LINES)))
    ax.set_xticklabels(CELL_LINES)
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes, fontsize=8)
    ax.set_title(f"Top {top_n} genes — 95th-pct Δcryptic fraction by cell line", fontsize=11)
    for i in range(len(top_genes)):
        for j in range(len(CELL_LINES)):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(v) < 0.5 * vmax else "white")
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE: PER GENE
# =============================================================================

def _get_canonical_positions(gene, canon):
    """Return list of (exon_start, exon_end) for each event in this gene."""
    events = canon[canon["gene_name"] == gene][["exon_start", "exon_end"]].drop_duplicates()
    return list(events.itertuples(index=False, name=None))


def _source_color(src):
    src = str(src).lower()
    for k, v in SOURCE_COLORS.items():
        if k in src:
            return v
    return "#888888"


def figure_per_gene(gene, var, wide, canon):
    """
    Per-gene figure:
      Top row (full width):  A — position scan bar chart (cell-line avg)
      Bottom row 3 panels:   B — Δlogit_avg vs Δcryptic_avg scatter
                             C — top cryptic junctions barh
                             D — mean Δcryptic per cell line (bar)
    """
    grp     = var[var["gene_name"] == gene].copy()
    grp_wide = wide[wide.get("gene_name", pd.Series(dtype=str)) == gene].copy() \
               if "gene_name" in wide.columns else pd.DataFrame()

    canon_positions = _get_canonical_positions(gene, canon)

    # Per-variant averages across cell lines
    var_avg = (
        grp.groupby("Parent_ref")
        .agg(
            avg_delta_cryptic=("delta_cryptic_fraction", "mean"),
            avg_delta_logit=("delta_logit", "mean"),
            seq_type=("seq_type", "first"),
            source=("source", "first"),
            snp=("snp", "first"),
        )
        .reset_index()
    )

    # Parse snp positions for singles
    parsed = var_avg["snp"].apply(parse_snp_position)
    var_avg["snp_pos"] = [x[0] for x in parsed]
    singles_avg = var_avg[var_avg["seq_type"] == "single"].dropna(subset=["snp_pos"]).copy()
    singles_avg["snp_pos"] = singles_avg["snp_pos"].astype(int)

    # ----------------------------------------------------------------
    # Layout: 2 rows — top full-width, bottom 3 cols
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"Cryptic splice site — {gene}", fontsize=13, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1, 1],
        hspace=0.48, wspace=0.35,
        left=0.07, right=0.97, top=0.92, bottom=0.09,
    )
    ax_pos  = fig.add_subplot(gs[0, :])       # A: spans all 3 cols
    ax_scat = fig.add_subplot(gs[1, 0])       # B
    ax_jxn  = fig.add_subplot(gs[1, 1])       # C
    ax_cl   = fig.add_subplot(gs[1, 2])       # D

    # ------------------------------------------------------------------
    # Panel A: Position scan bar chart — cell-line average Δcryptic
    # ------------------------------------------------------------------
    if len(singles_avg) > 0:
        # Average over all alt bases at each position
        pos_avg = (
            singles_avg.groupby("snp_pos")["avg_delta_cryptic"]
            .mean()
            .reindex(range(161), fill_value=0)
        )

        # Color each bar by region (intron1 / exon / intron2)
        # Use first event's canonical positions for coloring
        if canon_positions:
            exon_start_main, exon_end_main = canon_positions[0]
        else:
            exon_start_main, exon_end_main = None, None

        bar_colors = []
        for pos in pos_avg.index:
            if exon_start_main is not None and exon_end_main is not None:
                if pos < exon_start_main:
                    bar_colors.append(REGION_COLORS["intron1"])
                elif pos < exon_end_main:
                    bar_colors.append(REGION_COLORS["exon"])
                else:
                    bar_colors.append(REGION_COLORS["intron2"])
            else:
                bar_colors.append("#aaaaaa")

        ax_pos.bar(pos_avg.index, pos_avg.values, color=bar_colors,
                   width=1.0, linewidth=0, align="center")
        ax_pos.axhline(0, color="black", lw=0.7, linestyle="-")

        # Mark canonical splice site positions (one per event)
        line_styles = ["-.", "--", ":"]
        for idx, (es, ee) in enumerate(canon_positions):
            ls = line_styles[idx % len(line_styles)]
            lbl_suffix = f" (event {idx+1})" if len(canon_positions) > 1 else ""
            ax_pos.axvline(es, color="#333333", lw=1.5, linestyle=ls,
                           label=f"Canonical 3'SS{lbl_suffix} (pos {es})")
            ax_pos.axvline(ee, color="#555555", lw=1.5, linestyle=ls,
                           label=f"Canonical 5'SS{lbl_suffix} (pos {ee})")

        ax_pos.set_xlim(-1, 161)
        ax_pos.set_xlabel("Position within 161 nt construct", fontsize=9)
        ax_pos.set_ylabel("Mean Δcryptic fraction\n(avg across cell lines)", fontsize=9)
        ax_pos.set_title("A   Position scan — single variants (cell-line average)",
                         fontsize=9, loc="left")
        ax_pos.legend(fontsize=7, loc="upper right", framealpha=0.8, ncol=2)

        # Region legend patches
        patches = [
            mpatches.Patch(color=REGION_COLORS["intron1"], label="Intron 1"),
            mpatches.Patch(color=REGION_COLORS["exon"],    label="Exon"),
            mpatches.Patch(color=REGION_COLORS["intron2"], label="Intron 2"),
        ]
        ax_pos.legend(
            handles=patches + ax_pos.get_legend_handles_labels()[0][len(patches):],
            fontsize=7, loc="upper right", framealpha=0.8, ncol=3,
        )
        # rebuild with just splice site lines + region patches
        sj_handles, sj_labels = ax_pos.get_legend_handles_labels()
        ax_pos.legend(
            handles=patches + sj_handles,
            labels=["Intron 1", "Exon", "Intron 2"] + sj_labels,
            fontsize=7, loc="upper right", framealpha=0.8, ncol=3,
        )
    else:
        ax_pos.text(0.5, 0.5, "No single variants with position data",
                    ha="center", va="center", transform=ax_pos.transAxes, color="gray")
        ax_pos.set_title("A   Position scan", fontsize=9, loc="left")

    # ------------------------------------------------------------------
    # Panel B: avg Δlogit vs avg Δcryptic — one dot per variant
    # ------------------------------------------------------------------
    plot_b = var_avg.dropna(subset=["avg_delta_logit", "avg_delta_cryptic"])
    colors_b = plot_b["source"].apply(_source_color)

    ax_scat.scatter(
        plot_b["avg_delta_logit"],
        plot_b["avg_delta_cryptic"],
        c=colors_b, s=12, alpha=0.55, linewidths=0, rasterized=True,
    )
    ax_scat.axhline(0, color="black", lw=0.6, linestyle="--")
    ax_scat.axvline(0, color="black", lw=0.6, linestyle="--")

    if len(plot_b) >= 5:
        r, p = stats.pearsonr(plot_b["avg_delta_logit"], plot_b["avg_delta_cryptic"])
        ax_scat.text(0.05, 0.97, f"r = {r:.3f}\np = {p:.1e}",
                     transform=ax_scat.transAxes, fontsize=7, va="top",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    # Source legend
    seen_sources = plot_b["source"].dropna().unique()
    src_handles = [
        mpatches.Patch(color=_source_color(s), label=s) for s in seen_sources
    ]
    ax_scat.legend(handles=src_handles, fontsize=6, loc="lower right", framealpha=0.8)
    ax_scat.set_xlabel("Avg Δlogit(PSI) across cell lines", fontsize=8)
    ax_scat.set_ylabel("Avg Δcryptic fraction", fontsize=8)
    ax_scat.set_title("B   Avg Δlogit vs Δcryptic (per variant)", fontsize=9, loc="left")

    # ------------------------------------------------------------------
    # Panel C: Top cryptic junctions (barh)
    # ------------------------------------------------------------------
    jxn_counts = {}
    for cl in CELL_LINES:
        col = f"{cl}_dominant_cryptic_junction"
        if col not in grp_wide.columns:
            continue
        for jxn in grp_wide[col].dropna():
            jxn_counts[jxn] = jxn_counts.get(jxn, 0) + 1

    if jxn_counts:
        jxn_series = pd.Series(jxn_counts).sort_values(ascending=False).head(10)
        def rel(j):
            try:
                a, b = j.split(":")
                return f"{int(a) - UPSTREAM_5SS} : {int(b) - UPSTREAM_5SS}"
            except Exception:
                return j
        jxn_series.index = [rel(j) for j in jxn_series.index]
        ax_jxn.barh(jxn_series.index[::-1], jxn_series.values[::-1],
                    color="#4878cf", edgecolor="white", linewidth=0.4, height=0.7)
        ax_jxn.set_xlabel("# sequences (across cell lines)", fontsize=8)
        ax_jxn.set_ylabel("5'SS_rel : 3'SS_rel", fontsize=8)
        ax_jxn.set_title("C   Top cryptic junctions\n(positions rel. to const. 5'SS)",
                          fontsize=9, loc="left")
        ax_jxn.tick_params(axis="y", labelsize=7)
    else:
        ax_jxn.text(0.5, 0.5, "No junction data", ha="center", va="center",
                    transform=ax_jxn.transAxes, color="gray")
        ax_jxn.set_title("C   Top cryptic junctions", fontsize=9, loc="left")

    # ------------------------------------------------------------------
    # Panel D: Mean Δcryptic by cell line (bar chart)
    # ------------------------------------------------------------------
    cl_means = []
    cl_sems  = []
    for cl in CELL_LINES:
        vals = grp[grp["cell_line"] == cl]["delta_cryptic_fraction"].dropna()
        cl_means.append(vals.mean() if len(vals) > 0 else np.nan)
        cl_sems.append(vals.sem()  if len(vals) > 1 else 0)

    x = np.arange(len(CELL_LINES))
    bars = ax_cl.bar(x, cl_means, color=[CL_COLORS[c] for c in CELL_LINES],
                     edgecolor="white", linewidth=0.5, width=0.6)
    ax_cl.errorbar(x, cl_means, yerr=cl_sems, fmt="none",
                   color="black", capsize=3, lw=1.0)
    ax_cl.axhline(0, color="black", lw=0.7)
    ax_cl.set_xticks(x)
    ax_cl.set_xticklabels(CELL_LINES, fontsize=8)
    ax_cl.set_ylabel("Mean Δcryptic fraction (± SEM)", fontsize=8)
    ax_cl.set_title("D   Mean Δcryptic by cell line\n(all variants)", fontsize=9, loc="left")

    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=== Outlier Exon Family Analysis ===\n")
    var, wide, canon = load_data()

    print("\nIdentifying outlier genes...")
    summary = identify_outlier_genes(var)
    out_summary = SCRIPT_DIR / "outlier_gene_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"  {len(summary)} genes pass thresholds")
    top_genes = summary.head(TOP_N_GENES)["gene_name"].tolist()
    print(f"  Top {TOP_N_GENES}: {top_genes}")

    out_pdf = SCRIPT_DIR / "outlier_exon_families.pdf"
    from matplotlib.backends.backend_pdf import PdfPages

    print(f"\nGenerating PDF ({TOP_N_GENES + 1} pages)...")
    with PdfPages(out_pdf) as pdf:
        # Page 0: summary heatmap
        fig0 = figure_summary_heatmap(summary, var, top_n=TOP_N_GENES)
        pdf.savefig(fig0, bbox_inches="tight", dpi=150)
        fig0.savefig(SCRIPT_DIR / "outlier_fig0_summary_heatmap.png",
                     bbox_inches="tight", dpi=150)
        plt.close(fig0)
        print("  Page 1: summary heatmap")

        # One page per gene
        for i, gene in enumerate(top_genes, 1):
            print(f"  Page {i+1}: {gene}")
            fig = figure_per_gene(gene, var, wide, canon)
            pdf.savefig(fig, bbox_inches="tight", dpi=150)
            safe = gene.replace("/", "_")
            fig.savefig(SCRIPT_DIR / f"outlier_fig{i}_{safe}.png",
                        bbox_inches="tight", dpi=150)
            plt.close(fig)

    print(f"\nDone → {out_pdf}")


if __name__ == "__main__":
    main()
