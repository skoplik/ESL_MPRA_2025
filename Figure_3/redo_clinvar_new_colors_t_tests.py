import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import sys
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.stats import mannwhitneyu
import os
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

# === ClinVar color palette 
CLINVAR_COLORS = {
    "Pathogenic": "red",
    "Likely_pathogenic": "orange",
    "Uncertain_significance": "slateblue",
    "Likely_benign": "#9ACD32",
    "Benign": "#258d4c",
    "Conflicting": "orchid",
}

CLINVAR_LEGEND_ORDER = [
    "Benign",
    "Likely_benign",
    "Pathogenic",
    "Likely_pathogenic",
    "Conflicting",
    "Uncertain_significance"
]

CLINVAR_DISPLAY_NAMES = {
    "Benign": "Benign (B)",
    "Likely_benign": "Likely Benign (LB)",
    "Pathogenic": "Pathogenic (P)",
    "Likely_pathogenic": "Likely Pathogenic (LP)",
    "Uncertain_significance": "Uncertain Significance (VUS)",
    "Conflicting": "Conflicting (CL)"
}


ANNOTATE_REFERENCES = {
    33699: "1",
    45357: "2",
    60690: "3",
    29887: "4",
    40168: "5",
    47317: "6"
}

def parse_clinvar_significance(clinvar_item):
    sigs = set()
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
        else:
            normalized.add(s.replace(" ", "_"))
    if not normalized:
        return None
    elif len(normalized) > 1:
        return "Conflicting"
    else:
        return list(normalized)[0]

def run_mannwhitney_tests(df):
    print("\n=== Mann–Whitney U Tests ===")
    groups = {
        "B/LB": df[df["CLNSIG_category"].isin(["Benign", "Likely_benign"])]["mean_delta_logit"],
        "P/LP": df[df["CLNSIG_category"].isin(["Pathogenic", "Likely_pathogenic"])]["mean_delta_logit"],
        "CL/VUS": df[df["CLNSIG_category"].isin(["Uncertain_significance", "Conflicting"])]["mean_delta_logit"]
    }

    tests = [
        ("B/LB", "P/LP"),
        ("B/LB", "CL/VUS")
    ]

    pvals = {}
    for g1, g2 in tests:
        u_stat, p_val = mannwhitneyu(groups[g1], groups[g2], alternative="two-sided")
        pvals[(g1, g2)] = p_val
        print(f"{g1} vs {g2}: p = {p_val:.4e} (U = {u_stat})")
    return pvals, groups

def print_extremes_by_group(df_plot, output_prefix):
    print("\n=== Top and Bottom 15 Variants per ClinVar Category ===")
    all_extremes = []
    for category in CLINVAR_LEGEND_ORDER:
        group_df = df_plot[df_plot["CLNSIG_category"] == category]
        if group_df.empty:
            continue
        print(f"\n>>> {category} ({len(group_df)} variants)")
        top = group_df.sort_values("mean_delta_logit", ascending=False).head(15)
        bottom = group_df.sort_values("mean_delta_logit", ascending=True).head(15)

        print("\nTop 15 by mean_delta_logit:")
        print(top[["Reference", "gene_exon", "event_id", "mean_delta_logit"]].to_string(index=False))

        print("\nBottom 15 by mean_delta_logit:")
        print(bottom[["Reference", "gene_exon", "event_id", "mean_delta_logit"]].to_string(index=False))

        top["rank"] = "top"
        bottom["rank"] = "bottom"
        top["CLNSIG_group"] = category
        bottom["CLNSIG_group"] = category
        all_extremes.append(top)
        all_extremes.append(bottom)

    if all_extremes:
        out_df = pd.concat(all_extremes, axis=0)
        out_path = output_prefix + "_top_bottom_variants.tsv"
        out_df[["Reference", "gene_exon", "event_id", "CLNSIG_group", "rank", "mean_delta_logit"]].to_csv(out_path, sep="\t", index=False)
        print(f"\nExported top/bottom variant list to: {out_path}")

def main(input_csv, clinvar_csv, output_plot_pdf):
    df = pd.read_csv(input_csv)
    df_clinvar = pd.read_csv(clinvar_csv, sep="\t")
    df_clinvar["ClinVar_item"] = df_clinvar["ClinVar_item"].apply(ast.literal_eval)
    df_merged = df.merge(df_clinvar, left_on="Reference", right_on="mut_ref", how="left")
    df_merged = df_merged.drop_duplicates(subset="full_seq").copy()

    df_merged["CLNSIG_category"] = df_merged["ClinVar_item"].apply(parse_clinvar_significance)
    delta_cols = [c for c in df_merged.columns if c.endswith("delta_logit_pooled")]
    for c in delta_cols:
        df_merged[c] = pd.to_numeric(df_merged[c], errors="coerce")
    df_merged["mean_delta_logit"] = df_merged[delta_cols].mean(axis=1)

    df_plot = df_merged[
        df_merged["CLNSIG_category"].isin(CLINVAR_COLORS.keys()) &
        df_merged["mean_delta_logit"].notna()
    ].copy()

    df_plot["CLNSIG_bin"] = df_plot["CLNSIG_category"].map(lambda sig: (
        "B/LB" if sig in {"Benign", "Likely_benign"} else
        "P/LP" if sig in {"Pathogenic", "Likely_pathogenic"} else
        "CL/VUS"
    ))

    bin_order = ["B/LB", "P/LP", "CL/VUS"]
    df_plot["CLNSIG_category"] = pd.Categorical(df_plot["CLNSIG_category"], categories=CLINVAR_COLORS.keys(), ordered=True)
    pvals, groups = run_mannwhitney_tests(df_plot)

    ANNOTATE_REFERENCES = {
        33699: "1",
        45357: "2",
        60690: "3",
        29887: "4",
        40168: "5",
        47317: "6"
    }

    with PdfPages(output_plot_pdf) as pdf:
        plt.figure(figsize=(6.5, 4.5))
        ax = plt.gca()

        bin_yvals = {"B/LB": 0, "P/LP": 1, "CL/VUS": 2}
        for i, row in df_plot.iterrows():
            y = bin_yvals[row["CLNSIG_bin"]] + np.random.uniform(-0.2, 0.2)
            color = CLINVAR_COLORS[row["CLNSIG_category"]]
            ax.scatter(
                row["mean_delta_logit"], y,
                facecolor=to_rgba(color, alpha=0.6),
                edgecolor="none",
                linewidth=0,
                s=12,
                zorder=1
            )

            ref_id = row["Reference"]
            if ref_id in ANNOTATE_REFERENCES:
                ax.text(
                    row["mean_delta_logit"], y + 0.07,
                    ANNOTATE_REFERENCES[ref_id],
                    fontsize=7, ha="center", va="bottom", fontweight="bold", zorder=10
                )

        sns.boxplot(
            data=df_plot,
            y="CLNSIG_bin",
            x="mean_delta_logit",
            order=bin_order,
            showcaps=True,
            width=0.5,
            boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1.2, 'zorder': 5},
            whiskerprops={'linewidth': 2, 'zorder': 5},
            medianprops={'color': 'black', 'zorder': 5},
            showfliers=False,
            ax=ax,
            zorder=5
        )

        ax.set_xlim(df_plot["mean_delta_logit"].min() - 0.1, df_plot["mean_delta_logit"].max() + 0.1)

        # Put annotations under y-tick labels using ax.transAxes
        category_count = len(bin_order)
        spacing = 1.0 / category_count
        vertical_offset_n = -0.07
        vertical_offset_p = -0.15

        for bin_label in bin_order:
            y_idx = bin_yvals[bin_label]
            y_coord = (category_count - y_idx - 1) * spacing + spacing / 2

            n = len(groups[bin_label])
            p_val = (
                pvals[("B/LB", "P/LP")] if bin_label == "P/LP" else
                pvals[("B/LB", "CL/VUS")] if bin_label == "CL/VUS" else
                None
            )

            ax.text(
                -0.1, y_coord + vertical_offset_n,
                f"n = {n:,}",
                transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5
            )

            if p_val is not None:
                ax.text(
                    -0.1, y_coord + vertical_offset_p,
                    f"p = {p_val:.1e}",
                    transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.5
                )



        plt.axvline(0, color='black', linestyle='--', linewidth=0.5, zorder=0)
        plt.ylabel("ClinVar Category")
        plt.xlabel("Mean Δlogit(PSI)")
        plt.title("Δlogit(PSI) by ClinVar Group")

        y_offsets = {"B/LB": 0, "P/LP": 1, "CL/VUS": 2}
        # plt.text(
        #     x=df_plot["mean_delta_logit"].max() * 0.95,
        #     y=y_offsets["P/LP"] - 0.3,
        #     s=f"p = {pvals[('B/LB', 'P/LP')]:.2e}",
        #     va="center", ha="right", fontsize=8, zorder=6
        # )
        # plt.text(
        #     x=df_plot["mean_delta_logit"].max() * 0.95,
        #     y=y_offsets["CL/VUS"] - 0.3,
        #     s=f"p = {pvals[('B/LB', 'CL/VUS')]:.2e}",
        #     va="center", ha="right", fontsize=8, zorder=6
        # )

        legend_elements = [
            Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor=to_rgba(CLINVAR_COLORS[label], alpha=0.8),
                   markeredgecolor=to_rgba(CLINVAR_COLORS[label], alpha=1),
                   markersize=6,
                   label=CLINVAR_DISPLAY_NAMES[label])
            for label in CLINVAR_LEGEND_ORDER
        ]
        plt.legend(handles=legend_elements, title="ClinVar", loc="best", fontsize=6, title_fontsize=7)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    output_csv = os.path.splitext(output_plot_pdf)[0] + ".csv"
    df_export = df_plot[["Reference", "event_id", "full_seq", "gene_exon", "CLNSIG_category", "CLNSIG_bin", "mean_delta_logit"]].copy()
    df_export.to_csv(output_csv, index=False)
    print(f"Exported: {output_csv}")

    print_extremes_by_group(df_plot, os.path.splitext(output_plot_pdf)[0])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python clinvar_annots.py <input_csv> <clinvar_csv> <output_plot.pdf>")
        sys.exit(1)
    input_csv = sys.argv[1]
    clinvar_csv = sys.argv[2]
    output_plot = sys.argv[3]
    main(input_csv, clinvar_csv, output_plot)
