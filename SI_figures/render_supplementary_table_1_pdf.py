#!/usr/bin/env python3
"""Render Supplementary Table 1 as a paginated 8.5 x 11 PDF.

reportlab on this instance requires Python 3.8+ syntax and cannot be imported
under python3.7, so the table is drawn with matplotlib instead.

117 rows are split across letter pages with the header repeated on each.
"""
import os
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

SRC = "/ESL/ESL_MPRA/SI_figures/supp_tables/Supplementary_Table_1.csv"
OUT = "/ESL/ESL_MPRA/SI_figures/supp_tables/Supplementary_Table_1.pdf"
ROWS_PER_PAGE = 40

COLS = ["gene_exon", "event_id_161_hg38", "exon_start_hg38", "exon_end_hg38",
        "mean_reference_PSI", "min_avg_dlogit", "max_avg_dlogit", "n_variants"]
HEAD = ["Gene / exon", "Event ID (hg38, variable seq.)", "Exon start", "Exon end",
        "Mean ref.\nPSI", "Min avg\n$\\Delta$logit", "Max avg\n$\\Delta$logit", "n\nvariants"]
# fractions of the table width
WIDTHS = [0.155, 0.255, 0.105, 0.105, 0.095, 0.095, 0.095, 0.095]

TITLE = ("Supplementary Table 1 | Summary of reference PSI and "
         "$\\Delta$logit(PSI) ranges for near-saturation exons.")


def main():
    df = pd.read_csv(SRC)[COLS]
    n = len(df)
    pages = range(0, n, ROWS_PER_PAGE)
    npages = len(list(pages))
    print("rows %d -> %d pages" % (n, npages))

    with PdfPages(OUT) as pdf:
        for pi, start in enumerate(range(0, n, ROWS_PER_PAGE)):
            chunk = df.iloc[start:start + ROWS_PER_PAGE]
            fig = plt.figure(figsize=(8.5, 11))

            if pi == 0:
                fig.text(0.055, 0.962, TITLE, fontsize=8.5, fontweight="bold",
                         ha="left", va="top", wrap=True)
                fig.text(0.055, 0.941,
                         "All exons with at least 300 assayed variants (n = %d). "
                         "$\\Delta$logit minima and maxima are taken across all\n"
                         "variants for that exon, averaged over the five cell lines."
                         % n, fontsize=6.8, color="#444444", ha="left", va="top")
                top = 0.915
            else:
                top = 0.955

            ax = fig.add_axes([0.055, 0.045, 0.89, top - 0.045])
            ax.axis("off")

            tbl = ax.table(cellText=[[str(v) for v in r] for r in chunk.values.tolist()],
                           colLabels=HEAD, colWidths=WIDTHS,
                           cellLoc="right", loc="upper center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(6.4)

            ncol = len(HEAD)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_linewidth(0)
                if r == 0:
                    cell.set_text_props(fontweight="bold", fontsize=6.6)
                    cell.set_facecolor("white")
                    cell.visible_edges = "TB"
                    cell.set_linewidth(0.7)
                    cell.set_edgecolor("black")
                else:
                    cell.set_facecolor("#F4F6F8" if r % 2 == 0 else "white")
                    if r == len(chunk):
                        cell.visible_edges = "B"
                        cell.set_linewidth(0.7)
                        cell.set_edgecolor("black")
                if c in (0, 1):                      # text columns left-aligned
                    cell.get_text().set_horizontalalignment("left")
                cell.set_height(0.021)

            fig.text(0.945, 0.028, "Page %d of %d" % (pi + 1, npages),
                     fontsize=6.5, color="#666666", ha="right")
            pdf.savefig(fig)
            plt.close(fig)
            print("  page %d: rows %d-%d" % (pi + 1, start + 1,
                                             min(start + ROWS_PER_PAGE, n)))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
