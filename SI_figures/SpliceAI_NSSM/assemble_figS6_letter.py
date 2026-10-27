#!/usr/bin/env python3
"""Assemble Figure S6 (NSSM maps of disease-relevant exons) onto one US-Letter page.

The previous assembled file (SI_figures/SpliceAI_NSSM/assembled/additional_nssms_REPROCESSED.pdf)
was 9.10 x 17.93 inches - a single column of 12 panels, about twice the height of
a page. This lays the same 12 exons out 2 columns x 6 rows so the figure fits
8.5 x 11 with margins.

Panels are placed as vector content (no rasterisation), scaled uniformly so the
heatmap aspect ratio is preserved.
"""
import os
from PyPDF2 import PageObject, PdfReader, PdfWriter, Transformation

SRC = "/ESL/ESL_MPRA/Figure_4/outputs/ssm_heatmap_sep2026/ALL"
OUT = "/ESL/ESL_MPRA/SI_figures/SpliceAI_NSSM/assembled/figS6_nssm_maps_fitted.pdf"

# Order: ACMG exons named in the main text first, then the remainder.
EXONS = [
    "MUTYH_exon_6", "TNNT2_exon_4",
    "TNNT2_exon_5", "RYR1_exon_94",
    "TSC2_exon_32", "MYH11_exon_6",
    "MYH11_exon_42", "LMNA_exon_10",
    "SCN5A_exon_24", "SPINT2_exon_3",
    "HMBS_exon_2",  "BEST1_exon_8",
]

PAGE_W = 8.5 * 72
PAGE_H_MAX = 11.0 * 72          # hard ceiling: must not exceed one page
FIT_PAGE_TO_CONTENT = True      # shrink the page to the block instead of
                                # centring it on a full page and leaving
                                # ~4 in of dead space top and bottom
MARGIN_X, MARGIN_TOP, MARGIN_BOT = 0.35 * 72, 0.45 * 72, 0.35 * 72
NCOL, NROW = 2, 6
HGAP, VGAP = 0.12 * 72, 0.10 * 72


def find(stem):
    for fn in sorted(os.listdir(SRC)):
        if fn.startswith(stem + "_") and fn.endswith("_ALL_delta_logit.pdf"):
            return os.path.join(SRC, fn)
    raise FileNotFoundError(stem)


def main():
    avail_w = PAGE_W - 2 * MARGIN_X - (NCOL - 1) * HGAP
    cell_w = avail_w / NCOL

    # uniform scale from the first panel; all panels share a size
    r0 = PdfReader(find(EXONS[0])).pages[0]
    src_w = float(r0.mediabox.width)
    src_h = float(r0.mediabox.height)
    scale = cell_w / src_w
    cell_h = src_h * scale

    total_h = NROW * cell_h + (NROW - 1) * VGAP
    avail_h = PAGE_H_MAX - MARGIN_TOP - MARGIN_BOT
    if total_h > avail_h:                     # shrink to fit height if needed
        scale *= avail_h / total_h
        cell_w, cell_h = src_w * scale, src_h * scale
        total_h = NROW * cell_h + (NROW - 1) * VGAP

    if FIT_PAGE_TO_CONTENT:
        page_h = total_h + MARGIN_TOP + MARGIN_BOT
    else:
        page_h = PAGE_H_MAX
    page = PageObject.create_blank_page(width=PAGE_W, height=page_h)
    top = page_h - MARGIN_TOP

    print("panel %.2f x %.2f in | block %.2f x %.2f in | page %.2f x %.2f in"
          % (cell_w / 72, cell_h / 72,
             (NCOL * cell_w + HGAP) / 72, total_h / 72,
             PAGE_W / 72, page_h / 72))

    for i, stem in enumerate(EXONS):
        row, col = divmod(i, NCOL)
        src = PdfReader(find(stem)).pages[0]
        x = MARGIN_X + col * (cell_w + HGAP)
        y = top - (row + 1) * cell_h - row * VGAP
        src.add_transformation(
            Transformation().scale(scale).translate(x, y))
        page.merge_page(src)
        print("  %-16s -> col %d row %d" % (stem, col, row))

    w = PdfWriter()
    w.add_page(page)
    with open(OUT, "wb") as fh:
        w.write(fh)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
