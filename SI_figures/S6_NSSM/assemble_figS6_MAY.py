#!/usr/bin/env python3
"""Assemble Figure S6 (NSSM maps of disease-relevant exons) from the May 2026 panels.

Layout matches the V4 figure: one column, panel A = ACMG exons (9),
panel B = the remaining exons (3), on one 8.5 x 11 in page. Each panel is the
all-cell-line mean delta-logit map written by Figure_4/make_scn5a_ssm.py into
Figure_4/outputs/nssm_S6_MAY/ALL (see run_figS6_panels.sh for the per-exon calls).

Each panel is cropped to the heatmap itself: the per-panel ClinVar legend (left)
and colorbar (right) are cut off, and a blank column of RIGHT_RESERVE is left on
the right of the page for a single shared legend + colorbar added by hand.
The crop box is found from the panel's own text (legend labels, colorbar ticks,
title, axis ticks), so it follows the panel if the plotting code changes.

Panels are embedded as vector Form XObjects with PyMuPDF (pip install pymupdf).
PyPDF2's merge_page is NOT safe here: every matplotlib panel uses the same
resource names, and merged panels blanked each other out - this is why the
earlier assembled S6 PDFs rendered mostly empty.

Each panel's auto title ("<gene_exon> <event_id> delta_logit (ALL)") is covered
with white and replaced by the plain gene/exon name.
"""
import argparse, os, re
import fitz  # PyMuPDF
import matplotlib

# Use the panels' own font (matplotlib default DejaVu Sans) for added text.
FONT_FILE = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
FONT = fitz.Font(fontfile=FONT_FILE)

REPO = "/ESL/ESL_MPRA"
SRC = f"{REPO}/Figure_4/outputs/nssm_S6_MAY/ALL"
OUT = f"{REPO}/SI_figures/SpliceAI_NSSM/assembled/figS6_nssm_maps_MAY.pdf"

PANELS = {
    "A": ["LMNA_exon_10", "MUTYH_exon_6", "MYH11_exon_42", "MYH11_exon_6",
          "RYR1_exon_94", "SCN5A_exon_24", "TNNT2_exon_4", "TNNT2_exon_5",
          "TSC2_exon_32"],
    "B": ["BEST1_exon_8", "HMBS_exon_2", "SPINT2_exon_3"],
}

PAGE_W, PAGE_H = 8.5 * 72, 11.0 * 72
MX, MTOP, MBOT, LABEL_W = 0.4 * 72, 0.35 * 72, 0.35 * 72, 0.3 * 72
RIGHT_RESERVE = 1.6 * 72    # blank column for the hand-placed legend + colorbar
VGAP, GROUP_GAP = 3.0, 12.0
TITLE_KEY = "delta_logit (ALL)"
LEGEND_KEYS = ("Pathogenic", "benign", "Benign", "significance", "Conflicting")


def find(src, stem):
    hits = [f for f in sorted(os.listdir(src))
            if f.startswith(stem + "_chr") and f.endswith("_ALL_delta_logit.pdf")]
    if len(hits) != 1:   # fail loudly: one gene_exon label can span two loci
        raise SystemExit(f"{stem}: expected exactly 1 panel in {src}, found {hits}")
    return os.path.join(src, hits[0])


def crop_box(page, stem):
    """Heatmap-only crop: right of the legend, left of the colorbar,
    from the title down to the x-axis tick labels."""
    words = page.get_text("words")          # x0, y0, x1, y1, text, ...
    blocks = page.get_text("blocks")
    title = [b for b in blocks if TITLE_KEY in b[4]]
    legend = [w for w in words if any(k in w[4] for k in LEGEND_KEYS)]
    if len(title) != 1 or not legend:
        raise SystemExit(f"{stem}: cannot locate title/legend for cropping")
    left = max(w[2] for w in legend) + 2
    # x-axis ticks: the row of integers 0..160 under the heatmap
    xt = [w for w in words if w[0] > left and re.fullmatch(r"\d+", w[4])]
    if not any(w[4] == "160" for w in xt):
        raise SystemExit(f"{stem}: cannot locate x-axis ticks")
    right = max(w[2] for w in xt if w[4] == "160") + 2
    # colorbar ticks sit right of the axis; make sure the crop stops short of them
    cb = [w for w in words if w[0] > right - 2 and re.fullmatch(r"[−-]?\d+", w[4])]
    if cb:
        right = min(right, min(w[0] for w in cb) - 1)
    top = title[0][1] - 1
    bottom = max(w[3] for w in xt) + 1.5
    return fitz.Rect(left, top, right, bottom), fitz.Rect(title[0][:4])


def title_size(page):
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for sp in l["spans"]:
                if "delta_logit" in sp["text"]:
                    return sp["size"]
    raise SystemExit("cannot find title font size")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    stems = [s for g in PANELS.values() for s in g]
    files = {s: find(a.src, s) for s in stems}
    docs = {s: fitz.open(f) for s, f in files.items()}
    crops = {s: crop_box(docs[s][0], s) for s in stems}

    cw = max(c.width for c, _ in crops.values())
    ch_sum = sum(c.height for c, _ in crops.values())
    gaps = (len(stems) - 1) * VGAP + (len(PANELS) - 1) * GROUP_GAP
    avail_w = PAGE_W - 2 * MX - LABEL_W - RIGHT_RESERVE
    avail_h = PAGE_H - MTOP - MBOT - gaps
    scale = min(avail_w / cw, avail_h / ch_sum)

    out = fitz.open()
    page = out.new_page(width=PAGE_W, height=PAGE_H)   # PyMuPDF: y grows downward
    x0, y = MX + LABEL_W, MTOP
    for gi, (letter, group) in enumerate(PANELS.items()):
        if gi:
            y += GROUP_GAP
        page.insert_text((MX, y + 12), letter, fontsize=16, fontname="djv",
                         fontfile=FONT_FILE)
        for s in group:
            clip, t = crops[s]
            h = clip.height * scale
            page.show_pdf_page(fitz.Rect(x0, y, x0 + clip.width * scale, y + h),
                               docs[s], 0, clip=clip)
            # replace the auto title with the plain exon name
            tx0 = x0 + (max(t.x0, clip.x0) - clip.x0) * scale
            tx1 = x0 + (min(t.x1, clip.x1) - clip.x0) * scale
            ty0 = y + (t.y0 - clip.y0) * scale
            ty1 = y + (t.y1 - clip.y0) * scale
            page.draw_rect(fitz.Rect(tx0 - 1, ty0 - 1, tx1 + 1, ty1 + 1),
                           color=None, fill=(1, 1, 1), overlay=True)
            name = s.replace("_", " ")
            fs = title_size(docs[s][0]) * scale      # same size as the original title
            tw = FONT.text_length(name, fontsize=fs)
            base = ty1 - (t.y1 - t.y0) * 0.22 * scale   # baseline above descender
            page.insert_text(((tx0 + tx1) / 2 - tw / 2, base), name, fontsize=fs,
                             fontname="djv", fontfile=FONT_FILE)
            y += h + VGAP
            print(f"  {letter} {name:16s} crop {clip.width:.0f}x{clip.height:.0f} pt"
                  f" <- {os.path.basename(files[s])}")

    used_right = x0 + cw * scale
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.save(a.out, garbage=3, deflate=True)
    print(f"wrote {a.out}  (8.5 x 11 in, panel scale {scale:.2f}, "
          f"blank right column {(PAGE_W - used_right) / 72:.2f} in)")


if __name__ == "__main__":
    main()
