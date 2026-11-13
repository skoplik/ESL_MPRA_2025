#!/usr/bin/env python3
"""Assemble Figure S6 (NSSM maps of disease-relevant exons) from the May 2026 panels.

Layout matches the V4 figure: one column, panel A = ACMG exons (9),
panel B = the remaining exons (3). Each panel is the all-cell-line mean
delta-logit map written by Figure_4/make_scn5a_ssm.py into
Figure_4/outputs/nssm_S6_MAY/ALL (see run_figS6_panels.sh for the per-exon calls).

Panels are embedded as vector Form XObjects with PyMuPDF (pip install pymupdf).
PyPDF2's merge_page was tried first and is NOT safe here: every matplotlib panel
uses the same resource names, and merged panels blanked each other out - this is
why the earlier assembled S6 PDFs rendered mostly empty.

Each panel's auto-generated title ("<gene_exon> <event_id> delta_logit (ALL)") is
located by its text, covered with white, and replaced by the plain gene/exon name.
"""
import argparse, os
import fitz  # PyMuPDF

REPO = "/ESL/ESL_MPRA"
SRC = f"{REPO}/Figure_4/outputs/nssm_S6_MAY/ALL"
OUT = f"{REPO}/SI_figures/SpliceAI_NSSM/assembled/figS6_nssm_maps_MAY.pdf"

PANELS = {
    "A": ["LMNA_exon_10", "MUTYH_exon_6", "MYH11_exon_42", "MYH11_exon_6",
          "RYR1_exon_94", "SCN5A_exon_24", "TNNT2_exon_4", "TNNT2_exon_5",
          "TSC2_exon_32"],
    "B": ["BEST1_exon_8", "HMBS_exon_2", "SPINT2_exon_3"],
}

SCALE = 1.0          # panels at native size (404 x 110 pt each)
TITLE_H = 0.0
MX, MTOP, MBOT, LABEL_W = 18.0, 14.0, 10.0, 22.0
VGAP, GROUP_GAP = 2.0, 14.0


def find(src, stem):
    hits = [f for f in sorted(os.listdir(src))
            if f.startswith(stem + "_chr") and f.endswith("_ALL_delta_logit.pdf")]
    if len(hits) != 1:   # fail loudly: one gene_exon label can span two loci
        raise SystemExit(f"{stem}: expected exactly 1 panel in {src}, found {hits}")
    return os.path.join(src, hits[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    files = {s: find(a.src, s) for g in PANELS.values() for s in g}
    docs = {s: fitz.open(f) for s, f in files.items()}
    rects = {s: d[0].rect for s, d in docs.items()}
    w0 = max(r.width for r in rects.values())

    pw = w0 * SCALE
    heights = [rects[s].height * SCALE + TITLE_H
               for g in PANELS.values() for s in g]
    n = sum(len(g) for g in PANELS.values())
    page_h = MTOP + sum(heights) + (n - 1) * VGAP + (len(PANELS) - 1) * GROUP_GAP + MBOT
    page_w = MX + LABEL_W + pw + MX

    out = fitz.open()
    page = out.new_page(width=page_w, height=page_h)   # PyMuPDF: y grows downward
    y = MTOP
    for gi, (letter, stems) in enumerate(PANELS.items()):
        if gi:
            y += GROUP_GAP
        page.insert_text((MX, y + 14), letter, fontsize=16, fontname="hebo")
        for s in stems:
            r = rects[s]
            clip = r
            hit = [b for b in docs[s][0].get_text("blocks") if "delta_logit (ALL)" in b[4]]
            if len(hit) != 1:
                raise SystemExit(f"{s}: could not locate the auto title, found {len(hit)}")
            title = s.replace("_", " ")
            tw = fitz.get_text_length(title, fontname="helv", fontsize=7)
            x0 = MX + LABEL_W
            y += TITLE_H
            h = clip.height * SCALE
            page.show_pdf_page(fitz.Rect(x0, y, x0 + clip.width * SCALE, y + h),
                               docs[s], 0, clip=clip)
            bx0, by0, bx1, by1 = hit[0][:4]
            page.draw_rect(fitz.Rect(x0 + bx0 * SCALE - 1, y + by0 * SCALE - 1,
                                     x0 + bx1 * SCALE + 1, y + by1 * SCALE + 1),
                           color=None, fill=(1, 1, 1), overlay=True)
            cx = x0 + (bx0 + bx1) / 2 * SCALE
            page.insert_text((cx - tw / 2, y + by1 * SCALE - 1.5), title,
                             fontsize=7, fontname="helv")
            y += h + VGAP
            print(f"  {letter} {title:16s} <- {os.path.basename(files[s])}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.save(a.out, garbage=3, deflate=True)
    print(f"wrote {a.out}  ({page_w / 72:.2f} x {page_h / 72:.2f} in)")


if __name__ == "__main__":
    main()
