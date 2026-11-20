#!/usr/bin/env python3
"""
Figure 6 D/E panel elements, and the Data S2/S3 all-cluster versions.

Layout follows the V4 manuscript panel:
    [RBP names + motif logo] [square-celled per-cell heatmap] [bars + gene/exon]
with a horizontal colourbar underneath.

Run:
    python3 make_fig6_panel_elements.py            # 6D / 6E (manuscript clusters)
    python3 make_fig6_panel_elements.py --all      # Data S2 / S3 (every cluster)
"""
import os, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42      # editable vector text in Illustrator
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import logomaker

EFFECTS  = "/ESL/ESL_MPRA/Figure_6/outputs/effect_size_MAY_v2/out_MAY_v2_boots_0_%s.csv"
TRANSFAC = "/ESL/ESL_MPRA/Figure_6/inputs/motif_clusters/Root_motifs.tf"
CLUSTAB  = "/ESL/ESL_MPRA/Figure_6/inputs/motif_clusters/clusters.tab"
SUPER    = "/ESL/ESL_MPRA/Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25_strandfix.csv"
OUT      = "/ESL/ESL_MPRA/Figure_6/outputs/figure6_elements_MAY"

# cell order and labels exactly as the manuscript panel
CELLS  = ["HEK", "HeLa", "K562", "HMC3", "MCF7"]
LABELS = ["HEK293", "HeLa", "K562", "HMC3", "MCF7"]

CLUSTERS = {
    "exon":    ["cluster_050", "cluster_054", "cluster_023", "cluster_061"],
    "intron1": ["cluster_026", "cluster_143", "cluster_075", "cluster_118"],
}
PANEL = {"exon": "6D", "intron1": "6E"}
REGION_TITLE = {"exon": "Exon", "intron1": "5' Intron", "intron2": "3' Intron"}

CELL_IN = 0.17          # one heatmap cell, inches -> square cells
LOGO_H  = 0.42          # logo height, inches (uniform across blocks)
RBP_H   = 0.34          # space above each logo for the RBP list
BAR_W   = 2.6           # bar column width, inches
LOGO_W  = 1.9
LABEL_W = 1.25          # room for "TNNT2 exon 5" to the right of the bars


def parse_transfac(path):
    out, name, rows = {}, None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("ID"):
            if name and rows: out[name] = rows
            name, rows = line.split(None, 1)[1].strip(), []
        elif re.match(r"^\d+\s", line):
            p = line.split()
            try: rows.append([float(v) for v in p[1:5]])
            except ValueError: pass
        elif line.startswith("//"):
            if name and rows: out[name] = rows
            name, rows = None, []
    if name and rows: out[name] = rows
    pwm = {}
    for k, v in out.items():
        a = np.array(v, float); s = a.sum(axis=1, keepdims=True); s[s == 0] = 1
        pwm[k] = pd.DataFrame(a / s, columns=["A", "C", "G", "U"])
    return pwm


def cluster_rbps(path):
    m = {}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3: continue
        names = []
        for entry in p[-1].split(","):
            try:
                g = re.match(r"[A-Za-z0-9]+", entry.split(":", 1)[1])
                if g: names.append(g.group(0).upper())
            except Exception:
                pass
        m[p[0].strip()] = (sorted(set(names)), len(p[-1].split(",")))
    return m


def load(region):
    df = pd.read_csv(EFFECTS % region)
    gx = {}
    if os.path.exists(SUPER):
        st = pd.read_csv(SUPER, dtype=str, usecols=["event_id", "gene_exon"])
        gx = dict(zip(st.event_id, st.gene_exon))
    df["gene_exon"] = df["event_id"].map(gx).fillna(df["event_id"])
    return df


def wrap(names, per_line=4):
    return "\n".join(", ".join(names[i:i + per_line]) for i in range(0, len(names), per_line))


def build(region, ids, outfile, title):
    df = load(region)
    df = df[df["motif"].isin(ids)].copy()
    if df.empty:
        print("  no rows for", region); return

    mcols = [f"Mean {c} Average" for c in CELLS]
    df["_mean"] = df[mcols].mean(axis=1)
    df["_sem"]  = df[mcols].sem(axis=1)          # SEM across cell lines

    order = df.groupby("motif")["_mean"].mean().sort_values(ascending=False).index.tolist()
    blocks = []
    for c in order:
        sub = df[df.motif == c].sort_values("_mean", ascending=False)
        blocks.append((c, sub))

    pwm  = parse_transfac(TRANSFAC)
    rbps = cluster_rbps(CLUSTAB) if os.path.exists(CLUSTAB) else {}

    heights = [len(s) * CELL_IN + LOGO_H + RBP_H for _, s in blocks]
    fig_h = sum(heights) + 1.5
    fig_w = LOGO_W + len(CELLS) * CELL_IN + BAR_W + LABEL_W + 0.9
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(len(blocks), 3, figure=fig, height_ratios=heights,
                  width_ratios=[LOGO_W, len(CELLS) * CELL_IN, BAR_W + LABEL_W],
                  hspace=0.45, wspace=0.05)

    allv = df[mcols].to_numpy(float)
    v = np.nanmax(np.abs(allv)) if np.isfinite(allv).any() else 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)
    cmap = plt.get_cmap("coolwarm")

    for bi, (cl, sub) in enumerate(blocks):
        n = len(sub)
        mat = sub[mcols].to_numpy(float)

        # --- column 1: RBP list, then a uniform-height logo below it
        slot = fig.add_subplot(gs[bi, 0]); slot.axis("off")
        bb = slot.get_position()
        names, nmot = rbps.get(cl, ([], 0))
        head = "Number of Motifs in cluster %s: %d" % (cl.replace("cluster_", ""), nmot)
        slot.text(0, 1.0, head + ("\n" + wrap(names) if names else ""),
                  transform=slot.transAxes, ha="left", va="top", fontsize=6.5)
        lh = min(LOGO_H / fig_h, bb.height * 0.5)
        axl = fig.add_axes([bb.x0, bb.y0 + bb.height * 0.10, bb.width * 0.92, lh])
        if cl in pwm:
            logomaker.Logo(pwm[cl], ax=axl, color_scheme="classic")
        axl.set_xticks([]); axl.set_yticks([])
        for sp in axl.spines.values(): sp.set_visible(False)

        # --- column 2: heatmap with square cells
        axh = fig.add_subplot(gs[bi, 1])
        axh.imshow(mat, aspect="equal", cmap=cmap, norm=norm, interpolation="nearest")
        for i in range(n):
            for j in range(len(CELLS)):
                if not np.isfinite(mat[i, j]):
                    axh.text(j, i, "X", ha="center", va="center", fontsize=8, fontweight="bold")
        axh.set_yticks([])
        if bi == len(blocks) - 1:
            axh.set_xticks(range(len(CELLS)))
            axh.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=7)
        else:
            axh.set_xticks([])
        for sp in axh.spines.values():
            sp.set_linewidth(0.6)

        # --- column 3: bars (mean +/- SEM) with gene/exon to the right
        axb = fig.add_subplot(gs[bi, 2])
        y = np.arange(n)
        axb.barh(y, sub["_mean"], xerr=sub["_sem"], color=[cmap(norm(x)) for x in sub["_mean"]],
                 edgecolor="black", linewidth=0.5, capsize=2, error_kw={"lw": 0.7})
        axb.set_ylim(n - 0.5, -0.5)
        axb.set_xlim(-v - 0.35, v + 0.35)
        axb.axvline(0, color="black", lw=0.6)
        axb.set_yticks([])
        axb.tick_params(labelsize=7)
        for i, (_, r) in enumerate(sub.iterrows()):
            axb.text(v + 0.45, i, str(r["gene_exon"]), va="center", ha="left", fontsize=6.5)
        if bi == len(blocks) - 1:
            axb.set_xlabel("Mean Effect Size $\\pm$ SEM (All Cell Lines)", fontsize=8)
        else:
            axb.set_xticks([])
        for sp in axb.spines.values(): sp.set_visible(False)

    cax = fig.add_axes([0.36, 0.035, 0.30, 0.012])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cax.set_xlabel("$\\Delta$logit Effect Size", fontsize=8)
    cax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=11, y=0.995)

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print("  %-46s %3d rows / %2d clusters" % (os.path.basename(outfile), len(df), len(blocks)))


if __name__ == "__main__":
    if "--all" in sys.argv:
        for region, ds in (("exon", "S2"), ("intron1", "S3")):
            df = load(region)
            ids = df.groupby("motif")[[f"Mean {c} Average" for c in CELLS]].mean().mean(axis=1)\
                    .sort_values(ascending=False).index.tolist()
            build(region, ids, f"{OUT}/Data_{ds}_{region}_all_clusters.pdf",
                  "Motif Effects Across %s Contexts (Mean $\\pm$ SEM, All Cell Lines)" % REGION_TITLE[region])
    else:
        for region in ("exon", "intron1"):
            build(region, CLUSTERS[region], f"{OUT}/fig{PANEL[region]}_{region}_elements.pdf",
                  "Effect Sizes (All Cell Lines) - %s" % REGION_TITLE[region])
