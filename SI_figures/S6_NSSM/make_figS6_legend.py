#!/usr/bin/env python3
"""Shared key for Figure S6 (NSSM maps): ClinVar classes, wild-type base, and the
delta-logit colorbar. Written as separate files to place in the blank right column
left by assemble_figS6_MAY.py.

Colors come from the panel plotting script itself, and the colormap/range match
Figure_4/make_scn5a_ssm.py + make_heatmap (coolwarm, -6..6), so the key cannot
drift from the panels.
"""
import importlib.util, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/ESL/ESL_MPRA"
PANEL_SCRIPT = f"{REPO}/Figure_4/ssm_single_muts_logits_06_04_2025_with_clinvar_patched.py"
OUTDIR = f"{REPO}/SI_figures/SpliceAI_NSSM/assembled"
CMAP, VMIN, VMAX = "coolwarm", -6, 6

spec = importlib.util.spec_from_file_location("ssm_mod", PANEL_SCRIPT)
ssm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssm)          # top level is imports + defs only

os.makedirs(OUTDIR, exist_ok=True)

# --- ClinVar + WT legend ---
fig = plt.figure(figsize=(1.6, 1.5))
handles = [plt.Line2D([0], [0], marker="o", linestyle="None", color="w",
                      markerfacecolor=v, markeredgecolor=ssm.darken_color(v),
                      markeredgewidth=0.4, markersize=5, alpha=0.6,
                      label=k.replace("_", " "))
           for k, v in ssm.CLINVAR_COLORS.items()]
handles.append(plt.Line2D([0], [0], marker="s", linestyle="None", color="w",
                          markerfacecolor="black", markersize=5, label="Wild-type base"))
fig.legend(handles=handles, loc="center", title="ClinVar", title_fontsize=7,
           fontsize=6, frameon=False, handletextpad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUTDIR}/figS6_legend_clinvar.{ext}", dpi=300, transparent=True)
plt.close(fig)

# --- colorbar ---
fig = plt.figure(figsize=(0.8, 2.0))
cax = fig.add_axes([0.15, 0.08, 0.18, 0.84])
cmap = (matplotlib.colormaps[CMAP] if hasattr(matplotlib, "colormaps")
        else plt.cm.get_cmap(CMAP))
sm = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(VMIN, VMAX), cmap=cmap)
cb = fig.colorbar(sm, cax=cax)
cb.set_label("Δlogit(PSI)", fontsize=7, labelpad=3)
cb.ax.tick_params(labelsize=6, width=0.5)
cb.outline.set_linewidth(0.5)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUTDIR}/figS6_colorbar.{ext}", dpi=300, transparent=True)
plt.close(fig)
print("wrote", OUTDIR + "/figS6_legend_clinvar.{pdf,png}", "and figS6_colorbar.{pdf,png}")
