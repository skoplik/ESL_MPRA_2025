import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl

# === Fonts: keep text editable in Illustrator
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# === Config
supertable_file = "/ESL/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WITH_WT.csv"
exclusive_file = "/ESL/Figures_SK/sig_cell_types/out/exclusive_heatmaps/exclusive_psi_outliers_combined.tsv"
output_dir = "/ESL/Figures_SK/sig_cell_types/out/family_specificity_examples"
os.makedirs(output_dir, exist_ok=True)

cell_to_logit = {
    "HEK": "HEK_pooled_logit",
    "HeLa": "HeLa_pooled_logit",
    "K562": "K562_pooled_logit",
    "MCF7": "MCF7_pooled_logit",
    "HMC3": "HMC3_pooled_logit"
}
cell_to_psi = {
    "HEK": "HEK_pooled_psi_raw",
    "HeLa": "HeLa_pooled_psi_raw",
    "K562": "K562_pooled_psi_raw",
    "MCF7": "MCF7_pooled_psi_raw",
    "HMC3": "HMC3_pooled_psi_raw"
}
cell_to_dlogit = {
    "HEK": "HEK_delta_logit_pooled",
    "HeLa": "HeLa_delta_logit_pooled",
    "K562": "K562_delta_logit_pooled",
    "MCF7": "MCF7_delta_logit_pooled",
    "HMC3": "HMC3_delta_logit_pooled"
}

cell_lines = list(cell_to_logit.keys())
plot_labels = {"HEK": "HEK293", "HeLa": "HeLa", "K562": "K562", "MCF7": "MCF7", "HMC3": "HMC3"}
custom_rgb_colors = {
    'HEK293': '#E984B6',
    'HeLa': '#7FBE7E',
    'MCF7': '#807CB9',
    'HMC3': '#EF4025',
    'K562': '#F9AE33'
}

# === Special highlight sites
highlight_sites = {
    "MEF2C exon 8": 24,
    "DMD exon 71": 78
}
target_exons = set(highlight_sites.keys())

# === Load exclusive file and build set of allowed References (1-indexed)
exclusive = pd.read_csv(exclusive_file, sep="\t")
exclusive_refs = set(exclusive["Reference"].astype(str).unique())
print(f"[INFO] Loaded {len(exclusive_refs)} unique 1-indexed References from exclusive file")

# === Load supertable
df = pd.read_csv(supertable_file)
df["Reference"] = df["Reference"].astype(str)

# Require complete data in all 5 cell lines
required_cols = list(cell_to_logit.values()) + list(cell_to_psi.values()) + list(cell_to_dlogit.values())
mask_complete = df[required_cols].notnull().all(axis=1)

# Keep WTs and fully complete variants
mask_variants = (df["snp"] != "none") & mask_complete
mask_wt = (df["snp"] == "none")
df = df[mask_variants | mask_wt]

# Filter to just MEF2C exon 8 and DMD exon 71
df = df[df["gene_exon"].isin(target_exons)]
print(f"[INFO] Retained {len(df)} rows with full values for MEF2C exon 8 and DMD exon 71")

def snp_at_site(snp, site):
    """Check if any mutation in semicolon-separated list has position == site"""
    if pd.isna(snp) or snp == "none":
        return False
    for mut in str(snp).split(";"):
        mut = mut.strip()
        if not mut:
            continue
        try:
            pos = int(mut.split(":")[0])
            if pos == site:
                return True
        except Exception:
            continue
    return False

highlighted_variants = []

def plot_family(sub, eid, gene_exon, ydict, ylabel, suffix):
    highlight_site = highlight_sites.get(gene_exon, None)
    rows = sub[sub["snp"] != "none"].copy()
    if rows.empty:
        return

    vals, labs, snps, refs, eids, gexons = [], [], [], [], [], []
    for cl in cell_lines:
        cl_vals = rows[ydict[cl]].dropna().values
        cl_snps = rows["snp"].dropna().values
        cl_refs = rows["Reference"].dropna().values
        vals.extend(cl_vals)
        labs.extend([plot_labels[cl]] * len(cl_vals))
        snps.extend(cl_snps)
        refs.extend(cl_refs)
        eids.extend([eid] * len(cl_vals))
        gexons.extend([gene_exon] * len(cl_vals))

    plot_df = pd.DataFrame({
        "cell_line": labs,
        "value": vals,
        "snp": snps,
        "Reference": refs,
        "event_id": eids,
        "gene_exon": gexons
    })

    plt.figure(figsize=(3,4))
    # 1. Draw variant points (behind everything)
    ax = sns.stripplot(
        data=plot_df,
        x="cell_line", y="value",
        alpha=0.8, palette=custom_rgb_colors,
        size=3, jitter=True, zorder=1
    )
    # 2. Draw boxplot
    sns.boxplot(
        data=plot_df,
        x="cell_line", y="value",
        showcaps=True, width=0.4,
        boxprops={'facecolor':'none', 'edgecolor':'black'},
        whiskerprops={'color':'black'}, medianprops={'color':'black'},
        showfliers=False, zorder=5, ax=ax
    )
    # Force box patches on top of points
    for patch in ax.artists:
        patch.set_zorder(6)
        patch.set_edgecolor("black")
        patch.set_facecolor("none")

    # 3. WT baselines on top
    if not suffix.startswith("dlogit"):
        wt_row = sub[sub["snp"] == "none"]
        if not wt_row.empty:
            for i, cl in enumerate(cell_lines):
                col = ydict[cl]
                if col in wt_row.columns:
                    wt_val = wt_row[col].values[0]
                    ax.scatter(i, wt_val, color="black", s=14, zorder=10)

    # 4. Highlight variants on very top
    highlight_refs = {
        r for r, s in zip(rows["Reference"], rows["snp"])
        if snp_at_site(s, highlight_site) and (str(r) in exclusive_refs)
    }
    num_highlight = len(highlight_refs)
    print(f"[INFO] {gene_exon} - {ylabel}: {num_highlight} highlight References at site {highlight_site}")

    if num_highlight > 0:
        highlight_rows = plot_df[plot_df["Reference"].isin(highlight_refs)]
        highlighted_variants.extend(
            highlight_rows[["gene_exon", "Reference", "snp", "event_id"]]
            .drop_duplicates()
            .to_dict("records")
        )
        sns.stripplot(
            data=highlight_rows,
            x="cell_line", y="value",
            palette=custom_rgb_colors, size=3, jitter=True,
            edgecolor="black", linewidth=1.0, facecolor="none",
            zorder=10, ax=ax
        )

    ax.set_title(f"{gene_exon} ({eid}) - {ylabel}")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Cell line")
    plt.tight_layout()

    out_pdf = os.path.join(output_dir, f"{gene_exon.replace(' ', '_')}_{suffix}.pdf")
    out_svg = out_pdf.replace(".pdf", ".svg")
    plt.savefig(out_pdf, dpi=300)
    plt.savefig(out_svg, dpi=300)
    plt.close()
    print(f"[PLOT] Saved {out_pdf} and {out_svg}")

# === Generate plots for MEF2C exon 8 and DMD exon 71 only
for eid, sub in df.groupby("event_id"):
    gene_exon = sub["gene_exon"].iloc[0]
    if gene_exon not in target_exons:
        continue
    plot_family(sub, eid, gene_exon, cell_to_logit, "Logit(PSI)", "logit")
    plot_family(sub, eid, gene_exon, cell_to_psi, "PSI", "psi")
    plot_family(sub, eid, gene_exon, cell_to_dlogit, "ΔLogit(PSI)", "dlogit")

# === Save highlighted variants to TSV
if highlighted_variants:
    out_tsv = os.path.join(output_dir, "highlighted_variants.tsv")
    pd.DataFrame(highlighted_variants).drop_duplicates().to_csv(out_tsv, sep="\t", index=False)
    print(f"[INFO] Exported highlighted variants to {out_tsv}")
