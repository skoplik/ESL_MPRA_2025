import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr, spearmanr
import matplotlib as mpl


# === Fonts editable in Illustrator ===
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# === Input files
mfass_file = "/ESL/ESL_MPRA/SI_figures/MFASS_overlap/outputs/merged_mfass_supertable_overlap_summary.tsv"
supertable_file = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/ALL_WTS_VARS_NO_DELTAS.csv"
output_dir = "/ESL/ESL_MPRA/SI_figures/MFASS_overlap/outputs/corrs_out_r2"
os.makedirs(output_dir, exist_ok=True)

# === Load MFASS and Supertable
mfass_df = pd.read_csv(mfass_file, sep="\t")
mfass_df = mfass_df.rename(columns={"supertable_ref": "Reference"})

supertable_df = pd.read_csv(supertable_file)

# === Compute average PSI for HEK293
hek_cols = ["HEK_rep1_psi_raw", "HEK_rep2_psi_raw", "HEK_rep3_psi_raw", "HEK_rep4_psi_raw"]
supertable_df["HEK293_avg_psi"] = supertable_df[hek_cols].astype(float).mean(axis=1)

# === Compute pooled avg PSI across all 5 cell lines
cell_lines = {
    "HEK293": ["HEK_rep1_psi_raw", "HEK_rep2_psi_raw", "HEK_rep3_psi_raw", "HEK_rep4_psi_raw"],
    "HeLa": ["HeLa_rep1_psi_raw", "HeLa_rep2_psi_raw"],
    "K562": ["K562_rep1_psi_raw", "K562_rep2_psi_raw"],
    "MCF7": ["MCF7_rep1_psi_raw", "MCF7_rep2_psi_raw"],
    "HMC3": ["HMC3_rep1_psi_raw", "HMC3_rep2_psi_raw"]
}

all_reps = sum(cell_lines.values(), [])
supertable_df["AllCells_avg_psi"] = supertable_df[all_reps].astype(float).mean(axis=1)

# === Merge with MFASS
merged = mfass_df.merge(supertable_df[["Reference", "HEK293_avg_psi", "AllCells_avg_psi"]], on="Reference", how="left")
merged = merged.dropna(subset=["MFASS_index", "HEK293_avg_psi", "AllCells_avg_psi"])

# === Plotting function
def plot_corr(x, y, label_x, label_y, title_prefix, out_prefix):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=x, y=y, s=20, alpha=0.6, color="blue", rasterized=True)
    plt.axline((0, 0), slope=1, linestyle="--", color="gray", linewidth=1)

    r, p = pearsonr(x, y)
    rho, p_spearman = spearmanr(x, y)
    n = len(x)

    plt.title(f"{title_prefix}\nPearson r = {r:.2f}, r² = {r**2:.2f} | Spearman ρ = {rho:.2f}, n = {n}", fontsize=12)
    plt.xlabel(label_x, fontsize=12)
    plt.ylabel(label_y, fontsize=12)
    plt.tight_layout()

    pdf_path = os.path.join(output_dir, f"{out_prefix}.pdf")
    png_path = os.path.join(output_dir, f"{out_prefix}.png")
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    plt.close()

# === Plot MFASS vs HEK293
plot_corr(
    x=merged["MFASS_index"],
    y=merged["HEK293_avg_psi"],
    label_x="MFASS PSI (Index)",
    label_y="HEK293 PSI (Replicate Avg)",
    title_prefix="MFASS vs HEK293 PSI",
    out_prefix="mfass_vs_HEK293_PSI"
)

# === Plot MFASS vs All Cells Avg
plot_corr(
    x=merged["MFASS_index"],
    y=merged["AllCells_avg_psi"],
    label_x="MFASS PSI (Index)",
    label_y="All Cell Lines PSI (Avg)",
    title_prefix="MFASS vs All-Cell-Line Avg PSI",
    out_prefix="mfass_vs_AllCells_PSI"
)
