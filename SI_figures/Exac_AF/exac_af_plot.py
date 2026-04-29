import pandas as pd
import matplotlib.pyplot as plt

# Load your files
variant_df = pd.read_csv(
    "/ESL/Figures/Resources/supertable_dbSNP155_Gencode_v26/supertable_dbSNP155_Gencode_v26_overlap_dbSNP155_matches_ExAC.txt",
    sep="\t", dtype=str)
psi_df = pd.read_csv(
    "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/ALL_WITH_WT.csv",
    dtype=str)

# Merge on Reference
variant_df['supertable_1_num'] = variant_df['supertable_1_num'].astype(str)
psi_df['Reference'] = psi_df['Reference'].astype(str)
merged = psi_df.merge(variant_df[['supertable_1_num', 'study_altAlleleFreq']],
                      left_on='Reference', right_on='supertable_1_num', how='left')

# Convert allele freq and delta values to numeric
merged['study_altAlleleFreq'] = pd.to_numeric(merged['study_altAlleleFreq'], errors='coerce')
delta_logit_cols = [c for c in merged.columns if 'delta_logit_pooled' in c and not c.startswith('wt')]
dpsi_cols = [c for c in merged.columns if 'dpsi_pooled' in c and not c.startswith('wt')]
merged[delta_logit_cols + dpsi_cols] = merged[delta_logit_cols + dpsi_cols].apply(pd.to_numeric, errors='coerce')

# Compute mean values
merged['mean_delta_logit'] = merged[delta_logit_cols].mean(axis=1)
merged['mean_dpsi'] = merged[dpsi_cols].mean(axis=1)

# Prepare filtered dataframes
plot_df_logit = merged.dropna(subset=['study_altAlleleFreq', 'mean_delta_logit'])
plot_df_dpsi = merged.dropna(subset=['study_altAlleleFreq', 'mean_dpsi'])

# === Plot Δlogit full ===
plt.figure(figsize=(6, 5))
plt.scatter(plot_df_logit['mean_delta_logit'], plot_df_logit['study_altAlleleFreq'], s=10, alpha=0.6)
plt.xlim(-8, 8)
plt.xlabel("Mean Δlogit(PSI)")
plt.ylabel("Exac Allele Frequency")
plt.title("Mean Δlogit(PSI) vs Allele Frequency")
plt.tight_layout()
plt.savefig("/ESL/ESL_MPRA/SI_figures/Exac_AF/outputs/06_10_25_exac_af_logit.pdf")

# === Plot Δlogit with AF capped at 0.010 ===
plt.figure(figsize=(6, 5))
plt.scatter(plot_df_logit['mean_delta_logit'], plot_df_logit['study_altAlleleFreq'], s=10, alpha=0.6)
plt.ylim(0, 0.010)
plt.xlim(-8, 8)
plt.xlabel("Mean Δlogit(PSI)")
plt.ylabel("Exac Allele Frequency")
plt.title("Mean Δlogit(PSI) vs AF ≤ 0.010")
plt.tight_layout()
plt.savefig("/ESL/ESL_MPRA/SI_figures/Exac_AF/outputs/06_10_25_exac_af_logit_ycap.pdf")

# === Plot ΔPSI full ===
plt.figure(figsize=(6, 5))
plt.scatter(plot_df_dpsi['mean_dpsi'], plot_df_dpsi['study_altAlleleFreq'], s=10, alpha=0.6)
plt.xlabel("Mean ΔPSI")
plt.ylabel("Exac Allele Frequency")
plt.title("Mean ΔPSI vs Allele Frequency")
plt.tight_layout()
plt.savefig("/ESL/ESL_MPRA/SI_figures/Exac_AF/outputs/06_10_25_exac_af_dpsi.pdf")

# === Plot ΔPSI with AF capped at 0.010 ===
plt.figure(figsize=(6, 5))
plt.scatter(plot_df_dpsi['mean_dpsi'], plot_df_dpsi['study_altAlleleFreq'], s=10, alpha=0.6)
plt.ylim(0, 0.010)
plt.xlabel("Mean ΔPSI")
plt.ylabel("Exac Allele Frequency")
plt.title("Mean ΔPSI vs AF ≤ 0.010")
plt.tight_layout()
plt.savefig("/ESL/ESL_MPRA/SI_figures/Exac_AF/outputs/06_10_25_exac_af_dpsi_ycap.pdf")
