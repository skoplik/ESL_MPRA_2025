import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

# === Inputs ===
input_file = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/ALL_WITH_WT.csv"
output_dir = "/ESL/ESL_MPRA/SI_figures/Correlations_all_cell_lines/outputs/"
os.makedirs(output_dir, exist_ok=True)

# === Replicates to include
replicate_cols = {
    "HEK293": ["HEK_rep1", "HEK_rep2"],
    "HeLa": ["HeLa_rep1", "HeLa_rep2"],
    "K562": ["K562_rep1", "K562_rep2"],
    "MCF7": ["MCF7_rep1", "MCF7_rep2"],
    "HMC3": ["HMC3_rep1", "HMC3_rep2"]
}

# === Load input
df = pd.read_csv(input_file, low_memory=False)

# === Helpers
def get_all_replicates(df, suffix):
    reps = []
    for cl, rlist in replicate_cols.items():
        for r in rlist:
            col = f"{r}_{suffix}"
            if col in df.columns:
                reps.append((cl, r, col))
    return reps

def compute_delta(df, suffix):
    wt_rows = df[df["snp"] == "none"]
    wt_by_event = wt_rows.set_index("event_id")
    deltas = {}
    for cl, rlist in replicate_cols.items():
        for r in rlist:
            col = f"{r}_{suffix}"
            if col not in df.columns:
                continue
            v = df[["event_id", col]].copy()
            if col not in wt_by_event.columns:
                print(f"[WARN] Missing WT column: {col}")
                continue
            v["wt"] = v["event_id"].map(wt_by_event[col])
            v["delta"] = v[col] - v["wt"]
            delta_col = f"{cl} {r}"
            deltas[delta_col] = v["delta"]
    out_df = pd.DataFrame(deltas)
    out_df.index = df["event_id"]
    return out_df

def format_label(colname):
    for cl in replicate_cols:
        if colname.startswith(cl):
            for rep in replicate_cols[cl]:
                if rep in colname:
                    return f"{cl} Repl. {rep.split('_')[-1]}"
    return colname

def plot_full_matrix(data, title, filename, lim=None):
    reps = data.columns.tolist()
    n = len(reps)
    fig, axes = plt.subplots(n, n, figsize=(2*n, 2*n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
            if j < i:
                ax.axis("off")
                continue
            elif i == j:
                ax.annotate(format_label(reps[i]), xy=(0.5, 0.5), xycoords="axes fraction", ha="center", va="center", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            x = data[reps[j]]
            y = data[reps[i]]
            valid = x.notna() & y.notna()
            x = x[valid]
            y = y[valid]
            if len(x) < 2 or len(y) < 2:
                ax.annotate("n < 2", xy=(0.5, 0.5), xycoords="axes fraction", ha="center", va="center", fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            ax.scatter(x, y, s=1, alpha=0.05, color="blue", rasterized=True)
            r, p = pearsonr(x, y)
            ax.text(0.05, 0.95, f"$r^2$={r**2:.2f}", transform=ax.transAxes, fontsize=6, va="top")
            ax.text(0.05, 0.85, f"p={p:.1e}", transform=ax.transAxes, fontsize=6)
            ax.text(0.05, 0.75, f"n={len(x)}", transform=ax.transAxes, fontsize=6)
            if i == n - 1:
                ax.set_xlabel(format_label(reps[j]), fontsize=6, rotation=90)
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(format_label(reps[i]), fontsize=6)
            else:
                ax.set_yticks([])
            if lim:
                ax.set_xlim(*lim)
                ax.set_ylim(*lim)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_base = os.path.join(output_dir, filename.replace(".pdf", ""))
    fig.savefig(f"{out_base}.pdf", dpi=300)
    fig.savefig(f"{out_base}.png", dpi=300)
    plt.close(fig)

# === PSI replicate matrix
psi_meta = get_all_replicates(df, "psi_raw")
psi_cols = [x[2] for x in psi_meta]
psi_df = df[psi_cols].copy()
psi_df.columns = [f"{x[0]} {x[1]}" for x in psi_meta]

# === ΔPSI matrix
dpsi_df = compute_delta(df, "psi_raw")
dpsi_df = dpsi_df[[col for col in dpsi_df.columns if col in psi_df.columns]]  # Ensure matching columns

# === Debug
print("[DEBUG] dpsi_df shape:", dpsi_df.shape)
print(dpsi_df.notna().sum())

# === Plot PSI and ΔPSI only
plot_full_matrix(psi_df, "Replicate vs Replicate PSI", "supplement_fig1_fullmatrix_PSI.pdf", lim=(0, 1))
plot_full_matrix(dpsi_df, "Replicate vs Replicate ΔPSI", "supplement_fig1_fullmatrix_dPSI.pdf", lim=(-1, 1))
