import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load your file
df = pd.read_csv("/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/ALL_WITH_WT.csv")
print(len(df))
df = df.drop_duplicates(subset="full_seq", keep="first").copy()
print(len(df))
output_dir = "/ESL/ESL_MPRA/SI_figures/read_coverage_wts_vs_vars/outputs"
os.makedirs(output_dir, exist_ok=True)

cell_lines = ["HEK", "HeLa", "K562", "HMC3", "MCF7"]
ref_lines = [10, 100, 1000, 10000, 100000]
'''
# Subset labels for WT / Variant / All
subset_labels = [
    ("all", df),
    ("wt_only", df[df["snp"] == "none"]),
    ("variant_only", df[df["snp"] != "none"]),
]

# === Summary Table: Avg Read Depth by Subset ===
print("\n=== Average Pooled Read Depths by Cell Line and Subset ===\n")
summary_data = []
mean_tracker = {"all": [], "wt_only": [], "variant_only": []}

for cl in cell_lines:
    total_col = f"{cl}_total_pooled"
    if total_col not in df.columns:
        print(f"{cl}: Skipping, column not found.")
        continue

    for label, group_df in subset_labels:
        total_vals = group_df[total_col].replace([np.inf, -np.inf], np.nan).dropna()
        total_vals = total_vals[total_vals > 0]

        if total_vals.empty:
            print(f"{cl} - {label}: No valid values.")
            continue

        mean = total_vals.mean()
        median = total_vals.median()
        mean_tracker[label].append(mean)
        summary_data.append((cl, label, mean, median))
        print(f"{cl:<6} | {label:<12} | Mean = {mean:>8.2f} | Median = {median:>8.2f}")

# === Mean of Means ===
print("\n=== Mean of Means (Across Cell Lines) ===\n")
for label in mean_tracker:
    if mean_tracker[label]:
        mean_of_means = np.mean(mean_tracker[label])
        print(f"{label:<12}: Mean of Means = {mean_of_means:.2f}")
    else:
        print(f"{label:<12}: No valid data to compute mean of means.")


# === Pooled Total Histograms ===
print("\n=== Pooled Total Histogram Statistics ===\n")
for cl in cell_lines:
    total_col = f"{cl}_total_pooled"
    if total_col not in df.columns:
        print(f"Skipping {cl}: {total_col} not found.\n")
        continue

    for label, group_df in subset_labels:
        total_vals = group_df[total_col].replace([np.inf, -np.inf], np.nan).dropna()
        total_vals = total_vals[total_vals > 0]

        if total_vals.empty:
            print(f"{cl} - {label}: no valid values.\n")
            continue

        mean = total_vals.mean()
        median = total_vals.median()
        print(f"{cl} - {label}: Mean = {mean:.2f}, Median = {median:.2f}")

        # Plot
        plt.figure(figsize=(8, 6))
        bins = np.logspace(0, np.log10(total_vals.max() + 1), 100)
        plt.hist(total_vals, bins=bins, color='gray', edgecolor='black', alpha=0.8)

        plt.axvline(mean, color='blue', linestyle='--', linewidth=1.5, label=f"Mean = {mean:.1f}")
        plt.axvline(median, color='red', linestyle='-', linewidth=1.5, label=f"Median = {median:.1f}")

        for r in ref_lines:
            if r < total_vals.max():
                plt.axvline(r, color='black', linestyle=':', linewidth=1)
                plt.text(r, plt.ylim()[1]*0.9, f"{r}", ha='right', va='top', rotation=90, fontsize=8)

        plt.xscale('log')
        plt.xlabel("Total Pooled Reads (Included + Excluded)", fontsize=12)
        plt.ylabel("Number of Sequences", fontsize=12)
        plt.title(f"{cl} - {label.replace('_', ' ').title()}", fontsize=14)
        plt.legend()
        plt.tight_layout()

        out_name = f"{cl}_{label}_histogram.png"
        plt.savefig(os.path.join(output_dir, out_name), dpi=300)
        plt.close()

# === Replicate-Level Stats (no plots) ===
print("\n=== Replicate-Level Total Read Statistics ===\n")
replicates = {
    "HEK": 4,
    "HeLa": 2,
    "K562": 2,
    "HMC3": 2,
    "MCF7": 2
}

for cl, num_reps in replicates.items():
    for i in range(1, num_reps + 1):
        incl_col = f"{cl}_rep{i}_included"
        excl_col = f"{cl}_rep{i}_excluded"

        if incl_col not in df.columns or excl_col not in df.columns:
            print(f"{cl} rep{i}: missing columns")
            continue

        total_reads = df[incl_col] + df[excl_col]
        total_reads = total_reads.replace([np.inf, -np.inf], np.nan).dropna()

        if total_reads.empty:
            print(f"{cl} rep{i}: no valid values")
            continue

        mean = total_reads.mean()
        median = total_reads.median()
        print(f"{cl} rep{i}: Mean = {mean:.2f}, Median = {median:.2f}")

# === Read Depth by Quantile Bins (Equal-Size Binning) ===
print("\n=== Read Depth by Quantile Bins ===\n")
n_bins = 10

for cl in cell_lines:
    total_col = f"{cl}_total_pooled"
    if total_col not in df.columns:
        print(f"{cl}: Skipping, column not found.")
        continue

    for label, group_df in subset_labels:
        total_vals = group_df[total_col].replace([np.inf, -np.inf], np.nan).dropna()
        total_vals = total_vals[total_vals > 0]

        if total_vals.empty:
            print(f"{cl} - {label}: no valid values for binning.\n")
            continue

        # Quantile bins
        try:
            bins = pd.qcut(total_vals, q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            print(f"{cl} - {label}: too few unique values for qcut.")
            continue

        group_df = group_df.loc[total_vals.index].copy()
        group_df["bin"] = bins
        group_df["read_depth"] = total_vals

        # Boxplot by bin
        plt.figure(figsize=(10, 6))
        group_df.boxplot(column="read_depth", by="bin", grid=False, showfliers=False)
        plt.yscale("log")
        plt.xlabel("Quantile Bin (equal number of points)")
        plt.ylabel("Total Pooled Reads (log scale)")
        plt.title(f"{cl} - {label.replace('_', ' ').title()}")
        plt.suptitle("")

        out_name = f"{cl}_{label}_read_depth_by_bin.png"
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, out_name), dpi=300)
        plt.close()

print("\n=== Swarm Plot: WT vs Variant Read Depth by Bin (Log + Linear, No Seaborn) ===\n")
n_bins = 10

for cl in cell_lines:
    total_col = f"{cl}_total_pooled"
    if total_col not in df.columns:
        print(f"{cl}: Skipping, column not found.")
        continue

    wt_df = df[df["snp"] == "none"]
    var_df = df[df["snp"] != "none"]

    wt_vals = wt_df[total_col].replace([np.inf, -np.inf], np.nan).dropna()
    wt_vals = wt_vals[wt_vals > 0]
    var_vals = var_df[total_col].replace([np.inf, -np.inf], np.nan).dropna()
    var_vals = var_vals[var_vals > 0]

    if wt_vals.empty or var_vals.empty:
        print(f"{cl}: No valid WT or variant read values.")
        continue

    try:
        wt_bins = pd.qcut(wt_vals, q=n_bins, labels=False, duplicates="drop")
        var_bins = pd.qcut(var_vals, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        print(f"{cl}: Too few unique values to bin.")
        continue

    wt_df = pd.DataFrame({
        "ReadDepth": wt_vals,
        "Bin": wt_bins,
        "Type": "WT"
    })

    var_df = pd.DataFrame({
        "ReadDepth": var_vals,
        "Bin": var_bins,
        "Type": "Variant"
    })

    combined_df = pd.concat([wt_df, var_df])

    for scale in ["log", "linear"]:
        plt.figure(figsize=(12, 6))
        for b in range(n_bins):
            # WT points
            bin_wt = combined_df[(combined_df["Bin"] == b) & (combined_df["Type"] == "WT")]
            xvals_wt = np.random.normal(loc=b - 0.2, scale=0.05, size=len(bin_wt))
            plt.scatter(xvals_wt, bin_wt["ReadDepth"], color='blue', alpha=0.6, s=10, label='WT' if b == 0 else "")

            # Variant points
            bin_var = combined_df[(combined_df["Bin"] == b) & (combined_df["Type"] == "Variant")]
            xvals_var = np.random.normal(loc=b + 0.2, scale=0.05, size=len(bin_var))
            plt.scatter(xvals_var, bin_var["ReadDepth"], color='orange', alpha=0.6, s=10, label='Variant' if b == 0 else "")

        plt.xlabel("Quantile Bin (equal number of points)")
        plt.ylabel("Total Pooled Reads" + (" (log scale)" if scale == "log" else ""))
        plt.title(f"{cl}: WT vs Variant Read Depth per Bin ({scale})")
        plt.xticks(range(n_bins))
        if scale == "log":
            plt.yscale("log")
        plt.legend()
        plt.tight_layout()

        out_name = f"{cl}_swarmplot_manual_read_depth_by_bin_{scale}.png"
        plt.savefig(os.path.join(output_dir, out_name), dpi=300)
        plt.close()

'''
'''print("\n=== Swarm Plot: WT vs Variant Read Depth by Reference PSI Bin (Labeled by Mean PSI) ===\n")
n_bins = 10

for cl in cell_lines:
    total_col = f"{cl}_total_pooled"
    wt_psi_col = f"{cl}_pooled_psi_raw"
    if total_col not in df.columns or wt_psi_col not in df.columns:
        print(f"{cl}: Skipping, required columns not found.")
        continue

    # Subset to valid WT rows
    wt_df = df[df["snp"] == "none"].copy()
    wt_df = wt_df[(wt_df[wt_psi_col].notna()) & (wt_df[total_col] > 0)]

    if wt_df.empty:
        print(f"{cl}: No valid WT PSI values.")
        continue

    # Bin by quantiles of WT PSI
    # Add small jitter to PSI to avoid ties and guarantee 10 bins
    jittered_psi = wt_df[wt_psi_col] + np.random.uniform(-1e-10, 1e-10, size=len(wt_df))
    
    try:
        psi_bins = pd.qcut(jittered_psi, q=n_bins, labels=False, duplicates="raise")
    except ValueError:
        print(f"{cl}: Failed to create {n_bins} bins even after jittering.")
        continue
    
    wt_df["bin"] = psi_bins
    wt_df["ReadDepth"] = wt_df[total_col]
    wt_df["Type"] = "WT"

    # Get bin → mean PSI for labeling
    bin_mean_psi = wt_df.groupby("bin")[wt_psi_col].mean().round(2)
    bin_labels = [f"{bin_mean_psi.get(i, np.nan):.2f}" for i in range(n_bins)]

    # Map event_id → bin from WT
    event_bin_map = wt_df.set_index("event_id")["bin"].to_dict()

    # Assign variants to their reference’s bin
    var_df = df[df["snp"] != "none"].copy()
    var_df = var_df[(var_df[total_col] > 0) & (var_df["event_id"].isin(event_bin_map))].copy()
    var_df["bin"] = var_df["event_id"].map(event_bin_map)
    var_df["ReadDepth"] = var_df[total_col]
    var_df["Type"] = "Variant"

    # Combine
    combined_df = pd.concat([wt_df[["bin", "ReadDepth", "Type"]],
                             var_df[["bin", "ReadDepth", "Type"]]])

    for scale in ["log", "linear"]:
        plt.figure(figsize=(12, 6))

        for b in sorted(combined_df["bin"].dropna().unique()):
            b = int(b)
            bin_wt = combined_df[(combined_df["bin"] == b) & (combined_df["Type"] == "WT")]
            bin_var = combined_df[(combined_df["bin"] == b) & (combined_df["Type"] == "Variant")]

            xvals_wt = np.random.normal(loc=b - 0.2, scale=0.05, size=len(bin_wt))
            xvals_var = np.random.normal(loc=b + 0.2, scale=0.05, size=len(bin_var))

            plt.scatter(xvals_wt, bin_wt["ReadDepth"], color='blue', alpha=0.6, s=10, label='WT' if b == 0 else "")
            plt.scatter(xvals_var, bin_var["ReadDepth"], color='orange', alpha=0.6, s=10, label='Variant' if b == 0 else "")

        plt.xlabel("WT PSI Bin (Label = Mean WT PSI)")
        plt.ylabel("Total Pooled Reads" + (" (log scale)" if scale == "log" else ""))
        plt.title(f"{cl}: Read Depth by WT PSI Quantile Bin ({scale})")
        plt.xticks(range(n_bins), bin_labels, rotation=45)
        if scale == "log":
            plt.yscale("log")
        plt.legend()
        plt.tight_layout()

        out_name = f"{cl}_swarm_by_ref_psi_quantile_meanlabel_{scale}.png"
        plt.savefig(os.path.join(output_dir, out_name), dpi=300)
        plt.close()

print("\n=== Swarm Plot: WT vs Variant Read Depth by Fixed Reference PSI Bin (Log + Linear) ===\n")
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)  # [0.0, 0.1, ..., 1.0]

for cl in cell_lines:
    total_col = f"{cl}_total_pooled"
    wt_psi_col = f"{cl}_pooled_psi_raw"

    if total_col not in df.columns or wt_psi_col not in df.columns:
        print(f"{cl}: Skipping, required columns not found.")
        continue

    # Step 1: Get WT PSI and bin
    wt_df = df[df["snp"] == "none"].copy()
    wt_df = wt_df[(wt_df[wt_psi_col].notna()) & (wt_df[total_col] > 0)].copy()
    wt_df["bin"] = pd.cut(wt_df[wt_psi_col], bins=bin_edges, labels=False, include_lowest=True)
    wt_df["ReadDepth"] = wt_df[total_col]
    wt_df["Type"] = "WT"

    if wt_df.empty:
        print(f"{cl}: No valid WT rows.")
        continue

    # Step 2: Map event_id → bin from WT rows
    event_bin_map = wt_df.set_index("event_id")["bin"].to_dict()

    # Step 3: For each variant, assign same bin as WT
    var_df = df[df["snp"] != "none"].copy()
    var_df = var_df[(var_df[total_col] > 0) & (var_df["event_id"].isin(event_bin_map))].copy()
    var_df["bin"] = var_df["event_id"].map(event_bin_map)
    var_df["ReadDepth"] = var_df[total_col]
    var_df["Type"] = "Variant"

    # Combine
    combined_df = pd.concat([
        wt_df[["bin", "ReadDepth", "Type"]],
        var_df[["bin", "ReadDepth", "Type"]]
    ])

    for scale in ["log", "linear"]:
        plt.figure(figsize=(12, 6))
        for b in sorted(combined_df["bin"].dropna().unique()):
            b = int(b)
            bin_wt = combined_df[(combined_df["bin"] == b) & (combined_df["Type"] == "WT")]
            bin_var = combined_df[(combined_df["bin"] == b) & (combined_df["Type"] == "Variant")]

            xvals_wt = np.random.normal(loc=b - 0.2, scale=0.05, size=len(bin_wt))
            xvals_var = np.random.normal(loc=b + 0.2, scale=0.05, size=len(bin_var))

            plt.scatter(xvals_wt, bin_wt["ReadDepth"], color='blue', alpha=0.6, s=10, label='WT' if b == 0 else "")
            plt.scatter(xvals_var, bin_var["ReadDepth"], color='orange', alpha=0.6, s=10, label='Variant' if b == 0 else "")

        # Labels
        plt.xlabel("Reference PSI Bin (Fixed Width: 0.1)")
        plt.ylabel("Total Pooled Reads" + (" (log scale)" if scale == "log" else ""))
        plt.title(f"{cl}: Read Depth by WT PSI Bin ({scale})")
        plt.xticks(
            ticks=range(n_bins),
            labels=[f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(n_bins)],
            rotation=45
        )
        if scale == "log":
            plt.yscale("log")
        plt.legend()
        plt.tight_layout()

        out_name = f"{cl}_swarm_by_fixed_ref_psi_bin_{scale}.png"
        plt.savefig(os.path.join(output_dir, out_name), dpi=300)
        plt.close()
'''
'''
print("\n=== Swarm Plot: Read Depth by Starting PSI Quantile Bin (WT-defined, Equal Size) ===\n")
n_bins = 10

for cl in cell_lines:
    print(cl)
    psi_col = f"{cl}_pooled_psi_raw"
    depth_col = f"{cl}_total_pooled"
    if psi_col not in df.columns or depth_col not in df.columns:
        print(f"{cl}: Skipping, columns missing.")
        continue

    # 1. Subset WT rows and clean
    wt_df = df[df["snp"] == "none"].copy()
    wt_df = wt_df[(wt_df[psi_col].notna()) & (wt_df[depth_col] > 0)].copy()

    # 2. Add jitter and bin into quantiles
    jittered_psi = wt_df[psi_col] + np.random.uniform(-1e-10, 1e-10, size=len(wt_df))
    try:
        psi_bins = pd.qcut(jittered_psi, q=n_bins, labels=False, duplicates="raise")
    except ValueError:
        print(f"{cl}: Not enough unique PSI values to form {n_bins} bins.")
        continue

    wt_df["bin"] = psi_bins
    wt_df["ReadDepth"] = wt_df[depth_col]
    wt_df["Type"] = "WT"

    # 3. Map event_id → bin from WT
    event_to_bin = wt_df.set_index("event_id")["bin"].to_dict()

    # 4. Assign bin to variants based on their WT
    var_df = df[df["snp"] != "none"].copy()
    var_df = var_df[(var_df[depth_col] > 0) & (var_df["event_id"].isin(event_to_bin))].copy()
    var_df["bin"] = var_df["event_id"].map(event_to_bin)
    var_df["ReadDepth"] = var_df[depth_col]
    var_df["Type"] = "Variant"

    # 5. Combine
    combined_df = pd.concat([wt_df[["bin", "ReadDepth", "Type"]],
                             var_df[["bin", "ReadDepth", "Type"]]])

    # 6. Mean PSI per bin (for labeling)
    bin_means = wt_df.groupby("bin")[psi_col].mean().round(2)
    bin_labels = [f"{bin_means.get(i, np.nan):.2f}" for i in range(n_bins)]

    # 7. Plot
    for scale in ["log", "linear"]:
        plt.figure(figsize=(12, 6))
        for b in range(n_bins):
            bin_wt = combined_df[(combined_df["bin"] == b) & (combined_df["Type"] == "WT")]
            bin_var = combined_df[(combined_df["bin"] == b) & (combined_df["Type"] == "Variant")]

            # Scatter WT and Variant points first (behind)
            x_wt = np.random.normal(loc=b - 0.2, scale=0.05, size=len(bin_wt))
            x_var = np.random.normal(loc=b + 0.2, scale=0.05, size=len(bin_var))
            plt.scatter(x_wt, bin_wt["ReadDepth"], color='blue', alpha=0.6, s=10, label='WT' if b == 0 else "")
            plt.scatter(x_var, bin_var["ReadDepth"], color='orange', alpha=0.6, s=10, label='Variant' if b == 0 else "")
            
            # Black mean/median lines for WT
            if not bin_wt.empty:
                wt_mean = bin_wt["ReadDepth"].mean()
                wt_median = bin_wt["ReadDepth"].median()
                plt.hlines(wt_mean, b - 0.4, b, color='black', linestyle='--', linewidth=1.2, zorder=10,
                           label='Mean' if b == 0 else "")
                plt.hlines(wt_median, b - 0.4, b, color='black', linestyle='-', linewidth=1.2, zorder=11,
                           label='Median' if b == 0 else "")
            
            # Black mean/median lines for Variant
            if not bin_var.empty:
                var_mean = bin_var["ReadDepth"].mean()
                var_median = bin_var["ReadDepth"].median()
                plt.hlines(var_mean, b, b + 0.4, color='black', linestyle='--', linewidth=1.2, zorder=10)
                plt.hlines(var_median, b, b + 0.4, color='black', linestyle='-', linewidth=1.2, zorder=11)
            
        plt.xlabel("Starting PSI Quantile Bin (Mean WT PSI)")
        plt.ylabel("Total Pooled Reads" + (" (log scale)" if scale == "log" else ""))
        plt.title(f"{cl}: Read Depth by Starting PSI Bin ({scale})")
        plt.xticks(range(n_bins), bin_labels, rotation=45)
        if scale == "log":
            plt.yscale("log")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{cl}_read_depth_by_starting_psi_quantile_{scale}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        
        '''
'''      
print("\n=== Swarm Plot: Read Depth by Fixed PSI Bin (WT-based, with Black Mean/Median Lines) ===\n")
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)

for cl in cell_lines:
    psi_col = f"{cl}_pooled_psi_raw"
    depth_col = f"{cl}_total_pooled"
    if psi_col not in df.columns or depth_col not in df.columns:
        print(f"{cl}: Skipping, required columns not found.")
        continue

    # Step 1: Get WT and assign bins
    wt_df = df[df["snp"] == "none"].copy()
    wt_df = wt_df[(wt_df[psi_col].notna()) & (wt_df[depth_col] > 0)].copy()
    wt_df["bin"] = pd.cut(wt_df[psi_col], bins=bin_edges, labels=False, include_lowest=True, right=False)
    wt_df["ReadDepth"] = wt_df[depth_col]
    wt_df["Type"] = "WT"

    if wt_df.empty:
        print(f"{cl}: No valid WT rows.")
        continue

    # Step 2: Map event_id → bin
    event_bin_map = wt_df.set_index("event_id")["bin"].to_dict()

    # Step 3: Assign bin to variants via event_id
    var_df = df[df["snp"] != "none"].copy()
    var_df = var_df[(var_df[depth_col] > 0) & (var_df["event_id"].isin(event_bin_map))].copy()
    var_df["bin"] = var_df["event_id"].map(event_bin_map)
    var_df["ReadDepth"] = var_df[depth_col]
    var_df["Type"] = "Variant"

    # Combine
    combined_df = pd.concat([
        wt_df[["bin", "ReadDepth", "Type"]],
        var_df[["bin", "ReadDepth", "Type"]]
    ])

    for scale in ["log", "linear"]:
        plt.figure(figsize=(12, 6))
        for b in range(n_bins):
            bin_wt = combined_df[(combined_df["bin"] == b) & (combined_df["Type"] == "WT")]
            bin_var = combined_df[(combined_df["bin"] == b) & (combined_df["Type"] == "Variant")]

            # Plot points
            x_wt = np.random.normal(loc=b - 0.2, scale=0.05, size=len(bin_wt))
            x_var = np.random.normal(loc=b + 0.2, scale=0.05, size=len(bin_var))
            plt.scatter(x_wt, bin_wt["ReadDepth"], color='blue', alpha=0.6, s=10, label='WT' if b == 0 else "")
            plt.scatter(x_var, bin_var["ReadDepth"], color='orange', alpha=0.6, s=10, label='Variant' if b == 0 else "")
            
            # Annotate counts for WT and Variant
            wt_n = len(bin_wt)
            var_n = len(bin_var)
            label_text = f"WT n={wt_n}\nVar n={var_n}"
            plt.text(b, plt.ylim()[1] * 0.95, label_text, ha='center', va='top', fontsize=8, zorder=20)

            # Plot mean and median as black lines
            if not bin_wt.empty:
                wt_mean = bin_wt["ReadDepth"].mean()
                wt_median = bin_wt["ReadDepth"].median()
                plt.hlines(wt_mean, b - 0.4, b, color='black', linestyle='--', linewidth=1.2, zorder=10,
                           label='Mean' if b == 0 else "")
                plt.hlines(wt_median, b - 0.4, b, color='black', linestyle='-', linewidth=1.2, zorder=11,
                           label='Median' if b == 0 else "")
            if not bin_var.empty:
                var_mean = bin_var["ReadDepth"].mean()
                var_median = bin_var["ReadDepth"].median()
                plt.hlines(var_mean, b, b + 0.4, color='black', linestyle='--', linewidth=1.2, zorder=10)
                plt.hlines(var_median, b, b + 0.4, color='black', linestyle='-', linewidth=1.2, zorder=11)

        # X-tick labels
        bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(n_bins)]
        plt.xticks(range(n_bins), bin_labels, rotation=45)

        # Axes and output
        plt.xlabel("WT PSI Bin (Fixed Width: 0.1)")
        plt.ylabel("Total Pooled Reads" + (" (log scale)" if scale == "log" else ""))
        plt.title(f"{cl}: Read Depth by WT PSI Bin ({scale})")
        if scale == "log":
            plt.yscale("log")
        plt.legend()
        plt.tight_layout()

        out_name = f"{cl}_swarm_by_fixed_ref_psi_bin_black_lines_{scale}.png"
        plt.savefig(os.path.join(output_dir, out_name), dpi=300)
        plt.close()
'''
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import os

print("\n=== Plotting Δ Coverage (WT - Variant) vs. WT PSI — Stack-Normalized Vertical KDE ===\n")

for cl in ["HEK", "HeLa", "K562", "HMC3", "MCF7"]:
    psi_col = f"{cl}_pooled_psi_raw"
    cov_col = f"{cl}_total_pooled"

    if psi_col in df.columns and cov_col in df.columns:
        wt_df = df[df["snp"] == "none"].copy()
        var_df = df[df["snp"] != "none"].copy()

        for d in [wt_df, var_df]:
            d.loc[:, psi_col] = pd.to_numeric(d[psi_col], errors='coerce')
            d.loc[:, cov_col] = pd.to_numeric(d[cov_col], errors='coerce')

        merged = pd.merge(var_df, wt_df, on="event_id", suffixes=("_var", "_wt"))
        merged = merged.dropna(subset=[f"{cov_col}_var", f"{cov_col}_wt", f"{psi_col}_wt"])

        merged["delta_cov"] = merged[f"{cov_col}_wt"] - merged[f"{cov_col}_var"]
        x = merged[f"{psi_col}_wt"].to_numpy()
        y = merged["delta_cov"].to_numpy()

        # Round PSI to group into vertical stacks
        psi_round = np.round(x, 2)
        merged["psi_round"] = psi_round

        # Compute per-stack KDE + normalize max = 1
        z = np.zeros_like(y)
        for val in np.unique(psi_round):
            idx = (psi_round == val)
            if idx.sum() < 5:
                continue
            kde = gaussian_kde(y[idx], bw_method="silverman")
            densities = kde(y[idx])
            z[idx] = densities / densities.max()  # Normalize to [0, 1] within this stack

        # === Plot ===
        plt.figure(figsize=(12, 5))
        plt.scatter(x, y, c=z, cmap='viridis', alpha=0.8, s=10, edgecolor='none')
        plt.xlabel(f"WT PSI ({cl})")
        plt.ylabel("Δ Coverage (WT - Variant)")
        plt.title(f"Δ Coverage vs. WT PSI — Stack-Normalized Vertical KDE ({cl})")
        plt.colorbar(label='Normalized Vertical Density (max=1 per PSI bin)')
        plt.grid(True, linestyle='--', linewidth=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{cl}_delta_cov_vs_ref_psi_stacknorm_kde.pdf"), dpi=300)
        plt.close()

        print(f"{cl}: saved stack-normalized vertical KDE plot.")
    else:
        print(f"{cl}: Required columns not found in input.")


