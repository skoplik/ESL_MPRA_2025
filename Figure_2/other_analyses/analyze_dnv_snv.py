import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr, gaussian_kde
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def normalize_snp_set(snp_str):
    return frozenset(s.strip() for s in snp_str.split(";") if s.strip())

def snp_label(row):
    return f"{row['snp_snv1']} + {row['snp_snv2']} → {row['snp_double']}"

def plot_scatter(df, xcol, ycol, label, out_prefix, xlabel, ylabel):
    plt.figure(figsize=(6, 5), dpi=1200)
    plot_df = df[[xcol, ycol]].dropna()
    if plot_df.empty:
        print(f"[{label}] No data to plot.")
        return

    x = plot_df[xcol].values
    y = plot_df[ycol].values

    max_val = 10
    min_val = -15
    tick_range = np.arange(min_val, max_val + 1, 5)

    xy = np.vstack([x, y])
    z = gaussian_kde(xy, bw_method=1.2)(xy)
    z = np.log1p(z)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    norm = Normalize(vmin=z.min(), vmax=z.max())

    plt.scatter(x, y, c=z, s=8, cmap='plasma', norm=norm, alpha=0.3, edgecolors='none', rasterized=True)
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='plasma'), ax=plt.gca())
    cbar.set_label("Normalized Log-Scaled Density", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="grey", linewidth=1.2)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xticks(tick_range)
    plt.yticks(tick_range)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(label, fontsize=14)

    r, _ = pearsonr(x, y)
    plt.text(0.05, 0.95, f"r = {r:.3f}", transform=plt.gca().transAxes,
             ha="left", va="top", fontsize=8)
    print(f"[{label}] Pearson r = {r:.3f} (N={len(x)})")

    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=1200)
    plt.savefig(f"{out_prefix}.pdf")
    plt.close()

def process_delta_logit_file(csv_path, out_prefix="6_16_2025/snv_double_from_delta", deviation_threshold=1.0):
    out_dir = os.path.dirname(out_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df["snp"].notna() & (df["snp"] != "")]
    cell_lines = [col.replace("_delta_logit_pooled", "") for col in df.columns if col.endswith("_delta_logit_pooled")]
    delta_cols = [f"{cl}_delta_logit_pooled" for cl in cell_lines]
    wt_psi_cols = [f"{cl}_wt_pooled_psi_raw" for cl in cell_lines if f"{cl}_wt_pooled_psi_raw" in df.columns]
    print("Using delta_logit_pooled columns:", delta_cols)
    print("Using wt_psi columns:", wt_psi_cols)

    df["mean_delta_logit"] = df[delta_cols].mean(axis=1, skipna=True)

    bad_event_ids = set()
    if wt_psi_cols:
        wt_rows = df[df["snp"] == "none"]
        filter_mask = ~((wt_rows[wt_psi_cols] == 0.0).any(axis=1) | (wt_rows[wt_psi_cols] == 1.0).any(axis=1))
        good_event_ids = set(wt_rows[filter_mask]["event_id"])
        bad_event_ids = set(wt_rows["event_id"]) - good_event_ids
        print(f"Filtered WT PSI rows: keeping {len(good_event_ids)}, dropping {len(bad_event_ids)}")

    grouped = df.groupby("event_id")
    results = []
    for event_id, group in grouped:
        wt_row = group[group["snp"] == "none"]
        if wt_row.empty:
            continue
        snvs = group[(group["snp"] != "none") & (~group["snp"].str.contains(";"))]
        doubles = group[group["snp"].str.contains(";")]
        if len(snvs) < 2 or doubles.empty:
            continue
        double_map = {
            normalize_snp_set(row["snp"]): row
            for _, row in doubles.iterrows()
        }
        snv_list = snvs.to_dict("records")
        for i, j in itertools.combinations(range(len(snv_list)), 2):
            s1 = snv_list[i]
            s2 = snv_list[j]
            snp_key = normalize_snp_set(f"{s1['snp']};{s2['snp']}")
            double_row = double_map.get(snp_key)
            if double_row is None:
                continue
            delta1 = s1["mean_delta_logit"]
            delta2 = s2["mean_delta_logit"]
            delta_sum = delta1 + delta2
            delta_max = max(delta1, delta2)
            delta_double = double_row["mean_delta_logit"]
            if pd.isna(delta1) or pd.isna(delta2) or pd.isna(delta_double):
                continue
            results.append({
                "event_id": event_id,
                "Reference_double": double_row["Reference"],
                "Reference_snv1": s1["Reference"],
                "Reference_snv2": s2["Reference"],
                "snp_double": double_row["snp"],
                "snp_snv1": s1["snp"],
                "snp_snv2": s2["snp"],
                "delta_snv1": delta1,
                "delta_snv2": delta2,
                "delta_snv_sum": delta_sum,
                "delta_snv_max": delta_max,
                "delta_double": delta_double,
                "deviation": delta_double - delta_sum
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(f"{out_prefix}_full_results.tsv", sep="\t", index=False)

    nonadd_df = out_df[np.abs(out_df["deviation"]) > deviation_threshold]
    nonadd_df.to_csv(f"{out_prefix}_nonadditive.tsv", sep="\t", index=False)

    print(f"Total SNV1+SNV2 combos processed: {len(results)}")
    print(f"Non-additive (|Δ| > {deviation_threshold}): {len(nonadd_df)}")

    plot_scatter(
        out_df,
        "delta_snv_sum",
        "delta_double",
        "Additive Effects of SNVs",
        f"{out_prefix}_scatter_additive",
        "Sum of 2 SNV Δlogit(PSI)",
        "Double Variant Δlogit(PSI)"
    )

    plot_scatter(
        out_df,
        "delta_snv_max",
        "delta_double",
        "Max SNV vs. Double Variant",
        f"{out_prefix}_scatter_maxsnv",
        "Max SNV Δlogit(PSI)",
        "Double Variant Δlogit(PSI)"
    )

    if bad_event_ids:
        filtered_df = out_df[~out_df["event_id"].isin(bad_event_ids)].copy()
    else:
        filtered_df = out_df.copy()

    plot_scatter(
        filtered_df,
        "delta_snv_sum",
        "delta_double",
        "Additive Effects (WT PSI not 0 or 1)",
        f"{out_prefix}_scatter_additive_psi_filtered",
        "Sum of 2 SNV Δlogit(PSI)",
        "Double Variant Δlogit(PSI)"
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV file with *_delta_logit_pooled and snp info")
    parser.add_argument("--out", default="6_16_2025/snv_double_from_delta", help="Prefix for outputs")
    parser.add_argument("--threshold", type=float, default=1.0, help="Deviation threshold for non-additivity")
    args = parser.parse_args()

    process_delta_logit_file(args.csv, args.out, args.threshold)
