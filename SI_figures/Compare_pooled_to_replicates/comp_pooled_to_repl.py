import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from sklearn.metrics import r2_score

def plot_all_reps_vs_pooled(df, label, output_dir):
    pooled_col = f"{label}_pooled_logit"
    rep_cols = [col for col in df.columns if col.startswith(f"{label}_rep") and col.endswith("logit")]

    df = df.dropna(subset=[pooled_col])

    # === 1. Row-wise: Individual Replicates vs Pooled ===
    plt.figure(figsize=(7, 7))
    colors = plt.cm.tab10.colors
    for i, col in enumerate(rep_cols):
        cur_df = df[[col, pooled_col]].dropna()
        if cur_df.empty:
            continue
        r2 = r2_score(cur_df[pooled_col], cur_df[col])
        plt.scatter(cur_df[col], cur_df[pooled_col], label=f"{col} (R²={r2:.3f}), n = {len(cur_df[pooled_col])}",
                    alpha=0.3, s=40, color=colors[i % len(colors)])

    plt.axline((0, 0), slope=1, linestyle='--', color='gray')
    plt.xlabel("Replicate Logit PSI")
    plt.ylabel("Pooled Logit PSI")
    plt.title(f"{label} | Replicate vs Pooled Logit PSI (Row-wise)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{label}_all_reps_vs_pooled_rowwise.png"))
    plt.close()

    # === 2. Row-wise: Mean Replicate vs Pooled ===
    df['mean_repl_logit'] = df[rep_cols].mean(axis=1)
    mean_df = df[[pooled_col, 'mean_repl_logit']].dropna()
    if not mean_df.empty:
        r2_mean = r2_score(mean_df[pooled_col], mean_df['mean_repl_logit'])

        plt.figure(figsize=(7, 7))
        plt.scatter(mean_df['mean_repl_logit'], mean_df[pooled_col], alpha=0.3)
        plt.axline((0, 0), slope=1, linestyle='--', color='gray')
        plt.xlabel("Mean Replicate Logit PSI")
        plt.ylabel("Pooled Logit PSI")
        plt.title(f"{label} | Mean Replicates vs Pooled Logit PSI (Row-wise)\nR² = {r2_mean:.3f} \nn = {len(mean_df['mean_repl_logit'])}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{label}_mean_repl_vs_pooled_rowwise.png"))
        plt.close()

    # === 3. Event-wise: Mean per Event (Mean Repl vs Pooled) ===
    if 'event_id' not in df.columns:
        print("Missing 'event_id' column in input file — skipping event-wise plot")
        return

    eventwise = df.groupby('event_id')[[pooled_col] + rep_cols].mean().dropna()
    if not eventwise.empty:
        eventwise['mean_repl_logit'] = eventwise[rep_cols].mean(axis=1)
        r2_eventwise = r2_score(eventwise[pooled_col], eventwise['mean_repl_logit'])

        plt.figure(figsize=(7, 7))
        plt.scatter(eventwise['mean_repl_logit'], eventwise[pooled_col], alpha=0.3)
        plt.axline((0, 0), slope=1, linestyle='--', color='gray')
        plt.xlabel("Mean Replicate Logit PSI (per event_id)")
        plt.ylabel("Mean Pooled Logit PSI (per event_id)")
        plt.title(f"{label} | Mean Replicates vs Pooled Logit PSI (Event-wise)\nR² = {r2_eventwise:.3f} \nn = {len(eventwise['mean_repl_logit'])}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{label}_mean_repl_vs_pooled_eventwise.png"))
        plt.close()

def main():
    if len(sys.argv) != 4:
        print("Usage: python comp_pooled_to_repl.py <input_file.csv> <label> <output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    label = sys.argv[2]
    output_dir = sys.argv[3]

    os.makedirs(output_dir, exist_ok=True)
    print(f"→ Loading {input_file}")
    df = pd.read_csv(input_file)

    print(f"→ Generating plots for: {label}")
    plot_all_reps_vs_pooled(df, label, output_dir)
    print(f"→ Plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
