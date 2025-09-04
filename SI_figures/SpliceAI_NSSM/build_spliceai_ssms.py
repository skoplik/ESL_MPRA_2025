import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Base conversion maps
str_to_num = {'A': 3, 'C': 2, 'G': 1, 'T': 0}
num_to_str = {v: k for k, v in str_to_num.items()}

def parse_snp(snp):
    """Parse SNP strings, keep positions 0-based like SpliceAI."""
    if snp == "none":
        return []
    try:
        entries = snp.split(";")
        parsed = []
        for e in entries:
            pos, change = e.split(":")
            ref, alt = change.split(">")
            parsed.append((int(pos), ref.upper(), alt.upper()))  # keep 0-based
        return parsed
    except Exception:
        return []

def setup_axes(ax, title, length, splice_junctions=None):
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0.5, length + 0.5, 10))
    ax.set_xticklabels(np.arange(0, length, 10), fontsize=4)  # 0…160 labels
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(['A', 'C', 'G', 'T'], fontsize=3)
    ax.set_title(title, fontsize=6)
    ax.tick_params(axis='both', width=0.3, length=1, pad=0.5)
    if splice_junctions:
        sj1 = splice_junctions[0]
        sj2 = splice_junctions[0] + splice_junctions[1]
        ax.axvline(sj1, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(sj2, color='black', linestyle='--', linewidth=0.5)
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks([sj1, sj2])
        ax_top.set_xticklabels(["SA", "SD"], fontsize=5)
        ax_top.tick_params(width=0.5, length=10, pad=0.8)
        ax_top.spines["top"].set_visible(False)

def plot_data(fig, ax, matrix, wt_positions, cmap, vmin=-6, vmax=6):
    matrix = np.asarray(matrix, dtype='float64')
    masked = np.ma.masked_invalid(matrix)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    num_rows, num_cols = masked.shape
    wt_pos_set = set(wt_positions)

    for row in range(num_rows):
        for col in range(num_cols):
            if (col, row) in wt_pos_set:
                ax.add_patch(plt.Rectangle((col, num_rows - 1 - row), 1, 1,
                                           facecolor="black", edgecolor=None, zorder=1))
            else:
                value = masked[row, col]
                is_masked = masked.mask[row, col]
                color = 'white' if is_masked else cmap(norm(value))
                ax.add_patch(plt.Rectangle((col, num_rows - 1 - row), 1, 1,
                                           facecolor=color, edgecolor=None, zorder=1))
                if is_masked:
                    ax.text(col + 0.5, num_rows - row - 0.5, "X",
                            ha='center', va='center', fontsize=3, color='black', zorder=5)

    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    dummy = ax.imshow(np.array([[vmin, vmax]]), cmap=cmap,
                      visible=False, aspect='auto', extent=[0, 0, 0, 0])
    return dummy

def add_colorbar(fig, im, position, font_size=6, label=None):
    cax = fig.add_axes(position)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=font_size, width=0.5)
    cbar.outline.set_linewidth(0.5)
    if label:
        cbar.set_label(label, fontsize=font_size, labelpad=2)

def make_heatmap(df_event, output_prefix, cmap):
    if "none" not in df_event["snp_spliceai"].values:
        print(f"[SKIP] No WT for {df_event['event_id'].iloc[0]}")
        return

    wt_row = df_event[df_event["snp_spliceai"] == "none"].iloc[0]
    wt_logit = wt_row.get("spliceai_logit", np.nan)
    gene_name = str(wt_row.get("gene_exon", "NA")).replace(" ", "_")
    eid = wt_row["event_id"]

    intron1_len = len(str(wt_row.get("intron1", "")))
    exon_len = len(str(wt_row.get("exon", "")))
    intron2_len = len(str(wt_row.get("intron2", "")))
    splice_junctions = (intron1_len, exon_len, intron2_len)

    max_pos = 160
    mat = np.full((4, max_pos + 1), np.nan)

    # WT positions
    wt_positions = []
    wt_seq = wt_row.get("full_seq", "")
    for i, b in enumerate(wt_seq):
        if b in str_to_num:
            wt_positions.append((i, str_to_num[b]))

    for _, row in df_event.iterrows():
        if row["snp_spliceai"] == "none":
            continue
        snps = parse_snp(row["snp_spliceai"])
        if len(snps) != 1:
            continue
        pos, ref, alt = snps[0]
        if pos > max_pos or alt not in str_to_num:
            continue
        if pos < len(wt_seq) and wt_seq[pos] != ref:
            continue
        idx = str_to_num[alt]
        var_logit = row.get("spliceai_logit", np.nan)
        if pd.notnull(var_logit) and pd.notnull(wt_logit):
            dlogit = var_logit - wt_logit
            mat[idx, pos] = dlogit

    fig, ax = plt.subplots(figsize=(6, 2.5))
    im = plot_data(fig, ax, mat, wt_positions, cmap, vmin=-6, vmax=6)
    setup_axes(ax, f"{gene_name} Δlogit(PSI)", mat.shape[1], splice_junctions)
    fig.subplots_adjust(right=0.85)
    add_colorbar(fig, im, [0.87, 0.25, 0.01, 0.5], label="Δlogit(PSI)")
    os.makedirs(output_prefix, exist_ok=True)
    out_file = os.path.join(output_prefix, f"{gene_name}_spliceai_delta_logit.pdf")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

def main(spliceai_file, spliceai_missing_file, supertable_file, event_table_file, output_prefix):
    # Load main and missing spliceai predictions, concatenate
    df_spliceai = pd.read_csv(spliceai_file, sep="\t")
    df_missing = pd.read_csv(spliceai_missing_file, sep="\t")
    df_spliceai = pd.concat([df_spliceai, df_missing], ignore_index=True)

    df_spliceai["Reference"] = df_spliceai["Reference"].astype(str)
    df_spliceai = df_spliceai.rename(columns={"snp": "snp_spliceai"})

    # Load supertable
    df_super = pd.read_csv(supertable_file)
    df_super["Reference"] = df_super["Reference"].astype(str)

    # Load events filter
    df_events = pd.read_csv(event_table_file, sep="\t")
    keep_eids = set(df_events["event_id"].astype(str).unique())

    # Merge with metadata
    df = df_spliceai.merge(df_super, on=["Reference", "event_id"], how="left")
    df = df[df["event_id"].isin(keep_eids)]

    cmap = plt.cm.get_cmap("coolwarm").copy()
    cmap.set_bad("white")

    for eid, group in df.groupby("event_id"):
        make_heatmap(group, output_prefix, cmap)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 6:
        print("Usage: python script.py spliceai_psi_logit.tsv spliceai_psi_logit_missing_vars.tsv supertable.csv event_table.tsv output_dir")
        sys.exit(1)

    spliceai_file = sys.argv[1]
    spliceai_missing_file = sys.argv[2]
    supertable_file = sys.argv[3]
    event_table_file = sys.argv[4]
    output_prefix = sys.argv[5]
    main(spliceai_file, spliceai_missing_file, supertable_file, event_table_file, output_prefix)
