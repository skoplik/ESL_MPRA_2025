import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

str_to_num = {'A': 3, 'C': 2, 'G': 1, 'T': 0}

# === ClinVar color palette 
CLINVAR_COLORS = {
    "Pathogenic": "red",
    "Likely_pathogenic": "orange",
    "Likely_benign": "#9ACD32",
    "Benign": "#258d4c",
    "Uncertain_significance": "slateblue",
    "Conflicting": "orchid",
}

def darken_color(color, amount=0.7):
    c = mcolors.to_rgb(color)
    return tuple(max(0, min(1, channel * amount)) for channel in c)

def _str_to_num(seq):
    return np.array([str_to_num.get(b, -1) for b in seq], dtype='int64')

def fill_cell(positions, ax=None, **kwargs):
    ax = ax or plt.gca()
    for x, y in positions:
        if y >= 0:
            rect = plt.Rectangle((x + 0.25, y + 0.25), 0.5, 0.5, fill=True, **kwargs)
            ax.add_patch(rect)

def setup_axes(ax, title, splice_junctions=None):
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0.5, 161.5, 10))
    ax.set_xticklabels(np.arange(0, 161, 10), fontsize=4)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(['A', 'C', 'G', 'T'], fontsize=2.3)
    ax.set_title(title, fontsize=6)
    ax.tick_params(axis='both', width=0.3, length=1,pad=0.5)
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

def parse_snp(snp):
    try:
        entries = snp.split(";")
        return [(int(e.split(":")[0]), e.split(":")[1].split(">")[1]) for e in entries]
    except:
        return []

def parse_clinvar_significance(clinvar_item):
    sigs = set()

    if isinstance(clinvar_item, dict) and "ClinVar_info" in clinvar_item:
        clinvar_item = clinvar_item["ClinVar_info"]

    if isinstance(clinvar_item, list):
        for entry in clinvar_item:
            if isinstance(entry, tuple) and len(entry) == 4:
                meta = entry[3]
                if "CLNSIG=" in meta:
                    for part in meta.split(";"):
                        if part.startswith("CLNSIG="):
                            raw_sigs = part.split("=")[1].replace(" ", "_").split("|")
                            sigs.update(raw_sigs)
    elif isinstance(clinvar_item, str):
        for part in clinvar_item.split(";"):
            if part.startswith("CLNSIG="):
                raw_sigs = part.split("=")[1].replace(" ", "_").split("|")
                sigs.update(raw_sigs)

    normalized = set()
    for s in sigs:
        s_lower = s.strip().lower()
        if s_lower in {"conflicting_classifications_of_pathogenicity", "conflicting"}:
            normalized.add("Conflicting")
        elif s_lower == "not_provided":
            normalized.add("Not_provided")
        else:
            normalized.add(s.replace(" ", "_"))

    if not normalized:
        return None
    elif len(normalized) > 1:
        return "Conflicting"
    else:
        return list(normalized)[0]

def plot_data(fig, ax, matrix, cmap, wt_positions, splice_junctions=None, vmin=-6, vmax=6, clinvar_labels=None):
    matrix = np.asarray(matrix, dtype='float64')
    masked = np.ma.masked_invalid(matrix)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    num_rows, num_cols = masked.shape

    wt_pos_set = set((x, y) for x, y in wt_positions)

    # Create colormap-mapped heat array
    for row in range(num_rows):
        for col in range(num_cols):
            value = masked[row, col]
            is_masked = masked.mask[row, col]
            color = 'white' if is_masked else cmap(norm(value))
            rect = plt.Rectangle(
                (col, num_rows - 1 - row), 1, 1,
                facecolor=color,
                edgecolor=None,  # Remove grid lines
                zorder=1
            )
            ax.add_patch(rect)

            # Draw 'X' if NaN and not a WT position
            if is_masked and (col, row) not in wt_pos_set:
                ax.text(
                    col + 0.5, num_rows - row-0.4, 'X',
                    ha='center', va='center',
                    fontsize=2, color='black',
                    zorder=5
                )

    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # Draw WT squares fully
    for x, y in wt_positions:
        if 0 <= x < num_cols and 0 <= y < num_rows:
            ax.add_patch(
                plt.Rectangle((x, num_rows - 1 - y), 1, 1,
                              facecolor='black', edgecolor=None, zorder=10)
            )

    # Plot ClinVar labels
    if clinvar_labels:
        for pos, label, color in clinvar_labels:
            edge_color = darken_color(color)
            ax.plot(pos + 0.5, -1, marker='o',
                    transform=ax.transData, clip_on=False,
                    color=color,alpha=0.6, markersize=1.8,
                    markeredgecolor=edge_color, markeredgewidth=0.3, zorder=12)
            ax.text(pos + 0.5, -1.8, label,
                    transform=ax.transData, clip_on=False,
                    ha='center', va='bottom', fontsize=2.5,
                    color='black', rotation=90, zorder=13)

        handles = [
            plt.Line2D([0], [0], marker='o', linestyle='None',
                       color='w', markerfacecolor=v,
                       markeredgecolor=darken_color(v),
                       markeredgewidth=0.3, markersize=4,
                       label=k.replace("_", " "))
            for k, v in CLINVAR_COLORS.items()
        ]
        leg = fig.legend(
            handles=handles,
            loc='center left',           # use middle-left of legend
            bbox_to_anchor=(0.01, 0.6),  # halfway up the figure
            fontsize=3,
            frameon=False,
            borderpad=0,
            handletextpad=0.5
        )
        
        # shrink markers only in legend (not in the plot)
        for lh in leg.legendHandles:
            lh.set_markersize(2)  # smaller legend dots


    # Dummy image for colorbar
    dummy = ax.imshow(np.array([[vmin, vmax]]), cmap=cmap, visible=False, aspect='auto', extent=[0, 0, 0, 0])
    return dummy




def add_colorbar(fig, im, position, font_size=6, label=None):
    cax = fig.add_axes(position)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=font_size, width=0.5)
    cbar.outline.set_linewidth(0.5)
    if label:
        cbar.set_label(label, fontsize=font_size, labelpad=2)  # labelpad adjusts spacing


def make_heatmap(df_event, output_prefix, cmap, cmap_red):
    if "none" not in df_event["snp"].values:
        print(f"[SKIP] No WT for {df_event['event_id'].iloc[0]}")
        return

    wt_row = df_event[df_event["snp"] == "none"].iloc[0]
    seq = wt_row["full_seq"]
    wt_positions = [[i, b] for i, b in enumerate(_str_to_num(seq)) if b >= 0]
    splice_junctions = (len(wt_row["intron1"]), len(wt_row["exon"]), len(wt_row["intron2"]))
    gene_name = wt_row["gene_exon"]
    gene_name_file = "NA" if pd.isna(gene_name) else str(gene_name).replace(" ", "_")
    eid = wt_row["event_id"]
    single_mutants = df_event[df_event["snp"] != "none"]
    single_mutants = single_mutants[single_mutants["seq_type"] == 'single']
    total_valid = len(single_mutants)

    if total_valid < 150:
        print(f"[SKIP] {eid} — insufficient single mutants ({total_valid})")
        return

    cell_lines = ["HEK", "HeLa", "K562", "HMC3", "MCF7"]
    mat_stack, dpsi_stack = [], []
    clinvar_labels = []
    seen = set()

    for cl in cell_lines:
        mat = np.full((4, 161), np.nan)
        dpsi_mat = np.full((4, 161), np.nan)
        wt_logit = float(wt_row.get(f"{cl}_pooled_logit", np.nan))
        wt_psi = float(wt_row.get(f"{cl}_pooled_psi_raw", np.nan))

        for _, row in single_mutants.iterrows():
            ref_id = row["Reference"]
            snp = parse_snp(row["snp"])
            if not snp:
                continue
            pos, alt = snp[0]
            idx = str_to_num.get(alt, -1)
            if not (0 <= pos < 161 and idx >= 0):
                continue

            var_logit = float(row.get(f"{cl}_pooled_logit", np.nan))
            var_psi = float(row.get(f"{cl}_pooled_psi_raw", np.nan))
            if np.isnan(var_logit) or np.isnan(wt_logit) or np.isnan(var_psi) or np.isnan(wt_psi):
                continue

            mat[idx, pos] = np.nanmean([mat[idx, pos], var_logit - wt_logit]) if not np.isnan(mat[idx, pos]) else var_logit - wt_logit
            dpsi_mat[idx, pos] = np.nanmean([dpsi_mat[idx, pos], var_psi - wt_psi]) if not np.isnan(dpsi_mat[idx, pos]) else var_psi - wt_psi

            if pd.notna(row.get("mut_ref_mut_positions")):
                try:
                    mut_positions = ast.literal_eval(row["mut_ref_mut_positions"])
                    ref_base = wt_row["full_seq"][pos]
                    sig = parse_clinvar_significance(row.get("ClinVar_item", ""))
                    if sig in CLINVAR_COLORS:
                        for clin_pos, (clin_ref, clin_alt) in mut_positions:
                            if clin_pos == pos and clin_ref == ref_base:
                                key = (pos, clin_ref, sig)
                                if key not in seen:
                                    seen.add(key)
                                    matching_alts = [a for cp, (r, a) in mut_positions if cp == pos and r == clin_ref]
                                    unique_alts = sorted(set(matching_alts))
                                    label = f"{clin_ref}>{','.join(unique_alts)}"
                                    color = CLINVAR_COLORS[sig]
                                    clinvar_labels.append((pos, label, color))
                except Exception as e:
                    print(f"[WARN] Failed to parse ClinVar info for ref {ref_id}: {e}")

        mat_stack.append(mat)
        dpsi_stack.append(dpsi_mat)

        print(f"[PLOT] {eid} ({cl}) — {total_valid} single mutants")
        out_dir = os.path.join(output_prefix, cl)
        os.makedirs(out_dir, exist_ok=True)

        for name, matrix, vmin, vmax in [("delta_logit", mat, -6, 6), ("dpsi", dpsi_mat, -1, 1)]:
            fig, ax = plt.subplots(figsize=(6, 2.5))
            im = plot_data(fig, ax, matrix, cmap, wt_positions, splice_junctions, vmin=vmin, vmax=vmax, clinvar_labels=clinvar_labels)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            setup_axes(ax, f"{gene_name} {eid} {name}", splice_junctions)
            fig.subplots_adjust(right=0.85)
            add_colorbar(fig, im, [0.87, 0.25, 0.01, 0.5], label="Δlogit(PSI)")
            plt.savefig(os.path.join(out_dir, f"{gene_name_file}_{eid}_{cl}_{name}.pdf"), dpi=300, bbox_inches="tight")
            plt.close()

    if len(mat_stack) >= 2:
        print(f"[PLOT] {eid} (ALL cell lines)")
        all_out = os.path.join(output_prefix, "ALL")
        os.makedirs(all_out, exist_ok=True)

        for label, stack, vmin, vmax in [("delta_logit", mat_stack, -6, 6), ("dpsi", dpsi_stack, -1, 1)]:
            avg = np.nanmean(stack, axis=0)
            fig, ax = plt.subplots(figsize=(6, 2.5))
            im = plot_data(fig, ax, avg, cmap, wt_positions, splice_junctions, vmin=vmin, vmax=vmax, clinvar_labels=clinvar_labels)
            setup_axes(ax, f"{gene_name} {eid} {label} (ALL)", splice_junctions)
            fig.subplots_adjust(right=0.85)
            add_colorbar(fig, im, [0.87, 0.25, 0.01, 0.5])
            plt.savefig(os.path.join(all_out, f"{gene_name_file}_{eid}_ALL_{label}.pdf"), dpi=300, bbox_inches="tight")
            plt.close()

def main(input_csv, output_prefix, clinvar_csv):
    df = pd.read_csv(input_csv)
    df_clinvar = pd.read_csv(clinvar_csv, sep="\t")
    df_clinvar["ClinVar_item"] = df_clinvar["ClinVar_item"].apply(ast.literal_eval)
    df_merged = df.merge(df_clinvar, left_on="Reference", right_on="mut_ref", how="left")
    cmap = plt.cm.get_cmap("coolwarm").copy()  # use coolwarm and draw X for NaNs
    cmap.set_bad("white")
    cmap_red = LinearSegmentedColormap.from_list("white_red", ["white", "red"])

    for eid, group in df_merged.groupby("event_id"):
        make_heatmap(group, output_prefix, cmap, cmap_red)

if __name__ == "__main__":
    import sys
    input_csv = sys.argv[1]
    output_prefix = sys.argv[2]
    clinvar_csv = sys.argv[3]
    main(input_csv, output_prefix, clinvar_csv)
