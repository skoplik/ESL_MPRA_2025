import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
from collections import defaultdict

def stack_overlapping_boxes(boxes, pad=1):
    boxes = sorted(boxes, key=lambda x: x[1])
    tracks = []
    for box in boxes:
        placed = False
        for track in tracks:
            if all(box[1] > b[2] + pad or box[2] < b[1] - pad for b in track):
                track.append(box)
                placed = True
                break
        if not placed:
            tracks.append([box])
    return tracks

def plot_event_motifs(event_id, gene_exon, motif_boxes, output_path, exon_start, exon_end):
    clusters = sorted(set(box[0] for box in motif_boxes))
    cmap = cm.get_cmap("tab20", len(clusters))
    cluster_colors = {c: mcolors.to_hex(cmap(i)) for i, c in enumerate(clusters)}
    stacked_tracks = stack_overlapping_boxes(motif_boxes)

    plot_height = max(2, 0.3 + 0.2 * len(stacked_tracks))
    plt.figure(figsize=(10, plot_height))

    for track_idx, track in enumerate(stacked_tracks):
        for cluster, start, stop, motif_id in track:
            color = cluster_colors.get(cluster, 'gray')
            plt.hlines(track_idx + 1, start, stop, color=color, linewidth=3)
            label = f"{motif_id} | {cluster}"
            plt.text((start + stop) / 2, track_idx + 1.05, label,
                     ha='center', va='bottom', fontsize=5, color=color)

    plt.axvline(exon_start, color='purple', linestyle='--', linewidth=0.6)
    plt.axvline(exon_end, color='orange', linestyle='--', linewidth=0.6)
    plt.text(exon_start, 0.5, 'SJ1', fontsize=6, color='purple', ha='center')
    plt.text(exon_end, 0.5, 'SJ2', fontsize=6, color='orange', ha='center')

    plt.title(f"{gene_exon} — {event_id}", fontsize=8)
    plt.xlabel("Sequence Position (0–160)", fontsize=7)
    plt.yticks([])
    plt.xlim(0, 161)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] {output_path}")

def main(input_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    print(f"[INFO] Loaded dataframe with shape: {df.shape}")

    if 'motifs_in_cluster' not in df.columns:
        print("[ERROR] Missing 'motifs_in_cluster' column. Aborting.")
        return

    print(f"[INFO] Exploding motifs, total rows before explode: {len(df)}")
    df = df[df['motifs_in_cluster'].notna()].copy()
    df['motifs_in_cluster'] = df['motifs_in_cluster'].astype(str)
    df['motif_id_list'] = df['motifs_in_cluster'].str.split(',')
    df['cluster_id'] = df['motif_id']  # keep original cluster before overwrite

    df = df.drop(columns=['motif_id'], errors='ignore')
    df = df.explode('motif_id_list')
    df = df.rename(columns={'motif_id_list': 'motif_id'})
    df['motif_id'] = df['motif_id'].astype(str)

    print(f"[INFO] Rows after explode: {len(df)}")
    print(f"[DEBUG] Unique event_ids: {df['event_id'].nunique()}")
    print(f"[DEBUG] Unique motifs total: {df['motif_id'].nunique()}")

    motif_freq = (
        df.groupby(['event_id', 'motif_id', 'motif_start', 'motif_stop'])
        .size()
        .reset_index(name='count')
    )
    
    df = df.merge(motif_freq, on=['event_id', 'motif_id', 'motif_start', 'motif_stop'])
    df_valid = df[df['count'] >= 10].copy()
    df_valid = df_valid.drop_duplicates(subset=['event_id', 'motif_id', 'motif_start', 'motif_stop'])

    grouped = defaultdict(list)
    meta = {}

    for event_id, group in df_valid.groupby("event_id"):
        gene_exon_raw = group["gene_exon"].iloc[0] if "gene_exon" in group.columns else "NA"
        gene_exon = str(gene_exon_raw) if pd.notna(gene_exon_raw) else "NA"
        exon_start = int(group["exon_start"].iloc[0]) if "exon_start" in group.columns else 80
        exon_end = int(group["exon_end"].iloc[0]) if "exon_end" in group.columns else 140
        meta[event_id] = (gene_exon.replace(" ", "_"), exon_start, exon_end)

        print(f"\n[PROCESSING] event_id: {event_id} — {len(group)} motif hits")

        for _, row in group.iterrows():
            motif_id = row['motif_id']
            cluster = row['cluster_id']
            start = int(row['motif_start'])
            stop = int(row['motif_stop'])
            grouped[event_id].append((cluster, start, stop, motif_id))

    for event_id, boxes in grouped.items():
        gene_exon, exon_start, exon_end = meta.get(event_id, ("NA", 80, 140))
        fname = f"{event_id.replace(':','_')}_{gene_exon}_motifs_by_cluster.png"
        output_path = os.path.join(output_dir, fname)
        plot_event_motifs(event_id, gene_exon, boxes, output_path, exon_start, exon_end)

if __name__ == "__main__":
    import sys
    input_csv = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_csv, output_dir)
