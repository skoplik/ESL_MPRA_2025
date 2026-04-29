import re
import pandas as pd

def clean_meme_file(meme_file_path, ban_list_csv_path, output_file_path):
    ban_list = pd.read_csv(ban_list_csv_path, header=None)[0].tolist()
    matched_banned = set()

    with open(meme_file_path, "r") as f:
        lines = f.readlines()

    cleaned_lines = []
    inside_header = True
    kept_motifs = []
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        if inside_header:
            cleaned_lines.append(line)
            if line.startswith("Background letter frequencies"):
                inside_header = False
            idx += 1
            continue

        if line.startswith("MOTIF"):
            motif_name = line.strip().split(" ", 1)[1]
            motif_core = motif_name.split(":")[-1]
            matched = [b for b in ban_list if b in motif_core]

            buffer = [line]
            idx += 1
            while idx < len(lines) and not lines[idx].startswith("MOTIF"):
                buffer.append(lines[idx])
                idx += 1

            if matched:
                matched_banned.update(matched)
            else:
                cleaned_lines.extend(buffer)
                kept_motifs.append(motif_name)
        else:
            cleaned_lines.append(line)
            idx += 1

    with open(output_file_path, "w") as f:
        f.writelines(cleaned_lines)

    unmatched = [b for b in ban_list if b not in matched_banned]
    print(f"✓ Cleaned MEME written to: {output_file_path}")
    print(f"✓ Motifs kept: {len(kept_motifs)}")
    print(f"✓ Motifs removed: {len(matched_banned)}")
    if unmatched:
        print("⚠️ Banned motifs not found in MEME file:")
        for b in unmatched:
            print(f"  - {b}")
    return kept_motifs


def deduplicate_meme_headers(input_file_path, output_file_path):
    with open(input_file_path, "r") as f:
        lines = f.readlines()

    output_lines = []
    seen_header = False
    inside_header = True

    for line in lines:
        if line.startswith("MOTIF"):
            inside_header = False
            seen_header = True
            output_lines.append(line)
        elif not seen_header:
            output_lines.append(line)
        elif not inside_header:
            output_lines.append(line)

    with open(output_file_path, "w") as f:
        f.writelines(output_lines)

    print(f"✓ Deduplicated MEME saved to: {output_file_path}")

def split_and_propagate(annot_list):
    out = []
    for annot in annot_list:
        if '/' in annot:
            out.extend(annot.split('/'))
        else:
            out.append(annot)

    final = []
    current_prefix = None
    for item in out:
        if ':' in item:
            current_prefix = item.split(':')[0]
            final.append(item)
        else:
            final.append(f"{current_prefix}:{item}" if current_prefix else item)
    return final
    

def convert_meme_to_transfac(meme_file, transfac_file):
    with open(meme_file, "r") as f:
        lines = f.readlines()

    with open(transfac_file, "w") as out:
        idx = 0
        pwm = []
        motif_map = {}
        annotations_map = {}
        motif_count = 1
        motif_full = None

        while idx < len(lines):
            line = lines[idx]
            if line.startswith("MOTIF"):
                if motif_full and pwm:
                    short = f"motif{motif_count}"
                    write_transfac_block(short, pwm, out, annotations_list=split_and_propagate(motif_full.split(';')))

                    motif_map[short] = motif_full
                    annotations_map[short] = [a.strip() for a in motif_full.split(';') if a.strip()]
                    pwm = []
                    motif_count += 1
                motif_full = line.strip().split(" ", 1)[1]
                idx += 1
            elif "letter-probability matrix" in line:
                idx += 1
                while idx < len(lines) and lines[idx].strip():
                    row = [float(v) for v in lines[idx].strip().split()]
                    pwm.append(row)
                    idx += 1
            else:
                idx += 1

        if motif_full and pwm:
            short = f"motif{motif_count}"
            write_transfac_block(short, pwm, out)
            motif_map[short] = motif_full
            annotations_map[short] = [a.strip() for a in motif_full.split(';') if a.strip()]

    print(f"✓ TRANSFAC file saved to: {transfac_file}")
    return motif_map, annotations_map


def write_transfac_block(motif_id, pwm, out, annotations_list=None):
    out.write(f"AC {motif_id}\n")
    out.write("XX\n")
    if annotations_list:
        annot_str = ",".join(annotations_list)
        out.write(f"ID {motif_id}|{annot_str}\n")
    else:
        out.write(f"ID {motif_id}\n")
    out.write("XX\n")
    out.write("P0      A      C      G      T\n")
    for i, row in enumerate(pwm, 1):
        values = ["{:6.0f}".format(x * 100) for x in row]
        out.write("{:<2} {}\n".format(i, " ".join(values)))
    out.write("XX\n//\n\n")





def save_motif_mapping(motif_map, output_path):
    df = pd.DataFrame([
        {"short_name": k, "long_name": v} for k, v in motif_map.items()
    ])
    df.to_csv(output_path, sep="\t", index=False, header=False)
    print(f"✓ Motif name map saved to: {output_path}")


def save_motif_annotations(annotations_map, output_path):
    def split_and_propagate(annot_list):
        out = []
        for annot in annot_list:
            if '/' in annot:
                out.extend(annot.split('/'))
            else:
                out.append(annot)

        # Propagate prefix to untagged entries
        final = []
        current_prefix = None
        for item in out:
            if ':' in item:
                current_prefix = item.split(':')[0]
                final.append(item)
            else:
                # Prefix-less: prepend the last seen source
                if current_prefix:
                    final.append(f"{current_prefix}:{item}")
                else:
                    final.append(item)  # fallback
        return final

    rows = []
    for k, raw_annots in annotations_map.items():
        expanded = split_and_propagate(raw_annots)
        rows.append((k, str(expanded)))

    df = pd.DataFrame(rows, columns=["motif", "annotations"])
    df.to_csv(output_path, sep="\t", index=False, header=False)
    print(f"✓ Annotations file saved to: {output_path}")



# === Run ===

cleaned = clean_meme_file(
    meme_file_path="/ESL/ESL_MPRA/Figure_6/Cluster_motifs/merged_output_11_09_24.meme",
    ban_list_csv_path="/ESL/ESL_MPRA/Figure_6/Effect_size/banned_motifs_non_human_ornament_11_21_2024.csv",
    output_file_path="/ESL/ESL_MPRA/Figure_6/Cluster_motifs/output_cleaned_pwms/filtered_motifs_no_banned.meme"
)

deduplicate_meme_headers(
    input_file_path="/ESL/ESL_MPRA/Figure_6/Cluster_motifs/output_cleaned_pwms/filtered_motifs_no_banned.meme",
    output_file_path="/ESL/ESL_MPRA/Figure_6/Cluster_motifs/output_cleaned_pwms/filtered_motifs_no_banned_deduped.meme"
)

motif_map, annotations_map = convert_meme_to_transfac(
    meme_file="/ESL/ESL_MPRA/Figure_6/Cluster_motifs/output_cleaned_pwms/filtered_motifs_no_banned_deduped.meme",
    transfac_file="/ESL/ESL_MPRA/Figure_6/Cluster_motifs/output_cleaned_pwms/filtered_motifs_final.transfac"
)

save_motif_mapping(
    motif_map,
    output_path="/ESL/ESL_MPRA/Figure_6/Cluster_motifs/output_cleaned_pwms/filtered_motif_name_map.tsv"
)

save_motif_annotations(
    annotations_map,
    output_path="/ESL/ESL_MPRA/Figure_6/Cluster_motifs/output_cleaned_pwms/filtered_motif_annotations.tsv"
)
