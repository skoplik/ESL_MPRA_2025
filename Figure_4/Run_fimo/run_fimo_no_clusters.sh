#!/bin/bash

input_file="/ESL/Figures_SK/Cluster_motifs/output_cleaned_pwms/filtered_motifs_final.meme"
output_dir="/ESL/Figures_SK/effect_sizes_debug/fimo_not_on_clusters_no_rc/FIMO_idv_memes_04_30_2025_no_rc"

mkdir -p "$output_dir"

echo "Splitting MEME file into individual motif files..."

# Split file by MOTIF blocks into motif1.meme, motif2.meme, ... with ACGU alphabet
awk -v outdir="$output_dir" '
BEGIN {
    count = 0
    writing = 0
    header = "MEME version 5.5.0\n\nALPHABET= ACGU\n\nstrands: + -\n\nBackground letter frequencies:\nA 0.25 C 0.25 G 0.25 U 0.25"
}
/^MOTIF/ {
    count++
    filename = outdir "/motif" count ".meme"
    print header > filename
    print "" >> filename
    print $0 >> filename
    writing = 1
    next
}
writing {
    print $0 >> filename
}
' "$input_file"

echo "Split completed: individual .meme files written to $output_dir"

# Paths for FIMO run
fimopath="/ESL/Figures_SK/effect_sizes_debug/fimo_not_on_clusters_no_rc"
outpath="$fimopath/out_04_30_2025_no_rc"
memepath="$output_dir"
input_dna_fasta="$fimopath/input_fimo.fasta"
input_rna_fasta="$fimopath/input_fimo_rna.fasta"

mkdir -p "$outpath"

# Convert input FASTA from DNA (T) to RNA (U)
echo "Converting FASTA from T to U..."
sed 's/T/U/g' "$input_dna_fasta" > "$input_rna_fasta"

echo "Starting FIMO runs..."

# Run FIMO in batches of 2
max_jobs=2
job_count=0

for meme_file in $(find "$memepath" -name 'motif*.meme' | sort -V); do
    base=$(basename "$meme_file" .meme)
    echo "Running FIMO on $base"
    fimo --verbosity 1 --norc --o "$outpath/$base" "$meme_file" "$input_rna_fasta" &
    ((job_count++))

    if (( job_count >= max_jobs )); then
        wait
        job_count=0
    fi
done

wait
echo "All FIMO runs completed."

# Merge fimo.tsv outputs
echo "Merging fimo.tsv files..."

merged_output="$fimopath/merged_fimo_output_clusters.tsv"
first=1

for fimo_file in "$outpath"/motif*/fimo.tsv; do
    if [[ -f "$fimo_file" ]]; then
        if (( first )); then
            grep -v '^#' "$fimo_file" > "$merged_output"
            first=0
        else
            grep -v '^#' "$fimo_file" | tail -n +2 >> "$merged_output"
        fi
    else
        echo "Warning: $fimo_file not found."
    fi
done

echo "Merged output saved to $merged_output"
