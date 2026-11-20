def meme_to_transfac(meme_file, transfac_file):
    with open(meme_file, 'r') as infile, open(transfac_file, 'w') as outfile:
        lines = infile.readlines()
        
        motif_id = None
        pwm = []
        recording = False

        for line in lines:
            if line.startswith("MOTIF"):
                if motif_id and pwm:
                    # Write previous motif
                    write_transfac(motif_id, pwm, outfile)
                    pwm = []
                motif_id = line.strip().split()[1]
            
            elif line.startswith("letter-probability matrix"):
                recording = True
                continue

            elif recording:
                if line.strip() == "":
                    recording = False
                    if motif_id and pwm:
                        write_transfac(motif_id, pwm, outfile)
                        pwm = []
                    continue
                values = [float(v) for v in line.strip().split()]
                pwm.append(values)

        # Write last motif
        if motif_id and pwm:
            write_transfac(motif_id, pwm, outfile)

def write_transfac(motif_id, pwm, outfile):
    outfile.write("AC {}\n".format(motif_id))
    outfile.write("XX\n")
    outfile.write("ID {}\n".format(motif_id))
    outfile.write("XX\n")
    outfile.write("P0      A      C      G      T\n")
    for i, row in enumerate(pwm, start=1):
        # Round and convert to integer percentages if needed
        scaled_row = ["{:>6.0f}".format(val * 100) for val in row]
        outfile.write("{:<2} {}\n".format(i, " ".join(scaled_row)))
    outfile.write("XX\n//\n\n")

# Example usage:
meme_to_transfac("/ESL/ESL_MPRA/Figure_6/Cluster_motifs/filtered_motifs_final_shortnames_altname.meme", "/ESL/ESL_MPRA/Figure_6/Cluster_motifs/filtered_motifs_final_shortnames_altname.transfac")
