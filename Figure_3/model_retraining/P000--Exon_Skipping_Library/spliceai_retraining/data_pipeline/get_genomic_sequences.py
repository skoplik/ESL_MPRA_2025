
def skip_chromosome(buff):
    """
    skip the current chromosome
    second output denotes if the eof 
    was reached (empty string)
    """
    while True:
        line = buff.readline()
        if line and line[0] != ">":
            continue
        elif line:
            eof = False
            return line, eof
        else:
            eof = True
            return line, eof


def is_valid_chromosome(chrom_names, line):
    if line and line[0] != ">":
        return True
    elif not line:
        return False
    for chrom in chrom_names:
        if chrom in line:
            return True
    return False


def get_chrom_name(line):
    """
    e.g.
    parse >NC_00001.11 ... into chrom_name
    """
    try:
        chrom_name = line.split(".")[0].split(">")[1]
    except IndexError:
        breakpoint()
        " index error "
    return chrom_name


def parse_chromosome(chrom_list, file_pointer, chrom_name = ""):
    """
    Does the main heavy lifting of actually 
    going through the sequences and extracting
    those at the correct positions
    
    input: chrom_list where chrom_list[i][0] = start
                        and chrom_list[i][1] = end
    for each sequence i

    file_pointer TextIOWrapper to read from with file_pointer.read_line()
    reads it line-by-line to not result in OOM issues.
    
    file pointer will start one line after the chromosome header.
    
    output: The sequences extracted from the start, end positions
    for each item in chrom_list, including any extra fields after the 
    first two (i.e. for identifying the sequence source).
             The next chromosome header if any exists.

    """ 
    print("CALL PARSE CHROMOSOME", chrom_name)
    file_reads = 0
    debug = False
    curr_pos = 1
    all_sequences = []
    buff = [] # create buffer in case sequence overlaps
    for i in range(len( chrom_list)):
        # set up for finding the next sequence
        start = chrom_list[i][0]
        end = chrom_list[i][1]
        extra = chrom_list[i][2:]
        sequence = []
        buff_swap = []
        for buff_pos, buff_seq in buff:
            if buff_pos <= start <= buff_pos + len(buff_seq):
                buff_swap.append((buff_pos, buff_seq))
                sequence.append(buff_seq[start - buff_pos:])
            elif buff_pos >= start and buff_pos <= end:
                buff_swap.append((buff_pos, buff_seq))
                sequence.append(buff_seq)
            elif buff_pos + len(buff_seq) >= start:
                buff_swap.append((buff_pos, buff_seq))
        if buff_swap and debug:
            print(i, "using some buffer with size", len(buff_swap))
        buff = buff_swap
        # start reading from file
        while True:
            if (i == 4930 or i == 4929) and chrom_name == "NC_000003":
                debug = True
                " bug fix for empty sequence "
            else:
                debug = False
            if debug:
                print("current postion", curr_pos, "i", i)
            line = file_pointer.readline().split("\n")[0]
            file_reads += len(line)
            if file_reads != curr_pos + 79:
                print("file reads", file_reads, "curr_pos", curr_pos, "i", i)
                breakpoint()
                " unaligned curr pos"
            if debug:
                print("line len", len(line))
            if not line:
                _=[print("breaking due to empty line") if debug else ""]
                break
            if line[0] == ">":
                _=[print("breaking due to new chrom") if debug else ""]
                break

            ### Conditions
            if curr_pos >= start and curr_pos <= end: 
                # Can add line
                _=[print("case 1") if debug else ""]
                sequence.append(line.upper())
                buff.append((curr_pos, line.upper()))
                curr_pos += len(line)
            elif curr_pos + len(line) >= start and curr_pos <= end:
                # Can add line, but must align with start first
                _=[print("case 2") if debug else ""]
                indx = (start - curr_pos)
                sequence.append(line[indx:].upper())
                buff.append((curr_pos, line.upper()))
                curr_pos += len(line)
            elif curr_pos > end:
                buff.append((curr_pos, line.upper()))
                curr_pos += len(line)
                _=[print("breaking due to curr_pos > end") if debug else ""]
                break
            elif curr_pos < start:
                _=[print("case 3") if debug else ""]
                curr_pos += len(line)
            else:
                breakpoint()
                raise ValueError("Logical error in while loop")
        sequence_clipped = "".join(sequence)[:(end - start) + 1]
        _=[print("len seq clipped", len(sequence_clipped), "i", i) if debug else ""]
        if not len(sequence_clipped):
            breakpoint()
            "empty sequence" 
        all_sequences.append((start, end, *extra, "".join(sequence)[:(end - start) + 1]))
    if not line:
        return all_sequences, line
    while line and line[0] != ">":
        # throw out rest of chromosome if all sequences found
        line = file_pointer.readline()
    return all_sequences, line


def chrom_pos(file_, chrom_names):

    """
    takes in a fna file as input and 
    returns the sequences that start at pos
    and have size size

    for tpslice, chrom_names is a dictionary mapping the
    chromosome name to a list of (start, end, strand, id)

    chrom names need to be sorted by start

    """
    chrom = []
    curr_pos = 1
    all_chromosome_sequences = []
    with open(file_, "r") as f:
        # Get the header and check if the chromosome exists
        # in the queries
        line = f.readline() # reads the first chromosome
        while line:
            if not is_valid_chromosome(chrom_names, line):
                line, is_eof = skip_chromosome(f)
                continue 
            else:
                chrom_name = get_chrom_name(line)
                chromosome_sequences, line = parse_chromosome(chrom_names[chrom_name], f, chrom_name)

            all_chromosome_sequences.append(chromosome_sequences)
    return all_chromosome_sequences



if __name__ == "__main__":
    # file_ = "/storage/yu/Projects/P001--JTE-607_Sequence_Design/star/ncbi_dataset/data/GCF_000001405.26/hg38_and_plasmid_l3.fna"
    # pos = 73372929
    # size = 642
    # pos2 = 73374000
    # pos2end = pos2 + 100
    # chrom_name = "NC_000006.12"
    # chrom_names = {"NC_000006":[(pos, pos + size - 1), (pos2, pos2end)]}
    # chrom = (chrom_pos(file_, chrom_names))
    # print(chrom)
    
    file_ = "/storage/yu/Projects/P000--ESL_continuation/GitHub/P000--Exon_Skipping_Library/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna"
    pos1 = 248813163
    pos1end = 248814448
    chrom_name = "NC_000001.11"
    chrom_names = {"NC_000001":[(pos1, pos1end, 29, "GT_00062")]}
    chrom = (chrom_pos(file_, chrom_names))
    print(chrom)
    print(len(chrom[0][0][0]))
