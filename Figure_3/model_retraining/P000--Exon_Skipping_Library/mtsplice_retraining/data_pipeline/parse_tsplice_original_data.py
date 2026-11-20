import pickle
import time
import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from spliceai_retraining.data_pipeline.get_genomic_sequences import chrom_pos
from mtsplice_retraining.data_pipeline.data_split_one_hot_tsplice import encodeDNA, process_sequences_mt

cell_types = ["HEK293", "HeLa", "K562", "MCF7", "HMC3"]
cell_types2 = [i if i != "HEK293" else "HEK" for i in cell_types]
TISSUES = [
    'Retina - Eye', 'RPE/Choroid/Sclera - Eye', 'Subcutaneous - Adipose',
    'Visceral (Omentum) - Adipose', 'Adrenal Gland', 'Aorta - Artery',
    'Coronary - Artery', 'Tibial - Artery', 'Bladder', 'Amygdala - Brain',
    'Anterior cingulate - Brain', 'Caudate nucleus - Brain',
    'Cerebellar Hemisphere - Brain', 'Cerebellum - Brain', 'Cortex - Brain',
    'Frontal Cortex - Brain', 'Hippocampus - Brain', 'Hypothalamus - Brain',
    'Nucleus accumbens - Brain', 'Putamen - Brain',
    'Spinal cord (C1) - Brain', 'Substantia nigra - Brain',
    'Mammary Tissue - Breast', 'EBV-xform lymphocytes - Cells',
    'Leukemia (CML) - Cells', 'Xform. fibroblasts - Cells',
    'Ectocervix - Cervix', 'Endocervix - Cervix', 'Sigmoid - Colon',
    'Transverse - Colon', 'Gastroesoph. Junc. - Esophagus',
    'Mucosa - Esophagus', 'Muscularis - Esophagus', 'Fallopian Tube',
    'Atrial Appendage - Heart', 'Left Ventricle - Heart', 'Cortex - Kidney',
    'Liver', 'Lung', 'Minor Salivary Gland', 'Skeletal - Muscle',
    'Tibial - Nerve', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate',
    'Not Sun Exposed - Skin', 'Sun Exposed (Lower leg) - Skin',
    'Ileum - Small Intestine', 'Spleen', 'Stomach', 'Testis', 'Thyroid',
    'Uterus', 'Vagina', 'Whole Blood'
]

def process_sequences_mt(exon, intron1, intron2):
    original_sizes = (len(exon), len(intron1), len(intron2))
    if len(exon) < 100:
        exon_L = exon + "m"*(100-len(exon))
        exon_R = "m"*(100-len(exon)) + exon
    else:
        exon_L = exon[:100]
        exon_R = exon[-100:]
    assert len(exon_L) == len(exon_R) == 100
    if len(intron1) + len(SMN2_intron6) < 300: # if intron is really short
        padding = 300 - len(intron1) - len(SMN2_intron6)
        intron1 = "N" * padding + SMN2_intron6 + intron1
    elif len(intron1) < 300:
        intron1 = (SMN2_intron6 + intron1)[-300:]
    if len(intron2) < 300:
        intron2 = (intron2 + SMN2_intron7)[:300]
    if len(intron1) > 300:
        intron1 = intron1[-300:]
    if len(intron2) > 300:
        intron2 = intron2[:300]
    assert len(intron1) == len(intron2) == 300, f"Expected processed introns 1, 2 to be of size 300, 300 but got {len(intron1)},  {len(intron2)}. The original inputs to process_sequences_mt has the following sizes: exon size = {original_sizes[0]}, intron1 size = {original_sizes[1]}, intron2 size = {original_sizes[2]}"
    seqL = intron1 + exon_L
    seqR = exon_R + intron2
    return seqL, seqR


def process_sequences_ascot_mt(exon, intron1, intron2, context):
    """
    Support for two context modes
    SMN2 mode adds context from the synthetic intron
    genomic mode assumes the genomic context is already
    added and simply removes some part of the intron.
    It is necessary to query the genome with enough context,
    given introns have variable sizes.
    """
    match context:
        case "SMN2":
            return process_sequences_mt(exon, intron1, intron2)
        case "genomic":
            if len(exon) < 100:
                exon_L = exon + "m"*(100-len(exon))
                exon_R = "m"*(100-len(exon)) + exon
            else:
                exon_L = exon[:100]
                exon_R = exon[-100:]
            assert len(exon_L) == len(exon_R) == 100
            intron1 = intron1[-300:]
            intron2 = intron2[:300]
            assert len(intron1) == len(intron2) == 300
            seqL = intron1 + exon_L
            seqR = exon_R + intron2
            assert len(seqL) == len(seqR) == 400
            return seqL, seqR
        case _:
            raise ValueError(f"Expected context to be either SMN2 or genomic, got {context}")
    return 


def ncbi_chromosome_format(path, include_version_number = False):
    """
    Map the chromosome symbol to the hg38 symbol

    """
    keys = {}
    with open(path) as f:
        for line in f:
            if not line:
                break
            i = 4 # skip >NC_
            num = ""
            ver = ""
            while line[i] == "0": # skip leading zeros
                i += 1
            while line[i] != ".": # split the chrom nmber and version number
                num += line[i]
                i += 1
            while line[i] != " ": # ignore extra information
                ver += line[i]
                i += 1
            n = 6 - len(num)
            ncbi_name = f"NC_{'0' * n}{num}" # put into ncbi format e.g. NC_000001
            if include_version_number:
                ncbi_name += ver # e.g. NC_000001.11
            if int(num) == 23:
                num = "X"
            elif int(num) == 24:
                num = "Y"
            keys["chr"+num] = ncbi_name
    return keys


def sort_chrom_names(chrom_names):
    for chrom in chrom_names:
        chrom_names[chrom] = sorted(chrom_names[chrom], key = lambda x : x[0])
    return chrom_names


def get_chrom_names(path, cassette_exons_only = True, positive_strand_only = True, context="SMN2"):
    """

    Input: a pandas file containing the tsplice GTEx cell type
        specific PSIs from the ASCOT dataset
        (https://snaptron.cs.jhu.edu/data/ascot/)
    Output: a dictionary of chromosome names to start end end positions 
           , exon bounds and gene id
    """
    df=pd.read_csv(path)
    if cassette_exons_only:
        df = df[df["cassette_exon"] == "Yes"]
    if positive_strand_only:
        df = df[df["exon_strand"] == "+"]
    chromosomes = df["exon_location"].tolist()
    chromosome_names = [i.split(":")[0] for i in chromosomes]
    sample_ids = [i for i in range(len(df))]
    exon_ids = df["exon_id"].tolist()
    assert len(set(sample_ids)) == len(df), ["gene ids", len(set(sample_ids)), "df", len(df)]
    start_ends = [i.split(":")[1].split("-") for i in chromosomes]
    start_ends = [(int(i[0].split("|")[0]), int(i[1].split("|")[0])) for i in start_ends]
    bounds = df["exon_boundary"].tolist()
    bounds = [i.split(":")[1].split("-") for i in bounds]
    bounds = [(int(i[0]), int(i[1])) for i in bounds]
    psi_data =  df[df.keys()[12:]].to_numpy()
    strand = df["exon_strand"].tolist()
    chrom_names = {i:[] for i in chromosome_names}
    match context:
        case "SMN2":
            extra = 0
        case "genomic":
            extra = 300
        case _:
            raise ValueError(f"Expected context to be either SMN2 or genomic, got {context}")
    for i in range(len(df)):
        chrom_names[chromosome_names[i]].append((
                                             bounds[i][0] - extra, # 0
                                             bounds[i][1] + extra, # 1
                                             start_ends[i][0],     # 2
                                             start_ends[i][1],     # 3
                                             sample_ids[i],        # 4
                                             exon_ids[i],          # 5
                                             chromosome_names[i],  # 6
                                             strand[i],            # 7
                                           ))
    data_index_lookup = {"bounds_start" : 0,\
                         "bounds_end" : 1, \
                         "exon_start": 2,\
                         "exon_end": 3,\
                         "sample_id": 4,\
                         "exon_id" : 5,\
                         "chromosome" : 6, \
                         "strand" : 7, \
                         "full_sequence" : 8,\
                        }    

    return chrom_names, data_index_lookup, df


def validate_sequence_lengths(data, psi_data_path, data_index, df, context):
    bounds = df["exon_boundary"].tolist()
    bounds = [i.split(":")[1].split("-") for i in bounds]
    bounds = [(int(i[0]), int(i[1])) for i in bounds]
    extra = 0
    if context == "genomic":
        extra += 300
    
    for i, chromosome in enumerate(data):
        for j, sample in enumerate(chromosome):
            exon_id = sample[data_index["exon_id"]]
            sample_id = sample[data_index["sample_id"]]
            seq = sample[data_index["full_sequence"]]
            bound_query_start, bound_query_end = bounds[sample_id]
            bound_query_start -= extra
            bound_query_end += extra
            assert len(seq) == bound_query_end - bound_query_start + 1, ("i", i, "j", j, "len", len(seq), "true len", bound_query_end - bound_query_start + 1) 
    return 


# mapping of nucleotide pairing
complement_dict = {
    "A":"T",
    "U":"A",
    "G":"C",
    "C":"G",
    "T":"A",
    "N":"N",
    "R":"Y",
    "Y":"R",
    "M":"T", #normally M pairs with K, but M is the neutral alphabet in encodeDNA, so set M = A
    "K":"A",
    }


def rev_complement(seq):
    complement = []
    for i in range(len(seq) - 1, -1, -1):
        complement.append(complement_dict[seq[i]])
    return "".join(complement)



def split_sequences_to_intron_exon(sequence, exon_bounds, sequence_bounds, strand, chromosome):
    """
    Gets a sequence and its bounds,
    retruning the exon and intron1 (upstream / 5')
    and intron 2 (downstream / 3')
    """
    assert strand == "+" or strand == "-", f"expected strand to be '+' or '-' but got {strand} instead"
    exon_start_index = exon_bounds[0] - sequence_bounds[0]
    exon_end_index = exon_bounds[1] - sequence_bounds[0] 
    exon = sequence[exon_start_index:exon_end_index + 1]
    if strand == "+": 
        intron1 = sequence[:exon_start_index]
        intron2 = sequence[exon_end_index + 1:]
    elif strand == "-":
        # 5' and 3' introns are swapped
        intron2 = sequence[:exon_start_index]
        intron1 = sequence[exon_end_index + 1:]
        # get the reverse complement of each
        intron1 = rev_complement(intron1)
        intron2 = rev_complement(intron2)
        exon    = rev_complement(exon)
    return intron1, exon, intron2


def tsplice_metadata_to_csv(metadata, metadata_path):
    metadict = {
    "sample_id":[],
    "chromosome_id":[],
    "processed_sample_id":[],
    "exon_id":[],
    "chromosome":[],
    }
    for i in metadata:
        metadict["sample_id"].append(i[0])
        metadict["chromosome_id"].append(i[1])
        metadict["processed_sample_id"].append(i[2])
        metadict["exon_id"].append(i[3])
        metadict["chromosome"].append(i[4])
    df = pd.DataFrame(metadict)
    df.to_csv(metadata_path, index = False)
    return


def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return


def tsplice_data_to_csv(
        seq_L,
        seq_R,
        metadata,
        data_dir,
        metadata_path
    ):
        
    seq_L_concat = np.concatenate(seq_L, axis = 0)
    seq_R_concat = np.concatenate(seq_R, axis = 0)
    os.makedirs(data_dir, exist_ok=True)
    
    save_pickle(seq_L_concat, os.path.join(data_dir, "seq_L.p"))
    save_pickle(seq_R_concat, os.path.join(data_dir, "seq_R.p"))
    tsplice_metadata_to_csv(metadata, metadata_path)
    return 


def extract_psi_values(psi_df, sample_ids, cell_type_tissues):
    """
    Reorder to match the sequence order
    Replace -1.0 with NaN,
    then get mean, scaled to 0-1
    """
    psi_df_reordered = psi_df.iloc[sample_ids]
    psi_df_reordered_replaced = psi_df_reordered.replace(-1.0, np.nan)
    cell_type_psi = {}
    for cell in cell_type_tissues:
        tissue_PSIs = cell_type_tissues[cell]
        cell_type_psi[cell] = np.mean(psi_df_reordered_replaced[tissue_PSIs], axis = 1) / 100
    return cell_type_psi


def concatenate_psi_values(cell_to_psi_map):
    out_matrix = []
    for cell in cell_types2: # ensure correct order
        psi_tissues = cell_to_psi_map[cell+"_tissues"]
        if len(psi_tissues.shape) == 1:
            psi_tissues = np.reshape(psi_tissues, (psi_tissues.shape[0], 1)) 
        out_matrix.append(psi_tissues)
    return np.hstack(out_matrix)


def get_final_tsplice_data(data, data_indexes, context):
    start_time = time.perf_counter()
    """
    splits sequences into intron and exon and then
    puts in the SMN intron context from the ESL 
    
    """
    key_errors = []
    previous_size = 0
    metadata = []
    seqL_one_hot = []
    seqR_one_hot = []
    for chromosome_id, chromosome in enumerate(data):
        for processed_sample_id, sample in enumerate(chromosome):
            if len(seqL_one_hot) < previous_size:
                raise ValueError(f"Wrong size, {chromosome_id}, {processed_sample_id}")
            previous_size = len(seqL_one_hot)
            if len(seqL_one_hot) != len(seqR_one_hot):
                breakpoint()
                " mismatch L and R shapes "
            sequence = sample[data_indexes["full_sequence"]]
            bounds_start = sample[data_indexes["bounds_start"]]
            bounds_end = sample[data_indexes["bounds_end"]]
            exon_start = sample[data_indexes["exon_start"]]
            exon_end = sample[data_indexes["exon_end"]]
            exon_id = sample[data_indexes["exon_id"]]
            chromosome = sample[data_indexes["chromosome"]]
            strand = sample[data_indexes["strand"]]
            sequence_bounds = (bounds_start, bounds_end)
            exon_bounds = (exon_start, exon_end)
            intron1, exon, intron2 = split_sequences_to_intron_exon(
                                                                     sequence,\
                                                                     exon_bounds,\
                                                                     sequence_bounds,\
                                                                     strand,\
                                                                     chromosome)       
            try:
                seqL, seqR = process_sequences_ascot_mt(exon = exon, \
                                                  intron1 = intron1, \
                                                  intron2 = intron2, \
                                                  context = context) 
            except AssertionError as e:
                print(f"assertion error at sequence {chromosome_id, processed_sample_id}")
                breakpoint()
                raise e
            
            try:
                L = (encodeDNA(seqL))
                R = (encodeDNA(seqR))
                seqL_one_hot.append(L)
                seqR_one_hot.append(R)
            except KeyError as e:
                for letter in ["R", "W", "Y"]:
                    for sequence in [(seqL, "seqL"), (seqR, "seqR")]:
                        if letter in sequence[0]:
                            key_errors.append(f"{letter} in {sequence[1]} at data[{chromosome_id}][{processed_sample_id}]")
            
                seqL=seqL.replace("R", "A") # no good way to represent pyrimidine without 
                                            # knowing the PSI values for each.
                                            # fortunately only one of the sequences in the 
                                            # genome has this issue, and only at a single
                                            # nucleotide. Potentially could one-hot encode
                                            # as 0.5 for A and 0.5 for G, but this would
                                            # require a different encoding function which
                                            # would create inconsistensies with the MMSplice
                                            # code and the original MTSplice utils.
                seqL=seqL.replace("W", "A") # likewise for W, only a single sequence appears
                seqR=seqR.replace("Y", "T") # likewise for Y, only a single sequence appears
                seqL_one_hot.append(encodeDNA(seqL))
                seqR_one_hot.append(encodeDNA(seqR))
                if len(seqL_one_hot) != len(seqR_one_hot):
                    breakpoint()
                    " mismatch L and R shapes "

            # could add more identifiers, but it should still be possible
            # to recover any information as necessary
            metadata.append((sample[data_indexes["sample_id"]], # sample id in filtered df 
                                                 chromosome_id, # chromosome num in chrom_names
                                                 processed_sample_id, # sample index in chrom_name
                                                 exon_id,
                                                 chromosome)) 
    
    if key_errors:
        print(f"WARNING: replaced some non-canonical nucleotides:")
        print("\n".join(key_errors))
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    return seqL_one_hot, seqR_one_hot, metadata


def data_split_ascot_tsplice(metadata, psi_df): 
    """
    Reproduce the data splits from the paper
    'MTSplice predicts eﬀects of genetic variants on
    tissue-specific splicing' Cheng et al. 2020 Materials and Methods
    secition 

    training:   chromosomes 4, 6, 8, 10 to 23, X, Y
    validation: chromosomes 1, 7, 9
    test:       chromosomes 2, 3, 5

    returns a list of indexes from the metadata 
    (corresponds to seqL, seqR and psi data indexes)
    """
    train_nums = ["4", "6", "8", "X", "Y"]
    for i in range(10, 24):
        train_nums.append(str(i))
    train_nums = set(train_nums)
    val_nums = set(["1", "7", "9"])
    test_nums = set(["2","3", "5"])
    assert len(test_nums & train_nums) == len(val_nums & train_nums) == len(val_nums & test_nums) == 0
    train_split_ids = []
    val_split_ids = []
    test_split_ids = []
    for metadata_index, info in enumerate(metadata):
        sample_id = info[0]
        chromosome = psi_df.iloc[sample_id]["exon_location"].split(":")[0]
        chromosome_num = chromosome.split("chr")[1]
        if chromosome_num in train_nums:
            train_split_ids.append(metadata_index)
        elif chromosome_num in val_nums:
            val_split_ids.append(metadata_index)
        elif chromosome_num in test_nums:
            test_split_ids.append(metadata_index)
        else:
            raise ValueError(f"Unrecognized chromosome number '{chromosome_num}' when trying to set to train/val/test split")
    return train_split_ids, val_split_ids, test_split_ids


def execute_split(train_split_ids, val_split_ids, test_split_ids, *data):
    train_splitted = []
    val_splitted = []
    test_splitted = []
    for dataset in data:
        train_splitted.append(dataset[train_split_ids])
        val_splitted.append(dataset[val_split_ids])
        test_splitted.append(dataset[test_split_ids])
    return train_splitted, val_splitted, test_splitted 
        


def clip(x, clip_threshold=0.01):
    return np.clip(x, clip_threshold, 1 - clip_threshold)


def logit(x, clip_threshold=0.01):
    x = clip(x, clip_threshold=clip_threshold)
    return np.log(x) - np.log(1 - x)


def cell_delta_logit(cell_type_data, clip_threshold=0.01):
    """
    Turns numpy array of (samples x cell_types)
    into delta logits by the formula:
    dlogit(psi)_i = logit(cell_i_psi) - logit(mean(cell_i_psi))
    """
    if np.max(cell_type_data) > 1.0:
        "CHECK PSI"
        breakpoint()
        cell_type_data /= 100
    clipped_psi = clip(cell_type_data, clip_threshold=clip_threshold)
    cell_type_mean = np.nanmean(clipped_psi, axis=1).reshape((clipped_psi.shape[0],1))
    logit_data = logit(clipped_psi, clip_threshold=clip_threshold)
    dlogit_data = logit_data - logit(cell_type_mean, clip_threshold=clip_threshold)
    return dlogit_data


if __name__ == "__main__":
    psi_data_path = "/storage/yu/Projects/P000--ESL_continuation/GitHub/P000--Exon_Skipping_Library/models/data/gtex_psi.csv"
    genome_data_path = "/storage/yu/Projects/P000--ESL_continuation/GitHub/P000--Exon_Skipping_Library/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna"
    # fixing strand information, saving to tsplice_ascot_data2
    out_directory = "/storage/yu/Projects/P000--ESL_continuation/Data/ascot/tsplice_ascot_56"
    cassette_exons_only = True
    positive_strand_only = False
    drop_val_test_nans = False
    context = "genomic"
    output_cells = 56
    clip_threshold = 0.00001
    
    chrom_names, data_index_lookup, psi_df = get_chrom_names(psi_data_path, cassette_exons_only = cassette_exons_only, positive_strand_only=positive_strand_only, context=context)

    count = 0
    keys = ncbi_chromosome_format("fna_chromosome_names.txt")
    chrom_renamed = {}
    for i in chrom_names:
        if i in keys:
            chrom_renamed[keys[i]] = chrom_names[i]
    chrom_names = chrom_renamed
    del chrom_renamed
    for i in chrom_names:
        print(i)
        count = 0
        for j in chrom_names[i]:
            print(j)
            count +=1
            if count > 4:
                break
    chrom_names = sort_chrom_names(chrom_names)

    breakpoint()
    " reading from genome "
    data = chrom_pos(genome_data_path, chrom_names)   
    # (sequence, bounds start, bounds end, exon start, exon end, gene ids) 
    # validate sequence reading from genome was correct
    # validate_sequence_lengths(data, psi_data_path, data_index_lookup, psi_df, context)

    seq_L, seq_R, metadata = get_final_tsplice_data(data, data_index_lookup, context)

    tsplice_data_to_csv(seq_L, seq_R, metadata, out_directory, os.path.join(out_directory, "metadata.csv"))
    
    # get the psi values for ESL cell lines using the sample ids and tissue approximations
    sample_ids = [i[0] for i in metadata]
    cell_type_tissues = {
        "HEK_tissues"  : ["Cortex - Kidney"],
        "HeLa_tissues" : ["Uterus", 'Ectocervix - Cervix', 'Endocervix - Cervix'],
        "K562_tissues" : ["Leukemia (CML) - Cells"],
        "MCF7_tissues" : ["Mammary Tissue - Breast"],
        "HMC3_tissues" : [tissue for tissue in TISSUES if "Brain" in tissue], # average over all brain tissues
    }

    if output_cells == 5:
        y_psi = extract_psi_values(psi_df, sample_ids, cell_type_tissues)
        y_psi = concatenate_psi_values(y_psi)
    elif output_cells == 56:
        y_psi = psi_df.iloc[sample_ids]
        y_psi = y_psi.replace(-1.0, np.nan)
        y_psi = y_psi[TISSUES] / 100
        y_psi = y_psi.to_numpy()


    breakpoint()
    " check y_psi "

    seq_L_concat = np.stack(seq_L, axis = 0) # is currently a list, so change into numpy array
    seq_R_concat = np.stack(seq_R, axis = 0)

    # get the indexes to split seqL, seqR, psi
    train_split_ids, val_split_ids, test_split_ids = data_split_ascot_tsplice(metadata, psi_df)
    breakpoint()
    " testing split execution "

    train, val, test = execute_split(train_split_ids, val_split_ids, test_split_ids, seq_L_concat, seq_R_concat, y_psi) 

    train_y_dlogit = cell_delta_logit(train[2], clip_threshold=clip_threshold)
    val_y_dlogit = cell_delta_logit(val[2], clip_threshold=clip_threshold)
    test_y_dlogit = cell_delta_logit(test[2], clip_threshold=clip_threshold)

    from collections import Counter

    acceptors = {"+": Counter(), "-": Counter()}
    donors = {"+": Counter(), "-": Counter()}

    for i, meta in enumerate(metadata):
        sample_id = meta[0]
        strand = psi_df.iloc[sample_id]["exon_strand"]

        acceptors[strand][decoded_seqL[i][298:300]] += 1
        donors[strand][decoded_seqR[i][100:102]] += 1

    print(acceptors)
    print(donors)


    breakpoint()
    " test process NaNs "
    ### Processing NaNs
    # like with the ESL data, drop in train if all values are NaN
    # and drop in val/test if any values are NaN
    train_non_nan = np.sum(np.isnan(train_y_dlogit), axis = 1) != output_cells
    train_split_ids = [train_split_ids[i] for i in range(len(train_non_nan)) if train_non_nan[i]]

    if drop_val_test_nans:
        val_non_nan = np.sum(np.isnan(val_y_dlogit), axis = 1) == 0
        val_split_ids = [val_split_ids[i] for i in range(len(val_non_nan)) if val_non_nan[i]]

        test_non_nan = np.sum(np.isnan(test_y_dlogit), axis = 1) == 0
        test_split_ids = [test_split_ids[i] for i in range(len(test_non_nan)) if test_non_nan[i]]
    else:
        val_non_nan = np.sum(np.isnan(val_y_dlogit), axis = 1) != output_cells
        val_split_ids = [val_split_ids[i] for i in range(len(val_non_nan)) if val_non_nan[i]]

        test_non_nan = np.sum(np.isnan(test_y_dlogit), axis = 1) != output_cells
        test_split_ids = [test_split_ids[i] for i in range(len(test_non_nan)) if test_non_nan[i]]

    
    # recompute the splits with the dropped NaNs
    train, val, test = execute_split(train_split_ids, val_split_ids, test_split_ids, seq_L_concat, seq_R_concat, y_psi) 

    # extract the final splits and metadata
    train_x = [train[0], train[1]]
    train_y_psi = [train[2]]
    
    val_x = [val[0], val[1]]
    val_y_psi = [val[2]]

    test_x = [test[0], test[1]]
    test_y_psi = [test[2]]

    train_y_dlogit = [cell_delta_logit(train[2], clip_threshold=clip_threshold)]
    val_y_dlogit   = [cell_delta_logit(val[2], clip_threshold=clip_threshold)]
    test_y_dlogit  = [cell_delta_logit(test[2], clip_threshold=clip_threshold)]

    metadata_train = [metadata[i] for i in train_split_ids]
    metadata_val = [metadata[i] for i in val_split_ids]
    metadata_test = [metadata[i] for i in test_split_ids]


    size_train = train[0].shape[0]
    size_val = val[0].shape[0]
    size_test = test[0].shape[0]
    total_size = size_train + size_val + size_test
    percentage_train = size_train * 100 / total_size
    percentage_val = size_val * 100 / total_size
    percentage_test = size_test * 100 / total_size

    print("data sizes")
    print(f"number of samples in train: {size_train} ({percentage_train}%)")
    print(f"number of samples in val: {size_val} ({percentage_val}%)")
    print(f"number of samples in test: {size_test} ({percentage_test}%)")

    breakpoint()
    " will now save data "

    ### Save the final splits and metadata
    tsplice_metadata_to_csv(metadata_train,  os.path.join(out_directory, "metadata_train.csv"))
    tsplice_metadata_to_csv(metadata_val, os.path.join(out_directory, "metadata_val.csv"))
    tsplice_metadata_to_csv(metadata_test, os.path.join(out_directory, "metadata_test.csv"))

    save_pickle(y_psi, os.path.join(out_directory, "y_psi.p"))
    save_pickle(train_x, os.path.join(out_directory, "train_x.p"))
    save_pickle(train_y_psi, os.path.join(out_directory, "train_y_psi.p"))
    save_pickle(train_y_dlogit, os.path.join(out_directory, "train_y_dlogit.p"))
    save_pickle(val_x, os.path.join(out_directory, "val_x.p"))
    save_pickle(val_y_psi, os.path.join(out_directory, "val_y_psi.p"))
    save_pickle(val_y_dlogit, os.path.join(out_directory, "val_y_dlogit.p"))
    save_pickle(test_x, os.path.join(out_directory, "test_x.p"))
    save_pickle(test_y_psi, os.path.join(out_directory, "test_y_psi.p"))
    save_pickle(test_y_dlogit, os.path.join(out_directory, "test_y_dlogit.p"))
    print("done") 
