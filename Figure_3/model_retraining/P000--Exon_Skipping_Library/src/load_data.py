import os
import pickle

## Import Project
if __name__ == "__main__":
    import sys
    import os
    import getopt
    opts, args = getopt.getopt(sys.argv[1:],"", [
    "project_name="
    ]) # see if project name is provided, else use default project name
    opts = dict(opts)
    try:
        # try to import (e.g. if PYTHONPATH is set or installed via pip install -e)
        import src 
    except ModuleNotFoundError:
        # search parent directories for project name
        project_name = opts.get("--project_name", "P000--Exon_Skipping_Library")
        node = os.getcwd()
        for i in range(5): # max depth
            tree, node = os.path.split(node)
            if node == PROJECT_NAME:
                break 
            else:
                node = tree
        PROJECT_ROOT = os.path.join(tree, node)
        if node != PROJECT_NAME:
            # couldn't find project folder 
            raise ModuleNotFoundError(f"Could not find project directory {PROJECT_NAME}")
        if PROJECT_ROOT not in sys.path:
            sys.path.append(PROJECT_ROOT)

from src.esl import SMN2_INTRON6, SMN2_INTRON7, logit, cell_type_psi
from src.esl import cell_types, cell_types2
from collections import defaultdict
import pandas as pd

VARIANT_PATH = "/storage/yu/Projects/P000--ESL_continuation/Data/MMSplice_2025_07_23/double_variants/mmsplice_vs_supertable_vcfid_filtered.csv"

PSI_PATH = "/storage/yu/Projects/P000--ESL_continuation/Figures_SK/General_preprocessing/output_7_13_2025/07_18_2025_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"

SUPERTABLE_PATH = "/storage/yu/Projects/P000--ESL_continuation/Figures_SK/General_preprocessing/fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv"

MERGED_PATH = "/storage/yu/Projects/P000--ESL_continuation/Figures_SK/General_preprocessing/output_04_26_2026/merged_04_26_2026.csv"


def get_average_coverage(cov_df, c):
    cell_types = [i + "_coverage" for i in c]
    total = cov_df[cell_types].mean(axis=1)
    return total


def get_average_psi(cov_df, c):
    cell_types = [i + "_PSI" for i in c]
    total = cov_df[cell_types].mean(axis=1)
    return total


def read_supertable(supertable_file=SUPERTABLE_PATH, eid_col = 1):
    # Split sequences into groupings based on reference sequence, and notate which groups are only population variants (i.e. no ClinVar annotation in supertable.tsv)
    ref_supertable_dict = {}  # supertable reference -> seq_tuple
    ref_eventID_supertableRef_dict = {}  # supertable eventID -> supertable reference
    var_supertable_dict = defaultdict(list)  # supertable reference -> list of variants reference num
    ref_supertable_dict = {}  # supertable variant reference -> supertable reference
    supertable_gene_snp = {}  # supertable reference -> gene + SNP + event_id
    gene_name_dict = {}
    exon_only_var_supertable_ref_list = []
    intron_only_var_supertable_ref_list = []
    both_exon_intron_var_supertable_ref_list = []
    exac_geuvadis_refs = []
    ref_supertable_exons = {}
    sequence_name_list = []
    sequence_list = []

    # First pass read through of supertable for all references
    with open(supertable_file, "r") as f:
        f.readline()  # throw out header
        for line in f:
            var = line.split(",")
            curr_supertable_ref = str(int(var[0]) + 1)
            event_id = var[eid_col]
            gene_name = var[3]
            snp = var[8]
            exon = var[4]
            intron1 = var[5]
            intron2 = var[6]
            fullseq_supertable = intron1 + exon + intron2
            seq_tuple = (exon, intron1, intron2)
            source = var[10]
            # parse reference
            if snp == "none":
                ref_supertable_dict[curr_supertable_ref] = seq_tuple
                if event_id not in ref_eventID_supertableRef_dict:
                    ref_eventID_supertableRef_dict[event_id] = curr_supertable_ref
                supertable_gene_snp[curr_supertable_ref] = (gene_name, snp, event_id)
                # store sequences for getting correct coordinate
                if fullseq_supertable not in sequence_list:
                    sequence_name_list.append(var[0])
                    sequence_list.append(fullseq_supertable)
                    ref_supertable_exons[var[0]] = (exon, intron1, intron2)
            gene_name_dict[curr_supertable_ref] = gene_name
            # collect Exac and Geuvadis source references
            if source == "exac" or source == "geuvadis":
                exac_geuvadis_refs.append(curr_supertable_ref)

    # Second pass read through of supertable for all variants
    variant_positions = {}  # supertable ref -> [var_index]
    supertable_var_to_ref_dict = {}  # supertable var reference -> supertable reference
    var_supertable_seq_dict = {} # supertable var reference -> var intron1, exon, intron2
    with open(supertable_file, "r") as f:
        f.readline()  # throw out header
        for line in f:
            var = line.split(",")
            curr_supertable_ref = str(int(var[0]) + 1)
            event_id = var[eid_col]
            gene_name = var[3]
            snp = var[8]
            exon = var[4]
            intron1 = var[5]
            intron2 = var[6]
            fullseq_supertable = intron1 + exon + intron2
            seq_tuple = (exon, intron1, intron2)
            # parse ref
            if snp != "none":
                curr_supertable_ref2 = ref_eventID_supertableRef_dict[event_id]
                ref_exon, ref_5p_intron, ref_3p_intron = ref_supertable_dict[curr_supertable_ref2]
                # Ignore annotated variants but subsequence is the same as ref
                if fullseq_supertable == ref_5p_intron + ref_exon + ref_3p_intron:
                    continue
                # Count number of exon-only, intron-only, or both exon+intron variants in supertable
                if exon != ref_exon and intron1 == ref_5p_intron and intron2 == ref_3p_intron:
                    exon_only_var_supertable_ref_list.append(curr_supertable_ref)
                elif exon == ref_exon and (intron1 != ref_5p_intron or intron2 != ref_3p_intron):
                    intron_only_var_supertable_ref_list.append(curr_supertable_ref)
                else:
                    both_exon_intron_var_supertable_ref_list.append(curr_supertable_ref)
                # get variance positions
                var_positions = [i for i, chr_tup in enumerate(zip(fullseq_supertable, ref_5p_intron + ref_exon + ref_3p_intron)) if chr_tup[0] != chr_tup[1]]
                variant_positions[curr_supertable_ref] = var_positions
                
                # group variants by their reference number
                var_supertable_dict[curr_supertable_ref2].append(curr_supertable_ref)
                var_supertable_seq_dict[curr_supertable_ref] = (intron1, exon, intron2)
                supertable_gene_snp[curr_supertable_ref] = (gene_name, snp, event_id)
                ref_supertable_dict[int(curr_supertable_ref)] = int(curr_supertable_ref2)
                

    
                supertable_var_to_ref_dict[curr_supertable_ref] = curr_supertable_ref2

    out = {}
   
    out["ref_supertable_dict"] = ref_supertable_dict
    out["ref_eventID_supertableRef_dict"] =  ref_eventID_supertableRef_dict
    out["var_supertable_dict"] =  var_supertable_dict
    out["ref_supertable_dict"] =  ref_supertable_dict
    out["supertable_gene_snp"] = supertable_gene_snp
    out["gene_name_dict"] =  gene_name_dict
    out["exon_only_var_supertable_ref_list"] =   exon_only_var_supertable_ref_list
    out["intron_only_var_supertable_ref_list"] = intron_only_var_supertable_ref_list
    out["both_exon_intron_var_supertable_ref_list"] =  both_exon_intron_var_supertable_ref_list
    out["exac_geuvadis_refs"] =  exac_geuvadis_refs
    out["ref_supertable_exons"] =  ref_supertable_exons
    out["sequence_name_list"] =  sequence_name_list
    out["sequence_list"] =  sequence_list
    out["variant_positions"] = variant_positions
    out["var_supertable_seq_dict"] = var_supertable_seq_dict

    return out



def get_supertable_seq_dict(supertable_file, SMN2_intron6 = SMN2_INTRON6, SMN2_intron7 = SMN2_INTRON7):
    # Defining inputs
    # PSI_output_dir = "/JTE-607/ESL/Analysis/STAR_alignment/separate/"
    # supertable_file = "/storage/yu/Projects/P000--ESL_continuation/Figures_SK/General_preprocessing/fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv"

    # Load supertable and extract module seqs in expected order: 
    # inputs = [acceptor_intron_model.input, acceptor_model.input, exon_model.input, donor_model.input, donor_intron_model.input]
    supertable_seq_dict = {}
    overhang_5p = 129
    overhang_3p = 100


    with open(supertable_file, "r") as f:
        header = f.readline().split(",")  # throw out header
        for line in f:
            var = line.split(",")
            curr_supertable_ref = str(int(var[0]) + 1)
            exon = var[4]
            intron1 = var[5]
            intron2 = var[6]
            # ignoring long intron1 variants
            if len(intron1) > overhang_5p or len(intron2) > overhang_3p:
                continue
            acceptor_intron = (SMN2_intron6 + intron1)[-100:-6]
            acceptor = (SMN2_intron6 + intron1)[-50:] + exon[:3]
            donor = exon[-5:] + intron2[:13]
            donor_intron = (intron2 + SMN2_intron7)[6:100]

            supertable_seq_dict[curr_supertable_ref] = (acceptor_intron, acceptor, exon, donor, donor_intron)    

    return supertable_seq_dict


def read_csvs(splice_vs_super_path = VARIANT_PATH, merged_PSI_path = PSI_PATH, supertable_file = SUPERTABLE_PATH):
    splice_vs_super_double_variants = pd.read_csv(splice_vs_super_path)
    merged_PSI_cov_logit_df = pd.read_csv(merged_PSI_path)
    supertable = read_supertable(supertable_file)
    return splice_vs_super_double_variants, merged_PSI_cov_logit_df, supertable


def process_psi_csv(merged_PSI_cov_logit_df):
    # Load PSI data from each cell line
    cell_type_psi = ['HEK293_PSI', 'HeLa_PSI', 'K562_PSI', 'MCF7_PSI', 'HMC3_PSI']

    # load data
    merged_PSI_cov_logit_df["HEK_coverage"] = merged_PSI_cov_logit_df["HEK_pooled_included"] + merged_PSI_cov_logit_df["HEK_pooled_excluded"]
    merged_PSI_cov_logit_df["HeLa_coverage"] = merged_PSI_cov_logit_df["HeLa_pooled_included"] + merged_PSI_cov_logit_df["HeLa_pooled_excluded"]
    merged_PSI_cov_logit_df["K562_coverage"] = merged_PSI_cov_logit_df["K562_pooled_included"] + merged_PSI_cov_logit_df["K562_pooled_excluded"]
    merged_PSI_cov_logit_df["MCF7_coverage"] = merged_PSI_cov_logit_df["MCF7_pooled_included"] + merged_PSI_cov_logit_df["MCF7_pooled_excluded"]
    merged_PSI_cov_logit_df["HMC3_coverage"] = merged_PSI_cov_logit_df["HMC3_pooled_included"] + merged_PSI_cov_logit_df["HMC3_pooled_excluded"]

    in_all_cells_matchedrefvar = merged_PSI_cov_logit_df[['Reference', 'full_seq', \
                                            'HEK_pooled_psi_clipped', 'HEK_pooled_logit', 'HEK_coverage', \
                                                 'HeLa_pooled_psi_clipped', 'HeLa_pooled_logit', 'HeLa_coverage', \
                                            'K562_pooled_psi_clipped', 'K562_pooled_logit', 'K562_coverage', \
                                                 'MCF7_pooled_psi_clipped', 'MCF7_pooled_logit', 'MCF7_coverage', \
                                            'HMC3_pooled_psi_clipped', 'HMC3_pooled_logit', 'HMC3_coverage', "event_id", "snp",\
                                                         "intron1", "intron2", "exon"]]
    in_all_cells_matchedrefvar = in_all_cells_matchedrefvar.rename(columns={'HEK_pooled_psi_clipped': 'HEK293_PSI', 'HeLa_pooled_psi_clipped': 'HeLa_PSI', 'K562_pooled_psi_clipped': 'K562_PSI', \
                                                'MCF7_pooled_psi_clipped': 'MCF7_PSI', 'HMC3_pooled_psi_clipped': 'HMC3_PSI', \
                                               'HEK_pooled_logit': 'HEK293_logit', 'HeLa_pooled_logit': 'HeLa_logit', 'K562_pooled_logit': 'K562_logit',\
                                               'MCF7_pooled_logit': 'MCF7_logit', 'HMC3_pooled_logit': 'HMC3_logit', 'full_seq': 'sequence'})

    in_all_cells_matchedrefvar_no_nan = in_all_cells_matchedrefvar.dropna(subset = cell_type_psi).copy()
    # in_all_cells_matchedrefvar["Differential"] = in_all_cells_matchedrefvar["Reference"].isin(hits_df["Reference"])
    in_all_cells_matchedrefvar["Avg_logit"] = in_all_cells_matchedrefvar[["HEK293_PSI", "HeLa_PSI", "K562_PSI", "MCF7_PSI", "HMC3_PSI"]].mean(axis=1).apply(logit)
    in_all_cells_matchedrefvar["Avg_coverage"] = get_average_coverage(in_all_cells_matchedrefvar, cell_types2)
    in_all_cells_matchedrefvar["Avg_PSI"] = get_average_psi(in_all_cells_matchedrefvar, cell_types)

    in_all_cells_matchedrefvar_no_nan["Avg_logit"] = in_all_cells_matchedrefvar_no_nan[["HEK293_PSI", "HeLa_PSI", "K562_PSI", "MCF7_PSI", "HMC3_PSI"]].mean(axis=1).apply(logit)
    in_all_cells_matchedrefvar_no_nan["Avg_coverage"] = get_average_coverage(in_all_cells_matchedrefvar_no_nan, cell_types2)
    in_all_cells_matchedrefvar_no_nan["Avg_PSI"] = get_average_psi(in_all_cells_matchedrefvar_no_nan, cell_types)
    for i in cell_types:
        in_all_cells_matchedrefvar[f"{i}_delta_logit_mean_cell"] = in_all_cells_matchedrefvar[f"{i}_logit"] - in_all_cells_matchedrefvar["Avg_logit"]
        in_all_cells_matchedrefvar_no_nan[f"{i}_delta_logit_mean_cell"] = in_all_cells_matchedrefvar_no_nan[f"{i}_logit"] - in_all_cells_matchedrefvar_no_nan["Avg_logit"]
    in_all_cells_matchedrefvar[[f"{i}_PSI" for i in cell_types] + [f"{i}_delta_logit_mean_cell" for i in cell_types] + ["Avg_logit", "Avg_coverage", "Avg_PSI"]]
    return in_all_cells_matchedrefvar, in_all_cells_matchedrefvar_no_nan


def get_loaded_pickle(data_dir, data_names):
    """
    Helper function to load pickled numpy arrays.
    Mostly used for loading splits from disk.
    """
    dict_ = {}
    if not os.path.exists(os.path.join(data_dir, "x_train.p")):
        raise FileNotFoundError(os.path.join(data_dir, "x_train.p"))
        return False, {}
    else:
        for name in data_names:
            path = os.path.join(data_dir, name)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    dict_[name.split(".")[0]] = pickle.load(f)
            else:
                print("path", path, "does not exist")
    return True, dict_


def load_splits_dir(data_dir, data_names = None, extension = None):
    """
    Improved version of get_loaded_pickle
    loads both .p files and .csv files
    accepts specific extension
    accepts data names (only certain files)
    returns result of trying to load each file
    and a dictionary of each file name to the corresponding
    file
    """
    if data_names is not None:
        files = data_names
    else:
        files = os.listdir(data_dir)
    if extension is not None:
        files = [i for i in files if extension in i]
    loaded_result = {}
    loaded_files = {}
    for doc in files:
        path = os.path.join(data_dir, doc)
        if "." in doc:
            curr_extension = doc.split(".")[-1]
        else:
            loaded_result[doc] = False
            continue
        if not os.path.exists(path):
            loaded_result[doc] = False
        name = ".".join(doc.split(".")[:-1])
        if curr_extension == "p" or curr_extension == "pickle":
            with open(path, "rb") as f:
                loaded_files[name] = pickle.load(f)
                loaded_result[name] = True
        elif curr_extension == "csv":
            loaded_files[name] = pd.read_csv(path) 
            loaded_result[name] = True
        else:
            loaded_result[doc] = False
            continue
    return loaded_result, loaded_files




if __name__ == "__main__":
    print("testing data read")
