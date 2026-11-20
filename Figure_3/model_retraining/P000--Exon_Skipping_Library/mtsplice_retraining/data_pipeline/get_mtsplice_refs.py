if __name__ == "__main__":
    import sys
    import os
    PROJECT_NAME = "P000--Exon_Skipping_Library"
    node = os.getcwd()
    for i in range(5): # max depth
        tree, node = os.path.split(node)
        if node == PROJECT_NAME:
            break 
        else:
            node = tree
    PROJECT_ROOT = os.path.join(tree, node)
    if node != PROJECT_NAME:
        raise ModuleNotFoundError(f"Could not find project directory {PROJECT_NAME}")
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
from src.load_data import process_psi_csv, read_csvs
from src.load_data import VARIANT_PATH, PSI_PATH, SUPERTABLE_PATH
cell_type_psi = ['HEK293_PSI', 'HeLa_PSI', 'K562_PSI', 'MCF7_PSI', 'HMC3_PSI']


class MTSpliceRefPSI():
    def __init__(self, splice_vs_super_path = VARIANT_PATH, merged_PSI_path = PSI_PATH, supertable_file = SUPERTABLE_PATH, splice_vs_super_double_variants = None, in_all_cells_matchedrefvar = None, supertable = None, n_cell_types = 5):
        if splice_vs_super_double_variants is None or in_all_cells_matchedrefvar is None or supertable is None:
            splice_vs_super_double_variants,\
            merged_PSI_cov_logit_df,\
            supertable = read_csvs(splice_vs_super_path, merged_PSI_path, supertable_file)
        if in_all_cells_matchedrefvar is None:
            in_all_cells_matchedrefvar, in_all_cells_matchedrefvar_no_nan = process_psi_csv(merged_PSI_cov_logit_df)
        self.ref_eventID_supertableRef_dict = supertable["ref_eventID_supertableRef_dict"]
        self.ref_to_id = dict(zip(in_all_cells_matchedrefvar['Reference'], in_all_cells_matchedrefvar['event_id']))
        all_refs = in_all_cells_matchedrefvar['Reference'].tolist()
        in_all_cells_matchedrefvar.reset_index()
        self.ref_to_indx = {all_refs[i] : i for i in range(len(all_refs))}
        self.cell_type_psi_numpy = in_all_cells_matchedrefvar[(cell_type_psi)].to_numpy()
        self.n_cell_types = n_cell_types
        return 
    
    def find_cell_ref_psi(self, ref_nums):
        cell_ref_psi = np.zeros((len(ref_nums), self.n_cell_types))
        for i in range(len(ref_nums)):
            # get the event_id 
            event_id = self.ref_to_id[ref_nums[i]]
            # get the correct reference
            ref = self.ref_eventID_supertableRef_dict[event_id]
            # get the psi values
            psi_indx = self.ref_to_indx[int(ref)]
            psi = self.cell_type_psi_numpy[psi_indx]
            cell_ref_psi[i] += psi
        return cell_ref_psi


if __name__ == "__main__":
    references = MTSpliceRefPSI()
    cell_ref_psi=references.find_cell_ref_psi([1,2])
    print(cell_ref_psi)
