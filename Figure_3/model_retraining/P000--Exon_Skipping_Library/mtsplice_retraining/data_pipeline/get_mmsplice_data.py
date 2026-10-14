import pandas as pd
import numpy as np


# necessary to make columns into empty strings for assignment later
def prepare_seqs(x):
    x["acceptor_intron"] = [""] * len(x)
    x["acceptor"] = [""] * len(x)
    x["donor"] = [""] * len(x)
    x["donor_intron"] = [""] * len(x)
    return x


# run process_sequences on exon and introns to 
# get the appropriate donor and acceptors
def get_seqs(x):
    acceptor_intron, acceptor, exon, donor, donor_intron = process_sequences(x["exon"], x["intron1"], x["intron2"])
    x["acceptor_intron"] = acceptor_intron
    x["acceptor"] = acceptor
    x["exon"] = exon
    x["donor"] = donor
    x["donor_intron"] = donor_intron
    return 


# process the sequences as defined by MMSplice
def process_sequences(exon, intron1, intron2):
    acceptor_intron = (SMN2_intron6 + intron1)[-100:-6]
    acceptor = (SMN2_intron6 + intron1)[-50:] + exon[:3]
    donor = exon[-5:] + intron2[:13]
    donor_intron = (intron2 + SMN2_intron7)[6:100]
    return (acceptor_intron, acceptor, exon, donor, donor_intron)


def get_sequences_from_families_mm(dataset, splice_vs_super_full, ref_eventID_supertableRef_dict, var_supertable_dict, ref_supertable_dict):
    # ref_eventID_supertableRef_dict : event_id ->  reference num
    # var_supertable_dict : reference num -> (exon, intron1, intron2)
    # var_supertable_dict : reference num -> list(variant nums)
    # dataset : list of families in the form (family_size, total_avg_coverage, event_id)
    grouped_df = splice_vs_super_full.groupby("event_id").groups
    seqs_df = splice_vs_super_full
    seqs_df = prepare_seqs(seqs_df)
    splice_types = ["acceptor_intron", "acceptor", "exon", "donor", "donor_intron"]
    seqs_df = seqs_df[["intron1", "intron2"] + splice_types]
    seqs_df.apply(get_seqs, axis = 1)
    outputs = ["Supertable_Mean_Delta_Logit_PSI", "MMSplice_Delta_Logit_PSI"]
    for i in outputs:
        seqs_df[i] = splice_vs_super_full[i]
    info_out = splice_types + outputs
    data_variants = {}
    logits_ = []
    mmsplice_predictions = []
    for i in splice_types:
        data_variants[i] = []
    data_refs = {}
    for i in splice_types:
        data_refs[i] = []
    indx = 0
    for data in dataset:
        event_id = data[2]
        ref_num  = ref_eventID_supertableRef_dict[event_id]
        variant_nums = grouped_df[event_id]
        variants_family_df = seqs_df.iloc[variant_nums][info_out]
        ref_seq = process_sequences(*ref_supertable_dict[ref_num])
        curr_ref_seq_by_type = {}
        for i in range(len(ref_seq)):
            curr_ref_seq_by_type[splice_types[i]] = ref_seq[i]
        # populate the data with the sequences
        for i in splice_types:
            data_variants[i] += variants_family_df[i].tolist()
            data_refs[i] += [curr_ref_seq_by_type[i]] * len(variant_nums)
        logits_ += variants_family_df["Supertable_Mean_Delta_Logit_PSI"].tolist()
        mmsplice_predictions += variants_family_df["MMSplice_Delta_Logit_PSI"].tolist()    
    return data_variants, data_refs, logits_, mmsplice_predictions 



def get_refs_from_families_mm(dataset, splice_vs_super_full, ref_eventID_supertableRef_dict):
    # ref_eventID_supertableRef_dict : event_id ->  reference num
    # dataset : list of families in the form (family_size, total_avg_coverage, event_id)
    # splice_vs_super_full -> data containing filtered variants with no reference
    grouped_df = splice_vs_super_full.groupby("event_id").groups
    seqs_df = splice_vs_super_full
    outputs = ["Supertable_Mean_Delta_Logit_PSI",\
               "MMSplice_Delta_Logit_PSI",\
               "Reference",\
               "event_id",\
               "snp",]
    info_out = outputs
    logits_ = []
    mmsplice_predictions = []
    variant_ids = []
    ref_nums = []
    event_ids = []
    snps = []
    indx = 0
    for data in dataset:
        event_id = data[2]
        ref_num  = ref_eventID_supertableRef_dict[event_id]
        variant_nums = grouped_df[event_id]
        variants_family_df = seqs_df.iloc[variant_nums][info_out]
        logits_ += variants_family_df["Supertable_Mean_Delta_Logit_PSI"].tolist()
        mmsplice_predictions += variants_family_df["MMSplice_Delta_Logit_PSI"].tolist()    
        variant_ids += variants_family_df["Reference"].tolist()
        ref_nums += [ref_num] * len(variants_family_df)
        event_ids += [event_id] * len(variants_family_df)
        snps += variants_family_df["snp"].tolist()

    return variant_ids, ref_nums, logits_, mmsplice_predictions, event_ids, snps



