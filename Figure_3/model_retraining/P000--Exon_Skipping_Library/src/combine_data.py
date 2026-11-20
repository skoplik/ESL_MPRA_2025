import os
import pandas as pd
import numpy as np
from collections import defaultdict
import getopt
import sys

cell_types = ["HEK293", "HeLa", "K562", "MCF7", "HMC3"]
cell_types2 = ["HEK", "HeLa", "K562", "MCF7", "HMC3"]


def var_id_to_ref_id(var_ids, supertable):
    """
    uses supertable to find ref_id
    from the given var_ids list
    """
    ref_ids = []
    for i, var_id in enumerate(var_ids):
        ref_id = supertable["ref_supertable_dict"][var_id]
        ref_ids.append(ref_id)
    return ref_ids


def make_eid_to_ref_supertable(df, supertable = None, return_type = "int"):
    """
    makes dictionary mapping
    each eid to the reference sizes 
    in df. If supertable is provided, 
    will pull the seq tuple from that

    if return type is 'int', returns the size of each sequence
    if return type is 'string', returns the string itself
    """
    eid_to_ref = {}
    if supertable is not None:
        ref_supertable_dict = supertable["ref_supertable_dict"]
        ref_eventID_supertableRef_dict = supertable["ref_eventID_supertableRef_dict"]
        for eid in ref_eventID_supertableRef_dict:
            ref = ref_eventID_supertableRef_dict[eid]
            if return_type == "int":
                eid_to_ref[eid] = {"intron1" : len(ref_supertable_dict[ref][1]),
                                   "exon" : len(ref_supertable_dict[ref][0]),
                                   "intron2" : len(ref_supertable_dict[ref][2]),
                                   "Reference" : ref,
                                   }
            elif return_type == "string":
                eid_to_ref[eid] = {"intron1" : (ref_supertable_dict[ref][1]),
                                   "exon" : (ref_supertable_dict[ref][0]),
                                   "intron2" : (ref_supertable_dict[ref][2]),
                                   "Reference" : ref,
                                   }
            else:
                raise NotImplementedError("only int or string return types alllowed")
        return eid_to_ref
    eid_to_ref = {}
    if supertable is None:
        raise NotImplementedError("supertable must be provided")
    return eid_to_ref


def make_eid_to_ref_isomap(alt_isoforms):

    eid_to_alt_iso = {"ref":{},
                      "var":{},
                      "ref_index":{},
                      "var_index":{},
                      "intron1_size_ref":{},
                      "intron2_size_ref":{},
                      "exon_size_ref":{},
                      "intron1_size_var":{},
                      "intron2_size_var":{},
                      "exon_size_var":{},
                      "intron1" : {},
                      "exon" : {},
                      "intron2": {}}

    if "event_id_161" in alt_isoforms:
        eid_col = "event_id_161"
    else:
        eid_col = "event_id"

    for i in range(len(alt_isoforms)):
        row = alt_isoforms.iloc[i]
        eid = row[eid_col]
        if row["snp"] == "none": # reference seq
            if row[eid_col] not in eid_to_alt_iso["ref"]:
                for key in eid_to_alt_iso:
                    if "ref" in key:
                        eid_to_alt_iso[key][eid] = []
                eid_to_alt_iso["intron1"][eid] = []
                eid_to_alt_iso["exon"][eid] = []
                eid_to_alt_iso["intron2"][eid] = []
            eid_to_alt_iso["ref"][eid].append(row["Reference"])
            eid_to_alt_iso["ref_index"][eid].append(i)
            eid_to_alt_iso["intron1_size_ref"][eid].append(row["intron1_size"])
            eid_to_alt_iso["exon_size_ref"][eid].append(row["exon_size"])
            eid_to_alt_iso["intron2_size_ref"][eid].append(row["intron2_size"])
            # put the sizes of the reference sequences only to work with 
            # the check_size_and_seq and get alt iso functions
            eid_to_alt_iso["intron1"][eid].append(row["intron1_size"])
            eid_to_alt_iso["exon"][eid].append(row["exon_size"])
            eid_to_alt_iso["intron2"][eid].append(row["intron2_size"])
        else: # alt seq
            if row[eid_col] not in eid_to_alt_iso["var"]:
                for key in eid_to_alt_iso:
                    if "var" in key:
                        eid_to_alt_iso[key][eid] = []
            eid_to_alt_iso["var"][eid].append(row["Reference"])
            eid_to_alt_iso["var_index"][eid].append(i)
            eid_to_alt_iso["intron1_size_var"][eid].append(row["intron1_size"])
            eid_to_alt_iso["exon_size_var"][eid].append(row["exon_size"])
            eid_to_alt_iso["intron2_size_var"][eid].append(row["intron2_size"])
    return eid_to_alt_iso


def parse_eid(eid):
    chrom, startend, strand = eid.split(":")
    start, end = startend.split("-")
    start = int(start)
    end = int(end)
    return chrom, start, end, strand


def parse_snp(snp):
    out = []
    vs = snp.split(";")
    for p in vs:
        loc, change = p.split(":")
        old, new = change.split(">")
        out.append((int(loc), old.strip(), new.strip())) # e.g. multi snp '84:A>C ; 132:G>T' will introduce spaces
    return out
    

def exon_bound_to_seq_size(ex_start, ex_end, eid):
    chrom, start, end, strand = parse_eid(eid)
    intron1_size = ex_start - start
    exon_size = ex_end - ex_start + 1
    intron2_size = end - ex_end
    return intron1_size, exon_size, intron2_size


def get_seq_sizes(df):
    ex_start = df["alt_exon_start_hg38"]
    ex_end   = df["alt_exon_end_hg38"]
    if "event_id_161" in df:
        eid_col = "event_id_161"
    else:
        eid_col = "event_id"
    chrom, start, end, strand = parse_eid(df[eid_col])
    intron1_size = ex_start - start
    exon_size = ex_end - ex_start + 1
    intron2_size = end - ex_end
    assert intron1_size + exon_size + intron2_size == 161, (intron1_size, exon_size, intron2_size)
    df["intron1_size"] = intron1_size
    df["exon_size"] = exon_size
    df["intron2_size"] = intron2_size
    return df


def make_eid_to_ref(df):
    """
    makes dictionary mapping eid to ref 
    using only samples in the df (no supertable).
    """
    internal_map = {}
    if "intron1" in df:
        output_keys = "intron1", "exon", "intron2", "Reference"
    elif "intron1_size" in df:
        output_keys = "intron1_size", "exon_size", "intron2_size", "Reference"
    else:
        raise KeyError("expected either intron1 or intron1_size as key")
    
    if "event_id_161" in df:
        eid_col = "event_id_161"
    else:
        eid_col = "event_id"

    for i in range(len(df)):
        row = df.iloc[i]
        eid = row[eid_col]
        snp = row["snp"]
        if snp == "none":
            if eid not in internal_map:
                internal_map[eid] = {i : [] for i in output_keys}
                internal_map[eid]["index"] = []
            for key in output_keys:
                internal_map[eid][key].append(row[key])
            internal_map[eid]["index"].append(i)
    return internal_map 


def check_sizes(i1, exon, i2, info, idx):
    if len(i1[idx]) != info["intron1"]:
        return False
    if len(i2[idx]) != info["intron2"]:
        return False
    if len(exon[idx]) != info["exon"]:
        return False
    return True


def hamming_distance(a, b, max):
    p = 0
    q = 0
    mismatches = 0
    while True:
        if p > len(a) - 1 and q < len(b):
            mismatches += 1
            q += 1
        elif q > len(b) - 1 and p < len(a):
            mismatches += 1
            p += 1
        elif q > len(b) - 1 and p > len(a) - 1:
            return mismatches
        elif a[p] != b[q]:
            mismatches += 1
            p += 1
            q += 1
        elif a[p] == b[q]:
            p += 1
            q += 1
    return mismatches


def check_size_and_seq(i1, ex, i2, info, idx=None, snp=None, debug=False):

    """
    checks if the variant intron1, exon and intron2 matches the 
    reference.

    types check inputs, accepting either integers
    (seq length if length not provided or known)
    or strings (the sequences themselves).

    If the inputs are all strings, also checks the Hamming distance
    matches the expected changes in the snp (variant change), if provided.

    returns a tuple where the first element is a boolean 
    representing the result of the comparison
    (True if everything matches, False otherwise)
    and the second element is the failure condition, a string
    denoting where in the sequence it mismatched. 
    If the comparison passed, this will be 'pass'. If one of the
    sequences had a size mismatch, the output will be 'intron1', for exmaple.
    if it failed at the Hamming distance level, will output 'hamming'.

    """
    is_debug = debug
    debug = print if debug else lambda *x : x

    if idx is None:
        intron1 = i1
        exon = ex
        intron2 = i2
        debug("idx is none")
    else:
        intron1 = i1[idx]
        exon = ex[idx]
        intron2 = i2[idx]
        debug("idx is some")
   
    check_hamming_i1 = True
    check_hamming_ex = True
    check_hamming_i2 = True

    if isinstance(intron1, str):
        len_intron1 = len(intron1)
        debug("i1 is str")
    elif isinstance(intron1, int):
        len_intron1 = intron1
        check_hamming_i1 = False
    if isinstance(exon, str):
        len_exon = len(exon)
        check_hamming_ex = False
        debug("ex is str")
    elif isinstance(exon, int):
        len_exon = exon
        check_hamming_ex = False
    if isinstance(intron2, str):
        len_intron2 = len(intron2)
        debug("i2 is str")
    elif isinstance(intron2, int):
        len_intron2 = intron2
        check_hamming_i2 = False


    if isinstance(info["intron1"], str):
        len_intron1_var = len(info["intron1"])
    elif isinstance(info["intron1"], int):
        len_intron1_var = info["intron1"]
        check_hamming_i1 = False
    else:
        breakpoint()
        raise ValueError
    if isinstance(info["exon"], str):
        len_exon_var = len(info["exon"])
    elif isinstance(info["exon"], int):
        len_exon_var = info["exon"]
        check_hamming_ex = False
    if isinstance(info["intron2"], str):
        len_intron2_var = len(info["intron2"])
    elif isinstance(info["intron2"], int):
        len_intron2_var = info["intron2"]
        check_hamming_i2 = False

    debug("len_intron1", "len_intron1_var", len_intron1, len_intron1_var)
    
    if len_intron1 != len_intron1_var:
        return False, "intron1"
    if len_exon != len_exon_var:
        return False, "exon"
    if len_intron2 != len_intron2_var:
        return False, "intron2"

        
    if snp is not None:
        mismatches = len(snp[idx].split(";"))
        check_hamming_i1 = False
        check_hamming_ex = False
        check_hamming_i2 = False


    if check_hamming_i1:
        hamming_i1 = hamming_distance(i1[idx], info["intron1"], max = mismatches)
    if check_hamming_ex:
        hamming_ex = hamming_distance(exon[idx], info["exon"], max = mismatches)
    if check_hamming_i2:
        hamming_i2 = hamming_distance(i2[idx], info["intron2"], max = mismatches)

    if check_hamming_i1 and check_hamming_ex and check_hamming_i2:
        if hamming_i1 + hamming_ex + hamming_i2 != mismatches:
            return False, "hamming"
    
    return True, "pass"


def filter_variant_size(df, supertable = None):
    """
    checks if, for each event_id,
    there are variant intron1, exon or intron2 
    sequences that don't match the expected structure
    of the reference and removes those. e.g. if the 
    event_id maps to a position but a different isoform
    came up in sequencing
    """
    if "event_id_161" in df:
        eid_col = "event_id_161"
    else:
        eid_col = "event_id"
    eids = df[eid_col].tolist()
    intron1 = df["intron1"].tolist()
    exon = df["exon"].tolist()
    intron2 = df["intron2"].tolist()
    indx_to_keep = []
    # get all the reference sizes
    eid_to_ref = make_eid_to_ref_map(df, supertable)
    for i, eid in enumerate(eids):
        seq_info = eid_to_ref[eid]
        is_same = check_sizes(intron1, exon, intron2, seq_info, i)
        indx_to_keep.append(is_same)
    return indx_to_keep


def add_row_to_df(df, row):
    for i in row.keys():
        if i in df:
            df[i].append(row[i])
    return df


def add_row_to_df_by_key(df, row, keys, skip_map = None): 
    if skip_map is None:
        skip_map = set()
    for i in row.keys(): 
        if i in skip_map:
            continue
        if i in keys: 
            key = keys[i]
        else:
            key = i
        if key in df:
            df[key].append(row[i])
    return df


def check_matching_keys(df1, df2):
    unique_keys = [set(), set()]
    shared_keys = set()
    for i in df1:
        if i in df2:
            shared_keys.add(i)
        else:
            unique_keys[0].add(i)
    for i in df2:
        if i in df1:
            shared_keys.add(i)
        else:
            unique_keys[1].add(i)
    return unique_keys, shared_keys


def calc_sizes(cassette_start, cassette_end, exon_start, exon_end):
    """
    calculate the intron1, exon and intron2 sizes given
    the cassette start and end and exon start and end
    |     intron1    |  exon  |  intron 2  |
    cassette        exon     exon        cassette
    start           start    end         end
    where cassette_end - cassette_start + 1 = 161
    """
    intron1_size = exon_start - cassette_start
    exon_size = exon_end - exon_start + 1
    intron2_size = cassette_end - exon_end
    return intron1_size, exon_size, intron2_size


def get_alt_iso(eid_to_alt, eid, alt_df, variant_seq):
    """
    go through all possible reference isoforms and pull
    the correct row in alt_df

    eid_to_alt is a dictionary with 4 possible keys:
        ref, ref_index, var_index, var
    each map to another dictionary, mapping eid -> something
    where the something is denoted by the first key:
        ref: the ref nums of that event id
        var: likewise for the variant nums
        ref_index: indexes in alt_df of the corresponding event_id that
                   are reference sequences
        var_index: indexes in alt_df for the event id that are variants 

    eid is the eid to check reference isoforms. This function will then 
    pull information from alt_df and return it.

    variant_seq is a dictionary or pandas series containing 'intron1',
    'intron2' and 'exon' keys corresponding to the sequences of the variant
    to match with (i.e. find its corresponding reference in alt_df)

    returns found, row where found is a boolean indicating if a matching
    reference exists, and row contains the relavant information. If not found,
    will return None for the row
    """
    if "ref" not in eid_to_alt:
        eid_to_alt = {"ref" : eid_to_alt}
    if eid not in eid_to_alt["ref"]:
        return False, None
    references = eid_to_alt["ref"][eid]
    ref_inds   = eid_to_alt["ref_index"][eid]
    # get the cassette start and end from the event id
    chrom, eid_start, eid_end, strand = parse_eid(eid)
    if "event_id_161" in alt_df:
        eid_col = "event_id_161"
    else:
        eid_col = "event_id"
    for ref_i in ref_inds:
        row = alt_df.iloc[ref_i]
        assert row[eid_col] == eid, (row[eid_col],  eid)
        # get the alt isoform exon boundaries
        if ("intron1" in row) and ("exon" in row) and ("intron2" in row):
            i1_size = len(row["intron1"])
            ex_size = len(row["exon"])
            i2_size = len(row["intron2"])
        elif ("intron1_size" in row) and ("exon_size" in row) and ("intron2_size" in row):
            i1_size = row["intron1_size"]
            ex_size = row["exon_size"]
            i2_size = row["intron2_size"]
        else:
            alt_start, alt_end = row["alt_exon_start_hg38"], row["alt_exon_end_hg38"]
            # calculate the exon and intron sizes based on the cassette bounds 
            # and isoform exon boundaries
            i1_size, ex_size, i2_size = calc_sizes(eid_start, eid_end, alt_start, alt_end)
        # check if the size matches the given variant
        found, err = check_size_and_seq(i1 = int(i1_size), ex = int(ex_size), i2 = int(i2_size), info=variant_seq)
        if found:
            break
    if not found:
        row = None
    return found, row


def get_psi_keys(keys, methods):
    """
    methods:
        processed: either 'raw' or 'clipped'
        split_ref_var: if both ref and var psi in same row
        ref_identifier: identifier for reference. If None, assume default is ref.
        var_identifier: identifier for variant. If None, assume default is var.     
    """
    all_psi_keys = [i for i in keys if "pooled" in i and "psi" in i and methods["processed"] in i]
    if methods["split_ref_var"]:
        if "ref_identifier" not in methods or methods["ref_identifier"] is None:
            ref_keys = [i for i in all_psi_keys if methods["var_identifier"] not in i]
            var_keys = [i for i in all_psi_keys if methods["var_identifier"] in i]
        elif "var_identifier" not in methods or methods["var_identifier"] is None:
            var_keys = [i for i in all_psi_keys if methods["ref_identifier"] not in i]
            ref_keys = [i for i in all_psi_keys if methods["ref_identifier"] in i]
    else:
        return all_psi_keys, None 
    return ref_keys, var_keys


def rename_keys_by_cell(df, psi_keys, cell_types, processed, ref):
    renamer_dict = {}
    for c in cell_types:
        new_name = f"{c}_pooled_psi_{processed}_{ref}"
        for key in psi_keys:
            if c in key:
                renamer_dict[key] = new_name
    df = df.rename(columns = renamer_dict)
    return df



def rebuild_alt_isoform_seq(eid, seq_info, new_row):
    chrom, start, end, strand = parse_eid(eid)
    full_seq = "".join(
    [seq_info["intron1"],
    seq_info["exon"],
    seq_info["intron2"]]
    )
    alt_isoform_seqs = {}
    i1_size = int(new_row["intron1_size"])
    ex_size = int(new_row["exon_size"])
    i2_size = int(new_row["intron2_size"])
    i = 0
    alt_isoform_seqs["intron1"] = full_seq[i : i + i1_size]
    i += i1_size
    alt_isoform_seqs["exon"] = full_seq[i : i + ex_size] 
    i += ex_size
    alt_isoform_seqs["intron2"] = full_seq[i : i + i2_size]
    return alt_isoform_seqs


def rebuild_isoform_from_len(i1_len, **seq_info):
    full_seq = "".join(
    [seq_info["intron1"],
    seq_info["exon"],
    seq_info["intron2"]]
    )
    alt_isoform_seqs = {}
    ex_len = 161 - i1_len - 20
    i2_len = 20
    i = 0
    alt_isoform_seqs["intron1"] = full_seq[i : i + i1_len]
    i += i1_len
    alt_isoform_seqs["exon"] = full_seq[i : i + ex_len] 
    i += ex_len
    alt_isoform_seqs["intron2"] = full_seq[i : i + i2_len]
    return alt_isoform_seqs


def rebuild_isoform_from_eid(eid, **seq_info):
    chrom, ex_start, ex_end, strand = parse_eid(eid)
    if strand == "-":
        raise NotImplementedError()
    full_seq = "".join(
    [seq_info["intron1"],
    seq_info["exon"],
    seq_info["intron2"]]
    )
    alt_isoform_seqs = {}
    ex_size = int(ex_end) - int(ex_start) + 1
    i1_size = 161 - ex_size - 20
    i2_size = 20
    i = 0
    alt_isoform_seqs["intron1"] = full_seq[i : i + i1_size]
    i += i1_size
    alt_isoform_seqs["exon"] = full_seq[i : i + ex_size] 
    i += ex_size
    alt_isoform_seqs["intron2"] = full_seq[i : i + i2_size]
    return alt_isoform_seqs


def var_seq_from_snp(ref_seq_info, snp):
    # get the full reference sequence (161nt)
    i1_size = len(ref_seq_info["intron1"])
    ex_size = len(ref_seq_info["exon"])
    i2_size = len(ref_seq_info["intron2"])
    i1 = [i for i in ref_seq_info["intron1"]]
    ex = [i for i in ref_seq_info["exon"]]
    i2 = [i for i in ref_seq_info["intron2"]]
    snp = parse_snp(snp)
    full = [i for i in i1] + [i for i in ex] + [i for i in i2]
    # apply substitutions according to snp annotation
    for loc, old, new in snp:
        assert full[loc] == old, f"Expected to change {old} to {new} but found {full[loc]} at snp"
        full[loc] = new
    # get the correct exon boundaries using the full sequence
    # and the original intron1, exon and intron2 sizes
    var_isoforms = {}
    i = 0
    var_isoforms["intron1"] = "".join(full[i : i + i1_size])
    i += i1_size
    var_isoforms["exon"]    = "".join(full[i : i + ex_size])
    i += ex_size
    var_isoforms["intron2"] = "".join(full[i : i + i2_size])
    return var_isoforms


def validate_seq_len(df):
    for i in range(len(df)):
        row = df.iloc[i]
        assert len(row["intron1"]) == len(row["intron1_ref"])
        assert len(row["exon"]) == len(row["exon_ref"])
        assert len(row["intron2"]) == len(row["intron2_ref"]) 
    return


def validate_psi(new_df, old_df):
    from src.esl import cell_types2, clip
    old_df_keys  = [f"{cell}_pooled_psi_raw"     for cell in cell_types2]
    old_df_keys += [f"{cell}_wt_pooled_psi_raw"  for cell in cell_types2]
    new_df_keys  = [f"{cell}_pooled_psi_raw_var" for cell in cell_types2]
    new_df_keys += [f"{cell}_pooled_psi_raw_ref" for cell in cell_types2]
    ref_map = {old_df.iloc[i]["Reference"] : i for i in range(len(old_df))}
    for i in range(len(new_df)):
        row = new_df.iloc[i]
        ref = row["Reference"]
        old_i = ref_map[ref]
        old_row = old_df.iloc[old_i]
        for j, key in enumerate(old_df_keys):
            if np.isnan(clip(old_row[key])) and np.isnan(clip(row[new_df_keys[j]])):
                continue
            assert np.isclose(clip(old_row[key]), clip(row[new_df_keys[j]])), ("new index", i, "old_index", old_i, ":", old_row[key], row[new_df_keys[j]])
    return 


def compare(psi1, psi2):
    if np.isnan(psi1) and np.isnan(psi2):
        return True
    elif np.isnan(psi1) or np.isnan(psi2):
        return False
    return np.isclose(psi1, psi2)


def compare_raw_to_main(main_df, raw_df):
    ref_map = {(main_df["Reference"].iloc[i], len(main_df["intron1"].iloc[i])) : i for i in range(len(main_df))}
    same = []
    not_same = []
    dne = []
    by_cell = {c : {True:0, False:0} for c in cell_types2}
    for raw_i in range(len(raw_df)):
        raw_row = raw_df.iloc[raw_i]
        ref = int(raw_row["Reference"])
        i1_size = int(raw_row["intron1_size"])
        if (ref, i1_size) not in ref_map:
            dne.append((ref, i1_size))
            continue
        main_i = ref_map[(ref, i1_size)]
        main_row = main_df.iloc[main_i]
        is_same = True
        curr_mismatch = {"Reference": ref}
        for c, cell in enumerate(cell_types2):
            key = f"{cell}_pooled_psi_raw"
            psi1 = main_row[key]
            try:
                psi2 = raw_row[key]
            except KeyError:
                key = f"{cell_types[c]}_pooled_psi_raw"
                psi2 = raw_row[key]
            cell_same = compare(psi1, psi2)
            if cell_same:
                by_cell[cell][True] += 1
            else:
                is_same = False
                curr_mismatch[key] = (psi1, psi2)
                by_cell[cell][False] += 1

        if is_same:
            same.append(curr_mismatch)
        else:
            not_same.append(curr_mismatch)

    return same, not_same, by_cell, dne


def compare_main_to_merged(main_df, merged_df):
    if "intron1_size" in main_df:
        ref_map = {(main_df["Reference"].iloc[i], main_df["intron1_size"].iloc[i]) : i for i in range(len(main_df))}
    else:
        ref_map = {(main_df["Reference"].iloc[i], len(main_df["intron1"].iloc[i])) : i for i in range(len(main_df))}
    same = []
    not_same = []
    dne = []
    by_cell = {c : {True:0, False:0} for c in cell_types2}
    for merged_i in range(len(merged_df)):
        merged_row = merged_df.iloc[merged_i]
        ref = int(merged_row["ref_id"])
        var = int(merged_row["var_id"])
        for pair, seq in enumerate((ref, var)):
            i1_size = int(merged_row["intron1_size"])
            if (seq, i1_size) not in ref_map:
                dne.append((seq, i1_size, ("var" if pair else "ref")))
                continue
            main_i = ref_map[(seq, i1_size)]
            main_row = main_df.iloc[main_i]
            is_same = True
            curr_mismatch = {"Reference": seq,
                                "snp" : main_row["snp"],
                                "intron1_size":i1_size}
            suffix = "" if pair else "_ref"
            for c, cell in enumerate(cell_types2):
                key = f"{cell}_pooled_psi_raw"
                psi1 = main_row[key]
                try:
                    psi2 = merged_row[key+suffix]
                except KeyError:
                    key = f"{cell_types[c]}_pooled_psi_raw"
                    psi2 = merged_row[key+suffix]
                cell_same = compare(psi1, psi2)
                if cell_same:
                    by_cell[cell][True] += 1
                else:
                    is_same = False
                    curr_mismatch[key] = (psi1, psi2)
                    by_cell[cell][False] += 1
            if is_same:
                same.append(curr_mismatch)
            else:
                not_same.append(curr_mismatch)
    return same, not_same, by_cell, dne


def compare_alt_to_merged(alt_df, merged_df):
    ref_map = {(alt_df["Reference"].iloc[i], alt_df["alt_intron1_len"].iloc[i]) : i for i in range(len(alt_df))}
    same = []
    not_same = []
    dne = []
    by_cell = {c : {True:0, False:0} for c in cell_types2}
    for merged_i in range(len(merged_df)):
        merged_row = merged_df.iloc[merged_i]
        ref = int(merged_row["ref_id"])
        var = int(merged_row["var_id"])
        for pair, seq in enumerate((ref, var)):
            i1_size = int(merged_row["intron1_size"])
            if (seq, i1_size) not in ref_map:
                dne.append((seq, i1_size, ("var" if pair else "ref")))
                continue
            alt_i = ref_map[(seq, i1_size)]
            alt_row = alt_df.iloc[alt_i]
            is_same = True
            curr_mismatch = {"Reference": seq,
                                "snp" : alt_row["snp"],
                                "intron1_size":i1_size}
            suffix_merged = "" if pair else "_ref"
            is_wt = "" if pair else "wt_"
            for c, cell in enumerate(cell_types2):
                key_alt = f"{cell}_{is_wt}pooled_psi_raw_alt"
                key_merged = f"{cell}_pooled_psi_raw" if cell != "HEK" else "HEK293_pooled_psi_raw"
                psi_alt = alt_row[key_alt]
                psi_merged = merged_row[key_merged]
                cell_same = compare(psi_alt, psi_merged)
                if cell_same:
                    by_cell[cell][True] += 1
                else:
                    is_same = False
                    curr_mismatch[key_merged] = (psi_alt, psi_merged)
                    by_cell[cell][False] += 1
            if is_same:
                same.append(curr_mismatch)
            else:
                not_same.append(curr_mismatch)
    return same, not_same, by_cell, dne


def save_queries(queries, path):
    """
    save query ref nums for debugging
    """
    with open(path, "w") as f:
        for q in queries:
            f.write(str(q))
            f.write("\n")
    return


def load_queries(path):
    """
    load query ref nums for debugging
    """
    queries = []
    with open(path, "r") as f:
        for i in f:
            if i:
                queries.append(i.split("\n")[0])
    return queries


def collect_query_info(*all_events, supertable):
    var_ids = []
    eids = []
    intron1_sizes = []
    query_ref_nums = []
    for ambiguous in all_events:
        if "event_id_161" in ambiguous:
            eid_col = "event_id_161"
        else:
            eid_col = "event_id"
        var_ids += (ambiguous["Reference"])
        eids += ambiguous[eid_col]
        intron1_sizes += [len(i) for i in ambiguous["intron1"]]
        ref_ids = var_id_to_ref_id(ambiguous["Reference"], supertable)
        # intron size for ref should be the same
        intron1_sizes += [len(i) for i in ambiguous["intron1"]]
        query_ref_nums = query_ref_nums + ambiguous["Reference"] + ref_ids
    return query_ref_nums, intron1_sizes


# filter data using new format
    # check raw files for ALL ref numbers
    # go through main and alt and/or supertable, map each event_id_161 to 
    # references iff they exist in raw file ref numbers
    # if the reference *doesn't* exist, replace with the one that does in query_ref_nums
    # there should be only one reference id and one variant id for each event_id_161

def make_eid_to_psi_refs(main_df, alt_isoforms, ref_nums):
    """
    makes dictionary mapping each
    161-sequence event_id to all
    existing ref numbers in the raw
    data, separated by reference and
    variant numbers
    """
    if "event_id_161" in main_df:
        eid_col = "event_id_161"
    else:
        eid_col = "event_id"
    exist_map = defaultdict(set)
    raw_ref_lookup = defaultdict(None)
    for _, row in main_df.iterrows():
        eid = row[eid_col]
        ref = row["Reference"]
        snp = row["snp"]
        intron1_size = len(row["intron1"])
        if ref not in ref_nums and str(ref) not in ref_nums:
            continue
        exist_map[eid].add((ref, snp))

    for _, row in alt_isoforms.iterrows():
        eid = row[eid_col]
        ref = row["Reference"]
        ref_alt = row["alt_supertable_refs"]
        snp = row["snp"]
        if ref in ref_nums or str(ref) in ref_nums:
            exist_map[eid].add((ref, snp))
        if ref_alt in ref_nums or str(ref_alt) in ref_nums:
            exist_map[eid].add((ref_alt, snp))
    
    for eid in exist_map:
        for ref, snp in exist_map[eid]:
            raw_ref_lookup[(eid, snp)] = int(ref)
    return exist_map, raw_ref_lookup


def get_raw_psi_ref(exist_map, raw_ref_lookup, eid_col = "event_id_161"):
    """
    maps new reference back to old 
    in order to query the raw psi data
    using datastructures mapping each event_id to 
    known references in the raw data file 
    """
    def map_(df):
        snp = df["snp"]
        eid = df[eid_col]
        ref = df["Reference"]
        # check if the ref exists
        key = (ref, snp)
        if key in exist_map:
            return ref 
        ref = raw_ref_lookup[(eid, snp)]
        return ref
    return map_


def filter_variant_size_and_seq(df, supertable, alt_isoforms, psi_naming_method = None, collect_ambiguous = True):
    """
    checks if, for each event_id,
    there are variant intron1, exon or intron2 
    sequences that don't match the expected structure
    of the reference and removes those. e.g. if the 
    event_id maps to a position but a different isoform
    came up in sequencing
    """
    if "event_id_161" in df:
        eid_col = "event_id_161"
    else:
        eid_col = "event_id"
    eids = df[eid_col].tolist()
    intron1 = df["intron1"].tolist()
    exon = df["exon"].tolist()
    intron2 = df["intron2"].tolist()
    snp = df["snp"].tolist()
    indx_to_keep = []
    new_df = {}
    new_df["intron1_ref"] = []
    new_df["exon_ref"] = []
    new_df["intron2_ref"] = []
    new_df["intron1"] = []
    new_df["exon"] = []
    new_df["intron2"] = []
    new_df["var_id"] = []
    new_df["ref_id"] = []
    eids_mismatched = set()
    non_ambiguous = defaultdict(list)
    ambiguous = defaultdict(list)
    not_found = defaultdict(list)
    
    if "intron1" not in alt_isoforms:
        alt_isoforms = alt_isoforms.apply(get_seq_sizes, axis = 1)

    if psi_naming_method is None:
        psi_naming_method = {"split_ref_var": True,
                             "processed" : "raw",
                             "ref_identifier": "wt",
                             "var_identfier" : None
                             }

    unique_keys, shared_keys = check_matching_keys(alt_isoforms, df)
    ref_alt_raw_psi_keys, var_alt_raw_psi_keys = get_psi_keys(unique_keys[0], psi_naming_method)
    ref_df_raw_psi_keys, var_df_raw_psi_keys = get_psi_keys(unique_keys[1], psi_naming_method)
    
    if len(ref_df_raw_psi_keys) != len(var_df_raw_psi_keys) != len(ref_alt_raw_psi_keys) != len(ref_var_raw_psi_keys):
        print("ref_alt_raw_psi_keys", len( ref_alt_raw_psi_keys))
        print("ref_df_raw_psi_keys",  len( ref_df_raw_psi_keys))
        print("var_alt_raw_psi_keys", len( var_alt_raw_psi_keys))
        print("var_df_raw_psi_keys",  len( var_df_raw_psi_keys))
        raise ValueError("INCORRECT SIZES")

    from src.esl import cell_types2
    # standardize psi naming for both datasets
    alt_isoforms = rename_keys_by_cell(alt_isoforms, ref_alt_raw_psi_keys, cell_types2, psi_naming_method["processed"], "ref")
    df  = rename_keys_by_cell(df, ref_df_raw_psi_keys, cell_types2, psi_naming_method["processed"], "ref")
    alt_isoforms = rename_keys_by_cell(alt_isoforms, var_alt_raw_psi_keys, cell_types2, psi_naming_method["processed"], "var")
    df  = rename_keys_by_cell(df, var_df_raw_psi_keys, cell_types2, psi_naming_method["processed"], "var")
                
    unique_keys2, shared_keys2 = check_matching_keys(alt_isoforms, df)
    
    for shared in shared_keys2:
        new_df[shared] = []
    
    for cell in cell_types2:
        new_df[f"{cell}_total_pooled"] = []

    # get all the reference sequences for each variant
    eid_to_ref_supertable = make_eid_to_ref_supertable(df, supertable, return_type = "string")
    eid_to_ref_internal = make_eid_to_ref(df)
    eid_to_alt_iso = make_eid_to_ref_isomap(alt_isoforms)
    for i, eid in enumerate(eids):
        old_row = df.iloc[i].copy()

        if snp[i] == "none":
            continue


        # seq info maps the event id to the reference intron1, exon, intron2
        # from the supertable or from the same dataset, or from the alt isoforms dataset
        try:
            seq_info_supertable = eid_to_ref_supertable[eid]
        except KeyError:
            breakpoint()
        if eid in eid_to_ref_internal:
            seq_info_internal = eid_to_ref_internal[eid]

        # check match to supertable
        matches_supertable, err1 = check_size_and_seq(intron1, exon, intron2, seq_info_supertable, i, snp)
        for j in range(len(seq_info_internal["intron1"])):
            # check for potentially multiple refs for the same event id
            seq_info_curr = {key : seq_info_internal[key][j] for key in seq_info_internal}
            matches_reference, err2 = check_size_and_seq(intron1, exon, intron2, seq_info_curr, i, snp)
            if matches_reference:
                internal_ref_match = seq_info_internal["index"][j]
                seq_info_internal_index = j
                break

        # if the reference is in df and has matching size
        if matches_reference:
            matching_ref_row = df.iloc[internal_ref_match]
            assert seq_info_internal["Reference"][seq_info_internal_index] == matching_ref_row["Reference"], ("reference doesnt match:", seq_info_internal["Reference"][seq_info_internal_index], matching_ref_row["Reference"])
            old_row["intron1_ref"] = matching_ref_row["intron1"]
            old_row["exon_ref"]    = matching_ref_row["exon"]
            old_row["intron2_ref"] = matching_ref_row["intron2"]
            old_row["var_id"]  = old_row["Reference"]
            old_row["ref_id"]  = matching_ref_row["Reference"]
            new_df = add_row_to_df(new_df, old_row)

    
            if collect_ambiguous:
                for i in old_row.keys():
                    non_ambiguous[i].append(old_row[i])

        # no matching size reference in df, maybe in supertable
        elif matches_supertable:
            old_row["intron1_ref"] = seq_info_supertable["intron1"]
            old_row["exon_ref"]    = seq_info_supertable["exon"]
            old_row["intron2_ref"] = seq_info_supertable["intron2"]
            old_row["var_id"]  = old_row["Reference"]
            old_row["ref_id"]  = seq_info_supertable["Reference"]
            new_df = add_row_to_df(new_df, old_row)
            if collect_ambiguous:
                for i in old_row.keys():
                    non_ambiguous[i].append(old_row[i])

        # not in supertable either, look in alt isoform dataframe
        elif (not matches_reference) and (not matches_supertable):
            variant_seq = {"intron1" : intron1[i],
                           "exon"    : exon[i],
                           "intron2" : intron2[i]}
            # print(eid) if eid not in eids_mismatched else ""
            eids_mismatched.add(eid)
            matches_alt_isoform, new_row = get_alt_iso(eid_to_alt_iso, eid, alt_isoforms, variant_seq)
            if matches_alt_isoform:
                # remake exon boundaries using the reference seq, eid and size info
                ref_seq_alt_isoforms = rebuild_alt_isoform_seq(eid, seq_info_supertable, new_row) 
                old_row["intron1_ref"] = ref_seq_alt_isoforms["intron1"]
                old_row["exon_ref"]    = ref_seq_alt_isoforms["exon"]
                old_row["intron2_ref"] = ref_seq_alt_isoforms["intron2"]
                old_row["var_id"]      = old_row["Reference"]
                old_row["ref_id"]      = new_row["Reference"]
                new_df = add_row_to_df(new_df, old_row)
                passed, err = check_size_and_seq([ref_seq_alt_isoforms["intron1"]],
                                   [ref_seq_alt_isoforms["exon"]],
                                   [ref_seq_alt_isoforms["intron2"]],
                                   {"intron1":intron1[i],
                                    "exon" : exon[i],
                                    "intron2" : intron2[i]}, 0, [snp[i]])
                assert passed
                if collect_ambiguous:
                    for i in old_row.keys():
                        ambiguous[i].append(old_row[i])
            else:
                # didn't find the matching reference, so effectively drop the variant
                if collect_ambiguous:
                    for i in old_row.keys():
                        not_found[i].append(old_row[i])
                continue

    return new_df, non_ambiguous, ambiguous, not_found


def search_alt_isoforms(alt_isoforms, supertable):
    """
    searches alt_isoforms for all isoforms
    with existing annotation/data.

    returns references and intron1 sizes
    for querying the raw data and making raw_psi_df
    """
    references  = alt_isoforms["Reference"].tolist()
    exon_starts = alt_isoforms["alt_exon_start_hg38"].tolist()
    exon_ends   = alt_isoforms["alt_exon_end_hg38"].tolist()
    if "event_id_161" in alt_isoforms:
        eid_col = "event_id_161"
    else:
        eid_col = "event_id"
    event_ids   = alt_isoforms[eid_col].tolist()
    ref_out     = []
    i1_size_out = []
    for i, ref in enumerate(references):
        i1_size, ex_size, i2_size = exon_bound_to_seq_size(exon_starts[i], exon_ends[i], event_ids[i])
        ref_out.append(references[i])
        i1_size_out.append(i1_size)
        # get corresponding reference num from supertable
        if ref in supertable["ref_supertable_dict"]:
            ref_ = supertable["ref_supertable_dict"][ref]
            ref_out.append(ref_)
            i1_size_out.append(i1_size)
    return ref_out, i1_size_out


def add_missing_refs(combined_df, missing, main_df, alt_isoforms, rename_key_map):
    """
    Add missing refs by event_id_161 
    looking for the 'none' snp with 
    the same i1_len
    """
    for ref, var, alt_i, i1_size in missing:
        alt_row = alt_isoforms.iloc[alt_i[1]]
        eid_161 = alt_row.event_id_161
        ref_row = alt_isoforms[
                    (alt_isoforms.event_id_161 == eid_161) &
                    (alt_isoforms.snp == "none") &
                    (alt_isoforms.alt_intron1_len == i1_size[1])
                    ]
        if len(ref_row):
            combined_df = add_row_to_df_by_key(combined_df, ref_row, rename_key_map)
        else:
            if "intron1_len" not in main_df:
                main_df["intron1_len"] = main_df.intron1.apply(len)
            ref_row = main_df[
                        (main_df.event_id_161 == eid_161) &
                        (main_df.snp == "none") & 
                        (main_df.intron1_len == i1_size[1])
                        ]
            if len(ref_row): 
                combined_df = add_row_to_df_by_key(combined_df, ref_row, rename_key_map)
            else:
                breakpoint()
                " still missing " 
    return combined_df



def row_to_supertable_seqs(row, supertable):
    """
    Uses supertable to pull reference seq
    and uses snp from row to remake the alt
    sequences if missing.
    """
    eid = row["event_id"]
    snp = row["snp"]
    i1_len = row["intron1_len"] if "intron1_len" in row else row["alt_intron1_len"]
    eid_to_supertable_ref = supertable["ref_eventID_supertableRef_dict"]
    ref_supertable_dict = supertable["ref_supertable_dict"]
    ref_id = eid_to_supertable_ref[eid]
    exon_ref, intron1_ref, intron2_ref = ref_supertable_dict[ref_id]
    ref_seq = rebuild_isoform_from_len(i1_len, exon=exon_ref, intron1=intron1_ref, intron2=intron2_ref) 
    ref_seqs = {}
    ref_seqs["intron1_ref"] = intron1_ref = ref_seq["intron1"]
    ref_seqs["exon_ref"] = exon_ref = ref_seq["exon"]
    ref_seqs["intron2_ref"] = intron2_ref = ref_seq["intron2"]
    if (("intron1" not in row) or 
       ("exon" not in row) or 
       ("intron1" not in row) or 
       (snp == "none")):
        var_seqs = var_seq_from_snp({"intron1":intron1_ref,
                                     "exon":exon_ref,
                                     "intron2":intron2_ref}, snp)
    else:
        var_seqs = {"intron1":row["intron1"], "exon":row["exon"], "intron2":row["intron2"]}
    return ref_seqs | var_seqs


def get_unique_ids(df):
    df["unique_ref_id"] = df["ref_id"].astype(str) + "_" + df["intron1_len"].astype(str)
    df["unique_var_id"] = df["var_id"].astype(str) + "_" + df["intron1_len"].astype(str)
    df["pair_id"] = (
        df["ref_id"].astype(str) + "_" +
        df["var_id"].astype(str) + "_" +
        df["intron1_len"].astype(str)
    )
    return df


def combine_main_and_alt(main_df, alt_isoforms, supertable, eid_col = "event_id_161"):
    """
    Combine main and alt,  ( >= 5_11 format)
    and match reference to variant

    unqiue variants are identified by 
    (event_id_161, snp, intron1_len) pairs.
    other identifiers are possible, such as
    (variant_id, intron1_len). References 
    are matched using the event_id based
    supertable instead of on event_id_161,
    since event_id_161 may have multiple ref_ids.

    Returns combined_df, which has new columns
    ref_id, var_id, unique_ref_id, unique_var_id, pair_id
    that identify sequences. PSI values are given no 
    suffix if they correspond to the variant, and reference
    PSIs are given the _ref suffix. In addition, sequences
    are also given a new column separating variant from reference
    (e.g. intron1 vs intron1_ref). Finally, dataframe_source 
    denotes whether the variant with that isoform came from
    main_df or alt_isoforms. 
    """
    eid_to_supertable_ref = supertable["ref_eventID_supertableRef_dict"]

    main_df_refs = main_df["Reference"].tolist()
    main_df["intron1_len"] = main_df["intron1"].apply(len)
    main_df["exon_len"] = main_df["exon"].apply(len)
    main_df_sizes = main_df["intron1_len"].tolist()
    main_df_eids_161 = main_df["event_id_161"].tolist()
    main_df_snps = main_df["snp"].tolist()
    main_map = {(main_df_eids_161[i], main_df_snps[i],main_df_sizes[i]) : i for i in range(len(main_df_refs))}

    alt_refs = alt_isoforms["Reference"].tolist()
    alt_eids_161 = alt_isoforms["event_id_161"].tolist()
    alt_snps = alt_isoforms["snp"].tolist()
    alt_sizes = alt_isoforms["alt_intron1_len"].tolist()
    alt_map = {(alt_eids_161[i], alt_snps[i], alt_sizes[i]) : i for i in range(len(alt_refs))}

    main_psi_keys = [f"{cell}_pooled_psi_raw" for cell in cell_types2]
    main_psi_keys += [f"{cell}_wt_pooled_psi_raw" for cell in cell_types2]
    alt_psi_keys = [f"{cell}_pooled_psi_raw_alt" for cell in cell_types2]
    alt_psi_keys += [f"{cell}_wt_pooled_psi_raw_alt" for cell in cell_types2]

    rename_key_map = {}
    for key in main_psi_keys + alt_psi_keys:
        cell = key.split("_")[0]
        cell = cell if "HEK" not in cell else "HEK293"
        new_key = f"{cell}_pooled_psi_raw"
        if "_wt_" in key:
            new_key += "_ref"
        rename_key_map[key] = new_key


    combined_df = {}
    combined_df["ref_id"] = []
    combined_df["var_id"] = []
    combined_df["snp"] = []
    combined_df["event_id"] = []
    combined_df["event_id_161"] = []
    combined_df["gene_exon"] = []
    combined_df["source"] = []
    combined_df["intron1"] = []
    combined_df["exon"] = []
    combined_df["intron2"] = []
    combined_df["intron1_ref"] = []
    combined_df["exon_ref"] = []
    combined_df["intron2_ref"] = []
    combined_df["intron1_len"] = []
    combined_df["exon_len"] = []
    row_identifiers = [i for i in combined_df]
    for i in rename_key_map.values():
        combined_df[i] = []
    rename_key_map["Reference"] = "var_id"
    rename_key_map["alt_intron1_len"] = "intron1_len"
    rename_key_map["alt_exon_len"] = "exon_len"
    # prevent adding sequences twice
    skip_map = {"intron1", "exon", "intron2"}
    ref_rows = {}

    main_set = set()
    for i, row in main_df.iterrows():
        main_set.add((row.event_id_161, row.snp, row.intron1_len))
    alt_set = set()
    for i, row in alt_isoforms.iterrows():
        alt_set.add((row.event_id_161, row.snp, row.alt_intron1_len))
        alt_set.add((row.event_id_161, row.snp, row.main_intron1_len))
    all_with_deltas_set = main_set | alt_set
    all_variants = [i for i in all_with_deltas_set if i[1] != "none"]
    all_variants_set = set(all_variants)

    for (eid_161, snp, i1) in all_variants_set:
        if (eid_161, snp, i1) in main_map:
            row = main_df.iloc[main_map[(eid_161, snp, i1)]]
            df_source = "main"
        elif (eid_161, snp, i1) in alt_map:
            df_source = "alt"
            row = alt_isoforms.iloc[alt_map[(eid_161, snp, i1)]]
        else:
            breakpoint()
            " count not find variant in main or alt "
        seq_info = row_to_supertable_seqs(row, supertable)
        for i in seq_info:
            combined_df[i].append(seq_info[i])
        ref = int(eid_to_supertable_ref[row["event_id"]])
        combined_df = add_row_to_df_by_key(combined_df, row, rename_key_map, skip_map)
        combined_df["ref_id"].append(ref)
        combined_df["dataframe_source"] = df_source
        if len(combined_df["intron1"][-1]) != combined_df["intron1_len"][-1]:
            breakpoint()
            " mismatched size "

    # _=[print(i.ljust(25), len(combined_df[i])) for i in combined_df]
    # breakpoint()
    # " check combined_df "

    combined_df = pd.DataFrame(combined_df)
    assert len(combined_df) == len(all_variants_set)
    assert (combined_df.intron1.apply(len) != combined_df.intron1_len.values).sum() == 0
    assert (combined_df.exon.apply(len) != combined_df.exon_len).sum() == 0
    assert (combined_df.intron2.apply(len) != 20).sum() == 0
    assert (combined_df.intron1_ref.apply(len) != combined_df.intron1_len).sum() == 0
    assert (combined_df.intron2_ref.apply(len) != 20).sum() == 0
    assert (combined_df.exon_ref.apply(len) != combined_df.exon_len).sum() == 0

    from src.esl import clip, logit
    
    for cell in cell_types:
        try:
            combined_df[f"{cell}_pooled_psi_clipped"]     = clip(combined_df[f"{cell}_pooled_psi_raw"])
            combined_df[f"{cell}_pooled_logit"]           = logit(combined_df[f"{cell}_pooled_psi_clipped"])
            combined_df[f"{cell}_pooled_psi_clipped_ref"] = clip(combined_df[f"{cell}_pooled_psi_raw_ref"])
            combined_df[f"{cell}_pooled_logit_ref"]       = logit(combined_df[f"{cell}_pooled_psi_clipped_ref"])

            combined_df[f"{cell}_delta_logit_pooled"] = combined_df[f"{cell}_pooled_logit"] - combined_df[f"{cell}_pooled_logit_ref"]
        except KeyError as e:
            print(e)
            print(cell_types)
            breakpoint()
    combined_df  = get_unique_ids(combined_df)
    combined_df = pd.DataFrame(combined_df)
    combined_df["mean_dlogit"] = combined_df[[f"{cell}_delta_logit_pooled" for cell in cell_types]].mean(axis=1)
    return combined_df


def combine_all_dataframes(raw_df, main_df, matched_df, alt_isoforms, query_ref_nums, intron1_sizes, supertable):
    """
    use query_ref_nums and intron1_sizes pairs
    to get PSI values from raw_df (after validating)
    
    check each reference to see if it's a variant. If it is,
    used matched_df to get the correct reference number,
    and then get the reference PSI value from raw_df.

    TODO:
    Finally, validate that all of the (reference, size) pairs
    from alt_isoforms are included. 
    """
    raw_df_refs = raw_df["Reference"].tolist()
    raw_df_sizes = raw_df["intron1_size"].tolist()
    raw_map = {(raw_df_refs[i], raw_df_sizes[i]) : i for i in range(len(raw_df_refs))}
    
    main_df_refs = main_df["Reference"].tolist()
    main_df_sizes = [len(i) for i in main_df["intron1"]]
    main_df_snps = main_df["snp"].tolist()
    snp_map = {main_df_refs[i] : main_df_snps[i] for i in range(len(main_df_snps))}
    main_map = {main_df_refs[i] : i for i in range(len(main_df_refs))}
    
    matched_refs = matched_df["ref_id"].tolist()
    matched_vars = matched_df["var_id"].tolist()
    var_to_ref = {matched_vars[i] : matched_refs[i] for i in range(len(matched_refs))}

    alt_refs = alt_isoforms["Reference"].tolist()
    alt_sizes = (161 - 20 - (alt_isoforms["alt_exon_start_hg38"] - alt_isoforms["alt_exon_end_hg38"] + 1)).tolist()
    alt_map = {(alt_refs[i], alt_sizes[i]) : i for i in range(len(alt_refs))}
    alt_snp = alt_isoforms["snp"].tolist()
    snp_map_alt = {alt_refs[i] : alt_snp[i] for i in range(len(alt_snp))}
    
    complete_df = defaultdict(list)

    ref_supertable_dict = supertable["ref_supertable_dict"]
    ref_eventID_supertableRef_dict = supertable["ref_eventID_supertableRef_dict"]
    supertable_gene_snp = supertable["supertable_gene_snp"]
    seen = set()

    
    if "event_id_161" in main_df:
        eid_col = "event_id_161"
    else:
        eid_col = "event_id"

    for i, var in enumerate(query_ref_nums):
        i1_size = intron1_sizes[i]
        if var not in snp_map and var in snp_map_alt:
            snp = snp_map_alt[var]
        elif var in snp_map:
            snp = snp_map[var]
        else:
            snp = supertable_gene_snp[str(var)][1]

        if var not in var_to_ref:
            # assert snp == "none", "ref is in var_to_ref but has snp"
            continue
        if (var, i1_size) in seen:
            continue
        seen.add((var, i1_size))

        # add variant and reference IDs
        var_row = raw_df.iloc[raw_map[(var, i1_size)]]
        complete_df["var_id"].append(var_row["Reference"])
        ref = var_to_ref[var] 
        try:
            ref_row = raw_df.iloc[raw_map[(ref, i1_size)]]
        except KeyError as e:
            print("i is", i)
            breakpoint()
            raise e
        complete_df["ref_id"].append(ref_row["Reference"])

        # use main row for non-psi information
        main_row = main_df.iloc[main_map[var]]
        complete_df[eid_col].append(main_row[eid_col])
        complete_df["gene_exon"].append(main_row["gene_exon"])
        complete_df["source"].append(main_row["source"])
        complete_df["snp"].append(snp)

        # build sequences using supertable
        seq_info_supertable = {"intron1" : ref_supertable_dict[str(ref)][1],
                               "exon"    : ref_supertable_dict[str(ref)][0],
                               "intron2" : ref_supertable_dict[str(ref)][2],}

        # use supertable to build full sequence, then make the isoforms for 
        # the reference and variant. This can likely be optimized
        ref_seq_alt_isoforms = rebuild_alt_isoform_seq(main_row[eid_col], seq_info_supertable, var_row) 
        try:
            var_seq_alt_isoforms = var_seq_from_snp(ref_seq_alt_isoforms, snp)
        except ValueError as e:
            import traceback
            traceback.print_exc()
            breakpoint()
        cell_types = set()

        # add ref sequences
        for i in ref_seq_alt_isoforms:
            complete_df[i + "_ref"].append(ref_seq_alt_isoforms[i])
        # add var sequences
        for i in var_seq_alt_isoforms:
            complete_df[i].append(var_seq_alt_isoforms[i])
        # add var PSIs, included, excluded and total counts for all reps and pooled
        for i in var_row.keys():
            if "_psi_" in i or "_included" in i or "_excluded" in i or "_total" in i:
                cell_types.add(i.split("_")[0])
                complete_df[i].append(var_row[i])
        # add ref PSIs, included, excluded and total counts for all reps and pooled        
        for i in ref_row.keys():
            if "_psi_" in i or "_included" in i or "_excluded" in i or "_total" in i:
                complete_df[i + "_ref"].append(ref_row[i])
        # add sequence sizes
        complete_df["intron1_size"].append(ref_row["intron1_size"])
        complete_df["intron2_size"].append(ref_row["intron2_size"])
        complete_df["exon_size"].append(ref_row["exon_size"])

    try:
        complete_df = pd.DataFrame(complete_df)
    except ValueError as e:
        print(e)
        for i in complete_df:
            print(i.ljust(25), len(complete_df[i]))
        breakpoint()


    # get clipped psi values, logits, and dlogits for each cell
    for cell in cell_types:
        try:
            complete_df[f"{cell}_pooled_psi_clipped"]     = clip(complete_df[f"{cell}_pooled_psi_raw"])
            complete_df[f"{cell}_pooled_logit"]           = logit(complete_df[f"{cell}_pooled_psi_clipped"])
            complete_df[f"{cell}_pooled_psi_clipped_ref"] = clip(complete_df[f"{cell}_pooled_psi_raw_ref"])
            complete_df[f"{cell}_pooled_logit_ref"]       = logit(complete_df[f"{cell}_pooled_psi_clipped_ref"])

            complete_df[f"{cell}_delta_logit_pooled"] = complete_df[f"{cell}_pooled_logit"] - complete_df[f"{cell}_pooled_logit_ref"]
        except KeyError as e:
            print(e)
            print(cell_types)
            breakpoint()
    
    complete_df["mean_dlogit"] = complete_df[[f"{cell}_delta_logit_pooled" for cell in cell_types]].mean(axis=1)
    return complete_df


help_="""
Converts main dataframe into matched dataframe
where each row contains a variant and corresponding ref,
including PSI values.

Will try to pull PSI values from raw data if the refernce
is not found. If in validation mode, will pull PSIs from
all ambiguous events and see if they match the original.
"""
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:],"", [
    "raw_data_directory=",
    "main_dataframe=", 
    "alt_isoforms_dataframe=",
    "supertable_file=", 
    "output_path=",
    "validation",
    "debug",
    "help",
    ]) 
    opts = dict(opts)
    if "--help" in opts:
        print(help_)
        sys.exit(1)
    print(opts)
    raw_data_directory = opts["--raw_data_directory"]
    alt_isoforms_dataframe = opts["--alt_isoforms_dataframe"]
    main_dataframe = opts["--main_dataframe"]
    output_path = opts["--output_path"]
    supertable_file = opts["--supertable_file"]
    if "--debug" in opts:
        debug = True    
    else:
        debug = False

    if "--validation" in opts:
        validation = True
    else:
        validation = False

    sys.path.append(os.path.dirname(os.path.dirname(__file__)) )
    from src.load_data import read_supertable
    from src.esl import cell_types2, logit, clip
    from psi_from_raw_data import psi_from_raw_file_dir
    from psi_from_raw_data import refs_from_raw_file_dir
    from psi_from_raw_data import output_df as make_raw_psi_df
    from psi_from_raw_data import seq_len_to_junctions

    # supertable_file = "st_04_26_2026.csv" 
    # main_df = pd.read_csv("04_17_2026_1e-2_ALL_WITH_WT.csv")
    main_df = pd.read_csv(main_dataframe)
    eid_col = 22 if "event_id_161" in main_df else 1
    # alt_isoforms = pd.read_csv("SI_alt_transcript_psi.csv")
    alt_isoforms = pd.read_csv(alt_isoforms_dataframe)
    supertable = read_supertable(supertable_file = supertable_file, eid_col=eid_col)
    supertable2 = read_supertable(supertable_file = supertable_file, eid_col = 1)

    if debug:
        breakpoint()
        " filter variant size and seq "

    if validation:
        refs = refs_from_raw_file_dir(raw_data_directory)
        exist_map, raw_ref_lookup = make_eid_to_psi_refs(main_df, alt_isoforms, refs)
        # main_df.apply(get_raw_psi_ref(exist_map, raw_ref_lookup), axis = 1)

        matched_df, non_ambiguous, ambiguous, not_found = filter_variant_size_and_seq(main_df, supertable, alt_isoforms)

        if validation:
            query_ref_nums, intron1_sizes = collect_query_info(non_ambiguous, ambiguous, not_found, supertable=supertable)
        else:
            query_ref_nums, intron1_sizes = collect_query_info(not_found, supertable=supertable)

        if debug:
            breakpoint()
            " searching alt iso "

        alt_refs, alt_i1 = search_alt_isoforms(alt_isoforms, supertable)
        query_ref_nums += alt_refs
        intron1_sizes += alt_i1
        # check raw psi's against SI and main_df
        out = psi_from_raw_file_dir(raw_data_directory, query_ref_nums, sep = "\t")
        psi_by_ref = out["psi_by_ref"] 
        included_by_ref = out["included_by_ref"] 
        excluded_by_ref = out["excluded_by_ref"] 
        included_pooled_by_ref = out["included_pooled_by_ref"] 
        excluded_pooled_by_ref = out["excluded_pooled_by_ref"] 
        cell_reps = out["cell_reps"] 


        if debug:
            breakpoint()
            " make raw psi df "

        raw_psi_df = make_raw_psi_df(**out, references = query_ref_nums, intron1_sizes = intron1_sizes)
        matched_df = pd.DataFrame(matched_df)
        raw_psi_df = pd.DataFrame(raw_psi_df)

        if debug:
            breakpoint()
            " compare raw to main "

        same, not_same, by_cell, dne = compare_raw_to_main(main_df, raw_psi_df)
        same_refs = [i["Reference"] for i in same]
        not_same_refs = [i["Reference"] for i in not_same]
    else:
        matched_df = combine_main_and_alt(main_df, alt_isoforms, supertable2)
        matched_df.to_csv(output_path, index = False)
        print("done")
        sys.exit(0)

    for cell in cell_types2:
        matched_df[f"{cell}_dlogit"] = logit(clip(matched_df[f"{cell}_pooled_psi_raw_var"])) - logit(clip(matched_df[f"{cell}_pooled_psi_raw_ref"]))
    matched_df["mean_dlogit"] = matched_df[[f"{cell}_dlogit" for cell in cell_types2]].mean(axis=1)
    # matched_df.to_csv(output_path, index = False)

    if debug:
        if len(not_same):
            breakpoint()
            " not same "
            ref = not_same[0]["Reference"]
            psi1 = None
            psi2 = None
            for i in not_same[0]:
                if "psi" in i and not compare(float(not_same[0][i][0]), float(not_same[0][i][1])):
                    key = i
                    psi1 = not_same[i][0]
                    psi2 = not_same[i][1]
            from psi_from_raw_data import seq_len_to_junctions
            index_in_not_same = [i for i in range(len(query_ref_nums)) if query_ref_nums[i] == ref][0]
            i1_len = intron1_sizes[index_in_not_same]
            junctions = seq_len_to_junctions(i1_len=i1_len)
            junc_key = (junctions[0][1], junctions[1][0])
            main_df_row = main_df[main_df["Reference"] == ref]
            print("NOT SAME:")
            try:
                print("ref", ref)
                print(key)
                print("junction", junc_key)
                print("psi (MAIN)", psi1)
                print("psi (RAW)", psi2)
                print("psi_by_ref\n", psi_by_ref[str(ref)][junc_key])
            except Exception as e:
                print(e)

    if debug:
        breakpoint()
        " combining df "

    combined_df = combine_all_dataframes(raw_df=raw_psi_df,
                                         main_df=main_df,
                                         matched_df=matched_df,
                                         alt_isoforms=alt_isoforms,
                                         query_ref_nums=query_ref_nums,
                                         intron1_sizes=intron1_sizes,
                                         supertable=supertable)

    if debug:
        breakpoint()
        " saving combined "

    combined_df.to_csv(output_path, index=False)

    print("done")
