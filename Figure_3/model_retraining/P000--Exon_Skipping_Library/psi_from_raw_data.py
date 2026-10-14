import os
import numpy as np
import getopt
import sys
import pandas as pd
from collections import defaultdict


def parse_cell_file(cf):
    return os.path.split(cf)[-1].split("_")[:2]


def print_dict_r(d):
    if isinstance(d, dict):
        for i in d.keys():
            print(i)
            print_dict_r(d[i])
            break
    else:
        return


def parse_query_ref_nums(query_ref_nums):
    if isinstance(query_ref_nums, str) and query_ref_nums.lower() == "all":
        return None, None
    q = query_ref_nums.split(",")
    return set(q), q 


def parse_intron1_sizes(intron1_sizes):
    if not intron1_sizes:
        return []
    s = intron1_sizes.split(",")
    return s



def read_line(count_file_buffer, header, sep):
    """
    Reads line of the given buffer and returns
    the information as a dictionary
    """
    info = {}
    line = count_file_buffer.readline().split(sep) 
    if len(line) == 1:
        return info
    for i, h in enumerate(header):
        line[i] = line[i].split("\n")[0]
        info[h] = line[i]
    return info


def files_to_dir(directory_):
    dirs = []
    def search(directory):
        if not os.path.isdir(directory):
            return
        if "Rep" in directory:
            dirs.append(directory)
        for d in os.listdir(directory):
            search(os.path.join(directory, d))
    search(directory_)
    return dirs


def group_dirs_by_cell(dirs):
    cell_dirs = {}
    for d in dirs:
        cell, _ = parse_cell_file(os.path.split(d)[-1])
        if cell not in cell_dirs:
            cell_dirs[cell] = []
        cell_dirs[cell].append(d)
    return cell_dirs


def sort_by_rep(cell_dirs):
    for cell in cell_dirs:
        cell_dirs[cell] = sorted(cell_dirs[cell])
    return cell_dirs


def get_cell_file_names(cell_dirs):
    cell_file_names = []
    for cd in cell_dirs:
        cell_files = os.listdir(cd)
        for cf in cell_files:
            if cf.endswith("splicing_counts.txt"):
                cell_file_names.append(os.path.join(cd, cf))
    return cell_file_names


def remove_duplicate_files(file_names):
    """ 
    remove duplicate cell, rep pairs in files.
    """
    no_dupes = {}
    for name in file_names:
        path, file_name = os.path.split(name)
        cell, rep = parse_cell_file(file_name)
        if "HMC3" in path or "MCF7" in path:
            if "2024_" not in path:
                continue
        no_dupes[(cell, rep)] = os.path.join(path, file_name)
    return [no_dupes[key] for key in no_dupes]


def search_directory(directory):
    """
    searches raw file directory and separates file names
    by cell and replicate
    """
    dirs = files_to_dir(directory)
    cell_file_names = get_cell_file_names(dirs) 
    cell_file_names = remove_duplicate_files(cell_file_names)
    cell_file_names = sorted(cell_file_names)
    return cell_file_names


def sum_over_junctions(total_pooled):
    sums = {}
    for junction in total_pooled:
        for cell in total_pooled[junction]:
            if cell not in sums:
                sums[cell] = 0
            sums[cell] += total_pooled[junction][cell]
    return sums


def seq_len_to_junctions(i1_len = None, ex_len = None):
    if i1_len is None and ex_len is None:
        raise ValueError("at least one of 'i1_len' or 'ex_len' is a required input")
    if i1_len is None:
        i1_len = 161 - 20 - ex_len
    elif ex_len is None:
        ex_len = 161 - 20 - i1_len
    SHARED_5P = 286
    return (26, SHARED_5P + i1_len), (SHARED_5P + i1_len + ex_len + 1, 871), (26, 871)


def junction_to_exon_startend(junction):
    ja, jb = junction
    i1_len = ja - 286
    ex_start = i1_len
    ex_size = jb - ja - 1
    ex_end = ex_start + ex_size - 1
    return ex_start, ex_end


def junction_to_seq_sizes(junction):
    ja, jb = junction
    ex_size = jb - ja - 1
    i2_size = 20
    i1_size = 161 - 20 - ex_size
    return i1_size, ex_size, i2_size


def get_missing_sizes(i1_len = None, ex_len = None, i2_len = None, cassette_size = 161):
    """
    using the fact all sequences are length 161
    and i2_len is always 20, return the missing sizes
    requiring only i1_len or ex_len to be passed in.
    
    Allows for i2_len != 20 to be passed in, which shouldn't happen
    """
    if i2_len is None:
        i2_len = 20
    if i1_len is None and ex_len is None:
        raise ValueError("at least one of 'i1_len' or 'ex_len' is a required input")
    if i1_len is None:
        i1_len = cassette_size - i2_len - ex_len
    elif ex_len is None:
        ex_len = cassette_size - i2_len - i1_len
    return i1_len, ex_len, i2_len


def filter_junctions_to_exon_size(total_pooled):
    exon_sizes = {}
    for junction_a, junction_b in total_pooled:
        exon_size = junction_b - junction_a - 2
        if exon_size not in exon_sizes:
            exon_sizes[exon_size] = []
        exon_sizes[exon_size].append(((junction_a, junction_b), total_pooled[(junction_a, junction_b)]))
    return exon_sizes


def calc_cell_psi(splice_junctions, lim_5p = "26", lim_3p = "871", debug = False, min_rep = 2):
    """
    calc is short for calculate
    
    calculates raw psi for each splice isoform
    for each cell type
    splice_junctions:
        dict(i1, i2) -> dict(cell, true_rep) -> count

    junctions at (lim_5p, lim3p) are assumed to be excluded counts
    included counts use two junctions in splice_junctions,
    where the first starts at lim_5p and the other ends at lim_3p.

    the included count is defined as the minimum read count between
    the 5p and 3p junctions.

    psi = included_count / (included_count + excluded_count)
    total = included_count + excluded_count

    included_pooled is defined as the sum of included_count across replicates.

    psi_pooled = included_pooled / (included_pooled + excluded_pooled)
    
    returns psi, total, excluded_count for all cells (pooled and for each replicate)
  
    the debug option may be used to insert breakpoints in this function and 
    print intermediates
    """
    cell_psi = defaultdict(dict)
    included = {}
    excluded = {}
    included_pooled = {}
    excluded_pooled = {}
    total_pooled = {}
    cell_reps = set()
    junc_keys = set()
    debug_print = print if debug else lambda *x : x
    
    # first pass: get exclusion counts per rep
    for junction in splice_junctions:
        if junction == (lim_5p, lim_3p):
            for cell, rep in splice_junctions[junction]:
                cell_reps.add((cell, rep))
                assert (cell, rep) not in (excluded), (excluded, cell, rep)
                excluded[(cell, rep)] = int(splice_junctions[junction][(cell, rep)])

    debug_print("EXCLUDED POOLED\n", excluded_pooled)
    if debug:
        breakpoint()
        " start second pass "

    # second pass: calculate included counts for all pairs
    for junction1 in splice_junctions:
        for junction2 in splice_junctions:
            if junction1 == junction2:
                continue
            if junction1 == (lim_5p, lim_3p):
                excluded_junction = junction1 
            if junction1[0] != lim_5p or junction2[1] != lim_3p:
                continue
            s1, e1 = junction1
            s1 = int(s1)
            e1 = int(e1)
            s2, e2 = junction2
            s2 = int(s2)
            e2 = int(e2)
            if e1 >= s2:
                continue
            junc_key = (e1, s2)
            junc_keys.add(junc_key)
            for cell, rep in splice_junctions[junction1]:
                cell_reps.add((cell, rep))
                c1 = int(splice_junctions[junction1][(cell, rep)])
                try:
                    c2 = int(splice_junctions[junction2][(cell, rep)])
                except KeyError:
                    c2 = 0
                if junc_key not in included:
                    included[junc_key] = {}
                included[junc_key][(cell, rep)] = min(c1, c2)


    # third pass: calculate rep psi
    for cell, rep in cell_reps:
        for junction in junc_keys:
            try:
                incl = included[junction].get((cell, rep), 0)
                excl = excluded.get((cell, rep), 0)
                total = incl + excl
            except KeyError as e:
                print(e)
                breakpoint()
            if total < 10:
                cell_psi[junction][f"{cell}_rep{rep}"] = np.nan
            else:
                cell_psi[junction][f"{cell}_rep{rep}"] = incl / (incl + excl)
                 

    debug_print("INCLUDED\n", included)
    if debug:
        breakpoint()
        " start third pass "

    reps_counts = defaultdict(dict)


    # third pass: calculate pooled counts and pooled psi based on passing
    for junction in junc_keys:
        if junction not in included_pooled:
            included_pooled[junction] = {}
        if junction not in excluded_pooled:
            excluded_pooled[junction] = {} 

        for cell, rep in cell_reps:
            if cell not in reps_counts[junction]:
                reps_counts[junction][cell] = 0
            if cell not in included_pooled[junction]:
                included_pooled[junction][cell] = 0
            if cell not in excluded_pooled[junction]:
                excluded_pooled[junction][cell] = 0
            psi = cell_psi[junction][f"{cell}_rep{rep}"]

            if not np.isnan(psi):
                reps_counts[junction][cell] += 1
                if junction in included:
                    included_pooled[junction][cell] += included[junction].get((cell, rep), 0)
                if (cell, rep) in excluded:
                    excluded_pooled[junction][cell] += excluded[(cell, rep)]

        for cell in included_pooled[junction]:
            if reps_counts[junction][cell] < min_rep:
                cell_psi[junction][f"{cell}_pooled"] = np.nan
                included_pooled[junction][cell] = np.nan
                excluded_pooled[junction][cell] = np.nan
            else:
                inclp = included_pooled[junction].get(cell, 0)
                exclp = excluded_pooled[junction].get(cell, 0)
                cell_psi[junction][f"{cell}_pooled"] = inclp / (inclp + exclp)


    if debug:
        breakpoint()
        " returning "

    out = {}
    out["cell_psi"] = cell_psi
    out["included_pooled"] = included_pooled
    out["excluded_pooled"] = excluded_pooled
    out["included"] = included
    out["excluded"] = excluded
    return out


def file_reader(file_name, refs, sep, rep, cell, ref_key = "Parent_ref"):
    """
    for a given file, returns a dictionary
    mapping each reference number (along with the given cell name and replicate)
    to information from that file, using header as keys
    """
    with open(file_name, "r") as f:
        header = f.readline().split(sep) 
        header[-1] = header[-1].split("\n")[0]
        info_out = {ref : {h : [] for h in header} for ref in refs}
        while True:
            info = read_line(f, header, sep)
            if len(info) == 0:
                return info_out
            if refs is not None and info[ref_key] not in refs:
                continue
            else:
                ref_ = info[ref_key]
                for h in header:
                    info_out[ref_][h].append(info[h])
    return info_out


def splice_junc_from_ref(info, ref, cell, rep):
    data = info[ref]
    sjunc = {}
    for i in range(len(data["SJ_5"])):
        if (data["SJ_5"][i], data["SJ_3"][i]) in sjunc:
            raise KeyError(f"Duplicate key {(data['SJ_5'][i], data['SJ_3'][i])}") 
        sjunc[(data["SJ_5"][i], data["SJ_3"][i])] = data["Count"][i] 
    return sjunc


def splice_junc_from_file(cell_file, ref_nums, sep = "\t"):
    cell, rep = parse_cell_file(cell_file)
    info = file_reader(cell_file, ref_nums, sep, rep, cell, ref_key = "Parent_ref")
    ref_counts = {}
    for ref in info:
        ref_counts[ref] = splice_junc_from_ref(info, ref, cell, rep)
    return ref_counts, rep, cell


def refs_from_file(cell_file, sep = "\t"):
    with open(cell_file, "r") as f:
        header = f.readline().split(sep) 
        refs = set()
        while True:
            info = read_line(f, header, sep)
            if len(info) == 0:
                return refs
            refs.add(info["Parent_ref"])
    return refs


def true_rep_mapper(cell_file_names):
    counter  = {}
    true_map = {}
    for cf in cell_file_names:
        cell, rep = parse_cell_file(cf)
        if cell in counter:
            counter[cell].append(rep)
        else:
            counter[cell] = [rep]
    for cell in counter:
        counter[cell] = sorted(counter[cell])
        for i in range(len(counter[cell])):
            true_map[(cell, counter[cell][i])] = i + 1
    return true_map


def check_query_refs(query_ref_nums):
    """
    expected format of query_ref_nums 
    is a set of strings. Handles cases
    where it is a list and/or if the
    elements are integers.
    """
    new_queries = set()
    for i in query_ref_nums:
        new_queries.add(str(i))
    return new_queries


def group_file_info_by_ref(counts_by_ref, file_counts, true_rep, cell):
    for ref in file_counts:
        if ref not in counts_by_ref:
            counts_by_ref[ref] = {}
        for junc in file_counts[ref]:
            if junc not in counts_by_ref[ref]:
                counts_by_ref[ref][junc] = {}
            counts_by_ref[ref][junc][(cell, true_rep)] = file_counts[ref][junc]
    return counts_by_ref


def refs_from_raw_file_dir(directory, sep = "\t"):
    cell_file_names = search_directory(directory)
    true_rep_map = true_rep_mapper(cell_file_names)
    all_refs = set()
    for cell_file in cell_file_names:
        refs = refs_from_file(cell_file, sep)
        for r in refs:
            all_refs.add(r)
    return all_refs


def psi_from_raw_file_dir(directory, query_ref_nums, sep = "\t"):
    cell_file_names = search_directory(directory)
    true_rep_map = true_rep_mapper(cell_file_names)
    query_ref_nums = check_query_refs(query_ref_nums)
    # ref -> junction -> cell_psi_rep, cell_psi_pooled
    psi_by_ref = {}
    included_by_ref = {}
    counts_by_ref = {}
    excluded_by_ref = {}
    excluded_pooled_by_ref = {}
    included_pooled_by_ref = {}
    cell_reps = []

    for cell_file in cell_file_names:
        # ref -> junction -> count for (cell, rep)
        ref_counts, rep, cell = splice_junc_from_file(cell_file, query_ref_nums, sep)
        true_rep = true_rep_map[(cell, rep)]
        cell_reps.append((cell, true_rep))
        # ref -> junction -> (cell, true_rep) -> count
        counts_by_ref = group_file_info_by_ref(counts_by_ref, ref_counts, true_rep, cell)

    print("calculating PSIs")
    breakpoint()
    for ref in counts_by_ref:
        try:
            psi_results = calc_cell_psi(counts_by_ref[ref]) 
        except KeyError as e:
            import traceback
            print(ref)
            traceback.print_exc()
            breakpoint()
        psi_by_ref[ref]   = psi_results["cell_psi"]
        included_by_ref[ref] = psi_results["included"]
        excluded_by_ref[ref] = psi_results["excluded"]
        included_pooled_by_ref[ref] = psi_results["included_pooled"]
        excluded_pooled_by_ref[ref] = psi_results["excluded_pooled"]

    out = {}
    out["psi_by_ref"] = psi_by_ref
    out["included_by_ref"] = included_by_ref
    out["excluded_by_ref"] = excluded_by_ref
    out["included_pooled_by_ref"] = included_pooled_by_ref
    out["excluded_pooled_by_ref"] = excluded_pooled_by_ref
    out["cell_reps"] = cell_reps
    return out


def add_to_df(df,
              psi_by_ref,
              included_by_ref,
              excluded_by_ref,
              included_pooled_by_ref,
              excluded_pooled_by_ref,
              ref,
              junction,
              intron1_size=None,
              cell_reps=None,
              min_reps = 2):
    """
    Main logic to convert psi, total, and excluded dictionaries
    into rows of a pandas dataframe
    """
    if cell_reps is None:
        cell_reps = [i for i in excluded_by_ref.keys()]
    df["Reference"].append(int(ref))
    df["5p_Exon"].append(int(junction[0]))
    df["3p_Exon"].append(int(junction[1]))
    if intron1_size is not None:
        df["intron1_size"].append(int(intron1_size))
        i2_size = 20
        ex_size = 161 - int(intron1_size) - i2_size
    else:
        i1_size, ex_size, i2_size = junction_to_seq_sizes(junction)
        df["intron1_size"].append(i1_size)
    ex_start, ex_end = junction_to_exon_startend(junction)
    df["relative_exon_start"].append(int(ex_start))
    df["relative_exon_end"].append(int(ex_end))
    df["exon_size"].append(int(ex_size))
    df["intron2_size"].append(int(i2_size))

    cells = set()
    psi_pooled = {}
    total = {}
    total_included = {}
    total_excluded = {}
    missing_passed_reps = defaultdict(int)
    for cell, rep in cell_reps:
        cells.add(cell)
        rep_key = f"{cell}_rep{rep}"
        pool_key = f"{cell}_pooled"
        if junction not in psi_by_ref[ref]:
            if (cell, rep) in excluded_by_ref[ref]:
                psi = 0.0
                incl = 0
                excl = excluded_by_ref[ref][(cell, rep)]
            else:
                psi = np.nan
                incl = np.nan
                excl = np.nan

            # check if exclusion counts pass for non-existing junction
            # in at least min_reps
            if excluded_by_ref[ref].get((cell, rep), 0) >= 10:
                missing_passed_reps[cell] += 1
                if cell not in total_excluded:
                    total_excluded[cell] = 0
                total_excluded[cell] += excluded_by_ref[ref][(cell, rep)]

        elif junction in psi_by_ref[ref]:
            if rep_key in psi_by_ref[ref][junction]:
                psi = psi_by_ref[ref][junction][rep_key]
                if not np.isnan(psi):
                    incl = included_by_ref[ref][junction].get((cell, rep), 0)
                else:
                    incl = np.nan
                excl = excluded_by_ref[ref].get((cell, rep), 0)
            elif (cell, rep) in excluded_by_ref[ref]:
                psi = 0.0
                incl = 0
                excl = excluded_by_ref[ref]
            else:
                psi = np.nan
                incl = np.nan
                excl = np.nan
            if pool_key in psi_by_ref[ref][junction]:
                psi_pooled[cell] = psi_by_ref[ref][junction][pool_key]
                if not np.isnan(psi_by_ref[ref][junction][pool_key]):
                    total[cell]  = included_pooled_by_ref[ref][junction][cell]
                    total[cell] += excluded_pooled_by_ref[ref][junction][cell]
                    total_included[cell] = included_pooled_by_ref[ref][junction][cell]
                    total_excluded[cell] = excluded_pooled_by_ref[ref][junction][cell]
            elif junction in excluded_pooled_by_ref[ref] and cell in excluded_pooled_by_ref[ref][junction]:
                psi_pooled[cell] = 0.0
                total[cell] = excluded_pooled_by_ref[ref][cell]
                total_excluded[cell] = excluded_pooled_by_ref[ref][cell]
            else:
                psi_pooled[cell] = np.nan
                total[cell] = np.nan
                total_excluded[cell] = np.nan
                total_included[cell] = np.nan

        df[f"{rep_key}_psi_raw"].append(psi)
        df[f"{rep_key}_included"].append(incl)
        df[f"{rep_key}_excluded"].append(excl)

    for cell in cells:
        # only in the case the junction was missing and the 
        # excluded counts passed, make the psi zero, the inclusion count zero,
        # and pool the exclusion counts
        if missing_passed_reps[cell] >= min_reps:
            df[f"{cell}_pooled_psi_raw"].append(0.0)
            df[f"{cell}_pooled_total"].append(total_excluded[cell])
            df[f"{cell}_pooled_included"].append(0)
            df[f"{cell}_pooled_excluded"].append(total_excluded[cell])
        else:
            df[f"{cell}_pooled_psi_raw"].append(psi_pooled.get(cell, np.nan))
            df[f"{cell}_pooled_total"].append(total.get(cell, np.nan))
            df[f"{cell}_pooled_included"].append(total_included.get(cell, np.nan))
            df[f"{cell}_pooled_excluded"].append(total_excluded.get(cell, np.nan))

    return df


def get_expected_reps(cell_file_names, true_rep_map):
    cell_reps = []
    for cf in cell_file_names:
        cell, rep = parse_cell_file(cf)
        true_rep = true_rep_map((cell, rep))
        cell_reps.append((cell, true_rep))
    return cell_reps


def print_row(df):
    keys = df.keys()
    row = df.iloc[0]
    for i in sorted(keys):
        print(i.ljust(25), row[i])
    return


def output_df(psi_by_ref, included_by_ref, excluded_by_ref, included_pooled_by_ref, excluded_pooled_by_ref, references, intron1_sizes, cell_reps = None):
    df = defaultdict(list)
    dupes = set()
    if references is None:
        ref_list = list(psi_by_ref.keys())
        intron1_sizes = [] # can't match intron1 sizes 
    else:
        ref_list = [str(i) for i in references]
    added_times = 0

    for i, ref in enumerate(ref_list):
        if (ref, intron1_sizes[i]) in dupes:
            continue
        dupes.add((ref, intron1_sizes[i]))
        if intron1_sizes:
            sja, sjb, sje = seq_len_to_junctions(i1_len = int(intron1_sizes[i]))
            key = (sja[1], sjb[0])
            added_times += 1
            try:
                df = add_to_df(df, psi_by_ref=psi_by_ref,
                          included_by_ref=included_by_ref,
                          excluded_by_ref=excluded_by_ref,
                          included_pooled_by_ref=included_pooled_by_ref,
                          excluded_pooled_by_ref=excluded_pooled_by_ref,
                          ref=ref,
                          junction=key,
                          intron1_size = intron1_sizes[i],
                          cell_reps = cell_reps)
            except KeyError as err:
                import traceback
                traceback.print_exc()
                breakpoint()
                " key error in add to df "
        else:
            raise NotImplementedError("Need to debug getting PSIs from just the junctions")
            for key in total_by_ref[ref]:
                df = add_to_df(df, psi_by_ref=psi_by_ref,
                          total_by_ref=total_by_ref,
                          excluded_by_ref=excluded_by_ref,
                          excluded_pooled_by_ref=excluded_pooled_by_ref,
                          ref=ref,
                          junction=key,
                          cell_reps = cell_reps,
                          min_rep = min_ref)
    df = pd.DataFrame(df)
    df["intron1_size"] = df["intron1_size"].astype(int)
    df["exon_size"] = df["exon_size"].astype(int)
    df["intron2_size"] = df["intron2_size"].astype(int)
    df["relative_exon_start"] = df["relative_exon_start"].astype(int)
    return df


help_="""
Goes through the raw files with read counts and
computes PSI for each junction at the given ref
args:
directory      (input dir containing raw files)
query_ref_nums (return psis for specific ref nums)
output_path    (outputs psi for ref queries to this path)
"""
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:],"", [
    "directory=",
    "query_ref_nums=",
    "output_path=",
    "intron1_sizes=",
    "debug",
    "help",
    ]) 
    opts = dict(opts)
    print(opts)
    if "--debug" in opts:
        directory = "/storage/yu/Projects/P000--ESL_continuation/Analysis/STAR_alignment/separate_concat_2023_09_19_d1c_ms75_from_s3_2023_11_27/recount_SJs"
        query_ref_nums = "4"
        output_path = "TEST.csv"
        intron1_sizes = "54"
        debug = True    
    else:
        directory = opts["--directory"]
        query_ref_nums = opts["--query_ref_nums"]
        output_path = opts["--output_path"]
        intron1_sizes = opts.get("--intron1_sizes", "")
        debug = False
    query_ref_nums_set, query_ref_nums = parse_query_ref_nums(query_ref_nums)
    print(query_ref_nums_set)
    intron1_sizes  = parse_intron1_sizes(intron1_sizes)
    searched = search_directory(directory)
    out = psi_from_raw_file_dir(directory, query_ref_nums_set)
    psi_by_ref             = out["psi_by_ref"]
    included_by_ref        = out["included_by_ref"]
    excluded_by_ref        = out["excluded_by_ref"]
    included_pooled_by_ref = out["included_pooled_by_ref"]
    excluded_pooled_by_ref = out["excluded_pooled_by_ref"]
    cell_reps              = out["cell_reps"]


    if debug:
        breakpoint()
        " making df "
    df = output_df(psi_by_ref             = psi_by_ref,
                   included_by_ref        = included_by_ref,
                   excluded_by_ref        = excluded_by_ref,
                   included_pooled_by_ref = included_pooled_by_ref,
                   excluded_pooled_by_ref = excluded_pooled_by_ref,
                   references             = query_ref_nums,
                   intron1_sizes          = intron1_sizes,
                   cell_reps              = cell_reps)
    df = pd.DataFrame(df)
    df.to_csv(output_path, index = False)
    if debug:
        breakpoint()
    print("done")
