import sys
import os

# search parent directories for project name
project_name = "P000--Exon_Skipping_Library"
node = os.getcwd()
for i in range(5): # max depth
    tree, node = os.path.split(node)
    if node == project_name:
        break 
    else:
        node = tree
PROJECT_ROOT = os.path.join(tree, node)
if node != project_name:
    # couldn't find project folder 
    raise ModuleNotFoundError(
        f"Could not find project directory {PROJECT_NAME}"
    )
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.load_data import load_splits_dir
from src.esl import logit


import numpy as np
import getopt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def num_all_diff(x1, x2):
    num_diff = 0
    for i in range(len(x1)):
        for j in range(len(x1[0])):
            num_diff += np.sum(x1[i][j] != x2[i][j]) 
    return num_diff


def num_diff(x1, x2):
    return np.sum(x1 != x2)


def check_ref(x, meta):
    """
    verifies that, if those sequences
    share a ref id and isoform, then
    they are the exact same sequence.
    """
    first_refs = {}
    if "ref_id" in meta:
        id_key = "ref_id"
    else:
        id_key = "Reference"
    ref_ids = meta[id_key].tolist()
    intron1_lens = meta["intron1_len"].tolist()
    for i, ref in enumerate(meta[id_key].tolist()):
        if ref not in first_refs:
            first_refs[ref] = i

    for i in range(len(meta)):
        ref = ref_ids[i]
        for a, all_seqs in enumerate(ref_x):
            seq_first_ref = all_seqs[first_refs[ref]]
            seq = all_seqs[i]
            len1 = intron1_lens[i]
            len2 = intron1_lens[first_refs[ref]]
            if len1 != len2:
                continue
            n_diff = num_diff(seq, seq_first_ref)
            if n_diff:
                breakpoint()
            assert n_diff == 0, (i, n_diff)
    return


def check_var(x, meta):
    """
    Pairwise check every variant
    to see if they are different
    when expected (different id or isoform)
    and same when expected (same id and isoform).
    """
    var_ids = meta["var_id"].tolist()
    intron1_lens = meta["intron1_len"].tolist()
    var_id_map = {}
    var_x = x[5:10] + x[12:14]
    for i in range(len(meta)):
        for j in range(len(meta)):
            i1_i = intron1_lens[i]
            i1_j = intron1_lens[j]
            id_i = var_ids[i]
            id_j = var_ids[j]
            n_diff = 0
            for s, all_seqs in enumerate(var_x):
                seq_i = all_seqs[i]
                seq_j = all_seqs[j]
                n_diff += num_diff(seq_i, seq_j)
            if i1_i != i1_j:
                assert n_diff > 0, (i, j)
            elif i1_i == i1_j and id_i == id_j:
                assert n_diff == 0, (i, j)
            elif id_i != id_j:
                assert n_diff > 0, (i, j)
    return


def check_shapes_with_metadata(x, y, meta):
    """
    check data sizes match 
    between x, y and metadata
    """
    shapes = set()
    for i in x:
        shapes.add(i.shape[0])
    shapes.add(y.shape[0])
    shapes.add(len(meta))
    assert len(shapes) == 1
    return


def check_scale(y, scale="logit"):
    """
    for logit variant values
    checks if logit values are not dlogit values
    can't tell scale is dlogit or logit if the 
    values are small enough (i.e. tsplice cell type 
    effects are much lower, but forcing a minimum for
    dlogit values may wrongly label it as logit.
    """
    if scale == "logit":
        max_ = logit(1.0) + 1e-9
    elif scale == "dlogit":
        max_ = logit(1.0) - logit(0.0) + 1e-9
    assert np.nanmax(y) <= max_, (f"{np.nanmax(y)} not less than {max_}")
    assert np.nanmin(y) >= -max_, (f"{np.nanmin(y)} not more than {-max_}")
    return


help_="""
robusts check that data is OK
"""
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:],"", [
        "mtsplice_data_dir=",
        "tsplice_data_dir=",
        "debug",
        "help",
    ])
    opts = dict(opts)
    if "--help" in opts:
        print(help_)
        sys.exit(0)
    mtsplice_data_dir = opts["--mtsplice_data_dir"]
    tsplice_data_dir = opts["--tsplice_data_dir"]
    loaded, data_mt = load_splits_dir(mtsplice_data_dir)
    loaded, data_t = load_splits_dir(tsplice_data_dir)

    for i in ("train", "val", "test"):
        x = data_t["x_" + i]
        ref_x = x[0:2]
        y = data_t["y_" + i]
        m = data_t["metadata_" + i]
        check_ref(ref_x, m)
        check_shapes_with_metadata(x, y, m)
        check_scale(y, "dlogit")

        x = data_mt["x_" + i]
        ref_x = x[0:5] + x[10:12]
        y = data_mt["y_" + i]
        m = data_mt["metadata_" + i]
        check_ref(ref_x, m)
        check_shapes_with_metadata(x, y, m)
        check_scale(y, "logit")
        # check_var(x, m)
        

    logging.info("done")

