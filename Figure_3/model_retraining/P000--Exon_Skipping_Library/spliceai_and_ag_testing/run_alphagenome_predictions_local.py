# Imports (from alphagenome quickstart)
import os
import getopt
import sys
from collections import defaultdict
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

_help="""
Inputs:
    alphagenome_path
        where the alphagenome module is located, if it cannot be imported
    output_tsv
        path to output ag predictions and true values
    alphagenome_track
        tells which tracks to predict on 
        (SPLICE_SITES, SPLICE_SITE_USAGE or both)
    alphagenome_method 
        by default, predicts using predict_sequence
        on the variable region flanked
        by smn2 introns and citrine exons and 
        ~500kb context as N. Can also use
        'interval' to predict assuming the exon 
        is in the genomic context.
    alphagenome_context
        power of 2 amount of nucleotide context
        to add (including the input)
    use_strand_info
        if true, index the predictions using
        the metadata for different strands.
    acceptor_offset/donor_offset
        0 indicates to pick the first and last nucleotides of the exon
Outputs:
    the output is saved into output_tsv
"""


## Import Project
if __name__ == "__main__":
    import sys
    import os
    import getopt
    try:
        # try to import
        # (e.g. if PYTHONPATH is set or installed via pip install -e)
        import src 
    except ModuleNotFoundError:
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

try:
    from src.esl import logit
except ModuleNotFoundError:
    print("appending p000 path to sys")
    sys.path.append("..")

from src.utils.parsing import bool_map, ErrorWrapper
from src.load_data import read_csvs

from src.esl import (
    logit,
    cell_types,
    cell_types2
)
from src.esl import (
    SMN2_INTRON6,
    SMN2_INTRON7
)
from src.esl import (
    CITRINE_EXON1,
    CITRINE_EXON2
)

alphagenome_path = None
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:],"", [
        "alphagenome_path=",
        "merged_df_path=",
        "output_tsv=",
        "alphagenome_track=",
        "acceptor_offset=",
        "donor_offset=",
        "alphagenome_context=",
        "alphagenome_method=",
        "use_strand_info=",
        "parallel_split=",
        "help",
        "debug",
    ])
    opts = dict(opts)
    opts = ErrorWrapper(opts)
    print(opts, flush=True)
    if "--help" in opts:
        print(_help)
        sys.exit(0)

    if "--debug" in opts:
        debug = True
        alphagenome_path = os.path.join(
            "/storage", "yu", "Projects",
            "P000--ESL_continuation",
            "GitHub",
            "P000--Exon_Skipping_Library",
            "models", "alphagenome",
        )
        merged_df_path = os.path.join(
            "/storage", "yu", "Projects",
            "P000--ESL_continuation",
            "Figures_SK",
            "General_preprocessing",
            "output_5_23_2026",
            "merged_05_23_2026.csv",
        )
        output_tsv = "alphagenome_esl_predictions.tsv"
        alphagenome_track = "SPLICE_SITES"
        alphagenome_method = "default"
        alphagenome_context = 14
        prediction_acc_offset = 0
        prediction_don_offset = 0
        parallel_split = None
        use_strand_info = False
    # Load model (from alphagenome finetune)
    else:
        debug = False
        alphagenome_path = opts["--alphagenome_path"]
        merged_df_path = opts["--merged_df_path"]
        alphagenome_track = opts["--alphagenome_track"]
        valid_tracks = ["SPLICE_SITES", "SPLICE_SITE_USAGE"]
        if alphagenome_track not in valid_tracks:
            raise ValueError(
                f"Alpha genome track must be either "
                f"{valid_tracks[0]} or {valid_tracks[1]}"
            )
        if alphagenome_track == "SPLICE_SITE_USAGE":
            raise NotImplementedError("Splice site usage not yet implemented")
        valid_methods = ("default", "interval")
        alphagenome_method = opts["--alphagenome_method"]
        if alphagenome_method not in valid_methods:
            raise ValueError(
                f"AlphaGenome method must be either "
                f"{valid_methods[0]} or {valid_methods[1]}"
            ) 
        if alphagenome_method == "interval":
            raise NotImplementedError(
                "AlphaGenome 'inteval' method not yet implemented"
            )
        output_tsv = opts["--output_tsv"]
        alphagenome_context = int(opts["--alphagenome_context"])
        prediction_acc_offset = int(opts["--acceptor_offset"])
        prediction_don_offset = int(opts["--donor_offset"])
        use_strand_info = bool_map(opts["--use_strand_info"])
        parallel_split = opts.get("--parallel_split", "")
        if not parallel_split:
            parallel_split = None
        else:
            parallel_split = int(parallel_split)

    opts.check_inputs()
    print(opts, flush=True)


try:
    from alphagenome.data import gene_annotation
except ModuleNotFoundError:
    if alphagenome_path is not None:
        sys.path.append(alphagenome_path)
    else:
        raise ModuleNotFoundError(
            "Cannot import alphagenome and path not provided"
        )
    
from alphagenome.data import gene_annotation
from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.visualization import plot_components
from alphagenome_research.model import dna_model

import matplotlib
import numpy as np
import pandas as pd
import orbax.checkpoint as ocp
from alphagenome.models import dna_client


def get_splice_scores(seq):
    """
    Pad sequence to 1MB centered and return splice site values + offset.
    """
    padded = seq.center(SEQ_LEN, 'N')
    offset = CENTER - len(seq) // 2
    out = model.predict_sequence(
        sequence=padded,
        requested_outputs=[dna_model.OutputType.SPLICE_SITES],
        ontology_terms=None,  # tissue-agnostic, ontology ignored
    )
    return out.splice_sites.values, offset


N_CARDS = 4
def split_df_parallel(df, parallel_split, n_workers = N_CARDS):
    """
    Split dataframe into four to process in parallel.
    0-indexed (0, 1, 2, 3) for 4 workers
    """
    if parallel_split is None:
        return df
    N = len(df)
    tasks = N // n_workers
    if tasks == 0:
        return df
    start = parallel_split * tasks
    end = (parallel_split + 1) * tasks
    if parallel_split == n_workers - 1:
        end = len(df)
    return df.iloc[start : end]


def extract_ss_scores(
        values,
        offset,
        exon_start,
        exon_end,
        donor_idx,
        acceptor_idx
    ):
    """
    Extract 3'SS (acceptor) and 5'SS (donor) scores at exon boundaries.
    """
    pos_3ss = offset + exon_start - 1
    pos_5ss = offset + exon_end - 1
    acc_scores = values[pos_3ss, acceptor_idx]
    don_scores = values[pos_5ss, donor_idx]
    return acc_scores, don_scores


def compute_psi(values, offset, exon_start, exon_end):
    ACCEPTOR_IDX = 1
    DONOR_IDX = 0
    """
    PSI = SA * SD. AlphaGenome peaks at -1 from exon boundary (confirmed).
    """
    sa = float(values[offset + exon_start - 2, ACCEPTOR_IDX])
    sd = float(values[offset + exon_end   - 2, DONOR_IDX])
    return sa, sd, sa * sd


def get_splice_scores2(seq):
    seq_len = 2 ** 14
    center = seq_len // 2
    padded = seq.center(seq_len, 'N')
    offset = center - len(seq) // 2
    out = model.predict_sequence(
        sequence=padded,
        organism=dna_model.Organism.HOMO_SAPIENS,
        requested_outputs={dna_model.OutputType.SPLICE_SITES},
        ontology_terms=['UBERON:0000955'],
    ) #ontology term does not make a difference in prediction,
    # but this requires you have at least one, or pass None
    return out.splice_sites.values, offset


def indices_from_metadata(metadata, strand):
    pos_donor_idx_plus = metadata.index[
        (metadata['strand'] == '+')
        & (metadata['name'].str.contains('donor', case=False))
    ].tolist()
    pos_acceptor_idx_plus = metadata.index[
        (metadata['strand'] == '+')
        & (metadata['name'].str.contains('acceptor', case=False))
    ].tolist()
    pos_donor_idx_minus = metadata.index[
        (metadata['strand'] == '-')
        & (metadata['name'].str.contains('donor', case=False))
    ].tolist()
    pos_acceptor_idx_minus = metadata.index[
        (metadata['strand'] == '-')
        & (metadata['name'].str.contains('acceptor', case=False))
    ].tolist()
    if strand == "+":
        return pos_donor_idx_plus, pos_acceptor_idx_plus
    elif strand == "-":
        return pos_donor_idx_plus, pos_acceptor_idx_plus
    else:
        raise ValueError(
            f"Expected strand to be either '+' or '-', got {strand}"
            )


def compute_context_splice_scores(seq, context = 14):
    seq_len = min(context, 20)
    seq_len = 2 ** seq_len
    center = seq_len // 2
    padded = seq.center(seq_len, 'N')
    offset = center - len(seq) // 2
    out = model.predict_sequence(
        sequence=padded,
        organism=dna_model.Organism.HOMO_SAPIENS,
        requested_outputs={dna_model.OutputType.SPLICE_SITES},
        ontology_terms=None,
    )
    return out.splice_sites.values, out.splice_sites.metadata, offset


def extract_splice_site_probs(
        all_site_probs,
        exon_start,
        exon_end,
        context_offset,
        don_idx=0,
        acc_idx=1,
        index_offset_acc=-1,
        index_offset_don=-1
    ):
    """
    updated function to get ss scores with options
    for different nt offsets and different prediction indices
    the -1 is to get the offset at 0 to be the default SA and SD positions
    """
    acc_site = context_offset + exon_start - 1 + index_offset_acc
    don_site = context_offset + exon_end - 1 + index_offset_don
    acc_5p = all_site_probs[acc_site, acc_idx]
    don_5p = all_site_probs[acc_site, don_idx]
    acc_3p = all_site_probs[don_site, acc_idx]
    don_3p = all_site_probs[don_site, don_idx]
    don_scores = all_site_probs[don_site, don_idx]
    return {
        "5p_acceptor": acc_5p,
        "5p_donor": don_5p,
        "3p_acceptor": acc_3p,
        "3p_donor": don_3p,
    }


def joint_probability(ref_probabilities, var_probabilities):
    """
    computes the probability of both site usage
    by multiplying probabilities together
    """
    ref_acc = ref_probabilities["5p_acceptor"]
    ref_don = ref_probabilities["3p_donor"]
    var_acc = var_probabilities["5p_acceptor"]
    var_don = var_probabilities["3p_donor"]
    joint_ref = ref_acc * ref_don
    joint_var = var_acc * var_don
    logit_joint_ref = logit(joint_ref)
    logit_joint_var = logit(joint_var)
    dlogit = logit_joint_var - logit_joint_ref
    dpsi = joint_var - joint_ref
    outputs = {"dlogit":dlogit,
               "logit_ref":logit_joint_ref,
               "logit_var":logit_joint_var,
               "ref_psi":joint_ref,
               "var_psi":joint_var,
               "dpsi":dpsi}
    return outputs


def get_sequence_info(
        row,
        ref_var,
        ag_seq_context,
    ):
    """
    Builds full sequence from introns and exons
    plus AlphaGenome context.
    Returns the full sequence and the exon start ends
    in order to extract the splice site predictions.
    """
    key = "_ref" if ref_var == "ref" else ""
    core = (
        row["intron1" + key] 
        + row["exon" + key]
        + row["intron2" + key]
    )
    prefix, suffix = ag_seq_context
    full  = prefix + core.upper().replace("U", "T") + suffix
    exon_start = len(prefix) + len(row["intron1" + key])
    exon_end   = exon_start + len(row["exon" + key]) - 1
    return {
        "full" : full,
        "exon_start" : exon_start, 
        "exon_end" : exon_end,
    }  


def compute_probability(seq_info, ag_info):
    """
    Computes the splice acceptor
    and splice donor probabilities
    by predicting on the entire sequence
    and then querying positions of the splice
    sites within the 161nt exon skipping window
    """
    values, metadata, offset = compute_context_splice_scores(
        seq_info["full"],
        context=ag_info["context"],
    )
    exon_start = seq_info["exon_start"]
    exon_end = seq_info["exon_end"]
    probabilities = extract_splice_site_probs(
        values,
        exon_start,
        exon_end,
        context_offset=offset,
        don_idx=0,
        acc_idx=1,
        index_offset_acc=ag_info["prediction_acc_offset"],
        index_offset_don=ag_info["prediction_don_offset"],
    )
    probabilities["values"] = values
    probabilities["offset"] = offset
    return probabilities


def to_df(
        df,
        row,
        ag_results,
        ref_probabilities,
        var_probabilities,
        data_dlogit,
        write_path="",
        header_written=True,
    ):
    # TODO: reinplement header_written and write_path
    # functionality (write at each iteration)
    """
    turns current row alphagenome predictions
    and metadata into a dataframe, then adds to 
    df and appends to the given path if non empty.
    """
    df["ref_id"].append(str(row["ref_id"]))
    df["var_id"].append(str(row["var_id"]))
    df["snp"].append(str(row["snp"]))
    df["event_id"].append(str(row["event_id_161"]))
    df["alphagenome_dlogit"].append(float(ag_results["dlogit"]))
    df["alphagenome_dpsi"].append(float(ag_results["dpsi"]))
    df["alphagenome_ref_psi"].append(float(ag_results["ref_psi"]))
    df["alphagenome_var_psi"].append(float(ag_results["var_psi"]))
    for i in ref_probabilities:
        if i == "values" or i == "offset":
            continue
        df[f"alphagenome_ref_{i}"].append(float(ref_probabilities[i]))
    for i in var_probabilities:
        if i == "values" or i == "offset":
            continue
        df[f"alphagenome_var_{i}"].append(float(var_probabilities[i]))

    df["experimental_dlogit"].append(float(data_dlogit))
    if write_path:
        row.to_csv(
            write_path,
            sep='\t',
            mode='a',
            index=False,
            header=not header_written
        )
    return df


if __name__ == "__main__":
    # Flags to improve determinism.
    os.environ['XLA_FLAGS'] = ' '.join([
        '--xla_gpu_deterministic_ops',
        '--xla_gpu_enable_scatter_determinism_expander=True',
        '--xla_gpu_enable_triton_gemm=False',
    ])
    # Increase GPU and CPU memory to reduce out of memory errors.
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

    checkpointer = ocp.StandardCheckpointer()
    params_base, state_base = checkpointer.restore(alphagenome_path)
    default_settings_human = dna_model.default_organism_settings()[
        dna_model.Organism.HOMO_SAPIENS
    ]
    model = dna_model.create(
        alphagenome_path,
        organism_settings={
            dna_model.Organism.HOMO_SAPIENS: default_settings_human
        },
    )
    
    # read in ESL data
    merged_df = pd.read_csv(merged_df_path)
    merged_df = split_df_parallel(merged_df, parallel_split)
    if parallel_split is not None:
        output_tsv = output_tsv.replace(
            ".tsv",
            f"_split_{parallel_split}.tsv"
        )

    prefix = CITRINE_EXON1 + SMN2_INTRON6
    suffix = SMN2_INTRON7 + CITRINE_EXON2

    SEQ_LEN = dna_client.SEQUENCE_LENGTH_1MB
    CENTER  = SEQ_LEN // 2

    # === INSPECT TRACKS ONCE ===
    _test = model.predict_sequence(
        sequence = 'N' * SEQ_LEN,
        requested_outputs=[dna_model.OutputType.SPLICE_SITES],
        ontology_terms=None)

    # Get metadata
    splice_meta = _test.splice_sites.metadata
    delta_logit_pooled = [
        f"{cell}_delta_logit_pooled"
        for cell in cell_types
    ]
    
    header_written = False
    ag_info = {
        "prediction_don_offset" : prediction_don_offset,
        "prediction_acc_offset": prediction_acc_offset,
        "context" : alphagenome_context,
    }
    if debug:
        merged_df = merged_df[merged_df["event_id_161"] == "chr10:101775110-101775270:-"]
    cached_ref_values = {}
    ag_seq_context = [prefix, suffix]
    final_df = defaultdict(list)
    logging.info("Starting alpha genome predictions")
    from tqdm import tqdm
    for i, row in tqdm(
            merged_df.iterrows(), 
            total=len(merged_df),
            disable=not sys.stderr.isatty(),
        ):
        ref_id = row["ref_id"]
        if ref_id in cached_ref_values:
            ref_info = cached_ref_values[ref_id]
        else:
            ref_info = get_sequence_info(row, "ref", ag_seq_context)
            cached_ref_values[ref_id] = ref_info
        var_info = get_sequence_info(row, "var", ag_seq_context)
        ref_probabilities = compute_probability(ref_info, ag_info)
        var_probabilities = compute_probability(var_info, ag_info)
        ag_results = joint_probability(
            ref_probabilities=ref_probabilities,
            var_probabilities=var_probabilities
        )
        data_dlogit = np.mean(row[delta_logit_pooled])
        try:
            final_df = to_df(
                final_df, row, ag_results,
                ref_probabilities,
                var_probabilities,
                data_dlogit,
                write_path="",
                header_written=header_written,
            )
        except TypeError as e:
            if debug:
                print(e)
                breakpoint()
            else:
                raise e
        header_written = True
        if debug:
            breakpoint()

    final_df = pd.DataFrame(final_df)
    if not debug:
        final_df.to_csv(output_tsv, sep='\t', index=False) 
        


    logging.info("Done")

