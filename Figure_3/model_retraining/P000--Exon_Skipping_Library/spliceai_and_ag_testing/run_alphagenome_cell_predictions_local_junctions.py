import sys
import os
import getopt
from collections import defaultdict
import json
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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

cell_ontology_map = {
    "HEK293" : "UBERON:0001225", # kidney cortex
    "HeLa" : "EFO:0002791", # HeLa
    "K562" : "EFO:0002067", # K562
    "MCF7" : "EFO:0001203", # MCF7
    "HMC3" : "UBERON:0000955", # brain tissue
}

from src.utils.parsing import bool_map, ErrorWrapper
from src.load_data import read_csvs

from src.esl import (
    logit,
    clip,
    cell_types,
    cell_types2,
)
from src.esl import (
    SMN2_INTRON6,
    SMN2_INTRON7
)
from src.esl import (
    CITRINE_EXON1,
    CITRINE_EXON2
)

def load_json(file_):
    with open(file_, "r") as f:
        loaded_dict = json.load(f)
    return loaded_dict 

help_="""
"""
inputs = [
    "alphagenome_path=",
    "all_samples_path=",
    "output_tsv=",
    "alphagenome_context=",
    "conversion_method=",
    "cell_ontology_map_json=",
    "ontology_terms=",
    "parallel_split=",
    "help",
    "debug",
]
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:],"", inputs)
    opts = dict(opts)
    opts = ErrorWrapper(opts, max_printout_size=500)
    if "--help" in opts:
        print(help_)
        sys.exit(0)
    debug = "--debug" in opts
    if debug:
        alphagenome_path = os.path.join(
            "/storage", "yu", "Projects",
            "P000--ESL_continuation",
            "GitHub",
            "P000--Exon_Skipping_Library",
            "models", "alphagenome",
        )
        all_samples_path = os.path.join(
            "/storage/yu/Projects/"
            "P000--ESL_continuation/Figures_SK/"
            "General_preprocessing/output_5_23_2026/"
            "05_23_20261_e-2_ALL_WTS_VARS_NO_DELTAS_with_SI.csv"
        )
        output_tsv = os.path.join(
            "/storage/yu/Projects/P000--ESL_continuation"
            "/GitHub/P000--Exon_Skipping_Library/"
            "spliceai_and_ag_testing/alphagenome_predictions"
            "/cell_type_specific/psi_from_splice_junction_predictions/"
            "alphagenome_cell_predictions_16kb.tsv"
        )
        alphagenome_context = 14
        ontology_terms = None
        parallel_split = None
        
    else:
        alphagenome_path = opts["--alphagenome_path"]
        all_samples_path = opts["--all_samples_path"]
        output_tsv = opts["--output_tsv"]
        alphagenome_context = int(opts.get("--alphagenome_context", 14))
        ontology_terms = opts.get("--ontology_terms", "")
        if not ontology_terms:
            ontology_terms = None
        else:
            ontology_terms = ontology_terms.split(",")
        cell_ontology_map_json = opts.get("--cell_ontology_map_json", "")
        # overwrite default ontology if in inputs
        if cell_ontology_map_json:
            cell_ontology_map = load_json(cell_ontology_map_json)
        
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
        if alphagenome_path is not None and alphagenome_path not in sys.path:
            sys.path.insert(0, alphagenome_path)
            print("trying to import")
            try:
                import alphagenome
            except ModuleNotFoundError:
                print("Still can't import after adding to sys.path")
        elif alphagenome_path in sys.path:
            print("trying to import")
            try:
                import alphagenome
            except ModuleNotFoundError:
                print("Still can't import after adding to sys.path")
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
import tqdm


N_CARDS = 4
def split_df_parallel(df, parallel_split, n_workers = N_CARDS):
    """
    Split dataframe into four to process in parallel.
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


def ontology_terms_from_map(cell_ontology_map):
    return list(dict.fromkeys(cell_ontology_map.values()))


def build_metadata_keys(
        metadata,
        debug=False,
        cell_ontology_map=cell_ontology_map,
    ):
    """
    Using mapping from cell_type to 
    ontology term, make indices to query
    the alphagenome values for each cell type.
    """
    names = metadata["name"].tolist()
    if "strand" in metadata:
        strand = metadata["strand"].tolist()
    else:
        strand = ["+"] * len(names)
    keys_out = {}
    candidate_cell_keys = defaultdict(list)
    for i, c in enumerate(cell_types):
        ont = cell_ontology_map[c]
        for j, n in enumerate(names):
            if ont in n and strand[j] == "+" and "polyA plus" in n:
                keys_out[c] = j
                candidate_cell_keys[c].append(n)
    for c in candidate_cell_keys:
        assert len(candidate_cell_keys[c]) == 1
        if debug:
            print(c, ":", candidate_cell_keys[c][0])
    return keys_out


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
    exon_end = exon_start + len(row["exon" + key]) - 1
    return {
        "full" : full,
        "exon_start" : exon_start,
        "exon_end" : exon_end,
        "intron1_len" : len(row["intron1" + key]),
        "exon_len" : len(row["exon" + key]),
    }


def model_inference(
        seq_info,
        ag_info,
        ontology_terms,
    ):
    """
    Pads sequence with N's
    Calls model on splice junctions track
    """
    seq = seq_info["full"]
    context = ag_info["context"]
    seq_len = min(context, 20)
    seq_len = 2 ** seq_len
    center = seq_len // 2
    padded = seq.center(seq_len, 'N')
    offset = center - len(seq) // 2
    out = model.predict_sequence(
        sequence=padded,
        organism=dna_model.Organism.HOMO_SAPIENS,
        requested_outputs={dna_model.OutputType.SPLICE_JUNCTIONS},
        ontology_terms=ontology_terms,
    )
    return out.splice_junctions, out.splice_junctions.metadata, offset


def junction_from_context(ag_info):
    """
    Calculate the expected skipping/exclusion
    junction, which doesn't change given the constant
    synthetic context and 161-nt variable region
    """
    alphagenome_context = ag_info["context"]
    prefix_len = len(CITRINE_EXON1) + len(SMN2_INTRON6)
    suffix_len = len(SMN2_INTRON7) + len(CITRINE_EXON2)
    context_size = ((2 ** alphagenome_context)) - prefix_len - suffix_len - 161
    context_size = context_size // 2
    skipping_junction = (
        context_size + len(CITRINE_EXON1),
        context_size + prefix_len + 161 + suffix_len - len(CITRINE_EXON2)
    )
    return skipping_junction


def calc_junction_psi(
        seq_info,
        ag_info,
        tissue_info,
        ontology_terms=None,
    ):
    """
    Wrapper function for calling the model
    and computing PSI values from the predictions. 
    """
    context = ag_info["context"]
    skipping_junction = ag_info.get("skipping_junction", None)
    if skipping_junction is None:
        skipping_junction = junction_from_context(ag_info)
    junctions, metadata, offset = model_inference(
        seq_info,
        ag_info,
        ontology_terms
    )
    cell_psi = junctions_to_psi(
        junctions,
        seq_info,
        tissue_info,
        skipping_junction=skipping_junction,
    )
    return cell_psi


def junctions_to_psi(
        junctions,
        seq_info,
        tissue_info,
        skipping_junction=(7929, 8775),
    ):
    """
    Main function to compute PSI from the 
    SPLICE_JUNCIONS track.
   
    computes the usage using the minimum of 
    the PSI5 and PSI3 predictions, although
    in most cases only one such value is
    reported.

    The PSI is given by the included proportion
    divided by the sum of included and excluded
    predicted proportions.

    Excluded predicted proportions are given 
    by the predictions at the skipping junction.
    It should be defined once, but can be passed
    in as well.
 
    If excluded proportions are not found, the PSI
    will be np.nan, not 0. 
    
    If neither 5p not 3p junctions are reported by
    the model, the PSI will be 0.
    """
    junction_5p = (
        skipping_junction[0],
        skipping_junction[0] + len(SMN2_INTRON6) + seq_info["intron1_len"]
    )
    junction_3p = (
        (
            skipping_junction[0]
            + len(SMN2_INTRON6)
            + seq_info["intron1_len"]
            + seq_info["exon_len"]
        ),
        skipping_junction[1]
    )
    n_junctions = len(junctions.junctions)
    if not n_junctions:
        return {t: np.nan for t in tissue_info}
    n_tissues = junctions.values.shape[-1]
    total_reads = np.zeros((n_tissues, ))
    def tissue_array():
        return np.ones((n_tissues,)) * np.nan
    excl_reads = tissue_array()
    incl_reads_3p = tissue_array()
    incl_reads_5p = tissue_array()
    for j_idx in range(n_junctions):
        junction_start = junctions.junctions[j_idx].start
        junction_end = junctions.junctions[j_idx].end
        if (junction_start, junction_end) == skipping_junction:
            excl_reads = junctions.values[j_idx]
        elif (junction_start, junction_end) == junction_5p:
            incl_reads_5p = junctions.values[j_idx]
        elif (junction_start, junction_end) == junction_3p:
            incl_reads_3p = junctions.values[j_idx]
    cell_psi = {}
    for t in tissue_info:
        t_idx = tissue_info[t]
        if np.isnan(excl_reads[t_idx]):
            cell_psi[t] = np.nan
        else:
            # get the minimum and
            # change to 0 if both are nan
            incl = np.nanmin(
                [incl_reads_5p[t_idx],
                incl_reads_3p[t_idx]]
            )
            incl = np.nanmax(
                [incl,
                0.0]
            )
            excl = excl_reads[t_idx]
            psi = incl / (incl + excl)
            cell_psi[t] = psi
    return cell_psi


def get_ag_dlogits(ag_results, ag_predictions):
    """
    Compute cell dlogit
    (logit_cell - logit(mean(cell)))
    for the alphagenome predictions.

    """
    cell_raw_psi = list(ag_predictions.values())
    cell_mean_psi = np.mean(clip(cell_raw_psi, 0.01)) 
    cell_logit_mean = logit(cell_mean_psi, 0.01)
    for cell in ag_predictions:
        ag_results[f"{cell}_cell_dlogit"].append(
            logit(ag_predictions[cell], 0.01) - cell_logit_mean
        )
        ag_results[f"{cell}_psi_raw"].append(
            ag_predictions[cell]
        )
    return ag_results


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
    all_samples = pd.read_csv(all_samples_path)
    if debug:
        pass
    else:
        all_samples = split_df_parallel(all_samples, parallel_split)
    if parallel_split is not None:
        output_tsv = output_tsv.replace(".tsv", f"_split_{parallel_split}.tsv")


    ag_info = {
        "context": alphagenome_context,
    }
    ag_info["skipping_junction"] = junction_from_context(ag_info)
    prefix = CITRINE_EXON1 + SMN2_INTRON6
    suffix = SMN2_INTRON7 + CITRINE_EXON2
    ag_sequence_context = (
        prefix,
        suffix
    )
    if ontology_terms is None:
        ontology_terms = ontology_terms_from_map(cell_ontology_map)
    elif ontology_terms and ontology_terms[0].lower() == "none":
        ontology_terms = None


    metadata = None
    ag_results = defaultdict(list)

    for i, row in all_samples.iterrows():
        ref = row["Reference"]
        ag_results["Reference"].append(ref)
        ag_results["snp"].append(row["snp"])
        ag_results["event_id"].append(row["event_id"])
        ag_results["event_id_161"].append(row["event_id_161"])
        ag_results["gene_exon"].append(row["gene_exon"])
        seq_info = get_sequence_info(row, "var", ag_sequence_context)
        if metadata is None:
            _, metadata, _ = model_inference(
                seq_info,
                ag_info,
                ontology_terms=ontology_terms,
            )
            tissue_info = build_metadata_keys(
                metadata,
                cell_ontology_map=cell_ontology_map,
            )
        ag_predictions = calc_junction_psi(
            seq_info,
            ag_info,
            tissue_info,
            ontology_terms=ontology_terms,
        )
        ag_results = get_ag_dlogits(ag_results, ag_predictions)
        if debug:
            break

    

    psi_rows = [
        f"{cell}_pooled_psi_clipped"
        for cell in cell_types2
    ]
    all_samples["logit_mean_cell"] = all_samples[psi_rows].apply(clip).mean(axis=1).apply(logit)
    for cell in cell_types2:
        cell_name = cell.replace("HEK", "HEK293")
        if debug:
            ag_results[f"{cell_name}_experimental_cell_dlogit"] = (
                all_samples[f"{cell}_pooled_logit"] - all_samples["logit_mean_cell"]
            ).tolist()[:len(ag_results["Reference"])]
        else:
            ag_results[f"{cell_name}_experimental_cell_dlogit"] = (
                all_samples[f"{cell}_pooled_logit"] - all_samples["logit_mean_cell"]
            ).tolist()

    ag_results = pd.DataFrame(ag_results)
    if debug:
        breakpoint() 

    ag_results.to_csv(output_tsv, index=False, sep="\t")
     
    
    logging.info("done")

    
