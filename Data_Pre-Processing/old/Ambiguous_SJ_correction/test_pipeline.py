"""
Unit tests for the 04/26/2026 ambiguous-SJ pipeline outputs.

Run after stages 01 → 02 → 03 have produced files in:
  /ESL/Figures_SK/General_preprocessing/output_04_26_2026/

  pytest /ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/test_pipeline.py -v

Coverage:
  Per-row    1-6    : 161-nt invariant, intron2==20, full_seq concat, exon length vs hg38 coords
  Per-event  7-9    : single transcript/intron1_len/coords per ambiguous event,
                       chosen MANE transcript matches expected, lengths match GTF
  Cross-file 10-12  : supertable ↔ main CSV agreement on metadata, schema carry-through, row counts
  Non-ambig  13     : merged CSV matches 03_16 baseline on shared columns for non-ambiguous events
  PSI        14-17  : KCTD10 ref 99222 ≈ 130/145 in HEK rep1, CHANGED_EVENTS differ from baseline,
                       clip+logit identities
  Patched    18-19  : per-rep PSI text files unchanged for non-ambig refs; patched values match psi_at
  Safety     20     : source files (gz, 03_16/04_24 outputs, original PSI text files,
                       Ambiguous_SJ_correction/ + Post-process_STAR_PSIs/ scripts) untouched
"""

import gzip
import os
import pickle
import re

import numpy as np
import pandas as pd
import pytest

# ── Paths ─────────────────────────────────────────────────────────────────
APR26_DIR = "/ESL/Figures_SK/General_preprocessing/output_04_26_2026"
APR24_DIR = "/ESL/Figures_SK/General_preprocessing/output_04_24_2026"
MAR16_DIR = "/ESL/Figures_SK/General_preprocessing/output_03_16_2026"

SOURCE_GZ = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv.gz"
GTF       = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/gencode.v48.annotation.gtf"

ST_NEW    = f"{APR26_DIR}/st_04_26_2026.csv"
ND_NEW    = f"{APR26_DIR}/04_26_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"
WT_NEW    = f"{APR26_DIR}/04_26_2026_1e-2_ALL_WITH_WT.csv"  # not produced; merged ALL is the canonical
PATCHED_PSI_DIR = f"{APR26_DIR}/recount_PSIs_04_26_2026"
SI_NEW    = "/ESL/Figures_SK/ambiguous_sjs/SI_alt_transcript_psi_04_26_2026.csv"

MAR16_ND  = f"{MAR16_DIR}/03_16_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv"

# Pickles per replicate (1-indexed Reference) — used to validate patched PSI files.
PKL_BASE = "/ESL/Analysis/STAR_alignment/separate_concat_2023_09_19_d1c_ms75_from_s3_2023_11_27/recount_SJs"
PKL_2024 = PKL_BASE + "/2024_07_26"
WT_BASE  = "/ESL/Analysis/WT_Library/separate/recount_SJs"

REP_INPUTS = {
    "HeLa_Rep1":      ("HeLa_Rep1_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/HeLa_Rep1_separate_splicing_profiles/HeLa_Rep1_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/HeLa_Rep1_separate_splicing_profiles/HeLa_Rep1_separate_all_splicing_counts.p"),
    "HeLa_Rep2":      ("HeLa_Rep3_20231031_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/HeLa_Rep3_20231031_separate_splicing_profiles/HeLa_Rep3_20231031_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/HeLa_Rep3_20231031_separate_splicing_profiles/HeLa_Rep3_20231031_separate_all_splicing_counts.p"),
    "K562_Rep1":      ("K562_Rep1_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/K562_Rep1_separate_splicing_profiles/K562_Rep1_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/K562_Rep1_separate_splicing_profiles/K562_Rep1_separate_all_splicing_counts.p"),
    "K562_Rep2":      ("K562_Rep3_20231031_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/K562_Rep3_20231031_separate_splicing_profiles/K562_Rep3_20231031_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/K562_Rep3_20231031_separate_splicing_profiles/K562_Rep3_20231031_separate_all_splicing_counts.p"),
    "MCF7_Rep1":      ("MCF7_Rep1_recalc_PSIs_mincov10.txt",
                       PKL_2024 + "/MCF7_Rep1_separate_splicing_profiles/MCF7_Rep1_recalc_PSIs_mincov10.txt",
                       PKL_2024 + "/MCF7_Rep1_separate_splicing_profiles/MCF7_Rep1_all_splicing_counts.p"),
    "MCF7_Rep2":      ("MCF7_Rep2_recalc_PSIs_mincov10.txt",
                       PKL_2024 + "/MCF7_Rep2_separate_splicing_profiles/MCF7_Rep2_recalc_PSIs_mincov10.txt",
                       PKL_2024 + "/MCF7_Rep2_separate_splicing_profiles/MCF7_Rep2_all_splicing_counts.p"),
    "HMC3_Rep1":      ("HMC3_Rep1_recalc_PSIs_mincov10.txt",
                       PKL_2024 + "/HMC3_Rep1_separate_splicing_profiles/HMC3_Rep1_recalc_PSIs_mincov10.txt",
                       PKL_2024 + "/HMC3_Rep1_separate_splicing_profiles/HMC3_Rep1_all_splicing_counts.p"),
    "HMC3_Rep2":      ("HMC3_Rep2_recalc_PSIs_mincov10.txt",
                       PKL_2024 + "/HMC3_Rep2_separate_splicing_profiles/HMC3_Rep2_recalc_PSIs_mincov10.txt",
                       PKL_2024 + "/HMC3_Rep2_separate_splicing_profiles/HMC3_Rep2_all_splicing_counts.p"),
    "HEK293_Rep1":    ("HEK293_Rep2_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/HEK293_Rep2_separate_splicing_profiles/HEK293_Rep2_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/HEK293_Rep2_separate_splicing_profiles/HEK293_Rep2_separate_all_splicing_counts.p"),
    "HEK293_Rep2":    ("HEK293_Rep3_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/HEK293_Rep3_separate_splicing_profiles/HEK293_Rep3_separate_recalc_PSIs_mincov10.txt",
                       PKL_BASE + "/HEK293_Rep3_separate_splicing_profiles/HEK293_Rep3_separate_all_splicing_counts.p"),
    "HEK293_WT_Rep1": ("HEK_WT_Rep1_20231101_separate_recalc_PSIs_mincov10.txt",
                       WT_BASE  + "/HEK_WT_Rep1_20231101_separate_splicing_profiles/HEK_WT_Rep1_20231101_separate_recalc_PSIs_mincov10.txt",
                       WT_BASE  + "/HEK_WT_Rep1_20231101_separate_splicing_profiles/HEK_WT_Rep1_20231101_separate_all_splicing_counts.p"),
    "HEK293_WT_Rep2": ("HEK_WT_Rep2_20231101_separate_recalc_PSIs_mincov10.txt",
                       WT_BASE  + "/HEK_WT_Rep2_20231101_separate_splicing_profiles/HEK_WT_Rep2_20231101_separate_recalc_PSIs_mincov10.txt",
                       WT_BASE  + "/HEK_WT_Rep2_20231101_separate_splicing_profiles/HEK_WT_Rep2_20231101_separate_all_splicing_counts.p"),
}

# Construct geometry — fixed by MPRA library design.
SHARED_5P = 286
MINCOV    = 10
E_JXN     = (26, 871)

# 11 events whose chosen junction differs from the originally annotated transcript.
# Lifted verbatim from /ESL/Figures_SK/ambiguous_sjs/test_seqfix.py:50-62.
CHANGED_EVENTS = {
    "chr11:46367939-46368099:+",
    "chr5:103028098-103028258:+",
    "chr1:231229130-231229290:-",
    "chr6:138926381-138926541:-",
    "chr16:56667292-56667452:-",
    "chr9:83666330-83666490:-",
    "chr14:67592615-67592775:-",
    "chr8:143824293-143824453:-",
    "chr12:109457972-109458132:-",   # KCTD10
    "chr1:155061349-155061509:+",
    "chr16:55856357-55856517:-",
}

# Ambiguous event with zero reads at any junction in any rep, even after the
# variant-pool fallback. Stage 1 cannot collapse it; per-event consistency
# tests skip it.
SKIPPED_EVENTS = {
    "chr14:92096073-92096233:-",
}

# Number of rows expected in the merged main CSV. Differs from the 03_16/04_24
# baseline of 87,546 by 93 because some constructs newly pass MINCOV at the
# corrected MANE junction (e.g. KCTD10 ref 99222 in HEK rep1).
EXPECTED_ND_ROWS = 87639

# Expected MANE transcript per CHANGED_EVENT (from findings_04_24_2026.md).
EXPECTED_MANE = {
    "chr11:46367939-46368099:+":   "ENST00000456247",
    "chr5:103028098-103028258:+":  "ENST00000438793",
    "chr1:231229130-231229290:-":  "ENST00000366649",
    "chr6:138926381-138926541:-":  "ENST00000450536",
    "chr16:56667292-56667452:-":   "ENST00000379811",
    "chr9:83666330-83666490:-":    "ENST00000376395",
    "chr14:67592615-67592775:-":   "ENST00000216452",
    "chr8:143824293-143824453:-":  "ENST00000526683",
    "chr12:109457972-109458132:-": "ENST00000228495",   # KCTD10
    "chr1:155061349-155061509:+":  "ENST00000356955",
    "chr16:55856357-55856517:-":   "ENST00000290567",
}

PSI_TOL = 1e-5
KCTD10_REF = 99222


# ── Fixtures ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def st_new():
    return pd.read_csv(ST_NEW, low_memory=False)


@pytest.fixture(scope="session")
def nd_new():
    return pd.read_csv(ND_NEW, low_memory=False)


@pytest.fixture(scope="session")
def nd_mar16():
    return pd.read_csv(MAR16_ND, low_memory=False)


# ── 1-6: per-row invariants ────────────────────────────────────────────────
def _check_lengths(df):
    assert (df["intron1"].str.len() + df["exon"].str.len() + df["intron2"].str.len() == 161).all()
    assert (df["intron2"].str.len() == 20).all()
    assert (df["full_seq"].str.len() == 161).all()
    assert (df["intron1"] + df["exon"] + df["intron2"] == df["full_seq"]).all()


def test_supertable_lengths(st_new):
    _check_lengths(st_new)


def test_main_csv_lengths(nd_new):
    _check_lengths(nd_new)


def test_supertable_intron1_len_column(st_new):
    assert (st_new["intron1"].str.len() == st_new["intron1_len"]).all()
    assert (st_new["exon"].str.len() == st_new["exon_len"]).all()


def test_exon_length_matches_hg38_coords(st_new):
    sub = st_new[(st_new["exon_end_hg38"] > 0) & (st_new["exon_start_hg38"] > 0)].copy()
    width = (sub["exon_end_hg38"].astype(int) - sub["exon_start_hg38"].astype(int)).abs()
    # GTF-style exon widths: end - start + 1; hg38 coords here may be either
    # convention. Allow exact match OR off-by-one.
    diffs = (sub["exon"].str.len() - width).abs()
    assert (diffs <= 1).all(), f"exon length disagrees with hg38 coords for {(diffs>1).sum()} rows"


# ── 7-9: per-event invariants ──────────────────────────────────────────────
def test_one_transcript_per_event(st_new):
    sub = st_new[~st_new["event_id"].isin(SKIPPED_EVENTS)]
    g = sub.groupby("event_id")
    multi_tx = g["transcript_id"].nunique()
    bad = multi_tx[multi_tx > 1]
    assert bad.empty, f"events with multiple transcript_ids: {bad.to_dict()}"


def test_one_intron1_len_per_event(st_new):
    sub = st_new[~st_new["event_id"].isin(SKIPPED_EVENTS)]
    g = sub.groupby("event_id")
    bad = g["intron1_len"].nunique()
    bad = bad[bad > 1]
    assert bad.empty, f"events with multiple intron1_len: {bad.to_dict()}"


@pytest.mark.parametrize("event_id,expected_mane", sorted(EXPECTED_MANE.items()))
def test_changed_event_uses_mane(st_new, event_id, expected_mane):
    sub = st_new[st_new["event_id"] == event_id]
    tx = sub["transcript_id"].iloc[0]
    assert str(tx).split(".")[0] == expected_mane, f"{event_id} → got {tx}"


def _gtf_exon_widths_for_tx(tx_base):
    widths = set()
    with open(GTF) as f:
        for line in f:
            if "\texon\t" not in line:
                continue
            m = re.search(r'transcript_id "([^"]+)"', line)
            if not m:
                continue
            if m.group(1).split(".")[0] != tx_base:
                continue
            parts = line.split("\t")
            try:
                widths.add(int(parts[4]) - int(parts[3]) + 1)
            except (IndexError, ValueError):
                continue
    return widths


@pytest.mark.parametrize("event_id,expected_mane", sorted(EXPECTED_MANE.items()))
def test_chosen_lengths_match_gtf(st_new, event_id, expected_mane):
    sub = st_new[st_new["event_id"] == event_id]
    expected_exon_len = int(sub["exon_len"].iloc[0])
    gtf_widths = _gtf_exon_widths_for_tx(expected_mane)
    assert expected_exon_len in gtf_widths, (
        f"{event_id} chosen exon_len={expected_exon_len} not found in GTF "
        f"exon widths {sorted(gtf_widths)} for {expected_mane}"
    )


# ── 10-12: cross-file invariants ───────────────────────────────────────────
def test_supertable_main_metadata_agreement(st_new, nd_new):
    # 1-indexed Reference in main CSV; 0-indexed in supertable.
    cols = ["transcript_id", "exon_start_hg38", "exon_end_hg38", "variant_hg38"]
    st_view = st_new.assign(_Reference1=st_new["Reference"].astype(int) + 1)
    st_view = st_view.set_index(["_Reference1", "snp"])[cols]
    nd_view = nd_new.set_index(["Reference", "snp"])[cols]
    common = st_view.index.intersection(nd_view.index)
    assert len(common) > 0, "no shared (Reference, snp) keys"
    diffs = (st_view.loc[common].fillna("__NA__").astype(str)
             != nd_view.loc[common].fillna("__NA__").astype(str)).any(axis=1)
    assert not diffs.any(), f"{int(diffs.sum())} rows disagree on metadata"


def test_main_csv_carries_new_columns(nd_new):
    for col in ("event_id", "transcript_id", "exon_start_hg38", "exon_end_hg38", "variant_hg38"):
        assert col in nd_new.columns, f"missing column: {col}"


def test_main_csv_row_count(nd_new):
    assert len(nd_new) == EXPECTED_ND_ROWS


# ── 13: non-ambiguous events match March 16 baseline ───────────────────────
def test_non_ambiguous_events_match_baseline(nd_new, nd_mar16, st_new):
    ambig_eids = set(
        st_new[st_new["event_id"].isin(
            st_new.groupby("event_id")["intron1_len"].first().index
        )]["event_id"].unique()
    )
    # Ambiguity is now collapsed in st_new, so re-derive ambig_eids from source.
    with gzip.open(SOURCE_GZ, "rt") as f:
        st_src = pd.read_csv(f, low_memory=False)
    st_src["_i1"] = st_src["intron1"].str.len()
    ambig_eids = set(st_src.groupby("event_id")["_i1"].nunique().pipe(lambda s: s[s > 1]).index)

    nonambig_new   = nd_new[~nd_new["event_id"].isin(ambig_eids)].set_index("Reference")
    nonambig_mar16 = nd_mar16[~nd_mar16["event_id"].isin(ambig_eids)].set_index("Reference")

    common_refs = nonambig_new.index.intersection(nonambig_mar16.index)
    common_cols = nonambig_new.columns.intersection(nonambig_mar16.columns)
    a = nonambig_new.loc[common_refs, common_cols]
    b = nonambig_mar16.loc[common_refs, common_cols]
    # PSI columns: float compare with tolerance; everything else: exact.
    psi_cols = [c for c in common_cols if any(t in c for t in
                ("psi", "logit", "included", "excluded", "pooled", "dpsi", "delta", "_sd_"))]
    other_cols = [c for c in common_cols if c not in psi_cols]
    if other_cols:
        diff = (a[other_cols].fillna("__NA__").astype(str)
                != b[other_cols].fillna("__NA__").astype(str)).any(axis=1)
        assert not diff.any(), f"{int(diff.sum())} non-ambig rows differ on non-PSI columns"
    if psi_cols:
        diff = ~np.isclose(a[psi_cols].astype(float).values,
                           b[psi_cols].astype(float).values,
                           atol=PSI_TOL, equal_nan=True)
        assert not diff.any(), f"non-ambig rows differ on PSI columns at tolerance {PSI_TOL}"


# ── 14-17: PSI / regression checks ─────────────────────────────────────────
def test_kctd10_hek_rep1_psi(nd_new):
    row = nd_new[nd_new["Reference"] == KCTD10_REF]
    assert len(row) == 1, f"expected exactly 1 row for ref {KCTD10_REF}"
    inc = row["HEK_rep1_included"].iloc[0]
    exc = row["HEK_rep1_excluded"].iloc[0]
    psi = row["HEK_rep1_psi_raw"].iloc[0]
    assert inc == pytest.approx(130.0), f"expected 130 included reads, got {inc}"
    assert psi == pytest.approx(130.0 / (130.0 + exc), abs=PSI_TOL)


@pytest.mark.parametrize("event_id", sorted(CHANGED_EVENTS))
def test_changed_event_psi_differs_from_baseline(nd_new, nd_mar16, event_id):
    a = nd_new[nd_new["event_id"] == event_id].set_index("Reference")["HEK_pooled_psi_raw"]
    b = nd_mar16[nd_mar16["event_id"] == event_id].set_index("Reference")["HEK_pooled_psi_raw"]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        pytest.skip(f"{event_id} has no common Reference IDs between baselines")
    diff = ~np.isclose(a.loc[common].astype(float),
                      b.loc[common].astype(float),
                      atol=PSI_TOL, equal_nan=True)
    assert diff.any(), f"{event_id}: HEK_pooled_psi_raw identical to March 16 baseline"


def test_psi_raw_in_unit_interval(nd_new):
    psi_cols = [c for c in nd_new.columns if c.endswith("_psi_raw")]
    for col in psi_cols:
        vals = nd_new[col].dropna()
        bad = (vals < 0) | (vals > 1)
        assert not bad.any(), f"{col}: {int(bad.sum())} values outside [0,1]"


def test_psi_clipped_in_clip_interval(nd_new):
    psi_cols = [c for c in nd_new.columns if c.endswith("_psi_clipped")]
    for col in psi_cols:
        vals = nd_new[col].dropna()
        bad = (vals < 0.01) | (vals > 0.99)
        assert not bad.any(), f"{col}: {int(bad.sum())} values outside [0.01,0.99]"


def test_logit_identity(nd_new):
    # logit_PSI = log(clipped / (1 - clipped)) where clipped is _psi_clipped.
    pairs = [(c.replace("_logit", "_psi_clipped"), c)
             for c in nd_new.columns if c.endswith("_logit")
             and c.replace("_logit", "_psi_clipped") in nd_new.columns]
    assert pairs, "no logit/psi_clipped pairs found"
    for clipped_col, logit_col in pairs:
        sub = nd_new[[clipped_col, logit_col]].dropna()
        if sub.empty:
            continue
        expected = np.log(sub[clipped_col] / (1 - sub[clipped_col]))
        diff = np.abs(expected.astype(float) - sub[logit_col].astype(float))
        assert (diff < 1e-6).all(), f"{logit_col}: max diff {diff.max():g}"


# ── 18-19: patched per-rep PSI text-file invariants ────────────────────────
def _ambig_refs_1idx(st_new_df):
    """1-indexed Reference IDs that belong to ambiguous events with a chosen junction."""
    with gzip.open(SOURCE_GZ, "rt") as f:
        st_src = pd.read_csv(f, low_memory=False)
    st_src["_i1"] = st_src["intron1"].str.len()
    ambig_eids = set(st_src.groupby("event_id")["_i1"].nunique().pipe(lambda s: s[s > 1]).index)
    chosen_eids = set(st_new_df.loc[st_new_df["event_id"].isin(ambig_eids), "event_id"])
    return set((st_new_df.loc[st_new_df["event_id"].isin(chosen_eids), "Reference"].astype(int) + 1))


@pytest.mark.parametrize("rep_name", sorted(REP_INPUTS))
def test_patched_psi_nonambig_unchanged(st_new, rep_name):
    basename, orig_path, _ = REP_INPUTS[rep_name]
    new_path = os.path.join(PATCHED_PSI_DIR, basename)
    orig = pd.read_csv(orig_path, sep="\t").set_index("Reference")
    new  = pd.read_csv(new_path,  sep="\t").set_index("Reference")
    ambig = _ambig_refs_1idx(st_new)
    nonambig_refs = orig.index.difference(ambig)
    common = nonambig_refs.intersection(new.index)
    a = orig.loc[common]
    b = new.loc[common]
    diff_psi = ~np.isclose(a["PSI"].astype(float), b["PSI"].astype(float), atol=PSI_TOL, equal_nan=True)
    diff_cov = (a["Coverage"].astype(float) != b["Coverage"].astype(float))
    assert not diff_psi.any(), f"{rep_name}: {int(diff_psi.sum())} non-ambig refs changed PSI"
    assert not diff_cov.any(), f"{rep_name}: {int(diff_cov.sum())} non-ambig refs changed Coverage"
    # No non-ambig refs should be added/dropped.
    assert nonambig_refs.equals(new.index.difference(ambig)), \
        f"{rep_name}: non-ambig Reference set changed"


@pytest.mark.parametrize("rep_name", sorted(REP_INPUTS))
def test_patched_psi_ambig_match_pkl(st_new, rep_name):
    basename, _, pkl_path = REP_INPUTS[rep_name]
    new_path = os.path.join(PATCHED_PSI_DIR, basename)
    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f)

    new = pd.read_csv(new_path, sep="\t").set_index("Reference")

    # For each ambig ref, recompute (PSI, Coverage) at the chosen junction
    # using the supertable's intron1_len / exon_len for that event.
    ambig_refs = _ambig_refs_1idx(st_new)
    chosen_jxn_per_ref = {}
    for _, row in st_new.iterrows():
        ref1 = int(row["Reference"]) + 1
        if ref1 in ambig_refs:
            i1 = (26, SHARED_5P + int(row["intron1_len"]))
            i2 = (SHARED_5P + int(row["intron1_len"]) + int(row["exon_len"]) + 1, 871)
            chosen_jxn_per_ref[ref1] = (i1, i2)

    mismatches = 0
    for ref1, (i1, i2) in chosen_jxn_per_ref.items():
        if ref1 not in pkl:
            continue
        jd = pkl[ref1]
        inc = min(jd.get(i1, 0), jd.get(i2, 0))
        exc = jd.get(E_JXN, 0)
        tot = inc + exc
        if tot < MINCOV:
            # Should not appear in patched file
            assert ref1 not in new.index, f"{rep_name}: ref {ref1} in patched file but cov {tot} < MINCOV"
            continue
        expected_psi = inc / tot
        if ref1 not in new.index:
            mismatches += 1
            continue
        actual_psi = float(new.loc[ref1, "PSI"])
        actual_cov = float(new.loc[ref1, "Coverage"])
        if abs(actual_psi - expected_psi) > PSI_TOL or abs(actual_cov - tot) > 1e-6:
            mismatches += 1
    assert mismatches == 0, f"{rep_name}: {mismatches} ambig refs mismatch pickle-recomputed values"


# ── 20: source-file safety ─────────────────────────────────────────────────
SOURCE_FILES = [
    SOURCE_GZ,
    GTF,
    "/ESL/ESL_MPRA/Data_Pre-Processing/Ambiguous_SJ_correction/00_prepare_working_supertable.py",
    "/ESL/ESL_MPRA/Data_Pre-Processing/Ambiguous_SJ_correction/05_build_alt_transcript_si_table.py",
    "/ESL/ESL_MPRA/Data_Pre-Processing/Ambiguous_SJ_correction/07_apply_best_junction_ambiguous_events.py",
    "/ESL/ESL_MPRA/Data_Pre-Processing/Ambiguous_SJ_correction/08_update_sequences_for_chosen_junction.py",
    "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/merge_psi_07_18_2025_clipping_no_psuedo.py",
    "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/run_merge_psi_clipping_no_psuedo.sh",
    MAR16_ND,
] + [orig for (_, orig, _) in REP_INPUTS.values()]


@pytest.mark.parametrize("path", SOURCE_FILES)
def test_source_file_older_than_outputs(path):
    """Source files must be older than the corrected supertable (proof of no in-place edit)."""
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    src_mtime = os.path.getmtime(path)
    out_mtime = os.path.getmtime(ST_NEW)
    assert src_mtime < out_mtime, (
        f"{path} (mtime {src_mtime}) is newer than {ST_NEW} (mtime {out_mtime}) — possible in-place edit"
    )
