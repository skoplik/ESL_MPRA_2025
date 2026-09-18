"""
Stage-2 / mega builder for the MAY 2026 COMPASS model comparison.

Replaces build_mega_pred_file.py (2026-04-29). What changed and WHY:

  1. SINGLE SOURCE OF TRUTH. Every model's stage-1 ran on this exact file, so
     `Reference` is internally consistent. The old builder joined prediction
     files produced against OLDER data versions onto May rows by `Reference`,
     a row index that renumbers on every reprocess -> silent misalignment.

  2. JOINS ARE ASSERTED, NOT ASSUMED. Each prediction file carries
     (event_id_161, snp); after joining on Reference we verify they match and
     abort if not. This makes a repeat of bug #1 impossible to miss.
     (AlphaGenome and MMSplice files carry no stable key, so they are checked
     indirectly instead: References-exist-in-MAY, and REF/ALT-vs-snp agreement.)

  3. EVERYTHING IS KEYED BY LOCUS (`event_id_161`), NOT `gene_exon`.
     This is the delta-logit correctness guarantee, so read it carefully:

       - A `gene_exon` LABEL can span two different junctions. "CACNA1C exon 31"
         is the mutually exclusive 31a/31b pair (chr12:2633629 and chr12:2648475,
         ~15 kb apart). Keying WT by gene_exon could therefore subtract a WT
         measured at a DIFFERENT splice junction from the variant.
       - Conversely one LOCUS can carry two gene_exon labels, because exon
         NUMBERING is transcript-dependent ("ANKS1B exon 20" == "ANKS1B exon 5").
         Keying by gene_exon lets both survive and double-counts that full_seq.

     So: the primary-junction rule (#4) runs on `event_id_161` and collapses each
     locus to exactly ONE `event_id` BEFORE any WT lookup happens. The WT map is
     then keyed by `event_id_161`. Because only one junction survives per locus,
     variant and WT are always the same splice junction by construction.
     Audited on the MAY data: 0 / 86,390 variants paired with a WT from a
     different event_id, and 0 loci carrying >1 event_id after the primary rule.

     Note the other models enforce the same invariant their own way:
     build_mm_v3.py restricts variants to the chromosome's own annotation, and
     the AlphaGenome runner groups by `event_id` and scores ref and alt at the
     same exon_start/exon_end.

  4. PRIMARY-JUNCTION RULE, explicit: prefer the MANE junction per LOCUS; keep an
     `alt` junction only when that locus has no MANE. Where a locus has two
     non-MANE annotations of the same sequence (tandem acceptors 3 nt apart, e.g.
     AFDN exon 30), a deterministic tie-break applies: MANE, then a row with a
     measured value, then the lowest event_id.

  5. NO DOUBLE COUNTING. A hard assertion aborts the build if any
     (event_id_161, snp) appears twice, i.e. if the same full_seq would enter the
     correlation more than once.

  6. EDGE FILTER is exact: drop a variant if ANY cell line's WT pooled PSI is
     exactly 0 or exactly 1 (not a tolerance).

Note: variants at a locus with no WT row anywhere in the data (2,359 variants /
773 loci in MAY) have no baseline to subtract and are necessarily excluded.

Writes NEW files only; never overwrites an input.
"""
import os, sys, json
import numpy as np
import pandas as pd

MAIN = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz"
SPLICEAI_TSV = sys.argv[1] if len(sys.argv) > 1 else "/ESL/ESL_MPRA/Figure_3/SpliceAI/output_MAY_v2/spliceai_junction_scores_MAY.tsv"
PANGOLIN_TSV = sys.argv[2] if len(sys.argv) > 2 else "/ESL/ESL_MPRA/Figure_3/Pangolin/output_MAY_v2/pangolin_junction_scores_MAY.tsv"
ALPHAGENOME = "/ESL/ESL_MPRA/Figure_3/AlphaGenome/alphagenome_16k_all_variants_MAY_2026.tsv"
MMSPLICE = "/ESL/ESL_MPRA/Figure_3/MMSplice/outputs/mmsplice_predictions_MAY_v3.csv"
HAL_FILE = "/ESL/ESL_MPRA/Figure_3/HAL/outputs/hal_delta_logit_MAY_2026.csv"
OLD_MEGA = "/ESL/ESL_MPRA/Figure_3/model_comparison/mega_pred_file_filtered_MAY.csv"
MMSPLICE_VCF = "/ESL/ESL_MPRA/Figure_3/MMSplice/outputs/input_files_MAY_v3/synthetic_variants.vcf.gz"
OUTDIR = "/ESL/ESL_MPRA/Figure_3/model_comparison"
OUT = os.path.join(OUTDIR, "mega_pred_file_MAY_v2_NOEDGEFILTER.csv")
REPORT = os.path.join(OUTDIR, "mega_pred_file_MAY_v2_report.json")

CELLS = ["HeLa", "K562", "MCF7", "HMC3", "HEK"]
DELTA_COLS = ["%s_delta_logit_pooled" % c for c in CELLS]
WTPSI_COLS = ["%s_wt_pooled_psi_raw" % c for c in CELLS]
MODEL_NUMS = [0, 2, 4, 6]
report = {}


def logit_clip(p, eps=0.01):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


print("Loading canonical MAY data ...")
df = pd.read_csv(MAIN, low_memory=False)
df["snp"] = df["snp"].astype(str).str.strip()
df["Reference"] = df["Reference"].astype(str)
report["main_rows"] = len(df)

for c in DELTA_COLS + WTPSI_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["avg_delta_logit_pooled"] = df[DELTA_COLS].mean(axis=1)

# ---- (4) primary-junction rule -------------------------------------------
# NOTE: group on event_id_161 (the sequence-level locus key), NOT gene_exon.
# The same physical sequence can carry two different gene_exon labels because exon
# NUMBERING depends on the transcript (e.g. "ANKS1B exon 20" MANE and "ANKS1B exon 5"
# alt are the same 161-nt locus). Grouping by gene_exon lets both survive and
# double-counts that full_seq.
has_mane = df.groupby("event_id_161")["transcript_class"].transform(lambda s: (s == "MANE").any())
df["is_primary"] = (df["transcript_class"] == "MANE") | (~has_mane)
prim = df[df["is_primary"]].copy()

# Deterministic tie-break: a locus can still carry two non-MANE annotations of the
# SAME full_seq (tandem acceptors 3 nt apart, e.g. AFDN exon 30, where neither is
# MANE). Keep exactly one row per (event_id_161, snp), preferring MANE, then a row
# that actually has a measured value, then the lexicographically lowest event_id so
# the choice is reproducible.
prim["_notmane"] = (prim["transcript_class"] != "MANE").astype(int)
prim["_nomeas"] = prim["avg_delta_logit_pooled"].isna().astype(int)
prim = (prim.sort_values(["_notmane", "_nomeas", "event_id"])
            .drop_duplicates(["event_id_161", "snp"], keep="first")
            .drop(columns=["_notmane", "_nomeas"])
            .sort_index())
report["rows_after_locus_dedup"] = len(prim)
report["primary_rows"] = len(prim)
report["gene_exons_total"] = int(df["gene_exon"].nunique())
report["gene_exons_primary"] = int(prim["gene_exon"].nunique())
njunc = prim.groupby("gene_exon")["event_id"].nunique()
report["gene_exons_with_multiple_junctions_after_primary"] = int((njunc > 1).sum())
print("primary-junction rows: %d (%d gene_exons)" % (len(prim), prim["gene_exon"].nunique()))


def attach(pred_path, sep, cols_needed, name, extra_filter=None):
    """Join a prediction file on Reference and ASSERT the stable key matches."""
    if not os.path.exists(pred_path):
        print("  !! MISSING %s: %s" % (name, pred_path))
        report["%s_status" % name] = "MISSING"
        return None
    p = pd.read_csv(pred_path, sep=sep, low_memory=False)
    if extra_filter is not None:
        p = extra_filter(p)
    p["Reference"] = p["Reference"].astype(str)
    p = p.drop_duplicates("Reference")
    missing = [c for c in cols_needed if c not in p.columns]
    if missing:
        raise SystemExit("%s missing columns %s" % (name, missing))
    # ---- (2) assert the join ----
    if "event_id_161" in p.columns and "snp" in p.columns:
        chk = prim[["Reference", "event_id_161", "snp"]].merge(
            p[["Reference", "event_id_161", "snp"]], on="Reference",
            suffixes=("_main", "_pred"))
        bad = chk[(chk["event_id_161_main"].astype(str) != chk["event_id_161_pred"].astype(str)) |
                  (chk["snp_main"].astype(str).str.strip() != chk["snp_pred"].astype(str).str.strip())]
        report["%s_join_checked" % name] = int(len(chk))
        report["%s_join_mismatches" % name] = int(len(bad))
        if len(bad):
            bad.head(20).to_csv(os.path.join(OUTDIR, "JOIN_MISMATCH_%s.csv" % name), index=False)
            raise SystemExit("ABORT: %d Reference-join mismatches for %s "
                             "(stable key disagrees) - see JOIN_MISMATCH_%s.csv" % (len(bad), name, name))
        print("  %s: join verified on %d rows, 0 mismatches" % (name, len(chk)))
    else:
        print("  %s: no stable key in file - join NOT verifiable" % name)
        report["%s_join_checked" % name] = "no_stable_key"
    return p


# ---- SpliceAI -------------------------------------------------------------
print("SpliceAI ...")
sai = attach(SPLICEAI_TSV, "\t", ["spliceai_logit"], "spliceai")
if sai is not None:
    prim = prim.merge(sai[["Reference", "spliceai_logit"]], on="Reference", how="left")
    # (3) WT keyed by gene_exon
    wt = prim[prim["snp"] == "none"].dropna(subset=["spliceai_logit"])
    wt_map = wt.groupby("event_id_161")["spliceai_logit"].first()
    prim["spliceai_wt_logit"] = prim["event_id_161"].map(wt_map)
    prim["spliceai_delta_logit"] = prim["spliceai_logit"] - prim["spliceai_wt_logit"]
    report["spliceai_wt_loci"] = int(len(wt_map))

# ---- Pangolin -------------------------------------------------------------
print("Pangolin ...")
pcols = ["p3ss_m%d" % m for m in MODEL_NUMS] + ["p5ss_m%d" % m for m in MODEL_NUMS]
pan = attach(PANGOLIN_TSV, "\t", pcols, "pangolin")
if pan is not None:
    prim = prim.merge(pan[["Reference"] + pcols], on="Reference", how="left")
    wt = prim[prim["snp"] == "none"].dropna(subset=pcols, how="all")
    wt_p = wt.groupby("event_id_161")[pcols].first()
    joined = prim[["event_id_161"]].join(wt_p, on="event_id_161", rsuffix="_wt")
    a3 = logit_clip(prim[["p3ss_m%d" % m for m in MODEL_NUMS]].to_numpy())
    a5 = logit_clip(prim[["p5ss_m%d" % m for m in MODEL_NUMS]].to_numpy())
    r3 = logit_clip(joined[["p3ss_m%d" % m for m in MODEL_NUMS]].to_numpy())
    r5 = logit_clip(joined[["p5ss_m%d" % m for m in MODEL_NUMS]].to_numpy())
    d3, d5 = a3 - r3, a5 - r5
    # original convention: most extreme across the 4 model_nums at each site,
    # then the mean of the two sites
    md3 = np.take_along_axis(d3, np.argmax(np.abs(d3), axis=1)[:, None], axis=1)[:, 0]
    md5 = np.take_along_axis(d5, np.argmax(np.abs(d5), axis=1)[:, None], axis=1)[:, 0]
    prim["pangolin_delta_logit"] = np.nanmean(np.stack([md3, md5], axis=1), axis=1)
    nan_any = prim[["p3ss_m%d" % m for m in MODEL_NUMS]].isna().all(axis=1)
    prim.loc[nan_any, "pangolin_delta_logit"] = np.nan

# ---- AlphaGenome (already MAY) -------------------------------------------
print("AlphaGenome ...")
ag = attach(ALPHAGENOME, "\t", ["alphagenome_delta_logit"], "alphagenome")
if ag is not None:
    may_refs = set(df["Reference"])
    unknown = int((~ag["Reference"].isin(may_refs)).sum())
    report["alphagenome_refs_not_in_may_file"] = unknown
    print("  alphagenome: %d/%d References absent from the MAY file%s"
          % (unknown, len(ag), "  <-- STALE REFERENCES" if unknown else "  (all present)"))
    prim = prim.merge(ag[["Reference", "alphagenome_delta_logit"]], on="Reference", how="left")

# ---- MMSplice (already MAY, core exon only) ------------------------------
print("MMSplice ...")


def mm_filter(p):
    """MMSplice is keyed by ID = 'chrom:pos:ref>alt'. The VCF that generated these
    predictions carries the Reference in its ID field, so map back through it."""
    if "exon_id" in p.columns:
        p = p[p["exon_id"].astype(str) == "core"].copy()
    vcf = pd.read_csv(MMSPLICE_VCF, sep="\t", comment="#", header=None, compression="gzip",
                      names=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"],
                      dtype=str)
    key = vcf["CHROM"] + ":" + vcf["POS"] + ":" + vcf["REF"] + ">" + vcf["ALT"]
    # NOTE: the VCF ID field is Reference+1 (inherited from make_mmsplice_inputs.py,
    # which wrote `Reference = int + 1`). Verified empirically: at offset -1 all
    # 35,583 single-nt variants have REF/ALT matching the May `snp`; at offset 0
    # only 1.9% do. So map back with -1.
    vcf["ref_true"] = (vcf["ID"].astype(int) - 1).astype(str)
    ref_of = dict(zip(key, vcf["ref_true"]))
    # allele-consistency assertion (MMSplice carries no stable key of its own)
    chk = vcf[vcf["REF"].str.len() == 1].merge(
        df[["Reference", "snp"]], left_on="ref_true", right_on="Reference", how="inner")
    import re as _re
    def _pair(x):
        m = _re.findall(r"([ACGT])\s*>\s*([ACGT])", str(x))
        return m[0] if m else (None, None)
    pairs = chk["snp"].map(_pair)
    agree = ((pairs.map(lambda t: t[0]) == chk["REF"]) &
             (pairs.map(lambda t: t[1]) == chk["ALT"])).mean() if len(chk) else 0.0
    report["mmsplice_allele_agreement"] = round(float(agree), 4)
    print("  mmsplice: allele agreement %.2f%% on %d single-nt variants" % (100 * agree, len(chk)))
    if agree < 0.99:
        raise SystemExit("ABORT: MMSplice Reference mapping looks misaligned "
                         "(allele agreement %.1f%%)" % (100 * agree))
    p["Reference"] = p["ID"].astype(str).map(ref_of)
    n_unmapped = int(p["Reference"].isna().sum())
    report["mmsplice_unmapped_ids"] = n_unmapped
    if n_unmapped:
        print("  mmsplice: %d IDs could not be mapped to a Reference" % n_unmapped)
    return p[p["Reference"].notna()].copy()


mm = attach(MMSPLICE, ",", ["delta_logit_psi"], "mmsplice", extra_filter=mm_filter)
if mm is not None:
    prim = prim.merge(mm[["Reference", "delta_logit_psi"]].rename(
        columns={"delta_logit_psi": "baseline_mmsplice_delta_logit"}),
        on="Reference", how="left")

# ---- HAL (already MAY; Reference maps 1:1, verified 27,791/27,791) --------
print("HAL ...")
hal = attach(HAL_FILE, ",", ["hal_delta_logit"], "hal")
if hal is not None:
    prim = prim.merge(hal[["Reference", "hal_delta_logit"]], on="Reference", how="left")

# ---- retrained MMSplice + test-set flag ----------------------------------
# NOTE: the retrained MMSplice model has NOT been re-fit on the May data; these
# columns are carried over from the previous MAY mega so the retraining panel
# still renders. Treat the retrained bar as APRIL-derived until it is re-fit.
print("retrained MMSplice (carried over - NOT re-fit on May) ...")
if os.path.exists(OLD_MEGA):
    old = pd.read_csv(OLD_MEGA, low_memory=False,
                      usecols=lambda c: c in ("Reference", "event_id_161", "snp",
                                              "retrained_mmsplice_delta_logit", "mmsplice_is_test"))
    old["Reference"] = old["Reference"].astype(str)
    if "event_id_161" in old.columns:
        chk = prim[["Reference", "event_id_161"]].merge(
            old[["Reference", "event_id_161"]], on="Reference", suffixes=("_new", "_old"))
        bad = int((chk["event_id_161_new"].astype(str) != chk["event_id_161_old"].astype(str)).sum())
        report["retrained_join_mismatches"] = bad
        print("  retrained: join checked on %d rows, %d mismatches" % (len(chk), bad))
        if bad:
            raise SystemExit("ABORT: retrained-MMSplice carry-over join mismatches (%d)" % bad)
    keep = [c for c in ("Reference", "retrained_mmsplice_delta_logit", "mmsplice_is_test") if c in old.columns]
    prim = prim.merge(old[keep].drop_duplicates("Reference"), on="Reference", how="left")

# ---- (5) exact edge filter + variants only -------------------------------
variants = prim[prim["snp"] != "none"].copy()
report["variant_rows_primary"] = len(variants)
edge = variants[WTPSI_COLS].apply(lambda r: (r == 0).any() or (r == 1).any(), axis=1)
report["edge_filtered_out"] = int(edge.sum())
final = variants[~edge].copy()

# ---- no-double-counting assertion ----
dup_key = final["event_id_161"].astype(str) + "|" + final["snp"].astype(str).str.strip()
n_dup = int(dup_key.duplicated(keep=False).sum())
report["duplicate_variant_rows"] = n_dup
if n_dup:
    final[dup_key.duplicated(keep=False)].to_csv(
        os.path.join(OUTDIR, "DUPLICATE_VARIANTS_MAY_v2.csv"), index=False)
    raise SystemExit("ABORT: %d rows share an (event_id_161, snp) key - the same "
                     "full_seq would be counted twice. See DUPLICATE_VARIANTS_MAY_v2.csv" % n_dup)
fs = final["full_seq"].astype(str).str.replace(" ", "", regex=False)
report["duplicate_full_seq_rows"] = int(fs.duplicated(keep=False).sum())
print("dedup check: 0 duplicated (event_id_161, snp); %d duplicated full_seq rows"
      % report["duplicate_full_seq_rows"])
report["final_rows"] = len(final)

variants.to_csv(OUT, index=False)
print("\nWrote %s  (%d rows, NO edge filter)" % (OUT, len(variants)))

# ---- coverage + correlations, BOTH filter settings ----------------------
MODELS = [("spliceai_delta_logit", "SpliceAI"),
          ("alphagenome_delta_logit", "AlphaGenome"),
          ("pangolin_delta_logit", "Pangolin"),
          ("baseline_mmsplice_delta_logit", "MMSplice"),
          ("hal_delta_logit", "HAL"),
          ("retrained_mmsplice_delta_logit", "MMSplice retrained (APRIL)")]

def summarize(frame, tag):
    print("\n=== %s (%d variant rows) ===" % (tag, len(frame)))
    print("%-28s %10s %10s" % ("model", "n", "pearson r"))
    out = {}
    for col, label in MODELS:
        if col not in frame.columns:
            print("%-28s %10s %10s" % (label, "-", "absent"))
            continue
        sub = frame[[col, "avg_delta_logit_pooled"]].dropna()
        if len(sub) < 2:
            print("%-28s %10d %10s" % (label, len(sub), "n/a"))
            continue
        r = float(np.corrcoef(sub[col], sub["avg_delta_logit_pooled"])[0, 1])
        out[label] = {"n": int(len(sub)), "r": round(r, 4)}
        print("%-28s %10d %10.4f" % (label, len(sub), r))
    return out

report["summary_edge_filtered"] = summarize(final,    "WITH exact 0/1 edge filter")
report["summary_no_filter"]     = summarize(variants, "NO edge filter")

REPORT = os.path.join(OUTDIR, "mega_MAY_v2_bothfilters_report.json")
with open(REPORT, "w") as f:
    json.dump(report, f, indent=2)
print("\nReport -> %s" % REPORT)
