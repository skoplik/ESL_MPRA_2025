"""
Generate a markdown summary of the 16 events whose data changed between the
03_16/04_24 baseline and the 04_26 corrected pipeline:

  - 11 CHANGED_EVENTS  : chosen junction shifted to a different (MANE) transcript
  - 5  RESCUED_EVENTS  : primary reference was a synthesis dropout; chosen
                        junction selected via variant-pool fallback

Output: /ESL/Figures_SK/ambiguous_sjs/changed_events_summary_04_26_2026.md
"""

import gzip
import os
import sys
import numpy as np
import pandas as pd

APR26 = "/ESL/Figures_SK/General_preprocessing/output_04_26_2026"
MAR16 = "/ESL/Figures_SK/General_preprocessing/output_03_16_2026"
GZ    = "/ESL/Figures_SK/General_preprocessing/fix_supertable_2/st_final_with_snp_and_coords_05_30_25.csv.gz"

OUT_MD = "/ESL/Figures_SK/ambiguous_sjs/changed_events_summary_04_26_2026.md"

CELLS = ["HeLa", "K562", "MCF7", "HMC3", "HEK"]

# 11 events whose chosen junction differs from the originally annotated transcript.
CHANGED_EVENTS = [
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
]

# 5 events recovered via the variant-pool fallback (primary ref was a dropout).
RESCUED_EVENTS = [
    "chr12:98807824-98807984:-",
    "chr19:14152113-14152273:-",
    "chr5:163467608-163467768:+",
    "chr6:167954353-167954513:+",
    "chr8:132799032-132799192:+",
]

# (chr14:92096073-92096233:- was the only event with zero reads anywhere; not
# in either output, so not in this summary.)


def fmt_psi(v):
    return "—" if pd.isna(v) else f"{v:.3f}"


def fmt_seq(s, head=20, tail=20):
    if not isinstance(s, str):
        return "—"
    if len(s) <= head + tail + 3:
        return s
    return f"{s[:head]}…{s[-tail:]}"


def event_block(eid, src, st_new, mar16, new, label):
    sub_new   = st_new[st_new.event_id == eid]
    sub_src   = src[src.event_id == eid]
    if sub_new.empty:
        return f"### {eid}  *(no rows in corrected supertable)*\n\n"

    # Pick representative WT row for sequence display.
    wt_row = sub_new[sub_new.snp == "none"].sort_values("Reference").iloc[0]

    # Original (source) transcripts seen for this event.
    src_with_lens = sub_src.assign(
        intron1_len=sub_src["intron1"].str.len(),
        exon_len=sub_src["exon"].str.len(),
    )
    src_txs = src_with_lens.drop_duplicates("transcript_id")[
        ["transcript_id", "exon_start_hg38", "exon_end_hg38", "intron1_len", "exon_len"]
    ]

    chosen_tx     = wt_row["transcript_id"]
    chosen_mane   = wt_row.get("mane_status", "")
    chosen_i1l    = int(wt_row["intron1_len"])
    chosen_exl    = int(wt_row["exon_len"])
    chosen_start  = wt_row["exon_start_hg38"]
    chosen_end    = wt_row["exon_end_hg38"]
    gene_exon     = wt_row["gene_exon"]

    n_rows_event  = int((new.event_id == eid).sum())
    n_wt_event    = int(((new.event_id == eid) & (new.snp == "none")).sum())
    n_var_event   = n_rows_event - n_wt_event

    n_rows_old    = int((mar16.event_id == eid).sum())
    n_wt_old      = int(((mar16.event_id == eid) & (mar16.snp == "none")).sum())
    n_var_old     = n_rows_old - n_wt_old

    # WT pooled PSI per cell, old vs new.
    wt_old_psi = mar16[(mar16.event_id == eid) & (mar16.snp == "none")]
    wt_new_psi = new[(new.event_id == eid) & (new.snp == "none")]

    # SDV counts per cell on the new data: |delta_logit_pooled| >= 1.
    sdv_counts = {}
    for c in CELLS:
        col = f"{c}_delta_logit_pooled"
        v = new[(new.event_id == eid) & (new.snp != "none")][col].dropna()
        sdv_counts[c] = (v.abs() >= 1).sum()

    out = []
    out.append(f"### {eid}  *(gene_exon: {gene_exon} — {label})*")
    out.append("")
    out.append(f"- **Chosen transcript**: `{chosen_tx}` ({chosen_mane or 'no MANE'})")
    out.append(f"- **Chosen intron1_len / exon_len**: {chosen_i1l} nt / {chosen_exl} nt")
    out.append(f"- **Chosen exon coords (hg38)**: {chosen_start} – {chosen_end}")
    out.append(f"- **Source transcripts seen for this event**:")
    for _, t in src_txs.iterrows():
        marker = " ←chosen" if str(t['transcript_id']).split('.')[0] == str(chosen_tx).split('.')[0] else ""
        out.append(f"    - `{t['transcript_id']}` (i1l={int(t['intron1_len'])}, exl={int(t['exon_len'])}, "
                   f"exon {t['exon_start_hg38']}–{t['exon_end_hg38']}){marker}")
    out.append(f"- **Rows in main CSV** — 03_16: {n_rows_old} ({n_wt_old} WT, {n_var_old} variants);  "
               f"04_26: {n_rows_event} ({n_wt_event} WT, {n_var_event} variants)  "
               f"→ Δvariants: {n_var_event - n_var_old:+d}")
    out.append("")
    out.append(f"**Representative WT (Reference={int(wt_row['Reference'])}) sequences after correction:**")
    out.append("")
    out.append(f"```")
    out.append(f"intron1 ({chosen_i1l} nt): {wt_row['intron1']}")
    out.append(f"exon    ({chosen_exl} nt): {wt_row['exon']}")
    out.append(f"intron2 (20 nt): {wt_row['intron2']}")
    out.append(f"full_seq (161 nt): {wt_row['full_seq']}")
    out.append(f"```")
    out.append("")

    # PSI table per cell line — counts variants with non-NaN pooled PSI in
    # both 03_16 and 04_26 so the user can see how many measurements were
    # gained/lost per cell after the junction correction.
    out.append("**WT pooled PSI per cell line (03_16 baseline → 04_26 corrected):**")
    out.append("")
    out.append("| Cell | WT PSI 03_16 | WT PSI 04_26 | # variants 03_16 | # variants 04_26 | # SDVs 04_26 (\\|Δlogit\\|≥1) |")
    out.append("|------|-------------:|-------------:|-----------------:|-----------------:|----------------------------:|")
    for c in CELLS:
        col = f"{c}_pooled_psi_raw"
        old = wt_old_psi[col].mean() if col in wt_old_psi.columns else np.nan
        new_v = wt_new_psi[col].mean() if col in wt_new_psi.columns else np.nan
        n_var_old_c = mar16[(mar16.event_id == eid) & (mar16.snp != "none")][col].notna().sum() \
                      if col in mar16.columns else 0
        n_var_new_c = new[(new.event_id == eid) & (new.snp != "none")][col].notna().sum() \
                      if col in new.columns else 0
        out.append(f"| {c} | {fmt_psi(old)} | {fmt_psi(new_v)} | "
                   f"{int(n_var_old_c)} | {int(n_var_new_c)} | {int(sdv_counts[c])} |")
    out.append("")
    return "\n".join(out)


def main():
    print("Loading data...")
    with gzip.open(GZ, "rt") as f:
        src = pd.read_csv(f, low_memory=False)
    st_new = pd.read_csv(f"{APR26}/st_04_26_2026.csv", low_memory=False)
    new    = pd.read_csv(f"{APR26}/04_26_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv", low_memory=False)
    mar16  = pd.read_csv(f"{MAR16}/03_16_2026_1e-2_ALL_WTS_VARS_NO_DELTAS.csv", low_memory=False)

    parts = []
    parts.append("# Changed events — 04/26/2026 ambig-SJ correction\n")
    parts.append("Events whose chosen junction or PSI differs from the 03/16/2026 baseline. ")
    parts.append("All sequences shown are after the consistency rewrite: `intron1` / `exon` / `intron2` ")
    parts.append("derived by splitting `full_seq` at the chosen junction's `(intron1_len, exon_len)`. ")
    parts.append("PSI columns are pooled across replicates for the WT (snp=none) row of each event.\n")
    parts.append(f"\nGenerated by `build_changed_events_summary.py` from outputs in `{APR26}/`.\n")

    parts.append("\n## Summary table\n")
    parts.append("| Event | Gene | Class | Chosen tx | i1l | exl | # vars 03_16 | # vars 04_26 | Δ |")
    parts.append("|-------|------|-------|-----------|----:|----:|-------------:|-------------:|--:|")
    for eid in CHANGED_EVENTS + RESCUED_EVENTS:
        sub_new = st_new[st_new.event_id == eid]
        if sub_new.empty:
            continue
        wt_row = sub_new[sub_new.snp == "none"].sort_values("Reference").iloc[0]
        cls = "junction shifted to MANE" if eid in CHANGED_EVENTS else "rescued via variant-pool fallback"
        n_var_new   = int(((new.event_id == eid) & (new.snp != "none")).sum())
        n_var_old   = int(((mar16.event_id == eid) & (mar16.snp != "none")).sum())
        parts.append(f"| `{eid}` | {wt_row['gene_exon']} | {cls} | "
                     f"`{wt_row['transcript_id']}` | {int(wt_row['intron1_len'])} | "
                     f"{int(wt_row['exon_len'])} | {n_var_old} | {n_var_new} | {n_var_new - n_var_old:+d} |")
    parts.append("")

    parts.append("\n## 11 CHANGED_EVENTS (junction shifted to a MANE transcript)\n")
    for eid in CHANGED_EVENTS:
        parts.append(event_block(eid, src, st_new, mar16, new, "junction shifted to MANE"))

    parts.append("\n## 5 RESCUED events (primary ref was a synthesis dropout)\n")
    for eid in RESCUED_EVENTS:
        parts.append(event_block(eid, src, st_new, mar16, new, "rescued via variant-pool fallback"))

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(parts))
    print(f"Wrote {OUT_MD}  ({os.path.getsize(OUT_MD):,} bytes)")


if __name__ == "__main__":
    main()
