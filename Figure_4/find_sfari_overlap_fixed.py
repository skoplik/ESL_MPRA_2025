# Fig 4H SFARI overlap stripplot — FIXED. Based on find_sfari_overlap.py.
# Root cause of empty panel: the old supertable (st_final_..._05_30_25) uses the
# pre-May event_id scheme, which after the May reprocessing corresponds to
# allseq["event_id_161"], NOT allseq["event_id"]. The script joined variant_count
# from allseq on "event_id" (0 rows overlap) -> variant_count all-NaN ->
# the ">30 variants" filter removed every exon -> no x categories, no points.
# Fix: bridge the old scheme to the new one via event_id_161, then do ALL
# downstream work (variant_count, variants, avg_delta_logit, ClinVar merge) in the
# NEW event_id scheme, which is what allseq and the ClinVar swarm file both use.
import pandas as pd, numpy as np, os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42; mpl.rcParams['ps.fonttype']=42; mpl.rcParams['svg.fonttype']='none'
import matplotlib.pyplot as plt

sfari_file="/ESL/ESL_MPRA/Figure_4/SFARI-Gene_genes_07-08-2025release_08-20-2025export.csv"
hgnc_file="/ESL/ESL_MPRA/Figure_4/inputs/hgnc_complete_set.txt"
supertable_file="/ESL/ESL_MPRA/Data_Pre-Processing/st_final_with_snp_and_coords_05_30_25_strandfix.csv"
allseq_file="/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz"
clinvar_path="/ESL/ESL_MPRA/Figure_4/outputs/clinvar/swarm_delta_logit.csv"
output_dir="/ESL/ESL_MPRA/SI_figures/ACMG_overlap/outputs_fixed"
os.makedirs(output_dir, exist_ok=True)
output_plot=os.path.join(output_dir,"sfari_overlap_clinvar_stripplot.pdf")

CLINVAR_COLORS={"Pathogenic":"red","Likely_pathogenic":"orange","Uncertain_significance":"slateblue",
    "Likely_benign":"#9ACD32","Benign":"#258d4c","Conflicting":"orchid"}

sfari=pd.read_csv(sfari_file)
hgnc=pd.read_csv(hgnc_file, sep="\t", dtype=str, low_memory=False)
st=pd.read_csv(supertable_file, dtype=str, low_memory=False)
allseq=pd.read_csv(allseq_file, dtype=str, low_memory=False)
clinvar_df=pd.read_csv(clinvar_path)

sfari=sfari.rename(columns={"ensembl-id":"ensembl_gene_id","gene-symbol":"symbol"})
def expand_aliases(x):
    if pd.isna(x): return []
    return [a.strip() for a in str(x).split("|")]
alias_map={}
for _,row in hgnc.iterrows():
    for name in set([row["symbol"]]+expand_aliases(row["alias_symbol"])+expand_aliases(row["prev_symbol"])):
        if pd.notna(row["ensembl_gene_id"]): alias_map[name]=row["ensembl_gene_id"]
sfari["ensembl_gene_id_mapped"]=sfari["symbol"].map(alias_map)
sfari["ensembl_gene_id_final"]=sfari["ensembl_gene_id"].combine_first(sfari["ensembl_gene_id_mapped"])

# --- SFARI gene overlap, keyed by OLD event_id (== new event_id_161) ---
st_genes=st[["Reference","event_id","gene_exon","gene_name","hgnc_id"]].copy()
st_genes=st_genes.rename(columns={"event_id":"event_id_161"})   # old scheme == new event_id_161
st_genes=st_genes.merge(hgnc[["hgnc_id","ensembl_gene_id"]], how="left", on="hgnc_id")
overlap=st_genes.merge(sfari[["symbol","ensembl_gene_id_final","gene-score","syndromic","genetic-category"]],
    how="inner", left_on="ensembl_gene_id", right_on="ensembl_gene_id_final").drop_duplicates()
print("SFARI-overlapping exons (event_id_161 scheme):", overlap["event_id_161"].nunique())

# --- Bridge event_id_161 -> new event_id via allseq (which carries both) ---
bridge=allseq[["event_id","event_id_161","gene_exon"]].drop_duplicates()
sfari_161=set(overlap["event_id_161"].dropna())
sfari_events=bridge[bridge["event_id_161"].isin(sfari_161)][["event_id","gene_exon"]].drop_duplicates()
print("Bridged to NEW event_id:", sfari_events["event_id"].nunique(), "exons")

# --- variant counts in NEW scheme (variants only, snp != none) ---
vcount=allseq[allseq["snp"]!="none"]["event_id"].value_counts().rename("variant_count").reset_index()
vcount.columns=["event_id","variant_count"]
sfari_events=sfari_events.merge(vcount, on="event_id", how="left")
sfari_events["variant_count"]=sfari_events["variant_count"].fillna(0).astype(int)
sfari_events=sfari_events[sfari_events["variant_count"]>30].copy()
sfari_events=sfari_events.sort_values("gene_exon").reset_index(drop=True)
sfari_events["label"]=sfari_events["gene_exon"].astype(str)+", n="+sfari_events["variant_count"].astype(str)
n_exons=len(sfari_events)
print("Exons with >30 variants (x categories):", n_exons)
print("Distinct SFARI genes among them:", sfari_events["gene_exon"].str.split(" exon").str[0].nunique())

# --- avg delta logit across cell lines (NEW scheme) ---
dcols=[c for c in allseq.columns if c.endswith("_delta_logit_pooled")]
allseq[dcols]=allseq[dcols].apply(pd.to_numeric, errors="coerce")
allseq["avg_delta_logit"]=allseq[dcols].mean(axis=1)

variants=allseq[(allseq["event_id"].isin(sfari_events["event_id"]))&(allseq["snp"]!="none")].copy()
xpos={ev:i for i,ev in enumerate(sfari_events["event_id"])}
variants["x_pos"]=variants["event_id"].map(xpos)
rng=np.random.default_rng(seed=42)
variants["x_jitter"]=variants["x_pos"]+rng.uniform(-0.3,0.3,size=len(variants))
print("Total plotted variant points (grey):", variants["avg_delta_logit"].notna().sum())

# --- ClinVar merge (NEW scheme: clinvar_df.event_id matches allseq.event_id) ---
variants["Reference"]=variants["Reference"].astype(str)
clinvar_df["Reference"]=clinvar_df["Reference"].astype(str)
clinvar_df["event_id"]=clinvar_df["event_id"].astype(str)
annot=variants.merge(clinvar_df[["Reference","event_id","CLNSIG_category"]], on=["Reference","event_id"], how="inner")
print("Total ClinVar-annotated SFARI variants (n):", len(annot))
print("ClinVar breakdown:\n", annot["CLNSIG_category"].value_counts().to_string())

# --- Plot ---
fig,ax=plt.subplots(figsize=(11,6))
ax.scatter(variants["x_jitter"],variants["avg_delta_logit"],color="lightgrey",alpha=0.6,s=30,
           edgecolor="none",label="All variants",rasterized=True,zorder=1)
plot_order=["Likely_benign","Benign","Conflicting","Uncertain_significance","Likely_pathogenic","Pathogenic"]
for cat in plot_order:
    sub=annot[annot["CLNSIG_category"]==cat]
    if not sub.empty:
        ax.scatter(sub["x_jitter"],sub["avg_delta_logit"],facecolors=CLINVAR_COLORS[cat],edgecolors="none",
                   s=35,alpha=0.6,rasterized=True,zorder=2)
        ax.scatter(sub["x_jitter"],sub["avg_delta_logit"],facecolors="none",edgecolors=CLINVAR_COLORS[cat],
                   s=35,linewidth=0.6,alpha=1.0,label=cat,rasterized=True,zorder=2)
ax.set_xticks(range(n_exons))
ax.set_xticklabels(sfari_events["label"],rotation=45,ha="right",fontsize=8)
ax.axhline(0,color="black",linestyle="--",linewidth=1)
ax.set_ylabel("Average Δlogit(PSI) Across Cell Lines")
ax.set_xlabel("SFARI Gene / Exon (alphabetical)")
ax.set_title("SFARI Overlap Δlogit(PSI) with ClinVar Annotation")
handles,labels=ax.get_legend_handles_labels()
by_label=dict(zip(labels,handles))
ax.legend(by_label.values(),by_label.keys(),bbox_to_anchor=(1.01,1),loc="upper left",fontsize=8)
plt.tight_layout()
plt.savefig(output_plot,bbox_inches="tight",dpi=600)
plt.close()
sfari_events.to_csv(os.path.join(output_dir,"sfari_event_exons.csv"),index=False)
annot[["Reference","event_id","gene_exon","CLNSIG_category","avg_delta_logit"]].to_csv(os.path.join(output_dir,"sfari_clinvar_points.csv"),index=False)
print("Saved:", output_plot)
print("DONE")
