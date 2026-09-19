
import pandas as pd, ast, os
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["pdf.fonttype"]=42
mpl.rcParams["ps.fonttype"]=42
mpl.rcParams["svg.fonttype"]="none"
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Inputs (06_09_2025 supertable reproduces manuscript legend counts 1093/103/16 exactly)
main_csv="/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WITH_WT.csv.gz"
label_txt="/ESL/ESL_MPRA/Figure_4/inputs/supertable_dbSNP155_Gencode_v26_overlap_dbSNP155_matches_all_notes.txt"
clinvar_file="/ESL/ESL_MPRA/Figure_4/inputs/supertable_ClinVar_matched_in_supertable.txt"
outdir="/ESL/ESL_MPRA/Figure_4/outputs/regenerated_panels/fig4C"
os.makedirs(outdir,exist_ok=True)

CLINVAR_COLORS={"Pathogenic":"red","Likely_pathogenic":"orange","Uncertain_significance":"slateblue",
                "Likely_benign":"#9ACD32","Benign":"#258d4c","Conflicting":"orchid"}
# stacking order bottom->top
STACK_ORDER=["Benign","Likely_benign","Conflicting","Uncertain_significance","Likely_pathogenic","Pathogenic"]
DISPLAY={"Benign":"Benign (B)","Likely_benign":"Likely Benign (LB)","Pathogenic":"Pathogenic (P)",
         "Likely_pathogenic":"Likely Pathogenic (LP)","Uncertain_significance":"Uncertain Significance (VUS)",
         "Conflicting":"Conflicting (CL)"}
pop_labels=["commonAll","commonSome","rareAll"]
plot_order=["rareAll","commonSome","commonAll"]
disp={"commonAll":"common","commonSome":"sometimes rare","rareAll":"rare"}
cell_lines=["HeLa","K562","MCF7","HMC3","HEK"]

def load_label_map(fp):
    m={}
    for line in open(fp):
        parts=[p.strip() for p in line.strip().split(",") if p.strip()]
        if len(parts)<2: continue
        *labels,ref=parts
        try: m[int(ref)]=set(labels)
        except: pass
    return m
def parse_sig(x):
    sigs=set()
    try: it=ast.literal_eval(x)
    except: return None
    if isinstance(it,dict) and "ClinVar_info" in it: it=it["ClinVar_info"]
    if isinstance(it,list):
        for e in it:
            if isinstance(e,tuple) and len(e)==4 and "CLNSIG=" in e[3]:
                for part in e[3].split(";"):
                    if part.startswith("CLNSIG="):
                        sigs.update(s.strip().replace(" ","_") for s in part.split("=")[1].split("|"))
    keep={"benign":"Benign","likely_benign":"Likely_benign","pathogenic":"Pathogenic",
          "likely_pathogenic":"Likely_pathogenic","uncertain_significance":"Uncertain_significance"}
    m=set()
    for s in sigs:
        sl=s.lower()
        if sl in {"conflicting_classifications_of_pathogenicity","conflicting"}: m.add("Conflicting")
        elif sl in keep: m.add(keep[sl])
    if not m: return None
    return list(m)[0] if len(m)==1 else "Conflicting"

lm=load_label_map(label_txt)
def rbin(r):
    try: labs=lm.get(int(r),set())
    except: return None
    for l in pop_labels:
        if l in labs: return l
    return None

df=pd.read_csv(main_csv,low_memory=False)
cv=pd.read_csv(clinvar_file,sep="\t")[["mut_ref","ClinVar_item"]].dropna()
cv["CLNSIG_category"]=cv["ClinVar_item"].apply(parse_sig)
cv=cv[["mut_ref","CLNSIG_category"]].dropna().drop_duplicates().rename(columns={"mut_ref":"Reference"})
df=df.merge(cv,on="Reference",how="left")
df["rarity_bin"]=df["Reference"].apply(rbin)
df=df.dropna(subset=["rarity_bin","CLNSIG_category"])
df=df[df["CLNSIG_category"].isin(STACK_ORDER)].drop_duplicates("Reference")

ct=pd.crosstab(df["rarity_bin"],df["CLNSIG_category"]).reindex(index=plot_order,columns=STACK_ORDER).fillna(0).astype(int)
ct.to_csv(os.path.join(outdir,"fig4C_counts_crosstab.csv"))
totals=ct.sum(axis=1)
print("Per-category totals:",{disp[k]:int(v) for k,v in totals.items()})
print(ct.to_string())

xlabels=[f"{disp[r]}\n(n={int(totals[r])})" for r in plot_order]
colors=[CLINVAR_COLORS[c] for c in STACK_ORDER]

def make_bar(normalize, fname, ylab):
    data=ct.div(ct.sum(axis=1),axis=0) if normalize else ct
    fig,ax=plt.subplots(figsize=(4.2,4.0))
    bottom=[0.0]*len(plot_order)
    x=range(len(plot_order))
    for cls in STACK_ORDER:
        vals=data.loc[plot_order,cls].values
        ax.bar(x,vals,bottom=bottom,color=CLINVAR_COLORS[cls],width=0.7,edgecolor="white",linewidth=0.4,label=DISPLAY[cls])
        bottom=[b+v for b,v in zip(bottom,vals)]
    ax.set_xticks(list(x)); ax.set_xticklabels(xlabels,fontsize=9)
    ax.set_ylabel(ylab,fontsize=10)
    ax.set_xlabel("dbSNP155 allele-frequency category",fontsize=10)
    if normalize: ax.set_ylim(0,1)
    handles=[Patch(facecolor=CLINVAR_COLORS[c],label=DISPLAY[c]) for c in ["Benign","Likely_benign","Uncertain_significance","Conflicting","Likely_pathogenic","Pathogenic"]]
    ax.legend(handles=handles,fontsize=7,frameon=False,bbox_to_anchor=(1.01,1),loc="upper left")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,fname),dpi=300,bbox_inches="tight")
    fig.savefig(os.path.join(outdir,fname.replace(".pdf",".svg")),bbox_inches="tight")
    plt.close()
    print("wrote",fname)

make_bar(False,"fig4C_clinvar_composition_by_rarity_counts.pdf","Number of ClinVar-classified variants")
make_bar(True,"fig4C_clinvar_composition_by_rarity_normalized.pdf","Fraction of ClinVar-classified variants")
print("[DONE]")
