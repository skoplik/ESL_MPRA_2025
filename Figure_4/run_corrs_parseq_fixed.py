# Fig 4G ParSE-seq scatter — FIXED. Based on run_corrs_parseq_ci.py.
# Root cause of broken panel: the ClinVar classification join in the script that
# produced fig_parse_seq_sep2026/parseq_psi.pdf left every merged variant as
# "Not_provided", and plot_scatter only draws scatter() for rows whose
# classification is one of the 6 colored ClinVar categories -> zero points drawn
# while the regression line + "N=11" title still rendered.
# Fix: (1) correct ClinVar join on Reference==mut_ref against the CURRENT (May-repro)
# supertable, which populates real classes (VUS / Conflicting); (2) make the plotter
# robust so any uncategorized point is still drawn (default grey), never silently dropped.
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
import pandas as pd, numpy as np, os, ast
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

parse_seq_path = "/ESL/ESL_MPRA/Figure_4/41467_2024_52474_MOESM7_ESM.csv"
supertable_path = "/ESL/ESL_MPRA/Data_Pre-Processing/Post-process_STAR_PSIs/output/1e-2_ALL_WTS_VARS_NO_DELTAS.csv.gz"
clinvar_path = "/ESL/ESL_MPRA/Figure_4/inputs/supertable_ClinVar_matched_in_supertable.txt"
output_dir = "/ESL/ESL_MPRA/Figure_4/outputs/fig_parse_seq_fixed"
os.makedirs(output_dir, exist_ok=True)

CLINVAR_COLORS = {
    "Pathogenic": "red",
    "Likely_pathogenic": "orange",
    "Uncertain Significance (VUS)": "slateblue",
    "Likely_benign": "#9ACD32",
    "Benign": "#258d4c",
    "Conflicting (CL)": "orchid",
}
CLINVAR_ORDER = list(CLINVAR_COLORS.keys())
DEFAULT_COLOR = "0.6"  # grey for uncategorized / Not_provided

def parse_clinvar_significance(clinvar_item):
    sigs=set()
    if isinstance(clinvar_item,dict) and "ClinVar_info" in clinvar_item: clinvar_item=clinvar_item["ClinVar_info"]
    if isinstance(clinvar_item,list):
        for entry in clinvar_item:
            if isinstance(entry,tuple) and len(entry)==4:
                meta=entry[3]
                if "CLNSIG=" in meta:
                    for part in meta.split(";"):
                        if part.startswith("CLNSIG="):
                            sigs.update(part.split("=")[1].replace(" ","_").split("|"))
    elif isinstance(clinvar_item,str):
        for part in clinvar_item.split(";"):
            if part.startswith("CLNSIG="):
                sigs.update(part.split("=")[1].replace(" ","_").split("|"))
    norm=set()
    for s in sigs:
        sl=s.strip().lower()
        if sl in {"conflicting_classifications_of_pathogenicity","conflicting"}: norm.add("Conflicting (CL)")
        elif sl=="uncertain_significance": norm.add("Uncertain Significance (VUS)")
        elif sl=="not_provided": norm.add("Not_provided")
        else: norm.add(s.replace(" ","_"))
    if not norm: return "Not_provided"
    if len(norm)>1: return "Conflicting (CL)"
    return list(norm)[0]

def format_hgvs_to_variant_hg38(h):
    c,p,r,a=h.split("-"); return f"chr{c}:{p}:{r}>{a}"

def regression_ci_band(x_vals, y_vals, x_line, ci=0.95):
    slope, intercept, r, p, se = stats.linregress(x_vals, y_vals)
    y_fit = slope*x_line + intercept
    n=len(x_vals); x_mean=np.mean(x_vals)
    residuals=y_vals-(slope*x_vals+intercept)
    s_err=np.sqrt((residuals**2).sum()/(n-2))
    t_val=stats.t.ppf((1+ci)/2, df=n-2)
    x_dev=x_line-x_mean
    ci_band=t_val*s_err*np.sqrt(1.0/n + x_dev**2/((x_vals-x_mean)**2).sum())
    return y_fit, y_fit-ci_band, y_fit+ci_band

def plot_scatter(merged, x, y, filename, xlabel, ylabel):
    df = merged[[x, y, "ClinVar Classification", "gene_exon"]].dropna(subset=[x,y])
    if df.empty:
        print(f"Skipping {filename} — no data."); return 0
    gene_exon = df["gene_exon"].iloc[0]
    r, p = stats.pearsonr(df[x], df[y]); n=len(df)
    x_vals=df[x].values; y_vals=df[y].values
    x_line=np.linspace(x_vals.min(), x_vals.max(), 300)
    y_fit, lower, upper = regression_ci_band(x_vals, y_vals, x_line)
    plt.figure(figsize=(4.5,4.5)); ax=plt.gca()
    ax.fill_between(x_line, lower, upper, color="grey", alpha=0.2, zorder=0, label="95% CI")
    ax.plot(x_line, y_fit, color="black", linewidth=1, zorder=1, label="Regression")
    drawn=0
    # Draw known ClinVar categories in defined order
    for label in CLINVAR_ORDER:
        sub=df[df["ClinVar Classification"]==label]
        if not sub.empty:
            fc=mcolors.to_rgba(CLINVAR_COLORS[label], alpha=0.6)
            ec=mcolors.to_rgba(CLINVAR_COLORS[label], alpha=1.0)
            ax.scatter(sub[x], sub[y], s=40, facecolors=fc, edgecolors=ec,
                       linewidth=0.7, label=label, zorder=2)
            drawn+=len(sub)
    # Robustness: draw any point NOT in a known colored category (never silently drop)
    other=df[~df["ClinVar Classification"].isin(CLINVAR_ORDER)]
    if not other.empty:
        ax.scatter(other[x], other[y], s=40, facecolors=mcolors.to_rgba(DEFAULT_COLOR,alpha=0.6),
                   edgecolors=mcolors.to_rgba(DEFAULT_COLOR,alpha=1.0), linewidth=0.7,
                   label="Not in ClinVar", zorder=2)
        drawn+=len(other)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{gene_exon}\nr = {r:.3f}, p = {p:.1e}, N = {n}")
    handles,labels_list=ax.get_legend_handles_labels()
    by_label=dict(zip(labels_list,handles)); ordered=[]
    for entry in ["Regression","95% CI"]+CLINVAR_ORDER+["Not in ClinVar"]:
        if entry in by_label: ordered.append((entry,by_label[entry]))
    ax.legend([h for _,h in ordered],[l for l,_ in ordered], fontsize=7, loc="best")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, filename)); plt.close()
    print(f"Saved {filename}: N={n}, scatter points drawn={drawn}, r={r:.4f}, p={p:.2e}")
    return drawn

# === Load
parse_df=pd.read_csv(parse_seq_path)
psi_df=pd.read_csv(supertable_path, low_memory=False)
clin=pd.read_csv(clinvar_path, sep="\t")
print("supertable rows", len(psi_df), "| has variant_hg38:", "variant_hg38" in psi_df.columns)

clin["ClinVar_item"]=clin["ClinVar_item"].apply(ast.literal_eval)
clin["ClinVar Classification"]=clin["ClinVar_item"].apply(parse_clinvar_significance)

psi_df=pd.merge(psi_df, clin[["mut_ref","ClinVar Classification"]], left_on="Reference", right_on="mut_ref", how="left")
psi_df["ClinVar Classification"]=psi_df["ClinVar Classification"].fillna("Not_provided")

parse_df["variant_hg38"]=parse_df["HGVS Variant"].apply(format_hgvs_to_variant_hg38)
rep_cols=["HEK PSI Rep 1","HEK PSI Rep 2","HEK PSI Rep 3"]
for c in rep_cols: parse_df[c]=parse_df[c]/100
parse_df["parse_psi_mean"]=parse_df[rep_cols].mean(axis=1)
parse_df["parse_delta_psi"]=parse_df["delta_psi"]/100

merged=pd.merge(psi_df, parse_df, on="variant_hg38", how="inner", suffixes=("","_parse"))
print("merged rows (psi x parse):", len(merged))
hek_reps=[c for c in ["HEK_rep1_psi_raw","HEK_rep2_psi_raw","HEK_rep3_psi_raw","HEK_rep4_psi_raw"] if c in merged.columns]
merged["our_psi_mean"]=merged[hek_reps].mean(axis=1)
wt=psi_df[psi_df["snp"]=="none"][["event_id","HEK_pooled_psi_clipped"]].rename(columns={"HEK_pooled_psi_clipped":"our_wt_psi"})
merged=pd.merge(merged, wt, on="event_id", how="left")
merged["our_delta_psi"]=merged["our_psi_mean"]-merged["our_wt_psi"]
print("ClinVar class value_counts in merged:")
print(merged["ClinVar Classification"].value_counts().to_string())

n1=plot_scatter(merged, "our_psi_mean","parse_psi_mean","parseq_psi.pdf",
             "COMPASS HEK293 Mean PSI","ParSE-seq HEK293 Mean PSI")
n2=plot_scatter(merged, "our_delta_psi","parse_delta_psi","parseq_dpsi.pdf",
             "COMPASS HEK293 Mean ΔPSI","ParSE-seq HEK293 Mean ΔPSI")
merged[["variant_hg38","gene_exon","our_psi_mean","parse_psi_mean","our_delta_psi","parse_delta_psi","ClinVar Classification"]].to_csv(os.path.join(output_dir,"parseq_merged.csv"), index=False)
print("PSI panel points:", n1, "| dPSI panel points:", n2)
print("DONE")
