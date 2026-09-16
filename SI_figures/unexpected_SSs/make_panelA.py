import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["pdf.fonttype"]=42; mpl.rcParams["ps.fonttype"]=42; mpl.rcParams["svg.fonttype"]="none"
mpl.rcParams.update({
    "font.family":"sans-serif",
    "font.sans-serif":["Arial","Helvetica","Liberation Sans","DejaVu Sans"],
    "axes.titlesize":18,"axes.labelsize":17,"xtick.labelsize":15,"ytick.labelsize":15,
    "axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,
    "xtick.direction":"out","ytick.direction":"out",
})
import pandas as pd, numpy as np, matplotlib.pyplot as plt
D="/ESL/ESL_MPRA/SI_figures/unexpected_SSs"
CELLS=["HEK","HeLa","K562","MCF7","HMC3"]
df=pd.read_csv(f"{D}/cryptic_junctions_per_sequence_filtered.csv")
cols=[f"{c}_cryptic_fraction" for c in CELLS if f"{c}_cryptic_fraction" in df.columns]
avg=df[cols].mean(axis=1).dropna()
avg=avg[(avg>=0)&(avg<=1)]
fig,ax=plt.subplots(figsize=(7,5.2))
bins=np.linspace(0,1,50)
ax.hist(avg,bins=bins,color="#1F5FE0",edgecolor="white",linewidth=0.3)
ax.set_yscale("log")
ax.set_xlabel("Mean unexpected splicing frequency")
ax.set_ylabel("Number of sequences (log scale)")
ax.set_title("Noncanonical splicing across COMPASS constructs")
ax.set_xlim(0,1)
plt.tight_layout()
import os; os.makedirs(f"{D}/outputs_figS2",exist_ok=True)
for ext in ["png","pdf"]:
    fig.savefig(f"{D}/outputs_figS2/figA_unexpected_distribution.{ext}",bbox_inches="tight")
print("panel A n=",len(avg))
