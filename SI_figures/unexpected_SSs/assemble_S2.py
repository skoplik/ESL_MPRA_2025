
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"]=42; mpl.rcParams["ps.fonttype"]=42
import matplotlib.pyplot as plt
OUT="/ESL/ESL_MPRA/SI_figures/unexpected_SSs/outputs_nature"
panels={'A': 'figA_unexpected_distribution', 'B': 'fig1_cryptic_vs_wt_psi', 'C': 'fig2_cryptic_by_variant_class', 'D': 'fig4_top_exons', 'E': 'fig3_cryptic_position_map'}
imgs={k:plt.imread(f"{OUT}/_r_{k}.png") for k in panels}
asp={k:imgs[k].shape[1]/imgs[k].shape[0] for k in panels}   # w/h
margin=42.0; gap=34.0; colW=540.0
contentW=2*colW+gap; W=contentW+2*margin
h1=max(colW/asp["A"],colW/asp["C"]); hB=contentW/asp["B"]; h3=max(colW/asp["D"],colW/asp["E"])
H=h1+hB+h3+2*gap+2*margin
top1=H-margin
pl={}
pl["A"]=(margin, top1-colW/asp["A"], colW, colW/asp["A"])
pl["C"]=(margin+colW+gap, top1-colW/asp["C"], colW, colW/asp["C"])
top2=top1-h1-gap
pl["B"]=(margin, top2-hB, contentW, hB)
top3=top2-hB-gap
pl["D"]=(margin, top3-colW/asp["D"], colW, colW/asp["D"])
pl["E"]=(margin+colW+gap, top3-colW/asp["E"], colW, colW/asp["E"])
tops={"A":top1,"C":top1,"B":top2,"D":top3,"E":top3}
fig=plt.figure(figsize=(W/72.0,H/72.0))
for k in panels:
    x,y,w,h=pl[k]
    ax=fig.add_axes([x/W, y/H, w/W, h/H]); ax.imshow(imgs[k],aspect="auto"); ax.axis("off")
for k in panels:
    fig.text((pl[k][0]-4)/W,(tops[k]+8)/H,k,fontsize=28,fontweight="bold",va="bottom",ha="left")
fig.savefig(f"{OUT}/FigureS2_assembled.pdf")
fig.savefig(f"{OUT}/FigureS2_assembled.png",dpi=160)
print("WROTE assembled  page(in)=",round(W/72,2),round(H/72,2))
