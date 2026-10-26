import sys, os; sys.path.insert(0, "."); os.makedirs("figuras", exist_ok=True)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import analisis_revision as ar
allp=pd.concat([ar.load_v2(),ar.load_v3()],ignore_index=True)
P=[("efficientnetb1_v2","efficientnetb1_ft","EfficientNet-B1","#2a78d6","-","o"),
   ("vit32_as_run_v2","vit32_frozen_imagenet","ViT-B/32, frozen","#eb6834","--","s"),
   ("vit32_ft_ckpt","vit32_ft_ckpt","ViT-B/32, fine-tuned","#1baf7a","-.","D"),
   ("capsnet_r3_revised","capsnet_r3_revised","CapsNet-R3, revised","#e87ba4",(0,(5,1,1,1,1,1)),"^"),
   ("capsnet_r3_v2","capsnet_r3","CapsNet-R3, original","#eda100",":","v")]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
def style(a):
    for sp in ["top","right"]: a.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: a.spines[sp].set_color("#9a9a96")
    a.tick_params(colors="#4a4a46"); a.grid(color="#e6e6e3",lw=0.6); a.set_axisbelow(True)
# ROC split 2026
d=allp[(allp.split_seed==2026)&(allp.partition=="test")]
fig,ax=plt.subplots(figsize=(5.4,5.2),dpi=300); grid=np.linspace(0,1,501)
ax.plot([0,1],[0,1],color="#9a9a96",lw=1)
for m,_,name,c,ls,_ in P:
    tprs,aucs=[],[]
    for s,g in d[d.model==m].groupby("seed"):
        f,t,_=roc_curve(g.y_true,g.score); x=np.interp(grid,f,t); x[0]=0; tprs.append(x); aucs.append(roc_auc_score(g.y_true,g.score))
    mt=np.mean(tprs,0); mt[-1]=1
    ax.plot(grid,mt,color=c,ls=ls,lw=2,label=f"{name}  {np.mean(aucs):.3f} ± {np.std(aucs,ddof=1):.3f}")
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal"); style(ax)
ax.set_xlabel("1 − Specificity (false-positive rate)"); ax.set_ylabel("Sensitivity (true-positive rate)")
ax.legend(loc="lower right",frameon=False,fontsize=8.3,handlelength=3,title="Pipeline   AUC (mean ± SD)",title_fontsize=8.3)
fig.tight_layout(); fig.savefig("figuras/roc_split2026_v3.png",facecolor="white"); plt.close()
# repeated splits
b=allp[(allp.split_seed!=2026)&(allp.partition=="test")]
per=b.groupby(["model","split_seed","seed"]).apply(lambda g: roc_auc_score(g.y_true,g.score)).rename("auc").reset_index()
sm=per.groupby(["model","split_seed"]).auc.mean().reset_index()
orig=d.groupby(["model","seed"]).apply(lambda g: roc_auc_score(g.y_true,g.score)).groupby("model").mean()
fig,ax=plt.subplots(figsize=(7.2,4.2),dpi=300)
rng=np.random.default_rng(1)
for i,(m2026,mrep,name,c,_,mk) in enumerate(P):
    v=sm[sm.model==mrep].sort_values("split_seed").auc.values
    jit=np.linspace(-0.12,0.12,len(v))
    ax.scatter(i+jit,v,s=38,color=c,marker=mk,edgecolor="white",linewidth=0.8,zorder=3,label="Repeated splits (mean of 3 seeds)" if i==0 else None)
    ax.hlines(v.mean(),i-0.25,i+0.25,color="#3a3a38",lw=2,zorder=4,label="Mean across the 5 splits" if i==0 else None)
    ax.scatter(i+0.33,orig[m2026],s=46,facecolor="white",edgecolor=c,marker=mk,linewidth=1.5,zorder=3,label="Original split (mean of 5 seeds)" if i==0 else None)
    ax.text(i-0.28,v.mean(),f"{v.mean():.3f}",ha="right",va="center",fontsize=8.5,color="#3a3a38")
ax.set_xticks(range(len(P))); ax.set_xticklabels([p[2].replace(", ",",\n") for p in P],fontsize=9)
ax.set_ylabel("Test AUC"); ax.set_ylim(0.4,0.95); ax.set_xlim(-0.8,4.55); style(ax); ax.grid(axis="x",visible=False)
ax.axhline(0.5,color="#9a9a96",lw=1,ls=":"); ax.text(-0.75,0.507,"chance level",fontsize=8,color="#6b6b67",ha="left",va="bottom")
h,l=ax.get_legend_handles_labels()
for hh in h[:1]+h[2:]: pass
from matplotlib.lines import Line2D
hs=[Line2D([],[],marker="o",ls="",markerfacecolor="#6b6b67",markeredgecolor="white",markersize=7),
    Line2D([],[],color="#3a3a38",lw=2),
    Line2D([],[],marker="o",ls="",markerfacecolor="white",markeredgecolor="#6b6b67",markeredgewidth=1.5,markersize=7)]
ax.legend(hs,["Repeated splits (filled; mean of 3 seeds)","Mean across the 5 splits","Original split (open; mean of 5 seeds)"],loc="center",bbox_to_anchor=(0.4,0.42),frameon=False,fontsize=8.3)
fig.tight_layout(); fig.savefig("figuras/repeated_splits.png",facecolor="white"); plt.close()
print("ok")
