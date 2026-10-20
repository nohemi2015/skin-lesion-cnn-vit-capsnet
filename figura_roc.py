"""ROC curves on the test partition (Figure 1 of the article).

Usage: python figura_roc.py
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import argparse
ap=argparse.ArgumentParser(); ap.add_argument("--results_dir",default="."); ap.add_argument("--out",default="roc_stage2.png"); args=ap.parse_args()
import os
F=[("EfficientNet-B1","predicciones_efficientnetb1_finetuned.csv","#2a78d6","-"),
   ("ViT-Base/32","predicciones_vit_base32_finetuned.csv","#eb6834","--"),
   ("CapsNet-R3","predicciones_capsnet.csv","#1baf7a",":")]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
fig,ax=plt.subplots(figsize=(5.2,5.0),dpi=300)
ax.plot([0,1],[0,1],color="#9a9a96",lw=1,zorder=1)
ax.text(0.62,0.555,"Chance",color="#6b6b67",fontsize=8.5,rotation=45,ha="center",va="center")
grid=np.linspace(0,1,501)
for name,f,c,ls in F:
    d=pd.read_csv(os.path.join(args.results_dir,f)); d=d[d.partition=="test"]
    tprs,aucs=[],[]
    for s,g in d.groupby("seed_run"):
        fpr,tpr,_=roc_curve(g.y_true,g.score)
        ax.plot(fpr,tpr,color=c,lw=0.7,alpha=0.35,ls="-",zorder=2)
        t=np.interp(grid,fpr,tpr); t[0]=0; tprs.append(t); aucs.append(roc_auc_score(g.y_true,g.score))
        pass
    m=np.mean(tprs,0); m[-1]=1
    ax.plot(grid,m,color=c,lw=2,ls=ls,zorder=3,
            label=f"{name}  AUC {np.mean(aucs):.3f} ± {np.std(aucs,ddof=1):.3f}")
ax.set_xlim(0,1); ax.set_ylim(0,1.0); ax.set_aspect("equal")
ax.set_xlabel("1 − Specificity (false-positive rate)"); ax.set_ylabel("Sensitivity (true-positive rate)")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
for sp in ["left","bottom"]: ax.spines[sp].set_color("#9a9a96")
ax.tick_params(colors="#4a4a46"); ax.grid(color="#e6e6e3",lw=0.6); ax.set_axisbelow(True)
ax.legend(loc="lower right",frameon=False,fontsize=8.5,handlelength=2.6)
fig.tight_layout(); fig.savefig(args.out,facecolor="white")
