"""ROC empiricas out-of-fold de MobileNetV2 (split por lesion).
Usa las repeticiones 2 y 3: cada repeticion cubre una vez todas las imagenes
de Brown Skin; la repeticion 1 se excluye porque dos archivos de prediccion
(A y C40, fold 1) no coinciden con los AUC registrados.
Uso: python figura_roc_internas.py <carpeta_pred> <salida.png> <salida_auc.csv>"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
folder, out, out_csv = sys.argv[1:4]
REPS, FOLDS = [2, 3], [1, 2, 3, 4, 5]
def load(cfg):
    return pd.concat([pd.read_csv(os.path.join(folder, f"pred_{cfg}_rep{r}_fold{f}.csv"))
                      for r in REPS for f in FOLDS], ignore_index=True)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
panels = [
  ("(a) Adaptation strategy", [("A", "A: HAM10000 only", "#2a78d6", "-"),
                               ("B", "B: Brown Skin only", "#eb6834", "--"),
                               ("C40", "C40: HAM + FT (40)", "#1baf7a", "-"),
                               ("D", "D: HAM ∪ BS", "#eda100", ":"),
                               ("E", "E: HAM ∪ BS balanced", "#e87ba4", "-.")]),
  ("(b) Fine-tuning depth", [("C10", "C10", "#86b6ef", "-"), ("C20", "C20", "#5598e7", "-"),
                             ("C40", "C40", "#2a78d6", "-"), ("C60", "C60", "#1c5cab", "-"),
                             ("C80", "C80", "#104281", "-")]),
]
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.7), sharey=True)
rows = []
for ax, (title, cfgs) in zip(axes, panels):
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)))
    for cfg, label, color, ls in cfgs:
        d = load(cfg); fpr, tpr, _ = roc_curve(d.y_true, d.y_score)
        auc = roc_auc_score(d.y_true, d.y_score)
        ax.plot(fpr, tpr, color=color, lw=1.8, ls=ls, label=f"{label} ({auc:.3f})")
        if cfg not in [r["config"] for r in rows]:
            rows.append({"config": cfg, "pooled_oof_auc_rep2_3": round(auc, 4), "n": len(d)})
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("1 − Specificity", color=INK)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.95, edgecolor=GRID,
              title="Pooled OOF AUC", title_fontsize=7)
axes[0].set_ylabel("Sensitivity", color=INK)
fig.tight_layout(); fig.savefig(out, dpi=300, facecolor="white")
pd.DataFrame(rows).to_csv(out_csv, index=False); print(pd.DataFrame(rows))
