"""
analisis_por_diagnostico.py

Descriptive per-diagnosis results on the test partition (Table 4 of the article):
sensitivity per malignant diagnosis and specificity per benign diagnosis at the
validation-derived Youden threshold, AUC of each malignant diagnosis vs. all
benign test images, and AUC of MEL vs. all other test images.
Also prints the lesion-level composition by diagnosis and partition (Table 1).

Usage:
    python analisis_por_diagnostico.py --results_dir . --ledger split_ledger.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PIPELINES = [("EfficientNet-B1", "predicciones_efficientnetb1_finetuned.csv"),
             ("ViT-Base/32", "predicciones_vit_base32_finetuned.csv"),
             ("CapsNet-R3", "predicciones_capsnet.csv")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=".")
    ap.add_argument("--ledger", default="split_ledger.csv")
    args = ap.parse_args()

    L = pd.read_csv(args.ledger)
    print("== Images (lesions) by diagnosis and partition ==")
    img = pd.crosstab(L.dx_official, L.partition, margins=True)
    les = L.groupby(["dx_official", "partition"]).lesion_id.nunique().unstack(fill_value=0)
    print(img)
    print(les)

    dx = L.set_index("image_id").dx_official
    for name, f in PIPELINES:
        P = pd.read_csv(os.path.join(args.results_dir, f))
        T = P[P.partition == "test"].copy()
        T["dx"] = T.image_id.map(dx)
        rows = []
        for seed, g in T.groupby("seed_run"):
            pred = (g.score >= g.youden_threshold.iloc[0]).astype(int)
            ben = g.y_true == 0
            r = {"seed": seed}
            for d in ["mel", "bcc", "akiec"]:
                m = g.dx == d
                r[f"sens_{d}"] = 100 * pred[m].mean()
                yy = np.r_[np.ones(m.sum()), np.zeros(ben.sum())]
                r[f"auc_{d}_vs_benign"] = roc_auc_score(yy, np.r_[g.score[m], g.score[ben]])
            for d in ["nv", "bkl", "df"]:
                m = g.dx == d
                r[f"spec_{d}"] = 100 * (1 - pred[m].mean())
            r["auc_mel_vs_rest"] = roc_auc_score((g.dx == "mel").astype(int), g.score)
            rows.append(r)
        D = pd.DataFrame(rows).set_index("seed")
        print(f"\n== {name} (mean, SD across seeds) ==")
        print(D.agg(["mean", "std"]).T.round(3).to_string())


if __name__ == "__main__":
    main()
