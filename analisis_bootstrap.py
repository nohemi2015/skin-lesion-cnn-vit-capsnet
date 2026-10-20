"""
analisis_bootstrap.py

Seed-averaged test AUC with 95% lesion-clustered bootstrap CIs
(2000 replicates over test lesions) and paired differences between
pipelines. Reproduces the AUC intervals reported in the article.

Usage:
    python src/analisis_bootstrap.py --results_dir results --ledger data/split_ledger.csv
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
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--ledger", default="data/split_ledger.csv")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    L = pd.read_csv(args.ledger)
    print("== Partition composition ==")
    for p in ["train", "val", "test"]:
        s = L[L.partition == p]
        print(f"{p}: {len(s)} images, {s.lesion_id.nunique()} lesions, "
              f"{int(s.label.sum())} malignant images ({100 * s.label.mean():.1f}%)")
    print(pd.crosstab(L.dx_official, L.partition, margins=True))

    P = {}
    for name, f in PIPELINES:
        x = pd.read_csv(os.path.join(args.results_dir, f))
        P[name] = x[x.partition == "test"]
    first = P[PIPELINES[0][0]]
    base = first[first.seed_run == first.seed_run.min()][["image_id", "lesion_id", "y_true"]].reset_index(drop=True)
    seeds = sorted(first.seed_run.unique())
    S = {k: np.stack([t[t.seed_run == s].set_index("image_id").loc[base.image_id, "score"].values for s in seeds])
         for k, t in P.items()}
    y = base.y_true.values
    les = base.lesion_id.values
    ul = np.unique(les)
    idx = {l: np.where(les == l)[0] for l in ul}

    def mean_auc(sc, ii):
        return np.mean([roc_auc_score(y[ii], sc[s, ii]) for s in range(len(seeds))])

    rng = np.random.default_rng(args.seed)
    boots = {k: [] for k in S}
    for _ in range(args.n_boot):
        ii = np.concatenate([idx[l] for l in rng.choice(ul, len(ul), replace=True)])
        if y[ii].min() == y[ii].max():
            continue
        for k in S:
            boots[k].append(mean_auc(S[k], ii))
    allidx = np.arange(len(y))
    print("\n== Seed-averaged test AUC, 95% lesion-clustered bootstrap CI ==")
    for k in S:
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        print(f"{k}: {mean_auc(S[k], allidx):.3f} [{lo:.3f}, {hi:.3f}]")
    print("\n== Paired differences ==")
    names = list(S)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, c = names[i], names[j]
            d = np.array(boots[a]) - np.array(boots[c])
            obs = mean_auc(S[a], allidx) - mean_auc(S[c], allidx)
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"{a} - {c}: {obs:.3f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
