"""
analisis_revision.py

Analyses of the revision protocol. Reads the saved predictions only; no
training. Run it after all experiments (it skips parts whose files are absent).

  A. Original split (2026): every pipeline (v2 original runs + v3 runs), test
     AUC per seed, mean +/- SD, lesion-clustered bootstrap 95% CI of the
     seed-averaged AUC (2,000 replicates), paired differences against
     EfficientNet-B1 (v2), threshold metrics at the validation Youden threshold,
     and lesion-level AUC (mean score per lesion).
  B. Repeated lesion-grouped splits (3001-3005): test AUC per split and
     pipeline (mean over seeds); across-split mean and SD; within-split SD
     across seeds; paired differences against EfficientNet-B1 per split.
  C. Untouched confirmation cohort: seed-averaged AUC with lesion-clustered
     bootstrap CI, paired differences against EfficientNet-B1 (v2), and
     sensitivity/specificity at the validation-derived thresholds.

Usage:  python analisis_revision.py
Outputs: analisis_revision/*.csv and analisis_revision/resumen_analisis.txt
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = "analisis_revision"
B = 2000
REF = "efficientnetb1_v2"
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def load_v2(res_dir="resultados_split_final_v2"):
    names = {"efficientnetb1_finetuned": "efficientnetb1_v2", "vit_base32_finetuned": "vit32_as_run_v2",
             "capsnet": "capsnet_r3_v2"}
    out = []
    for tag, name in names.items():
        p = os.path.join(res_dir, f"predicciones_{tag}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            res = pd.read_csv(os.path.join(res_dir, f"resultados_{tag}.csv"))
            seedmap = dict(zip(res.seed_run, res.seed_value))
            out.append(pd.DataFrame({"model": name, "split_seed": 2026, "seed": d.seed_run.map(seedmap),
                                     "partition": d.partition, "image_id": d.image_id,
                                     "lesion_id": d.lesion_id, "y_true": d.y_true, "score": d.score,
                                     "thr": d.youden_threshold}))
    return pd.concat(out) if out else pd.DataFrame()


def load_v3(root="resultados_v3"):
    out = []
    for p in glob.glob(os.path.join(root, "split_*", "predicciones_*.csv")):
        d = pd.read_csv(p)
        if "test" not in set(d.partition):
            continue                                    # validation-only grid runs
        out.append(pd.DataFrame({"model": d.tag, "split_seed": d.split_seed, "seed": d.seed_value,
                                 "partition": d.partition, "image_id": d.image_id, "lesion_id": d.lesion_id,
                                 "y_true": d.y_true, "score": d.score, "thr": d.youden_threshold}))
    return pd.concat(out) if out else pd.DataFrame()


def score_matrix(d, ids):
    """seeds x images matrix of scores in the order of ids."""
    seeds = sorted(d.seed.unique())
    return np.stack([d[d.seed == s].set_index("image_id").score.loc[ids].values for s in seeds])


def boot_analysis(models, y, les, label, rng_seed=2026):
    """models: dict name -> seeds x images. Lesion-clustered bootstrap of seed-averaged AUC."""
    ul = np.unique(les)
    idx = {l: np.where(les == l)[0] for l in ul}
    mean_auc = lambda S, ii: np.mean([roc_auc_score(y[ii], S[s, ii]) for s in range(S.shape[0])])
    rng = np.random.default_rng(rng_seed)
    boots = {k: [] for k in models}
    for _ in range(B):
        ii = np.concatenate([idx[l] for l in rng.choice(ul, len(ul), replace=True)])
        if y[ii].min() == y[ii].max():
            continue
        for k, S in models.items():
            boots[k].append(mean_auc(S, ii))
    allidx = np.arange(len(y))
    rows = []
    for k, S in models.items():
        per = [roc_auc_score(y, S[s]) for s in range(S.shape[0])]
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        row = {"analysis": label, "model": k, "n_seeds": S.shape[0], "auc_mean": np.mean(per),
               "auc_sd_seeds": np.std(per, ddof=1) if len(per) > 1 else np.nan, "ci_low": lo, "ci_high": hi}
        if REF in models and k != REF:
            dd = np.array(boots[REF]) - np.array(boots[k])
            row.update({"diff_vs_ref": mean_auc(models[REF], allidx) - mean_auc(S, allidx),
                        "diff_ci_low": np.percentile(dd, 2.5), "diff_ci_high": np.percentile(dd, 97.5)})
        rows.append(row)
    return pd.DataFrame(rows)


def thr_metrics(d):
    rows = []
    for (m, s), g in d.groupby(["model", "seed"]):
        # compare in float32, the precision of the scores and thresholds at training time
        pred = (g.score.astype(np.float32) >= np.float32(g.thr.iloc[0])).astype(int)
        tp = ((pred == 1) & (g.y_true == 1)).sum(); tn = ((pred == 0) & (g.y_true == 0)).sum()
        fp = ((pred == 1) & (g.y_true == 0)).sum(); fn = ((pred == 0) & (g.y_true == 1)).sum()
        rows.append({"model": m, "seed": s, "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                     "precision": tp / (tp + fp) if tp + fp else np.nan, "accuracy": (tp + tn) / len(g)})
    r = pd.DataFrame(rows)
    return r.groupby("model")[["sensitivity", "specificity", "precision", "accuracy"]].agg(["mean", "std"])


def part_a(allp):
    d = allp[(allp.split_seed == 2026) & (allp.partition == "test")]
    if d.empty:
        return
    say("=" * 70 + "\nA. ORIGINAL SPLIT (2026), test partition\n" + "=" * 70)
    ids = sorted(set(d.image_id))
    base = d.drop_duplicates("image_id").set_index("image_id").loc[ids]
    y, les = base.y_true.values, base.lesion_id.values
    models = {m: score_matrix(g, ids) for m, g in d.groupby("model")}
    res = boot_analysis(models, y, les, "image-level")
    # lesion level: mean score per lesion
    ul = np.unique(les)
    y_l = np.array([y[les == l][0] for l in ul])
    models_l = {k: np.stack([[S[s, les == l].mean() for l in ul] for s in range(S.shape[0])])
                for k, S in models.items()}
    res_l = boot_analysis(models_l, y_l, ul, "lesion-level")
    res = pd.concat([res, res_l])
    res.to_csv(os.path.join(OUT, "A_split2026_auc.csv"), index=False)
    say(res.round(3).to_string(index=False))
    tm = thr_metrics(d)
    tm.to_csv(os.path.join(OUT, "A_split2026_umbral.csv"))
    say("\nThreshold metrics (validation Youden threshold), mean and SD across seeds:")
    say(tm.round(3).to_string())


def part_b(allp):
    d = allp[(allp.split_seed != 2026) & (allp.partition == "test")]
    if d.empty:
        return
    say("\n" + "=" * 70 + "\nB. REPEATED LESION-GROUPED SPLITS, test partitions\n" + "=" * 70)
    per = d.groupby(["model", "split_seed", "seed"]).apply(
        lambda g: roc_auc_score(g.y_true, g.score)).rename("auc").reset_index()
    per.to_csv(os.path.join(OUT, "B_splits_auc_por_semilla.csv"), index=False)
    split_mean = per.groupby(["model", "split_seed"]).auc.agg(["mean", "std", "count"]).reset_index()
    split_mean.to_csv(os.path.join(OUT, "B_splits_auc_por_split.csv"), index=False)
    say("Test AUC per split (mean over seeds):")
    say(split_mean.pivot(index="split_seed", columns="model", values="mean").round(3).to_string())
    summ = split_mean.groupby("model").agg(auc_mean_over_splits=("mean", "mean"),
                                           sd_across_splits=("mean", "std"),
                                           min_split=("mean", "min"), max_split=("mean", "max"),
                                           mean_within_split_sd_seeds=("std", "mean"),
                                           n_splits=("mean", "count")).reset_index()
    ref = "efficientnetb1_ft"
    if ref in set(split_mean.model):
        piv = split_mean.pivot(index="split_seed", columns="model", values="mean")
        diffs = piv[ref].values[:, None] - piv.values
        dd = pd.DataFrame(diffs, index=piv.index, columns=piv.columns)
        dd.to_csv(os.path.join(OUT, "B_splits_diferencias_vs_efficientnet.csv"))
        summ["diff_vs_efficientnet_mean"] = summ.model.map(dd.mean())
        summ["diff_vs_efficientnet_min"] = summ.model.map(dd.min())
        summ["diff_vs_efficientnet_max"] = summ.model.map(dd.max())
    summ.to_csv(os.path.join(OUT, "B_splits_resumen.csv"), index=False)
    say("\nSummary across splits:")
    say(summ.round(3).to_string(index=False))


def part_c():
    p = os.path.join("confirmacion", "predicciones_confirmacion.csv")
    if not os.path.exists(p):
        return
    say("\n" + "=" * 70 + "\nC. UNTOUCHED CONFIRMATION COHORT\n" + "=" * 70)
    d = pd.read_csv(p).rename(columns={"seed_value": "seed", "youden_threshold": "thr"})
    ids = sorted(set(d.image_id))
    base = d.drop_duplicates("image_id").set_index("image_id").loc[ids]
    y, les = base.y_true.values, base.lesion_id.values
    say(f"{len(ids)} images, {len(np.unique(les))} lesions, {y.sum()} malignant images")
    models = {m: score_matrix(g, ids) for m, g in d.groupby("model")}
    res = boot_analysis(models, y, les, "confirmation")
    res.to_csv(os.path.join(OUT, "C_confirmacion_auc.csv"), index=False)
    say(res.round(3).to_string(index=False))
    tm = thr_metrics(d)
    tm.to_csv(os.path.join(OUT, "C_confirmacion_umbral.csv"))
    say("\nSensitivity/specificity at the validation-derived thresholds (prevalence differs from the subset):")
    say(tm.round(3).to_string())


def main():
    os.makedirs(OUT, exist_ok=True)
    allp = pd.concat([load_v2(), load_v3()], ignore_index=True)
    if not allp.empty:
        part_a(allp)
        part_b(allp)
    part_c()
    with open(os.path.join(OUT, "resumen_analisis.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved in {OUT}/")


if __name__ == "__main__":
    main()
