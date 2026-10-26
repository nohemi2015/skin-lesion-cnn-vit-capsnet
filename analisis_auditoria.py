"""Additional statistics requested in the reporting audit (no training).
Reads the archived predictions; lesion-clustered bootstrap (2,000 replicates).
Run in the working folder that contains resultados_split_final_v2/, resultados_v3/ and confirmacion/.
Outputs: analisis_auditoria/*.csv and stats_rev3.json"""
import os, glob, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
R = "."; B = 2000; OUT = "analisis_auditoria"; os.makedirs(OUT, exist_ok=True)
NAMES = {"efficientnetb1_v2": "EfficientNet-B1", "vit32_as_run_v2": "ViT-B/32, frozen, ImageNet norm.",
         "vit32_frozen_ckpt": "ViT-B/32, frozen, reference norm.", "vit32_ft_ckpt": "ViT-B/32, fine-tuned, reference norm.",
         "vit32_ft_imagenet": "ViT-B/32, fine-tuned, ImageNet norm.", "capsnet_r3_v2": "CapsNet-R3, original",
         "capsnet_r3_revised": "CapsNet-R3, revised"}
ORDER = list(NAMES)

def load_2026():
    out = []
    m = {"efficientnetb1_finetuned": "efficientnetb1_v2", "vit_base32_finetuned": "vit32_as_run_v2", "capsnet": "capsnet_r3_v2"}
    for tag, name in m.items():
        d = pd.read_csv(f"{R}/resultados_split_final_v2/predicciones_{tag}.csv")
        res = pd.read_csv(f"{R}/resultados_split_final_v2/resultados_{tag}.csv")
        sm = dict(zip(res.seed_run, res.seed_value))
        out.append(pd.DataFrame({"model": name, "seed": d.seed_run.map(sm), "partition": d.partition,
                                 "image_id": d.image_id, "lesion_id": d.lesion_id, "y": d.y_true, "score": d.score, "thr": d.youden_threshold}))
    for t in ["vit32_frozen_ckpt", "vit32_ft_ckpt", "vit32_ft_imagenet", "capsnet_r3_revised"]:
        d = pd.read_csv(f"{R}/resultados_v3/split_2026/predicciones_{t}.csv")
        out.append(pd.DataFrame({"model": t, "seed": d.seed_value, "partition": d.partition, "image_id": d.image_id,
                                 "lesion_id": d.lesion_id, "y": d.y_true, "score": d.score, "thr": d.youden_threshold}))
    return pd.concat(out, ignore_index=True)

def load_cohort():
    d = pd.read_csv(f"{R}/confirmacion/predicciones_confirmacion.csv")
    return d.rename(columns={"seed_value": "seed", "youden_threshold": "thr", "y_true": "y"})

def mats(d, ids):
    seeds = sorted(d.seed.unique())
    S = np.stack([d[d.seed == s].set_index("image_id").score.loc[ids].values for s in seeds]).astype(np.float32)
    T = np.array([np.float32(d[d.seed == s].thr.iloc[0]) for s in seeds])
    return seeds, S, T

def counts(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return tp, fn, tn, fp

def rates(y, P):
    """P: seeds x images binary predictions -> seed-averaged sens, spec, prec, npv."""
    tp = ((P == 1) & (y == 1)).sum(1); fn = ((P == 0) & (y == 1)).sum(1)
    tn = ((P == 0) & (y == 0)).sum(1); fp = ((P == 1) & (y == 0)).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.stack([tp / (tp + fn), tn / (tn + fp), tp / (tp + fp), tn / (tn + fn)])
    return np.nanmean(r, axis=1) * 100

def boot_rates(models, y, les, seed=2026):
    ul = np.unique(les); idx = {l: np.where(les == l)[0] for l in ul}
    rng = np.random.default_rng(seed); bs = {k: [] for k in models}
    for _ in range(B):
        ii = np.concatenate([idx[l] for l in rng.choice(ul, len(ul), replace=True)])
        for k, P in models.items():
            bs[k].append(rates(y[ii], P[:, ii]))
    out = {}
    for k, P in models.items():
        est = rates(y, P); lo, hi = np.percentile(np.array(bs[k]), [2.5, 97.5], axis=0)
        out[k] = {m: (est[i], lo[i], hi[i]) for i, m in enumerate(["sens", "spec", "prec", "npv"])}
    return out

def fixed_sens_thr(yv, sv, target):
    """Highest threshold whose validation sensitivity >= target."""
    fpr, tpr, thr = roc_curve(yv, sv)
    ok = np.where(tpr >= target)[0]
    return np.float32(thr[ok[0]])

res = {}
d = load_2026()
test = d[d.partition == "test"]; val = d[d.partition == "val"]
ids = sorted(set(test.image_id)); base = test.drop_duplicates("image_id").set_index("image_id").loc[ids]
y, les = base.y.values, base.lesion_id.values
P = {}; Sm = {}
for m in ORDER:
    seeds, S, T = mats(test[test.model == m], ids); Sm[m] = (seeds, S)
    P[m] = (S >= T[:, None]).astype(int)
res["test_ci"] = boot_rates(P, y, les)

c = load_cohort()
cids = sorted(set(c.image_id)); cb = c.drop_duplicates("image_id").set_index("image_id").loc[cids]
cy, cles = cb.y.values, cb.lesion_id.values
CP = {}; CS = {}
cohort_2x2 = []
for m in ORDER:
    seeds, S, T = mats(c[c.model == m], cids); CS[m] = (seeds, S)
    CP[m] = (S >= T[:, None]).astype(int)
    for k, s in enumerate(seeds):
        tp, fn, tn, fp = counts(cy, CP[m][k])
        cohort_2x2.append({"model": m, "seed": s, "thr": float(T[k]), "auc": roc_auc_score(cy, S[k]), "TP": tp, "FN": fn, "TN": tn, "FP": fp,
                           "sens": 100 * tp / (tp + fn), "spec": 100 * tn / (tn + fp), "prec": 100 * tp / (tp + fp), "npv": 100 * tn / (tn + fn)})
res["cohort_ci"] = boot_rates(CP, cy, cles)
pd.DataFrame(cohort_2x2).to_csv(f"{OUT}/cohort_2x2_per_seed.csv", index=False)

# fixed-sensitivity operating points: threshold set on validation, applied to test and cohort
fs = []
vids = sorted(set(val.image_id)); vb = val.drop_duplicates("image_id").set_index("image_id").loc[vids]
for m in ORDER:
    if m == "capsnet_r3_v2": continue
    seeds, SV, _ = mats(val[val.model == m], vids)
    for tgt in [0.90, 0.95]:
        T = np.array([fixed_sens_thr(vb.y.values, SV[k], tgt) for k in range(len(seeds))])
        Pt = (Sm[m][1] >= T[:, None]).astype(int); Pc = (CS[m][1] >= T[:, None]).astype(int)
        rt = rates(y, Pt); rc = rates(cy, Pc)
        fs.append({"model": m, "target": tgt, "test_sens": rt[0], "test_spec": rt[1], "cohort_sens": rc[0], "cohort_spec": rc[1], "cohort_npv": rc[3]})
fs = pd.DataFrame(fs); fs.to_csv(f"{OUT}/fixed_sensitivity.csv", index=False)

# error overlap on the original test partition: image misclassified by majority of seeds
maj = {m: (P[m].mean(0) >= 0.5).astype(int) for m in ORDER}
wrong = {m: maj[m] != y for m in ORDER}
pre = ["efficientnetb1_v2", "vit32_as_run_v2", "vit32_ft_ckpt"]
allw = np.logical_and.reduce([wrong[m] for m in pre]); anyw = np.logical_or.reduce([wrong[m] for m in pre])
ov = {"n_wrong": {m: int(wrong[m].sum()) for m in ORDER}, "all3_pretrained_wrong": int(allw.sum()), "any3_pretrained_wrong": int(anyw.sum()),
      "all3_FN": int((allw & (y == 1)).sum()), "all3_FP": int((allw & (y == 0)).sum()),
      "all4_incl_caps": int((allw & wrong["capsnet_r3_revised"]).sum())}
dx = test.drop_duplicates("image_id").set_index("image_id")
# dx not in v2 files; take from v3
dxmap = pd.read_csv(f"{R}/resultados_v3/split_2026/predicciones_vit32_ft_ckpt.csv").drop_duplicates("image_id").set_index("image_id").dx
ov["all3_by_dx"] = pd.Series([dxmap[i] for i, w in zip(ids, allw) if w]).value_counts().to_dict()
# candidates for qualitative figure: images wrong in all 3 pretrained pipelines, ranked by mean score distance from threshold
meansc = np.mean([Sm[m][1].mean(0) for m in pre], axis=0)
cand = pd.DataFrame({"image_id": ids, "lesion_id": les, "dx": [dxmap[i] for i in ids], "y": y, "mean_score": meansc, "all3_wrong": allw})
cand = cand[cand.all3_wrong]
fnc = cand[cand.y == 1].sort_values("mean_score").head(6); fpc = cand[cand.y == 0].sort_values("mean_score", ascending=False).head(6)
pd.concat([fnc.assign(kind="false negative"), fpc.assign(kind="false positive")]).to_csv(f"{OUT}/error_examples.csv", index=False)
res["overlap"] = ov

# repeated splits: threshold metrics at validation Youden thresholds
rows = []
for p in glob.glob(f"{R}/resultados_v3/split_30*/predicciones_*.csv"):
    t = pd.read_csv(p); t = t[t.partition == "test"]
    for (tag, s), g in t.groupby(["tag", "seed_value"]):
        pr = (g.score.astype(np.float32) >= np.float32(g.youden_threshold.iloc[0])).astype(int)
        tp, fn, tn, fp = counts(g.y_true.values, pr.values)
        rows.append({"model": tag, "split": g.split_seed.iloc[0], "seed": s, "TP": tp, "FN": fn, "TN": tn, "FP": fp})
rs = pd.DataFrame(rows); rs.to_csv(f"{OUT}/repeated_2x2_per_run.csv", index=False)
rs["sens"] = 100 * rs.TP / (rs.TP + rs.FN); rs["spec"] = 100 * rs.TN / (rs.TN + rs.FP)
res["repeated"] = rs.groupby("model")[["sens", "spec"]].agg(["mean", "std", "min", "max", "count"]).round(1).to_dict()
res["repeated_counts"] = rs.groupby("model")[["TP", "FN", "TN", "FP"]].sum().to_dict()

def fmt(o):
    if isinstance(o, dict): return {str(k): fmt(v) for k, v in o.items()}
    if isinstance(o, (tuple, list)): return [fmt(v) for v in o]
    if isinstance(o, (np.floating, float)): return round(float(o), 2)
    if isinstance(o, np.integer): return int(o)
    return o
json.dump(fmt(res), open(f"{OUT}/stats_rev3.json", "w"), indent=1)
print(json.dumps(fmt(res), indent=1)); print(fs.round(1).to_string())
