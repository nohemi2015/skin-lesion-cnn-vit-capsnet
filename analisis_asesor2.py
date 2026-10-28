"""Additional exploratory analyses requested by the second advisor (no training).
Run in the working folder (resultados_split_final_v2/, resultados_v3/, confirmacion/).
  1. Calibration of the sigmoid-output pipelines (test partition and held-out cohort):
     Brier score, calibration intercept (slope fixed at 1) and slope, ECE (10 bins);
     on the cohort also after a prior-shift correction from the training prevalence.
  2. Decision-curve analysis on the held-out cohort (prior-shift-corrected probabilities).
  3. Per-diagnosis AUC on the held-out cohort with lesion-clustered bootstrap CIs.
  4. Case-mix-adjusted AUC on the cohort (diagnoses reweighted to the subset mix).
Outputs: analisis_asesor2/*.csv, analisis_asesor2/resumen.json, figuras/decision_curves.png
"""
import os, json, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import brentq
from scipy.special import expit, logit
sys.path.insert(0, ".")
OUT = "analisis_asesor2"; os.makedirs(OUT, exist_ok=True); os.makedirs("figuras", exist_ok=True)
B = 2000
SIGMOID = ["efficientnetb1_v2", "vit32_as_run_v2", "vit32_frozen_ckpt", "vit32_ft_ckpt", "vit32_ft_imagenet"]
MAIN = ["efficientnetb1_v2", "vit32_as_run_v2", "vit32_ft_ckpt", "capsnet_r3_revised"]
PI_TRAIN = 725 / 1404          # malignant share of the training partition (original split)
SUBSET_MIX = {1: {"akiec": 177, "bcc": 248, "mel": 575}, 0: {"bkl": 138, "df": 16, "nv": 827, "vasc": 19}}

def load_test():
    import analisis_revision as ar
    d = pd.concat([ar.load_v2(), ar.load_v3()], ignore_index=True)
    return d[(d.split_seed == 2026) & (d.partition == "test")].rename(columns={"y_true": "y"})

def load_cohort():
    d = pd.read_csv(os.path.join("confirmacion", "predicciones_confirmacion.csv"))
    return d.rename(columns={"seed_value": "seed", "y_true": "y", "dx_official": "dx"})

def mat(d, m, ids):
    g = d[d.model == m]; seeds = sorted(g.seed.unique())
    return np.stack([g[g.seed == s].set_index("image_id").score.loc[ids].values for s in seeds]).astype(np.float64)

def calib(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); lp = logit(p)
    brier = np.mean((p - y) ** 2)
    a = brentq(lambda a: expit(a + lp).mean() - y.mean(), -20, 20)        # calibration-in-the-large
    # slope: logistic regression of y on logit(p) (Newton)
    X = np.c_[np.ones_like(lp), lp]; w = np.zeros(2)
    for _ in range(50):
        mu = expit(X @ w); W = mu * (1 - mu)
        w += np.linalg.solve(X.T @ (X * W[:, None]) + 1e-9 * np.eye(2), X.T @ (y - mu))
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    return brier, a, w[1], ece

def prior_shift(p, pi_new, pi_old=PI_TRAIN):
    o = p / (1 - p) * (pi_new / (1 - pi_new)) / (pi_old / (1 - pi_old))
    return o / (1 + o)

res = {}
t = load_test(); tid = sorted(set(t.image_id)); tb = t.drop_duplicates("image_id").set_index("image_id").loc[tid]; ty = tb.y.values
c = load_cohort(); cid = sorted(set(c.image_id)); cb = c.drop_duplicates("image_id").set_index("image_id").loc[cid]
cy, cles, cdx = cb.y.values, cb.lesion_id.values, cb.dx.values
pi_c = cy.mean()

# 1. calibration
rows = []
for m in SIGMOID:
    for setname, S, y, adj in [("test", mat(t, m, tid), ty, False), ("cohort", mat(c, m, cid), cy, False), ("cohort, prior-corrected", mat(c, m, cid), cy, True)]:
        vals = np.array([calib(y, prior_shift(S[k], pi_c) if adj else S[k]) for k in range(S.shape[0])])
        rows.append({"model": m, "set": setname, "brier": vals[:, 0].mean(), "intercept": vals[:, 1].mean(), "slope": vals[:, 2].mean(), "ece": vals[:, 3].mean()})
cal = pd.DataFrame(rows); cal.to_csv(f"{OUT}/calibration.csv", index=False)

# 2. decision curves on the cohort
pts = np.round(np.arange(0.01, 0.301, 0.01), 2); dca = []
for m in SIGMOID:
    S = mat(c, m, cid); P = np.stack([prior_shift(S[k], pi_c) for k in range(S.shape[0])])
    for pt in pts:
        nb = [((P[k] >= pt) & (cy == 1)).mean() - ((P[k] >= pt) & (cy == 0)).mean() * pt / (1 - pt) for k in range(P.shape[0])]
        dca.append({"model": m, "pt": pt, "net_benefit": np.mean(nb)})
for pt in pts:
    dca.append({"model": "treat_all", "pt": pt, "net_benefit": pi_c - (1 - pi_c) * pt / (1 - pt)})
    dca.append({"model": "treat_none", "pt": pt, "net_benefit": 0.0})
dca = pd.DataFrame(dca); dca.to_csv(f"{OUT}/decision_curves.csv", index=False)

# 3 & 4. per-diagnosis and case-mix-adjusted AUC with lesion-clustered bootstrap
ul = np.unique(cles); idx = {l: np.where(cles == l)[0] for l in ul}
w = np.zeros(len(cy))
for cls, mix in SUBSET_MIX.items():
    tot = sum(mix.values())
    for dx, n in mix.items():
        sel = (cdx == dx); w[sel] = (n / tot) / sel.sum()
def stats(S, ii):
    y = cy[ii]; dx = cdx[ii]; out = {}
    ben = (y == 0)
    for mdx in ["mel", "bcc", "akiec"]:
        sel = ben | (dx == mdx)
        out[f"auc_{mdx}_vs_benign"] = np.mean([roc_auc_score(y[sel], S[k, ii][sel]) for k in range(S.shape[0])])
    out["auc_mel_vs_rest"] = np.mean([roc_auc_score((dx == "mel").astype(int), S[k, ii]) for k in range(S.shape[0])])
    out["auc_casemix"] = np.mean([roc_auc_score(y, S[k, ii], sample_weight=w[ii]) for k in range(S.shape[0])])
    out["auc_unweighted"] = np.mean([roc_auc_score(y, S[k, ii]) for k in range(S.shape[0])])
    return out
rng = np.random.default_rng(2026); boots = {m: [] for m in MAIN}; Smain = {m: mat(c, m, cid) for m in MAIN}
for _ in range(B):
    ii = np.concatenate([idx[l] for l in rng.choice(ul, len(ul), replace=True)])
    for m in MAIN:
        boots[m].append(stats(Smain[m], ii))
rows = []
for m in MAIN:
    est = stats(Smain[m], np.arange(len(cy))); bt = pd.DataFrame(boots[m])
    for k, v in est.items():
        lo, hi = np.percentile(bt[k], [2.5, 97.5]); rows.append({"model": m, "metric": k, "estimate": v, "ci_low": lo, "ci_high": hi})
auc = pd.DataFrame(rows); auc.to_csv(f"{OUT}/cohort_auc_by_dx_casemix.csv", index=False)
counts = {dx: int((cdx == dx).sum()) for dx in np.unique(cdx)}

# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)
sty = {"efficientnetb1_v2": ("EfficientNet-B1", "#2a78d6", "-"), "vit32_as_run_v2": ("ViT-B/32, frozen", "#eb6834", "--"),
       "vit32_ft_ckpt": ("ViT-B/32, fine-tuned", "#1baf7a", "-.")}
for m, (lab, col, ls) in sty.items():
    g = dca[dca.model == m]; ax.plot(g.pt, g.net_benefit, color=col, ls=ls, lw=1.8, label=lab)
g = dca[dca.model == "treat_all"]; ax.plot(g.pt, g.net_benefit, color="#6b6b67", lw=1.2, label="Treat all")
ax.axhline(0, color="#3a3a38", lw=1.0, label="Treat none")
ax.set_xlim(0.01, 0.30); ax.set_ylim(-0.01, 0.085); ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig("figuras/decision_curves.png", facecolor="white")

json.dump({"cohort_counts": counts, "pi_cohort": pi_c, "pi_train": PI_TRAIN}, open(f"{OUT}/resumen.json", "w"), indent=1)
pd.set_option("display.width", 200)
print(cal.round(3).to_string()); print(auc.round(3).to_string())
print(dca[dca.pt.isin([0.05, 0.1, 0.2])].pivot(index="pt", columns="model", values="net_benefit").round(4).to_string()); print(counts)
