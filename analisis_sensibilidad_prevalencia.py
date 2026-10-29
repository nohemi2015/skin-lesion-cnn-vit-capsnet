"""Sensitivity analysis of the prior-shift correction: HAM10000 overall prevalence (1,954/10,015)
instead of the held-out cohort's own prevalence. Output: analisis_asesor2/prior_sensitivity.csv"""
import sys, numpy as np, pandas as pd
sys.argv=[sys.argv[0]]
src=open("analisis_asesor2.py").read().split("res = {}")[0]
exec(src)
c = load_cohort(); cid = sorted(set(c.image_id)); cb = c.drop_duplicates("image_id").set_index("image_id").loc[cid]; cy = cb.y.values
rows=[]
for pi_name, pi in [("cohort", cy.mean()), ("HAM10000", 1954/10015)]:
    for m in SIGMOID:
        S = mat(c, m, cid); P = np.stack([prior_shift(S[k], pi) for k in range(S.shape[0])])
        cal = np.array([calib(cy, P[k]) for k in range(P.shape[0])])
        nb = {pt: np.mean([((P[k] >= pt) & (cy == 1)).mean() - ((P[k] >= pt) & (cy == 0)).mean() * pt / (1 - pt) for k in range(P.shape[0])]) for pt in np.round(np.arange(0.01,0.301,0.01),2)}
        ta = {pt: cy.mean() - (1-cy.mean())*pt/(1-pt) for pt in nb}
        rows.append({"prior": pi_name, "pi": pi, "model": m, "brier": cal[:,0].mean(), "intercept": cal[:,1].mean(), "slope": cal[:,2].mean(), "ece": cal[:,3].mean(),
                     "nb_10": nb[0.1], "nb_above_treat_all_and_none_all_pt": all(nb[p] > max(ta[p],0) for p in nb)})
r=pd.DataFrame(rows); r.to_csv("analisis_asesor2/prior_sensitivity.csv", index=False); print(r.round(3).to_string())
