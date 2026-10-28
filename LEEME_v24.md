# Release v2.4 — exploratory analyses added after internal review (October 2026)

No model was retrained; the script reads the predictions archived in v1.0 and v2.2.

| File | Content |
|---|---|
| `analisis_asesor2.py` | Calibration (Brier, intercept, slope, ECE; raw and prior-shift corrected), decision curves on the held-out cohort, per-diagnosis AUC on the cohort and case-mix-adjusted AUC, all with lesion-clustered bootstrap CIs (2,000 replicates). Run in the folder that contains `resultados_split_final_v2/`, `resultados_v3/`, `confirmacion/` and `analisis_revision.py`. |
| `calibration.csv` | Table S8 |
| `decision_curves.csv`, `decision_curves.png` | Figure 5 |
| `cohort_auc_by_dx_casemix.csv` | Table 8 |
| `resumen.json` | Cohort counts and prevalences used for the prior-shift correction |
