# Release v2.3 — analyses added after the reporting audit (October 2026)

No model was retrained. Everything here reads the predictions archived in v1.0 and v2.2.

## Scripts
| File | What it does |
|---|---|
| `analisis_auditoria.py` | Lesion-clustered bootstrap CIs for sensitivity, specificity, precision and NPV (test partition and untouched cohort); seed-level confusion matrices on the cohort; operating points for 90% / 95% validation sensitivity; error overlap across pipelines; pooled counts on the repeated splits. Run in the folder that contains `resultados_split_final_v2/`, `resultados_v3/` and `confirmacion/`. Output: `analisis_auditoria/`. |
| `duplicados.py` | Search for near-duplicate HAM10000 images with different `lesion_id` (EfficientNet-B1 ImageNet embeddings, cosine similarity, 1,000 most similar pairs, contact sheet of the top 60). |
| `copiar_ejemplos.py` | Copies the 12 example images listed in `ejemplos_errores.csv`. |
| `PASO5_duplicados.bat` | Runs the two scripts above (Windows). |
| `figuras_resultados.py` | Figures 3 and 4 of the manuscript (needs `analisis_revision.py` from v2.2). |
| `figura_flujo.py` | Figure 1 (flow of images and lesions). |

## Outputs
| File | Content |
|---|---|
| `stats_rev3.json` | All CIs, error overlap and repeated-split summaries |
| `cohort_2x2_per_seed.csv` | Table B.2 |
| `repeated_2x2_per_run.csv` | Counts behind Table B.1 |
| `fixed_sensitivity.csv` | Table 9 |
| `ejemplos_errores.csv` | Shared errors used as examples |
| `duplicados_pares_top1000.csv`, `duplicados_resumen.txt` | Duplicate search; the 60 most similar pairs were reviewed visually: 7 same-lesion pairs, none crossing a train/test boundary |
