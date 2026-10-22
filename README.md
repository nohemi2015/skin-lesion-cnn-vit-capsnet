# Benchmarking CNN, ViT and CapsNet pipelines for binary skin-lesion classification

Code, split and prediction-level outputs for the article
*"Benchmarking CNN, Vision Transformer, and Capsule Network Pipelines for Binary
Skin-Lesion Classification Under Lesion-Grouped Holdout Evaluation"*
(N. Sánchez-Medel, S. Zapotecas-Martínez, R. Díaz-Hernández, L. Altamirano-Robles; INAOE).

Three pipelines are compared for malignant (MEL, BCC, AKIEC) vs. benign (NV, BKL, DF, VASC)
classification on a 2,000-image subset of HAM10000:

| Pipeline | Initialization | Test AUC (mean of 5 seeds) [95% CI] |
|---|---|---|
| EfficientNet-B1 | ImageNet, full fine-tuning | 0.859 [0.819–0.899] |
| ViT-Base/32 | ImageNet-21k, full fine-tuning | 0.844 [0.796–0.888] |
| CapsNet-R3 (3 routing iterations) | random | 0.486 [0.462–0.512] |

## Contents

```
curar_dataset_grande.py          builds the 1,000 + 1,000 image subset from HAM10000 (seed 2026)
modelos_reducidos.py             model builders (EfficientNet-B1, ViT-Base/32, CapsNet-R3)
entrenar_split_final_v2.py       lesion-grouped 70/15/15 split (seed 2026), training with
                                 5 seeds (2027–2031), Youden threshold on validation, test metrics
analisis_bootstrap.py            seed-averaged AUC with lesion-clustered bootstrap CIs
figura_roc.py                    ROC figure (Figure 1)
analisis_por_diagnostico.py      per-diagnosis results (Table 4) and lesion-level composition (Table 1)
split_ledger.csv                 every image used: image_id, lesion_id, diagnosis, label, partition
resultados_*.csv                 per-seed metrics (Youden threshold and 0.5)
resumen_*.csv                    mean and SD across seeds
predicciones_*.csv               per-image scores on validation and test for every seed
roc_stage2.png                   Figure 1
```

## Data

The HAM10000 images are **not** redistributed here. Download them from the Harvard Dataverse
(https://doi.org/10.7910/DVN/DBW86T) under its terms of use. `split_ledger.csv` lists the
exact images, lesions and partitions used.

## Reproducing the results

```bash
pip install -r requirements.txt

# 1. Build the subset (needs the HAM10000 images and metadata)
python curar_dataset_grande.py --ham10000_images_dir <HAM10000_images> \
    --metadata_oficial HAM10000_metadata.csv --n_per_class 1000 \
    --out_dir HAM_subset_grande --manifest_out subset_grande_con_lesion_id.csv

# 2. Train and evaluate (5 seeds each)
python entrenar_split_final_v2.py --manifest_csv subset_grande_con_lesion_id.csv --arch efficientnetb1 --unfreeze_backbone
python entrenar_split_final_v2.py --manifest_csv subset_grande_con_lesion_id.csv --arch vit_base32 --unfreeze_backbone
python entrenar_split_final_v2.py --manifest_csv subset_grande_con_lesion_id.csv --arch capsnet

# 3. Statistics and figure from the saved predictions (no GPU needed)
python analisis_bootstrap.py
python figura_roc.py
python analisis_por_diagnostico.py
```

Step 3 runs directly on the CSV files in this repository and reproduces the AUCs and confidence intervals
reported in the article. Experiments were run on an NVIDIA GeForce GTX 1660 Ti (6 GB) with
Python 3.11.9, TensorFlow 2.21.0, Keras 3.15.0, transformers 4.44.2 and scikit-learn 1.9.0;
retraining on other hardware may give slightly different numbers.

## License

Code: MIT (see `LICENSE`). HAM10000 data: subject to its original license.

---

## Revision v2.0 (October 2026)

The folder `revision_v2/` contains the additional experiments of the revised manuscript. They followed the protocol in `revision_v2/PROTOCOLO_REVISION.md`, which was fixed before the experiments were run. The files of release v1.0 (above) are unchanged.

**Implementation finding.** In the original ViT pipeline (`modelos_reducidos.py`), the Hugging Face ViT (a tf-keras model) is wrapped in a Keras 3 layer. Keras 3 does not track its variables, so only the dense head (98,561 parameters) was trained and the backbone stayed frozen. `revision_v2/code/verificar_vit_original.py` reproduces this check. The original ViT results therefore correspond to a frozen-backbone pipeline. `revision_v2/code/modelos_v3.py` builds the ViT entirely in tf-keras, so that fine-tuning works.

```
revision_v2/
  PROTOCOLO_REVISION.md           pre-specified revision protocol
  code/                           modelos_v3.py, entrenar_v3.py, capsnet_diagnostico.py,
                                  seleccionar_capsnet.py, cohorte_confirmacion.py,
                                  analisis_revision.py, verificar_vit_original.py,
                                  PASO1-4 .bat files and LEEME.md (instructions, Spanish)
  diagnostico_capsnet/            CapsNet-R3 diagnosis (gates G1-G6)
  resultados_v3.zip               per-seed results, per-image predictions, per-epoch histories
                                  and split ledgers: original split 2026 (new pipelines and
                                  CapsNet selection grid) and repeated splits 3001-3005
  predicciones_confirmacion.zip   per-image scores of the original-split models on the
                                  untouched cohort (7,124 HAM10000 images, 5,740 lesions)
  confirmacion_manifest.csv       images, lesions and diagnoses of the untouched cohort
  analisis_revision/              outputs of analisis_revision.py (AUCs, bootstrap CIs,
                                  repeated-split summaries, confirmation cohort)
  figures/                        figures of the revised manuscript
```

To re-run the statistics without training: copy `revision_v2/code/*.py` next to the v1.0 files, unzip `resultados_v3.zip`, put `predicciones_confirmacion.csv` in a folder `confirmacion/`, create `resultados_split_final_v2/` with the v1.0 `resultados_*.csv`, `predicciones_*.csv` and `split_ledger.csv`, and run `python analisis_revision.py`.

| Pipeline | Original split AUC | Repeated splits, mean AUC (5 splits) | Untouched cohort AUC |
|---|---|---|---|
| EfficientNet-B1 | 0.859 | 0.873 | 0.903 |
| ViT-B/32, frozen (original pipeline) | 0.844 | 0.875 | 0.903 |
| ViT-B/32, fine-tuned (reference normalization) | 0.846 | 0.885 | 0.908 |
| CapsNet-R3, original (lr 1e-3) | 0.486 | 0.532 | 0.473 |
| CapsNet-R3, revised (lr 1e-4, normalized inputs) | 0.836 | 0.838 | 0.871 |
