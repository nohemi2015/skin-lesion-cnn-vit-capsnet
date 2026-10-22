# Revision protocol (v3) — pre-specified before any new experiment

**Manuscript:** Benchmarking CNN, Vision Transformer, and Capsule Network Pipelines for Binary Skin-Lesion Classification Under Lesion-Grouped Holdout Evaluation
**Date fixed:** 20 October 2026, before running any experiment described here.
**Purpose:** respond to the major-revision report (capsule failure, ViT preprocessing, single split, evaluation unit) with a design fixed in advance, so that no decision is driven by the existing test results.

## 0. Frozen original record

- The original results (v2) are kept unchanged and archived: Zenodo release v1.0, DOI 10.5281/zenodo.23512171 (code, split ledger, per-seed results, per-image predictions).
- The original code (`modelos_reducidos.py`, `entrenar_split_final_v2.py`) is not modified. New code is in separate files (`modelos_v3.py`, `entrenar_v3.py`, ...).
- No configuration is selected using test-partition results of any split. Selection uses training and validation data only.
- Analyses added during revision are labelled as such in the manuscript.

## 1. Implementation finding recorded before the new experiments

While preparing this protocol, a test of the original ViT builder showed that the Hugging Face ViT backbone (a `tf_keras` object) wrapped inside a Keras 3 layer is not tracked by Keras 3: the optimizer only updated the dense head (4 trainable tensors), and the 40 backbone tensors did not change after training, although `train_base=True`. The original "ViT-Base/32, fully fine-tuned" pipeline was therefore a **frozen ViT-Base/32 backbone with a trained head**. This is confirmed on the original machine with `verificar_vit_original.py` and by the reload check of `cohorte_confirmacion.py` (the original ViT models, rebuilt with unmodified pretrained weights, must reproduce their saved predictions). The manuscript will describe the original ViT pipeline accordingly.

`modelos_v3.py` builds the ViT entirely in `tf_keras`, so that `train_base=True` fine-tunes the backbone (verified: backbone tensors change; frozen variant: they do not).

## 2. Pipelines

| ID | Name (`--pipeline`) | Description |
|---|---|---|
| P1 | `efficientnetb1_ft` | EfficientNet-B1, ImageNet-1k, fully fine-tuned (unchanged; v2 runs = original split) |
| P2 | `vit32_frozen_imagenet` | ViT-B/32 frozen backbone + head, ImageNet statistics (= the original ViT pipeline as actually run; v2 runs = original split) |
| P3 | `vit32_frozen_ckpt` | as P2 but with the checkpoint's reference normalization (mean = std = 0.5) — isolates the normalization effect with all else fixed |
| P4 | `vit32_ft_ckpt` | ViT-B/32 fully fine-tuned, reference normalization — the intended fine-tuned ViT pipeline |
| P5 | `vit32_ft_imagenet` | ViT-B/32 fully fine-tuned, ImageNet statistics — isolates normalization under fine-tuning |
| P6 | `capsnet_r3` | CapsNet-R3 original configuration (v2 runs = original split) |
| P7 | `capsnet_r3_revised` | CapsNet-R3 with the configuration selected in gate G6, if any |

All pipelines share: 224×224 inputs in [0, 1], the same augmentation, AdamW (weight decay 0.004), batch 16, early stopping on validation loss (patience 10, max 200 epochs, best weights restored), Youden threshold from validation. Learning rates: 1e-4 (P1–P5), 1e-3 (P6), selected value (P7). No result of P3/P4/P5 is assumed to be better than P2.

## 3. CapsNet-R3 diagnosis (gates, in order; training/validation data of split 2026 only)

- **G1 Data and evaluation:** labels vs. diagnoses, no mixed-label lesions, no duplicate images, files present, one-hot order [benign, malignant], input range [0, 1], batch-order and batch-size invariance of predictions, synthetic metric controls (AUC 1 / 0 / 0.5, Youden threshold). Any failure stops further CapsNet work until fixed.
- **G2 Mathematics:** squash, dynamic routing (3 iterations) and margin loss compared with an independent NumPy reference (max abs difference < 1e-5); coupling coefficients sum to 1 over output capsules.
- **G3 Gradient flow:** finite, non-zero gradients for every trainable tensor; every tensor updated after one optimizer step. Gradients propagate through all routing iterations (no stop-gradient).
- **G4 Memorization:** 32 training images (16 malignant, 16 benign), no augmentation, no weight decay, 100 epochs, learning rates 1e-3, 1e-4, 3e-5, 1e-5, plus a synthetic trivially separable control. A learning rate passes when, at the last epoch, the capsule decision rule (longer capsule) classifies ≥ 95% of the 32 images and the margin loss is ≤ 0.10. (Engineering tolerances, not literature benchmarks.)
- **G5 Saturation localization (descriptive):** step-by-step trace (first 60 steps) of primary-capsule lengths and fraction > 0.9, prediction-vector and pre-squash norms, coupling coefficients, lengths of both output capsules and per-layer gradient norms, for lr 1e-3 with 3 and 1 routing iterations, and lr 1e-5 with 3 iterations.
- **G6 Validation-only selection:** grid lr ∈ {1e-4, 3e-5, 1e-5} × input normalization ∈ {none, ImageNet statistics}, one seed (2027), full training protocol (augmentation and weight decay restored), split 2026, validation only (`--no_test`). Candidates are configurations whose lr passed G4. Select the highest validation AUC (ties < 0.005 → lower validation loss). The selection is accepted only if validation AUC ≥ 0.60; otherwise P7 is not trained and the capsule results are reported as a characterized failure.

Pilot evidence obtained before this protocol, on synthetic images only (not HAM10000): with the original lr 1e-3, primary-capsule lengths rose from 0.02 to > 0.9 within two optimizer steps, both output capsules saturated near 1, and the model failed to fit a trivially separable synthetic set; with lr 1e-4 it fitted it. This motivates the grid but is not a result on the study data.

**Amendment window:** after G1–G5 and before G6 results are seen, the G6 grid may be amended only if G1–G3 reveal an implementation defect; any amendment is recorded here with its date and reason.

## 4. Original split (2026): new runs

P3, P4, P5 and (if selected) P7, five seeds (2027–2031), same split as v2 (the script verifies that the split reproduces the v2 ledger exactly). Test partition evaluated once per seed.

## 5. Repeated lesion-grouped splits

Five new outer splits with seeds 3001–3005, produced by the same two-step `GroupShuffleSplit` procedure (test 15%, then validation 15% of the total), shared by all pipelines (matched partitions). Pipelines: P1, P2, P4, P6, and P7 if selected. Training seeds 2027, 2028, 2029 per split. Model and threshold selection inside each split's training/validation partitions only. Reported: test AUC per split (mean over seeds), mean and SD across splits, mean within-split SD across seeds, and paired differences with P1 per split (mean, min, max).
If the total training time proves impractical, the number of training seeds per split may be reduced to one (seed 2027) for all pipelines; this decision may depend only on elapsed time, never on results, and must be recorded here before any repeated-split result is inspected.

## 6. Untouched confirmation cohort

All HAM10000 images whose `lesion_id` is not among the 1,730 lesions of the 2,000-image subset (no image of these lesions was used for training, selection, thresholds or earlier evaluation). The models trained on split 2026 (v2: P1, P2, P6; v3: P3, P4, P5, P7) are applied unchanged with their validation-derived thresholds. Before evaluation, each reloaded model must reproduce its own saved test predictions (max abs difference ≤ 1e-3). Reported: seed-averaged image-level AUC with lesion-clustered bootstrap 95% CI (2,000 replicates), paired differences with P1, and sensitivity/specificity at the validation thresholds (noting the different prevalence).

## 7. Evaluation unit

The primary estimand remains image-level AUC. Lesion-level sensitivity analysis: lesion score = mean of its image scores (all images of a lesion share the label), AUC per seed, averaged across seeds, lesion bootstrap CI. HAM10000 has no patient identifiers; partitions are lesion-disjoint, not necessarily patient-disjoint.

## 8. Reporting

All pipelines and analyses above are reported regardless of their results, including null results and failures. The comparison is presented as a comparison of training pipelines on HAM10000 internal data; no clinical claim is made.
