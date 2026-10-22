"""
cohorte_confirmacion.py

Confirmation on an untouched cohort (revision protocol, step 5): all HAM10000
images whose lesion is NOT among the 1,730 lesions of the 2,000-image subset.
None of these lesions was used for training, model selection, threshold
selection or any earlier evaluation.

The models trained on the original split (split seed 2026) are applied without
any change; each seed keeps the Youden threshold selected on its validation
partition. Nothing is re-trained or re-tuned.

  - Original runs (v2): EfficientNet-B1, ViT-Base/32 as run (frozen backbone),
    CapsNet-R3; rebuilt with modelos_reducidos.py and loaded from
    modelos_split_final_v2/*.keras
  - Revision runs (v3) on split 2026: every pipeline found in
    resultados_v3/split_2026/ (grid runs excluded); loaded from
    modelos_v3/split_2026/*.weights.h5

Usage (Windows):
    python cohorte_confirmacion.py --metadata HAM10000_metadata.csv ^
        --ham_dirs "C:\\ruta\\HAM10000_images_part_1,C:\\ruta\\HAM10000_images_part_2" ^
        --manifest_csv subset_grande_con_lesion_id.csv
Outputs: confirmacion/manifest_confirmacion.csv, confirmacion/predicciones_confirmacion.csv
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import modelos_v3 as mv
from entrenar_v3 import make_dataset

MALIGNANT_DX = {"mel", "bcc", "akiec"}
OUT = "confirmacion"


def index_images(dirs):
    """image_id -> path, searching every sub-folder of the given folders."""
    idx = {}
    for d in dirs:
        if not os.path.isdir(d):
            raise SystemExit(f"Folder not found: {d}")
        for root, _, files in os.walk(d):
            for f in files:
                stem, ext = os.path.splitext(f)
                if ext.lower() in (".jpg", ".jpeg", ".png") and stem not in idx:
                    idx[stem] = os.path.join(root, f)
    return idx


def build_cohort(args):
    meta = pd.read_csv(args.metadata)
    subset = pd.read_csv(args.manifest_csv)
    used = set(subset.lesion_id)
    coh = meta[~meta.lesion_id.isin(used)].copy()
    dirs = [d.strip().strip('"') for d in args.ham_dirs.split(",")]
    idx = index_images(dirs)
    print(f"Images found under --ham_dirs (all sub-folders): {len(idx)}")
    coh["filepath"] = coh.image_id.map(idx)
    missing = coh.filepath.isna().sum()
    if missing:
        ex = ", ".join(coh[coh.filepath.isna()].image_id.head(5))
        raise SystemExit(f"{missing} of {len(coh)} cohort images were not found under --ham_dirs "
                         f"(e.g. {ex}). HAM10000 has 10,015 images; check that all of them are there.")
    coh["label"] = coh.dx.isin(MALIGNANT_DX).astype(int)
    coh = coh.rename(columns={"dx": "dx_official"})[["filepath", "image_id", "lesion_id", "dx_official", "label"]]
    assert not set(coh.lesion_id) & used
    coh.to_csv(os.path.join(OUT, "manifest_confirmacion.csv"), index=False)
    print(f"Confirmation cohort: {len(coh)} images, {coh.lesion_id.nunique()} lesions, "
          f"{coh.label.sum()} malignant images ({100 * coh.label.mean():.1f}%), "
          f"{coh[coh.label == 1].lesion_id.nunique()} malignant lesions")
    return coh


def model_list(args):
    """(name, seed, loader, threshold, is_capsnet) for every model to evaluate."""
    items = []
    v2_res, v2_mod = args.v2_results, args.v2_models
    if os.path.isdir(v2_mod):
        import modelos_reducidos as mr
        v2 = [("efficientnetb1_v2", "efficientnetb1_finetuned", lambda: mr.build_efficientnetb1(train_base=True), False),
              ("vit32_as_run_v2", "vit_base32_finetuned", lambda: mr.build_vit_base32(train_base=True), False),
              ("capsnet_r3_v2", "capsnet", lambda: mr.build_capsnet(), True)]
        for name, tag, builder, caps in v2:
            res = pd.read_csv(os.path.join(v2_res, f"resultados_{tag}.csv"))
            for _, r in res.iterrows():
                path = os.path.join(v2_mod, f"{tag}_seed{int(r.seed_run)}.keras")
                ref = os.path.join(v2_res, f"predicciones_{tag}.csv")
                items.append((name, int(r.seed_value), builder, path, float(r.youden_threshold), caps,
                              (ref, "seed_run", int(r.seed_run))))
    v3dir = os.path.join(args.v3_results, "split_2026")
    for path in sorted(glob.glob(os.path.join(v3dir, "resultados_*.csv"))):
        res = pd.read_csv(path)
        if "test_auc" not in res.columns:          # validation-only grid runs are not evaluated
            continue
        for _, r in res.iterrows():
            pipe, tag = r.pipeline, r.tag
            lr = None if pd.isna(r.get("capsnet_lr")) else float(r.capsnet_lr)
            norm = None if pd.isna(r.get("capsnet_input_norm")) else r.capsnet_input_norm
            builder = (lambda p=pipe, l=lr, n=norm: mv.build_pipeline(p, lr=l, input_norm=n)[0])
            w = os.path.join(args.v3_models, "split_2026", f"{tag}_seed{int(r.seed_value)}.weights.h5")
            ref = os.path.join(v3dir, f"predicciones_{tag}.csv")
            items.append((tag, int(r.seed_value), builder, w, float(r.youden_threshold),
                          pipe.startswith("capsnet"), (ref, "seed_value", int(r.seed_value))))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--ham_dirs", required=True)
    ap.add_argument("--manifest_csv", required=True, help="manifest of the 2,000-image subset")
    ap.add_argument("--v2_results", default="resultados_split_final_v2")
    ap.add_argument("--v2_models", default="modelos_split_final_v2")
    ap.add_argument("--v3_results", default="resultados_v3")
    ap.add_argument("--v3_models", default="modelos_v3")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--check_n", type=int, default=32, help="test images used for the reload check")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    coh = build_cohort(args)
    preds_path = os.path.join(OUT, "predicciones_confirmacion.csv")
    done = set()
    if os.path.exists(preds_path):
        d = pd.read_csv(preds_path, usecols=["model", "seed_value"])
        done = set(zip(d.model, d.seed_value))

    for name, seed, builder, wpath, thr, caps, (ref_path, ref_col, ref_val) in model_list(args):
        if (name, seed) in done:
            print(f"{name} seed {seed}: already evaluated")
            continue
        if not os.path.exists(wpath):
            raise SystemExit(f"Missing weights: {wpath}")
        model = builder()
        model.load_weights(wpath)
        # Reload check: the model must reproduce its own saved test predictions.
        ref = pd.read_csv(ref_path)
        ref = ref[(ref[ref_col] == ref_val) & (ref.partition == "test")].head(args.check_n)
        man = pd.read_csv(args.manifest_csv)
        ref = ref.merge(man[["image_id", "filepath"]], on="image_id").assign(label=0)
        pr = model.predict(make_dataset(ref, caps, False, args.batch_size, 0), verbose=0)
        pr = pr[:, 1] if caps else pr.ravel()
        err = float(np.max(np.abs(pr - ref.score.values)))
        if err > args.tol:
            raise SystemExit(f"{name} seed {seed}: reloaded model does not reproduce its saved test "
                             f"predictions (max abs difference {err:.2e}). Check the weights files.")
        ds = make_dataset(coh, caps, False, args.batch_size, 0)
        p = model.predict(ds, verbose=0)
        score = p[:, 1] if caps else p.ravel()
        out = coh[["image_id", "lesion_id", "dx_official", "label"]].rename(columns={"label": "y_true"}).assign(
            model=name, seed_value=seed, score=score, youden_threshold=thr)
        out.to_csv(preds_path, mode="a", header=not os.path.exists(preds_path), index=False)
        from sklearn.metrics import roc_auc_score
        print(f"{name} seed {seed}: reload check max diff {err:.1e} | "
              f"confirmation AUC = {roc_auc_score(out.y_true, out.score):.3f}")
        del model
    print(f"\nPredictions: {preds_path}")


if __name__ == "__main__":
    main()
