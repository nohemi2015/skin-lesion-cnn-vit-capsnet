"""
entrenar_v3.py

Training and evaluation for the revision experiments (protocol v3).
Same protocol as entrenar_split_final_v2.py (70/15/15 lesion-grouped split,
AdamW, batch 16, early stopping on val_loss with patience 10, max 200 epochs,
Youden threshold from validation), with these additions:

  --split_seed   seed of the lesion-grouped split (2026 = original split;
                 3001..3005 = repeated splits of the protocol)
  --seeds        training seeds, e.g. 2027,2028,2029
  --no_test      selection mode: train and evaluate on VALIDATION only; the
                 test partition is never loaded or predicted
  Per-epoch training history (loss, val_loss, ...) is saved for every run.
  For CapsNet the lengths of BOTH output capsules are saved.

Outputs (one folder per split):  resultados_v3/split_<seed>/
    split_ledger.csv
    resultados_<tag>.csv          one row per training seed
    predicciones_<tag>.csv        per-image scores (val, and test unless --no_test)
    historial_<tag>_seed<s>.csv   per-epoch history
    entorno.txt
Weights: modelos_v3/split_<seed>/<tag>_seed<s>.weights.h5

Examples (Windows):
    python entrenar_v3.py --manifest_csv subset_grande_con_lesion_id.csv ^
        --pipeline vit32_ft_ckpt --split_seed 2026 --seeds 2027,2028,2029,2030,2031

    python entrenar_v3.py --manifest_csv subset_grande_con_lesion_id.csv ^
        --pipeline capsnet_r3_revised --capsnet_lr 1e-5 --capsnet_input_norm imagenet ^
        --split_seed 2026 --seeds 2027 --no_test --tag capsnet_grid_lr1e-5_imagenet
"""

import argparse
import os
import platform
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

import modelos_v3 as mv

EARLY_STOPPING_PATIENCE = 10
MAX_EPOCHS = 200


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    keras.utils.set_random_seed(seed)


def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, mv.IMG_SIZE)                 # bilinear
    img = tf.cast(img, tf.float32) / 255.0                 # [0, 1]
    return img, label


def make_dataset(df, is_capsnet, augment, batch_size, seed):
    paths = df["filepath"].values
    if is_capsnet:
        labels = np.stack([1 - df["label"].values, df["label"].values], axis=1).astype("float32")
    else:
        labels = df["label"].values.astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        aug = keras.Sequential([
            keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            keras.layers.RandomRotation(0.25, seed=seed),
            keras.layers.RandomTranslation(0.1, 0.1, seed=seed),
            keras.layers.RandomBrightness(0.1, value_range=(0.0, 1.0), seed=seed),
            keras.layers.RandomContrast(0.1, seed=seed),
        ])
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 1.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=len(df), seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def youden_threshold(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    best = int(np.argmax(tpr - fpr))          # first maximum = highest threshold among ties
    t = thr[best]
    return float(t) if np.isfinite(t) else 0.5


def threshold_metrics(y_true, y_score, threshold, prefix):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = (2 * prec * sens / (prec + sens)
          if np.isfinite(prec) and np.isfinite(sens) and (prec + sens) > 0 else np.nan)
    return {f"{prefix}threshold": threshold, f"{prefix}accuracy": (tp + tn) / (tp + tn + fp + fn),
            f"{prefix}sensitivity": sens, f"{prefix}specificity": spec,
            f"{prefix}precision": prec, f"{prefix}f1": f1,
            f"{prefix}tp": int(tp), f"{prefix}tn": int(tn), f"{prefix}fp": int(fp), f"{prefix}fn": int(fn)}


def grouped_split(df, test_size, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    a, b = next(splitter.split(df, groups=df["lesion_id"]))
    return df.iloc[a].reset_index(drop=True), df.iloc[b].reset_index(drop=True)


def make_split(df, split_seed):
    """Identical procedure to the original script: test first, then validation."""
    trainval, test = grouped_split(df, 0.15, split_seed)
    train, val = grouped_split(trainval, 0.15 / 0.85, split_seed)
    sets = {"train": set(train.lesion_id), "val": set(val.lesion_id), "test": set(test.lesion_id)}
    for a in sets:
        for b in sets:
            if a < b and sets[a] & sets[b]:
                raise RuntimeError(f"Lesion leakage between {a} and {b}")
    return train, val, test


def load_manifest(path):
    df = pd.read_csv(path)
    df["label"] = (df["label_from_folder"] == "malignant").astype(int)
    df["image_id"] = df["filepath"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    return df


def save_environment(path):
    lines = [f"python: {platform.python_version()}", f"tensorflow: {tf.__version__}",
             f"keras: {keras.__version__}", f"numpy: {np.__version__}", f"pandas: {pd.__version__}"]
    for mod in ["sklearn", "transformers", "tf_keras"]:
        try:
            lines.append(f"{mod}: {__import__(mod).__version__}")
        except Exception:
            pass
    for g in tf.config.list_physical_devices("GPU"):
        try:
            lines.append("gpu: " + tf.config.experimental.get_device_details(g).get("device_name", g.name))
        except Exception:
            lines.append("gpu: " + g.name)
    lines.append(f"os: {platform.platform()}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def predict_scores(model, ds, is_capsnet):
    pred = model.predict(ds, verbose=0)
    if is_capsnet:
        return pred[:, 1], pred[:, 0]          # malignant length, benign length
    return pred.ravel(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--pipeline", required=True, choices=list(mv.PIPELINES))
    ap.add_argument("--split_seed", type=int, default=2026)
    ap.add_argument("--seeds", default="2027,2028,2029,2030,2031")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--capsnet_lr", type=float, default=None)
    ap.add_argument("--capsnet_input_norm", default=None, choices=["none", "imagenet"])
    ap.add_argument("--capsnet_config", default=None,
                    help="JSON written by seleccionar_capsnet.py (gate G6) for capsnet_r3_revised")
    ap.add_argument("--no_test", action="store_true", help="validation only (selection runs)")
    ap.add_argument("--tag", default=None, help="name for the output files (default: pipeline)")
    ap.add_argument("--check_ledger", default=None,
                    help="existing split_ledger.csv that this split must reproduce (e.g. the v2 one)")
    ap.add_argument("--results_root", default="resultados_v3")
    ap.add_argument("--models_root", default="modelos_v3")
    args = ap.parse_args()

    tag = args.tag or args.pipeline
    is_caps = args.pipeline.startswith("capsnet")
    if args.capsnet_config:
        import json
        with open(args.capsnet_config) as f:
            sel = json.load(f)
        if not sel.get("selected"):
            print("Gate G6 did not select a functioning revised CapsNet configuration; "
                  "capsnet_r3_revised is not trained (see protocol).")
            return
        args.capsnet_lr, args.capsnet_input_norm = float(sel["lr"]), sel["input_norm"]
        print(f"Revised CapsNet configuration from {args.capsnet_config}: lr={args.capsnet_lr}, "
              f"input_norm={args.capsnet_input_norm}")
    if args.pipeline == "capsnet_r3_revised" and (args.capsnet_lr is None or args.capsnet_input_norm is None):
        raise SystemExit("capsnet_r3_revised needs --capsnet_lr and --capsnet_input_norm "
                         "(the configuration selected in gate G6).")
    res_dir = os.path.join(args.results_root, f"split_{args.split_seed}")
    mod_dir = os.path.join(args.models_root, f"split_{args.split_seed}")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(mod_dir, exist_ok=True)
    save_environment(os.path.join(res_dir, "entorno.txt"))

    df = load_manifest(args.manifest_csv)
    train_df, val_df, test_df = make_split(df, args.split_seed)
    ledger = pd.concat([train_df.assign(partition="train"), val_df.assign(partition="val"),
                        test_df.assign(partition="test")])
    key = lambda d: set(zip(d["image_id"], d["partition"]))
    if args.check_ledger:
        ref = pd.read_csv(args.check_ledger)
        if key(ref) != key(ledger):
            raise RuntimeError("This split does not reproduce " + args.check_ledger)
        print(f"Split {args.split_seed} reproduces {args.check_ledger} exactly.")
    ledger_path = os.path.join(res_dir, "split_ledger.csv")
    if os.path.exists(ledger_path):
        if key(pd.read_csv(ledger_path)) != key(ledger):
            raise RuntimeError("Split differs from the existing " + ledger_path)
    else:
        ledger.drop(columns=["filepath"]).to_csv(ledger_path, index=False)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name:5s}: {len(part)} images, {part.lesion_id.nunique()} lesions, "
              f"{part.label.sum()} malignant")

    results_path = os.path.join(res_dir, f"resultados_{tag}.csv")
    preds_path = os.path.join(res_dir, f"predicciones_{tag}.csv")
    done = set(pd.read_csv(results_path)["seed_value"].astype(int)) if os.path.exists(results_path) else set()

    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        if seed in done:
            print(f"[{tag}] seed {seed} already done, skipping.")
            continue
        print(f"\n{'=' * 60}\n{tag} | split {args.split_seed} | seed {seed}\n{'=' * 60}")
        set_seeds(seed)
        model, framework = mv.build_pipeline(args.pipeline, lr=args.capsnet_lr if is_caps else None,
                                             input_norm=args.capsnet_input_norm if is_caps else None)
        K = keras
        if framework == "tf_keras":
            import tf_keras as K
        print(f"Trainable parameters seen by the optimizer: {mv.count_trainable(model):,}")

        train_ds = make_dataset(train_df, is_caps, True, args.batch_size, seed)
        val_ds = make_dataset(val_df, is_caps, False, args.batch_size, seed)
        weights_path = os.path.join(mod_dir, f"{tag}_seed{seed}.weights.h5")
        hist_path = os.path.join(res_dir, f"historial_{tag}_seed{seed}.csv")
        callbacks = [
            K.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                                      restore_best_weights=True, verbose=1),
            K.callbacks.ModelCheckpoint(weights_path, monitor="val_loss", save_best_only=True,
                                        save_weights_only=True, verbose=0),
            K.callbacks.CSVLogger(hist_path),
        ]
        t0 = time.time()
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.max_epochs,
                            callbacks=callbacks, verbose=2)
        minutes = (time.time() - t0) / 60
        n_run = len(history.history["loss"])
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1

        y_val = val_df["label"].values
        s_val, b_val = predict_scores(model, val_ds, is_caps)
        thr = youden_threshold(y_val, s_val)
        row = {"pipeline": args.pipeline, "tag": tag, "split_seed": args.split_seed, "seed_value": seed,
               "capsnet_lr": args.capsnet_lr, "capsnet_input_norm": args.capsnet_input_norm,
               "trainable_params": mv.count_trainable(model), "train_minutes": round(minutes, 1),
               "n_epochs_run": n_run, "best_epoch": best_epoch,
               "best_val_loss": float(np.min(history.history["val_loss"])),
               "val_auc": roc_auc_score(y_val, s_val), "youden_threshold": thr}
        parts = [pd.DataFrame({"partition": "val", "image_id": val_df.image_id, "lesion_id": val_df.lesion_id,
                               "dx": val_df.dx_official, "y_true": y_val, "score": s_val})]
        if is_caps:
            parts[0]["score_benign_capsule"] = b_val

        if not args.no_test:
            test_ds = make_dataset(test_df, is_caps, False, args.batch_size, seed)
            y_test = test_df["label"].values
            s_test, b_test = predict_scores(model, test_ds, is_caps)
            row.update({"test_auc": roc_auc_score(y_test, s_test), "test_n": len(y_test),
                        "test_malignant": int(y_test.sum()), "test_lesions": test_df.lesion_id.nunique()})
            row.update(threshold_metrics(y_test, s_test, thr, "youden_"))
            t = pd.DataFrame({"partition": "test", "image_id": test_df.image_id, "lesion_id": test_df.lesion_id,
                              "dx": test_df.dx_official, "y_true": y_test, "score": s_test})
            if is_caps:
                t["score_benign_capsule"] = b_test
            parts.append(t)

        pd.DataFrame([row]).to_csv(results_path, mode="a", header=not os.path.exists(results_path), index=False)
        pd.concat(parts).assign(pipeline=args.pipeline, tag=tag, split_seed=args.split_seed,
                                seed_value=seed, youden_threshold=thr).to_csv(
            preds_path, mode="a", header=not os.path.exists(preds_path), index=False)
        msg = f"seed {seed}: val AUC={row['val_auc']:.3f}"
        if not args.no_test:
            msg += f" | TEST AUC={row['test_auc']:.3f}"
        print(msg + f" | best epoch {best_epoch}/{n_run} | {minutes:.1f} min")
        del model
        K.backend.clear_session()

    print(f"\nResults: {results_path}\nPredictions: {preds_path}")


if __name__ == "__main__":
    main()
