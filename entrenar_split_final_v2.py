"""
entrenar_split_final_v2.py

Entrenamiento y evaluacion de los tres pipelines (version corregida).

Diseno: UN split fijo 70/15/15 agrupado por lesion (semilla de particion
2026) y 5 semillas de entrenamiento (2027-2031) sobre ese mismo split.
La variabilidad reportada es entre semillas de entrenamiento, no entre
particiones. El test se evalua una sola vez por semilla.

Cambios respecto a la version anterior:
  1. BUG CORREGIDO: RandomBrightness usaba value_range=(0,255) por
     defecto con imagenes en [0,1]; ~la mitad de las imagenes de
     entrenamiento quedaban negras y el resto desplazadas a valores
     de ~1 a ~25. Ahora value_range=(0,1) y se recorta a [0,1].
  2. Umbral de Youden calculado SOLO en validacion y aplicado al test.
     Se reportan tambien las metricas a 0.5 para comparar.
  3. Se guardan las predicciones por imagen (val y test) de cada
     semilla: necesarias para IC bootstrap, ROC y la tabla suplementaria.
  4. Se registran mejor epoca, umbral, versiones de software y GPU.
  5. Resultados en carpetas nuevas (_v2) para no mezclar con los viejos.

Uso (Windows):
    python entrenar_split_final_v2.py ^
        --manifest_csv subset_grande_con_lesion_id.csv ^
        --arch efficientnetb1 ^
        --n_seeds 5

Requiere: numpy, pandas, scikit-learn, tensorflow, transformers, tf-keras
"""

import argparse
import os
import platform
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

from modelos_reducidos import MODEL_BUILDERS, IMG_SIZE

SPLIT_SEED = 2026          # semilla de la particion (fija)
BASE_SEED = 2026           # semillas de entrenamiento = BASE_SEED + 1..n
EARLY_STOPPING_PATIENCE = 10
MAX_EPOCHS = 200
RESULTS_DIR = "resultados_split_final_v2"
MODELS_DIR = "modelos_split_final_v2"


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)          # bilineal (el articulo debe decir bilineal)
    img = tf.cast(img, tf.float32) / 255.0        # imagenes en [0, 1]
    return img, label


def make_dataset(df, arch, augment, batch_size, seed):
    paths = df["filepath"].values
    if arch == "capsnet":
        labels = np.stack([1 - df["label"].values, df["label"].values], axis=1).astype("float32")
    else:
        labels = df["label"].values.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        aug = keras.Sequential([
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.25),                       # hasta +/-90 grados
            keras.layers.RandomTranslation(0.1, 0.1),
            keras.layers.RandomBrightness(0.1, value_range=(0.0, 1.0)),  # CORREGIDO
            keras.layers.RandomContrast(0.1),
        ])
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 1.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=len(df), seed=seed)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def predict_scores(model, ds, arch):
    """Score continuo de la clase MALIGNA (antes de umbralizar)."""
    pred = model.predict(ds, verbose=0)
    if arch == "capsnet":
        return pred[:, 1]          # longitud del capsule maligno
    return pred.ravel()            # salida sigmoide = P(maligno)


def youden_threshold(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    best = int(np.argmax(j))
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
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {f"{prefix}threshold": threshold, f"{prefix}accuracy": acc,
            f"{prefix}sensitivity": sens, f"{prefix}specificity": spec,
            f"{prefix}precision": prec, f"{prefix}f1": f1,
            f"{prefix}tp": int(tp), f"{prefix}tn": int(tn),
            f"{prefix}fp": int(fp), f"{prefix}fn": int(fn)}


def grouped_split(df, test_size, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_a, idx_b = next(splitter.split(df, groups=df["lesion_id"]))
    return df.iloc[idx_a].reset_index(drop=True), df.iloc[idx_b].reset_index(drop=True)


def save_environment(path):
    lines = [f"python: {platform.python_version()}",
             f"tensorflow: {tf.__version__}",
             f"keras: {keras.__version__}",
             f"numpy: {np.__version__}",
             f"pandas: {pd.__version__}"]
    try:
        import sklearn
        lines.append(f"scikit-learn: {sklearn.__version__}")
    except ImportError:
        pass
    try:
        import transformers
        lines.append(f"transformers: {transformers.__version__}")
    except ImportError:
        pass
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            name = tf.config.experimental.get_device_details(g).get("device_name", g.name)
        except Exception:
            name = g.name
        lines.append(f"gpu: {name}")
    if not gpus:
        lines.append("gpu: ninguna (CPU)")
    lines.append(f"os: {platform.platform()}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True,
                    help="CSV generado por curar_dataset_grande.py")
    ap.add_argument("--arch", required=True, choices=list(MODEL_BUILDERS.keys()))
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--unfreeze_backbone", action="store_true",
                    help="Descongela el backbone preentrenado (fine-tuning completo). "
                         "No aplica a CapsNet, que siempre se entrena desde cero.")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_environment(os.path.join(RESULTS_DIR, "entorno.txt"))

    df = pd.read_csv(args.manifest_csv)
    df["label"] = (df["label_from_folder"] == "malignant").astype(int)
    df["image_id"] = df["filepath"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    print(f"Total: {len(df)}  (malignant={df['label'].sum()}, benign={(1 - df['label']).sum()})")
    print(f"Lesiones distintas: {df['lesion_id'].nunique()}")

    trainval_df, test_df = grouped_split(df, test_size=0.15, seed=SPLIT_SEED)
    train_df, val_df = grouped_split(trainval_df, test_size=0.15 / 0.85, seed=SPLIT_SEED)

    sets = {"train": set(train_df["lesion_id"]), "val": set(val_df["lesion_id"]),
            "test": set(test_df["lesion_id"])}
    for a in sets:
        for b in sets:
            if a < b and (sets[a] & sets[b]):
                raise RuntimeError(f"Fuga de lesion entre {a} y {b}: {sets[a] & sets[b]}")

    print("\nSplit fijo (agrupado por lesion, sin fuga verificada):")
    for name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"  {name:5s}: {len(part)} imgs, {part['lesion_id'].nunique()} lesiones, "
              f"malignas={part['label'].sum()} ({100 * part['label'].mean():.1f}%)")

    ledger_path = os.path.join(RESULTS_DIR, "split_ledger.csv")
    ledger = pd.concat([train_df.assign(partition="train"),
                        val_df.assign(partition="val"),
                        test_df.assign(partition="test")])
    if os.path.exists(ledger_path):
        old = pd.read_csv(ledger_path)
        same = (set(zip(old["filepath"], old["partition"]))
                == set(zip(ledger["filepath"], ledger["partition"])))
        if not same:
            raise RuntimeError("El split no coincide con split_ledger.csv existente. "
                               "Revisa que el manifest sea el mismo.")
    else:
        ledger.to_csv(ledger_path, index=False)

    suffix = "_finetuned" if args.unfreeze_backbone else ""
    tag = f"{args.arch}{suffix}"
    results_path = os.path.join(RESULTS_DIR, f"resultados_{tag}.csv")
    preds_path = os.path.join(RESULTS_DIR, f"predicciones_{tag}.csv")

    done_seeds = set()
    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        done_seeds = set(existing["seed_run"].astype(int))
        print(f"\nReanudando: semillas ya completas: {sorted(done_seeds)}")

    for seed_i in range(1, args.n_seeds + 1):
        if seed_i in done_seeds:
            print(f"Semilla {seed_i}/{args.n_seeds} ya completada, se salta.")
            continue

        seed = BASE_SEED + seed_i
        print(f"\n{'=' * 60}\n{tag} | Semilla {seed_i}/{args.n_seeds} (seed={seed})\n{'=' * 60}")
        set_seeds(seed)

        train_ds = make_dataset(train_df, args.arch, augment=True, batch_size=args.batch_size, seed=seed)
        val_ds = make_dataset(val_df, args.arch, augment=False, batch_size=args.batch_size, seed=seed)
        test_ds = make_dataset(test_df, args.arch, augment=False, batch_size=args.batch_size, seed=seed)

        if args.arch == "capsnet":
            model = MODEL_BUILDERS[args.arch]()  # CapsNet: inicializacion aleatoria
        else:
            model = MODEL_BUILDERS[args.arch](train_base=args.unfreeze_backbone)
        model_path = os.path.join(MODELS_DIR, f"{tag}_seed{seed_i}.keras")

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                                          restore_best_weights=True, verbose=1),
            keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss",
                                            save_best_only=True, verbose=0),
        ]
        history = model.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS,
                            callbacks=callbacks, verbose=2)
        n_epochs_run = len(history.history["loss"])
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1

        # Umbral SOLO con validacion
        y_val = val_df["label"].values
        s_val = predict_scores(model, val_ds, args.arch)
        thr = youden_threshold(y_val, s_val)

        # Test: una sola evaluacion
        y_test = test_df["label"].values
        s_test = predict_scores(model, test_ds, args.arch)

        row = {"arch": args.arch, "finetuned": bool(args.unfreeze_backbone),
               "seed_run": seed_i, "seed_value": seed,
               "n_epochs_run": n_epochs_run, "best_epoch": best_epoch,
               "val_auc": roc_auc_score(y_val, s_val),
               "test_auc": roc_auc_score(y_test, s_test),
               "test_n": len(y_test), "test_malignant": int(y_test.sum()),
               "test_lesions": test_df["lesion_id"].nunique()}
        row.update(threshold_metrics(y_test, s_test, thr, "youden_"))
        row.update(threshold_metrics(y_test, s_test, 0.5, "t05_"))
        pd.DataFrame([row]).to_csv(results_path, mode="a",
                                   header=not os.path.exists(results_path), index=False)

        preds = pd.concat([
            pd.DataFrame({"partition": "val", "image_id": val_df["image_id"],
                          "lesion_id": val_df["lesion_id"], "y_true": y_val, "score": s_val}),
            pd.DataFrame({"partition": "test", "image_id": test_df["image_id"],
                          "lesion_id": test_df["lesion_id"], "y_true": y_test, "score": s_test}),
        ]).assign(arch=args.arch, finetuned=bool(args.unfreeze_backbone),
                  seed_run=seed_i, youden_threshold=thr)
        preds.to_csv(preds_path, mode="a", header=not os.path.exists(preds_path), index=False)

        print(f"Semilla {seed_i} -> TEST AUC={row['test_auc']:.3f} | umbral val={thr:.3f} | "
              f"Sens={row['youden_sensitivity']:.3f} Spec={row['youden_specificity']:.3f} | "
              f"mejor epoca={best_epoch}/{n_epochs_run}")
        keras.backend.clear_session()

    res = pd.read_csv(results_path)
    cols = ["test_auc", "youden_accuracy", "youden_sensitivity", "youden_specificity",
            "youden_precision", "youden_f1", "t05_accuracy", "t05_sensitivity", "t05_specificity"]
    summary = res[cols].agg(["mean", "std"]).T
    summary.to_csv(os.path.join(RESULTS_DIR, f"resumen_{tag}.csv"))
    print(f"\n{'=' * 60}\nRESUMEN {tag} ({len(res)} semillas, TEST)\n{'=' * 60}")
    print(summary.round(4))
    print(f"\nResultados por semilla: {results_path}\nPredicciones por imagen: {preds_path}")


if __name__ == "__main__":
    main()
