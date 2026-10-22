"""
capsnet_diagnostico.py

Staged diagnosis of the CapsNet-R3 training failure (revision protocol, gates
G1-G5). Uses ONLY the training and validation partitions of the original split
(split seed 2026); the test partition is never loaded.

  G1  Data and evaluation checks: labels vs. diagnoses, one-hot class order,
      input range, batch-order invariance, synthetic metric controls.
  G2  Mathematical implementation: squash, dynamic routing and margin loss
      compared with an independent NumPy reference.
  G3  Gradient flow: finite, non-zero gradients for every trainable tensor and
      actual parameter updates after one optimizer step.
  G4  Training-only memorization: 32 training images (16 + 16), no
      augmentation, no weight decay, 100 epochs, for each learning rate of the
      protocol; plus a synthetic, trivially separable control. A learning rate
      passes when, at the last epoch, the capsule decision rule (longer capsule)
      classifies >= 95% of the 32 images correctly and the margin loss is
      <= 0.10 (engineering tolerances, not literature benchmarks).
  G5  Saturation localization: step-by-step trace of primary-capsule lengths,
      pre-squash norms, coupling coefficients, output lengths of BOTH capsules
      and gradient norms during the first steps, for the original configuration
      and controlled perturbations (learning rate, routing iterations).

Gate G6 (validation-only selection of a revised configuration) is run with
entrenar_v3.py --no_test and seleccionar_capsnet.py.

Usage (Windows):
    python capsnet_diagnostico.py --manifest_csv subset_grande_con_lesion_id.csv
    python capsnet_diagnostico.py --manifest_csv subset_grande_con_lesion_id.csv --gates G1,G2,G3
Outputs: diagnostico_capsnet/  (CSV files + resumen_diagnostico.txt)
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import roc_auc_score

import modelos_v3 as mv
from entrenar_v3 import load_manifest, make_split, load_image, make_dataset, set_seeds, youden_threshold

OUT = "diagnostico_capsnet"
PROTOCOL_LRS = [1e-3, 1e-4, 3e-5, 1e-5]
MALIGNANT_DX = {"mel", "bcc", "akiec"}
summary = []


def log(msg):
    print(msg, flush=True)
    summary.append(msg)


def load_images(df):
    imgs = [load_image(p, 0)[0].numpy() for p in df["filepath"]]
    return np.stack(imgs).astype("float32")


def onehot(y):
    return np.stack([1 - y, y], axis=1).astype("float32")


# ----------------------------------------------------------------- G1
def gate_g1(df, train, val):
    ok = True
    wrong = df[(df.dx_official.isin(MALIGNANT_DX)).astype(int) != df.label]
    log(f"G1 labels consistent with diagnoses: {len(wrong) == 0} ({len(wrong)} mismatches)")
    ok &= len(wrong) == 0
    mixed = (df.groupby("lesion_id").label.nunique() > 1).sum()
    log(f"G1 lesions with mixed labels: {mixed}")
    ok &= mixed == 0
    dup = df.image_id.duplicated().sum()
    log(f"G1 duplicated image_id: {dup}")
    ok &= dup == 0
    missing = sum(not os.path.exists(p) for p in df.filepath)
    log(f"G1 missing image files: {missing}")
    ok &= missing == 0

    ds = make_dataset(train.head(64), True, False, 64, 0)
    _, y = next(iter(ds))
    y = y.numpy()
    order_ok = np.array_equal(y[:, 1], train.head(64).label.values) and np.allclose(y.sum(1), 1)
    log(f"G1 one-hot order [benign, malignant] matches labels: {order_ok}")
    ok &= order_ok

    x = load_images(train.head(32))
    log(f"G1 input range: min={x.min():.3f} max={x.max():.3f} mean={x.mean():.3f} (expected within [0, 1])")
    ok &= x.min() >= 0 and x.max() <= 1

    set_seeds(2027)
    m = mv.build_capsnet()
    p1 = m.predict(x, batch_size=16, verbose=0)
    perm = np.random.default_rng(0).permutation(len(x))
    p2 = m.predict(x[perm], batch_size=8, verbose=0)[np.argsort(perm)]
    inv = np.allclose(p1, p2, atol=1e-5)
    log(f"G1 batch-order / batch-size invariance of predictions: {inv}")
    ok &= inv

    yt = np.r_[np.ones(50), np.zeros(50)]
    ctrl = {"perfect": roc_auc_score(yt, yt), "inverted": roc_auc_score(yt, 1 - yt),
            "constant": roc_auc_score(yt, np.full(100, 0.7))}
    thr = youden_threshold(yt, np.r_[np.full(50, 0.8), np.full(50, 0.2)])
    ctrl_ok = ctrl == {"perfect": 1.0, "inverted": 0.0, "constant": 0.5} and 0.2 < thr <= 0.8
    log(f"G1 synthetic metric controls {ctrl}, Youden threshold={thr:.2f}: {ctrl_ok}")
    ok &= ctrl_ok
    log(f"G1 RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------- G2
def np_squash(s):
    n2 = (s ** 2).sum(-1, keepdims=True)
    return n2 / (1 + n2) * s / np.sqrt(n2 + 1e-7)


def np_routing(u_hat, iters):
    b = np.zeros(u_hat.shape[:3])
    for i in range(iters):
        c = np.exp(b) / np.exp(b).sum(2, keepdims=True)
        s = (c[..., None] * u_hat).sum(1)
        v = np_squash(s)
        if i < iters - 1:
            b = b + (u_hat * v[:, None]).sum(-1)
    return v


def np_margin(y, L):
    return np.mean(np.sum(y * np.maximum(0, 0.9 - L) ** 2 + 0.5 * (1 - y) * np.maximum(0, L - 0.1) ** 2, 1))


def gate_g2():
    rng = np.random.default_rng(1)
    ok = True
    s = rng.normal(size=(4, 7, 8)).astype("float32") * rng.choice([0.01, 1, 50], size=(4, 7, 1))
    d = np.abs(mv.squash(tf.constant(s)).numpy() - np_squash(s.astype("float64"))).max()
    log(f"G2 squash max abs diff vs NumPy: {d:.2e}")
    ok &= d < 1e-5
    u = rng.normal(size=(3, 50, 8)).astype("float32")
    layer = mv.DigitCaps(2, 16, routing_iters=3)
    v_tf = layer(tf.constant(u)).numpy()
    W = layer.W.numpy()[0]                                      # (in, out, 16, 8)
    u_hat = np.einsum("ijdk,bik->bijd", W.astype("float64"), u.astype("float64"))
    d = np.abs(v_tf - np_routing(u_hat, 3)).max()
    log(f"G2 routing (3 iterations) max abs diff vs NumPy: {d:.2e}")
    ok &= d < 1e-5
    c_sum = layer.route(layer.predictions(tf.constant(u)))[2].numpy().sum(2)
    log(f"G2 coupling coefficients sum to 1 over output capsules: {np.allclose(c_sum, 1, atol=1e-5)}")
    ok &= np.allclose(c_sum, 1, atol=1e-5)
    y = onehot(rng.integers(0, 2, 10))
    L = rng.random((10, 2)).astype("float32")
    d = abs(float(mv.margin_loss(tf.constant(y), tf.constant(L))) - np_margin(y, L))
    log(f"G2 margin loss abs diff vs NumPy: {d:.2e}")
    ok &= d < 1e-5
    log(f"G2 RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------- G3
def gate_g3(train):
    set_seeds(2027)
    m = mv.build_capsnet()
    sub = pd.concat([train[train.label == 1].head(8), train[train.label == 0].head(8)])
    x, y = load_images(sub), onehot(sub.label.values)
    with tf.GradientTape() as tape:
        loss = mv.margin_loss(y, m(x, training=True))
    grads = tape.gradient(loss, m.trainable_variables)
    rows, ok = [], True
    for v, g in zip(m.trainable_variables, grads):
        fin = g is not None and bool(np.all(np.isfinite(g.numpy())))
        nz = g is not None and float(tf.norm(g)) > 0
        rows.append({"variable": v.path, "grad_norm": float(tf.norm(g)) if g is not None else np.nan,
                     "weight_norm": float(tf.norm(v)), "finite": fin, "nonzero": nz})
        ok &= fin and nz
    before = [v.numpy().copy() for v in m.trainable_variables]
    m.train_on_batch(x, y)
    changed = sum(not np.allclose(b, v.numpy()) for b, v in zip(before, m.trainable_variables))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "G3_gradientes.csv"), index=False)
    log(f"G3 initial loss {float(loss):.4f}; all gradients finite and non-zero: {ok}; "
        f"variables updated after one step: {changed}/{len(before)}")
    ok &= changed == len(before)
    log(f"G3 RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------- G4
def memorize(x, y01, lr, epochs, label):
    set_seeds(2027)
    m = mv.build_capsnet(lr=lr, weight_decay=0.0)
    Y = onehot(y01)
    rows = []
    for ep in range(1, epochs + 1):
        h = m.fit(x, Y, batch_size=16, epochs=1, shuffle=True, verbose=0)
        L = m.predict(x, verbose=0)
        auc = roc_auc_score(y01, L[:, 1]) if L[:, 1].std() > 0 else 0.5
        acc = float(np.mean((L[:, 1] > L[:, 0]).astype(int) == y01))     # capsule decision rule
        rows.append({"data": label, "lr": lr, "epoch": ep, "train_loss": h.history["loss"][0],
                     "train_auc": auc, "train_acc_argmax": acc, "len_mal_mean": L[:, 1].mean(), "len_mal_std": L[:, 1].std(),
                     "len_ben_mean": L[:, 0].mean(), "len_ben_std": L[:, 0].std()})
    return rows


def gate_g4(train, epochs):
    sub = pd.concat([train[train.label == 1].sample(16, random_state=0),
                     train[train.label == 0].sample(16, random_state=0)])
    x, y = load_images(sub), sub.label.values
    rng = np.random.default_rng(0)
    xs = (rng.random((32, 224, 224, 3)) * 0.6).astype("float32")
    ys = np.r_[np.ones(16), np.zeros(16)].astype(int)
    xs[ys == 1, :, :, 0] += 0.3                                  # trivially separable control
    rows, passed = [], {}
    for lr in PROTOCOL_LRS:
        for label, xx, yy in [("real_32", x, y), ("synthetic_32", xs, ys)]:
            t0 = time.time()
            r = memorize(xx, yy, lr, epochs, label)
            rows += r
            last = r[-1]
            memorized = last["train_acc_argmax"] >= 0.95 and last["train_loss"] <= 0.10
            log(f"G4 {label:12s} lr={lr:.0e}: final train accuracy={last['train_acc_argmax']:.3f}, "
                f"final loss={last['train_loss']:.4f} (initial {r[0]['train_loss']:.4f}), "
                f"malignant length {last['len_mal_mean']:.4f}+/-{last['len_mal_std']:.4f}, "
                f"benign {last['len_ben_mean']:.4f}+/-{last['len_ben_std']:.4f} -> "
                f"{'memorized' if memorized else 'not memorized'} ({time.time() - t0:.0f}s)")
            if label == "real_32":
                passed[lr] = memorized
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "G4_memorizacion.csv"), index=False)
    log("G4 learning rates that memorize 32 real training images (final accuracy >= 0.95 and loss <= 0.10): "
        + str([f"{k:.0e}" for k, v in passed.items() if v]))
    with open(os.path.join(OUT, "G4_lrs_aprobadas.json"), "w") as f:
        json.dump({f"{k:.0e}": bool(v) for k, v in passed.items()}, f, indent=1)
    ok = any(passed.values())
    log(f"G4 RESULT: {'PASS (at least one learning rate)' if ok else 'FAIL (no learning rate memorizes)'}")
    return ok


# ----------------------------------------------------------------- G5
def trace(train, lr, routing_iters, steps):
    set_seeds(2027)
    m = mv.build_capsnet(lr=lr, routing_iters=routing_iters)
    prim = m.get_layer("primary_caps")
    dig = m.get_layer("digit_caps")
    probe = keras.Model(m.input, prim.output)
    ds = make_dataset(train, True, True, 16, 2027)
    x_fixed = load_images(pd.concat([train[train.label == 1].head(16), train[train.label == 0].head(16)]))
    y_fixed = np.r_[np.ones(16), np.zeros(16)]
    rows = []
    it = iter(ds.repeat())
    for step in range(steps + 1):
        if step > 0:
            xb, yb = next(it)
            with tf.GradientTape() as tape:
                loss = mv.margin_loss(yb, m(xb, training=True))
            grads = tape.gradient(loss, m.trainable_variables)
            m.optimizer.apply_gradients(zip(grads, m.trainable_variables))
            gnorm = {v.path.split("/")[0]: float(tf.norm(g)) for v, g in zip(m.trainable_variables, grads)
                     if "kernel" in v.path or "routing" in v.path}
        else:
            loss, gnorm = np.nan, {}
        p = probe(x_fixed)
        plen = tf.norm(p, axis=-1).numpy()
        u_hat = dig.predictions(p)
        v, s, c = dig.route(u_hat)
        L = tf.norm(v, axis=-1).numpy()
        cnp = c.numpy()
        rows.append({"lr": lr, "routing_iters": routing_iters, "step": step,
                     "batch_loss": float(loss),
                     "primary_len_mean": plen.mean(), "primary_frac_len_gt_0.9": (plen > 0.9).mean(),
                     "uhat_norm_mean": float(tf.reduce_mean(tf.norm(u_hat, axis=-1))),
                     "presquash_norm_mal": float(tf.reduce_mean(tf.norm(s[:, 1], axis=-1))),
                     "presquash_norm_ben": float(tf.reduce_mean(tf.norm(s[:, 0], axis=-1))),
                     "coupling_max_mean": cnp.max(2).mean(),
                     "len_mal_mean": L[:, 1].mean(), "len_mal_std": L[:, 1].std(),
                     "len_ben_mean": L[:, 0].mean(), "len_ben_std": L[:, 0].std(),
                     "auc_fixed_batch": roc_auc_score(y_fixed, L[:, 1]) if L[:, 1].std() > 0 else 0.5,
                     **{f"grad_{k}": g for k, g in gnorm.items()}})
    return rows


def gate_g5(train, steps):
    rows = []
    for lr, r in [(1e-3, 3), (1e-3, 1), (1e-5, 3)]:
        rr = trace(train, lr, r, steps)
        rows += rr
        first_sat = next((q["step"] for q in rr if q["primary_frac_len_gt_0.9"] > 0.5), None)
        log(f"G5 lr={lr:.0e} routing={r}: primary length {rr[0]['primary_len_mean']:.3f} -> "
            f"{rr[-1]['primary_len_mean']:.3f}; >50% primary capsules saturated (>0.9) from step "
            f"{first_sat}; final malignant length {rr[-1]['len_mal_mean']:.4f}+/-{rr[-1]['len_mal_std']:.4f}, "
            f"benign {rr[-1]['len_ben_mean']:.4f}+/-{rr[-1]['len_ben_std']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "G5_traza_saturacion.csv"), index=False)
    log("G5 RESULT: descriptive (see G5_traza_saturacion.csv)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--gates", default="G1,G2,G3,G4,G5")
    ap.add_argument("--g4_epochs", type=int, default=100)
    ap.add_argument("--g5_steps", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = load_manifest(args.manifest_csv)
    train, val, _ = make_split(df, 2026)          # test partition is not used
    gates = args.gates.split(",")
    log(f"CapsNet-R3 diagnosis | split 2026 | train {len(train)} images, val {len(val)} images | gates {gates}")
    if "G1" in gates:
        gate_g1(df, train, val)
    if "G2" in gates:
        gate_g2()
    if "G3" in gates:
        gate_g3(train)
    if "G4" in gates:
        gate_g4(train, args.g4_epochs)
    if "G5" in gates:
        gate_g5(train, args.g5_steps)
    with open(os.path.join(OUT, f"resumen_diagnostico_{'_'.join(gates)}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    print(f"\nSaved in {OUT}/")


if __name__ == "__main__":
    main()
