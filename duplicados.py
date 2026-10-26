"""
duplicados.py  (PASO 5, inference only, no training)

Searches HAM10000 for near-duplicate images that belong to DIFFERENT lesion_id
values (the metadata does not link them), following the approach of Abhishek
et al. (Sci Data 2025;12:196): image embeddings, cosine similarity over all
pairs, and visual review of the most similar pairs.

Embeddings: EfficientNet-B1 with ImageNet weights (global average pooling),
images resized to 240x240. For every image, the embedding of the image and of
its horizontal flip are averaged.

For each candidate pair it records where each image was used:
  original split (train/val/test), repeated splits 3001-3005, untouched cohort,
  or not used; and whether the pair crosses a training/test boundary of any
  evaluation.

Outputs (folder duplicados/):
  pares_top1000.csv     1,000 most similar cross-lesion pairs, with partitions
  pares_revisar.png     the 60 most similar pairs side by side, for review
  resumen.txt           counts

Usage:  python duplicados.py --metadata HAM10000_metadata.csv --ham_dirs "carpeta1,carpeta2"
"""
import argparse, os
import numpy as np, pandas as pd

OUT = "duplicados"


def index_images(dirs):
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


def embeddings(paths):
    import tensorflow as tf
    from tensorflow import keras
    net = keras.applications.EfficientNetB1(include_top=False, weights="imagenet", pooling="avg",
                                            input_shape=(240, 240, 3))
    def load(p):
        img = tf.io.decode_jpeg(tf.io.read_file(p), channels=3)
        return tf.image.resize(tf.cast(img, tf.float32), (240, 240))
    ds = tf.data.Dataset.from_tensor_slices(paths).map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(64)
    out = []
    for i, x in enumerate(ds):
        e = net(x, training=False).numpy() + net(tf.image.flip_left_right(x), training=False).numpy()
        out.append(e)
        if i % 20 == 0:
            print(f"  embeddings: {min((i + 1) * 64, len(paths))}/{len(paths)}", flush=True)
    E = np.concatenate(out).astype(np.float32)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def roles(meta):
    """image_id -> dict evaluation -> role."""
    r = {i: {} for i in meta.image_id}
    led = pd.read_csv(os.path.join("resultados_split_final_v2", "split_ledger.csv"))
    for i, p in zip(led.image_id, led.partition):
        r.setdefault(i, {})["s2026"] = p
    for s in range(3001, 3006):
        f = os.path.join("resultados_v3", f"split_{s}", "split_ledger.csv")
        if os.path.exists(f):
            l = pd.read_csv(f)
            for i, p in zip(l.image_id, l.partition):
                r.setdefault(i, {})[f"s{s}"] = p
    f = os.path.join("confirmacion", "manifest_confirmacion.csv")
    if os.path.exists(f):
        for i in pd.read_csv(f).image_id:
            r.setdefault(i, {})["cohort"] = "cohort"
    return r


def crosses(ra, rb):
    """True if the pair puts one image in training/validation and the other in a test set of the same evaluation."""
    out = []
    for k in set(ra) | set(rb):
        if k == "cohort":
            continue
        a, b = ra.get(k), rb.get(k)
        if a and b and a != b and "test" in (a, b):
            out.append(k)
    # cohort is evaluated with models trained on the train/val partitions of split 2026
    for x, y in ((ra, rb), (rb, ra)):
        if x.get("cohort") and y.get("s2026") in ("train", "val"):
            out.append("cohort")
    return sorted(set(out))


def describe(rr):
    if not rr:
        return "not used"
    return "; ".join(f"{k}:{v}" for k, v in sorted(rr.items()))


def contact_sheet(pairs, path_of, fname, n=60):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    pairs = pairs.head(n); rows = int(np.ceil(len(pairs) / 3))
    fig, axs = plt.subplots(rows, 6, figsize=(18, 2.6 * rows))
    axs = np.atleast_2d(axs)
    for k, (_, p) in enumerate(pairs.iterrows()):
        r, c = divmod(k, 3)
        for j, col in enumerate(("image_a", "image_b")):
            ax = axs[r, 2 * c + j]
            ax.imshow(Image.open(path_of[p[col]]).convert("RGB")); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"#{k + 1}  {p[col]}\n{p['lesion_' + col[-1]]} {p['dx_' + col[-1]]}", fontsize=7)
        axs[r, 2 * c].set_ylabel(f"sim {p.similarity:.3f}", fontsize=8)
    for ax in axs.flat[len(pairs) * 2:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--ham_dirs", required=True)
    ap.add_argument("--top", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    meta = pd.read_csv(a.metadata)
    idx = index_images([d.strip() for d in a.ham_dirs.split(",") if d.strip()])
    miss = [i for i in meta.image_id if i not in idx]
    if miss:
        raise SystemExit(f"{len(miss)} images not found, e.g. {miss[:3]}")
    ids = meta.image_id.values; les = meta.lesion_id.values; dx = meta.dx.values
    print(f"{len(ids)} images; computing embeddings (several minutes on CPU)...", flush=True)
    E = embeddings([idx[i] for i in ids])
    np.save(os.path.join(OUT, "embeddings.npy"), E)
    # top pairs with different lesion_id, computed in blocks
    cand = []
    for s in range(0, len(ids), 1000):
        S = E[s:s + 1000] @ E.T
        for r in range(S.shape[0]):
            i = s + r
            S[r, :i + 1] = -1                     # each pair once
            S[r, les == les[i]] = -1              # same lesion: already linked
        k = min(a.top, S.size)
        flat = np.argpartition(S.ravel(), -k)[-k:]
        rr, cc = np.unravel_index(flat, S.shape)
        cand += [(s + x, y, S[x, y]) for x, y in zip(rr, cc)]
    cand = sorted(cand, key=lambda t: -t[2])[:a.top]
    R = roles(meta)
    rows = []
    for i, j, sim in cand:
        cr = crosses(R[ids[i]], R[ids[j]])
        rows.append({"rank": len(rows) + 1, "similarity": round(float(sim), 4),
                     "image_a": ids[i], "lesion_a": les[i], "dx_a": dx[i], "use_a": describe(R[ids[i]]),
                     "image_b": ids[j], "lesion_b": les[j], "dx_b": dx[j], "use_b": describe(R[ids[j]]),
                     "same_dx": dx[i] == dx[j], "crosses_train_test_in": ",".join(cr)})
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(OUT, "pares_top1000.csv"), index=False)
    contact_sheet(P, idx, os.path.join(OUT, "pares_revisar.png"))
    with open(os.path.join(OUT, "resumen.txt"), "w", encoding="utf-8") as f:
        for n in (20, 60, 200, 1000):
            sub = P.head(n)
            f.write(f"top {n}: similarity >= {sub.similarity.min():.3f}; "
                    f"pairs crossing a train/test boundary: {(sub.crosses_train_test_in != '').sum()}\n")
    print(open(os.path.join(OUT, "resumen.txt"), encoding="utf-8").read())
    print(f"Saved in {OUT}/")


if __name__ == "__main__":
    main()
