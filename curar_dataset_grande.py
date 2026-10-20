"""
curar_dataset_grande.py

Arma un subset balanceado mas grande (750-1000 por clase, configurable)
desde el HAM10000 COMPLETO, en vez de las 200 imagenes originales.

Reglas de seleccion (documentadas para el Methods del articulo):
    - Maligno: todas las imagenes de mel, bcc, akiec disponibles (hasta
      el limite pedido). Estas 3 categorias son las mismas que ya usaba
      el diseno original.
    - Benigno: muestreo aleatorio (semilla fija) de nv, bkl, df, vasc,
      hasta igualar el numero de malignas seleccionadas.
    - Se conserva el lesion_id de cada imagen desde el metadata oficial
      (no hay que recuperarlo despues como con las 200 originales).
    - Si una lesion tiene multiples imagenes en HAM10000 y mas de una
      cae en la seleccion, todas se conservan CON el mismo lesion_id
      (para que el split agrupado las mantenga juntas).

Uso:
    python curar_dataset_grande.py ^
        --ham10000_images_dir "C:\\ruta\\a\\HAM10000_images" ^
        --metadata_oficial HAM10000_metadata.csv ^
        --n_per_class 1000 ^
        --out_dir "C:\\CASEIB2026_proyecto\\datos_piel\\HAM_subset_grande" ^
        --manifest_out subset_grande_con_lesion_id.csv

Nota: HAM10000 distribuye las imagenes en dos carpetas
(HAM10000_images_part_1 y part_2); si es tu caso, junta ambas en una
sola carpeta antes de correr esto, o pasa --ham10000_images_dir dos
veces separado por coma.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

MALIGNANT_DX = {"mel", "bcc", "akiec"}
BENIGN_DX = {"nv", "bkl", "df", "vasc"}
SEED = 2026


def find_image_file(image_id, search_dirs):
    for d in search_dirs:
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(d, image_id + ext)
            if os.path.exists(candidate):
                return candidate
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ham10000_images_dir", required=True,
                     help="Una o mas carpetas (separadas por coma) con las imagenes .jpg de HAM10000")
    ap.add_argument("--metadata_oficial", required=True)
    ap.add_argument("--n_per_class", type=int, default=1000,
                     help="Imagenes por clase a seleccionar (default 1000 -> ~2000 total)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--manifest_out", required=True)
    args = ap.parse_args()

    search_dirs = [d.strip() for d in args.ham10000_images_dir.split(",")]

    meta = pd.read_csv(args.metadata_oficial)
    print(f"Metadata oficial: {len(meta)} imagenes totales")
    print(meta["dx"].value_counts())

    malignant_pool = meta[meta["dx"].isin(MALIGNANT_DX)].copy()
    benign_pool = meta[meta["dx"].isin(BENIGN_DX)].copy()

    n_malignant_available = len(malignant_pool)
    n_target = min(args.n_per_class, n_malignant_available)
    if args.n_per_class > n_malignant_available:
        print(f"[aviso] Pediste {args.n_per_class} malignas, pero solo hay "
              f"{n_malignant_available} disponibles en total. Se usaran todas las malignas.")

    rng = np.random.default_rng(SEED)

    if len(malignant_pool) > n_target:
        malignant_sel = malignant_pool.sample(n=n_target, random_state=SEED)
    else:
        malignant_sel = malignant_pool

    benign_sel = benign_pool.sample(n=n_target, random_state=SEED)

    selected = pd.concat([
        malignant_sel.assign(label="malignant"),
        benign_sel.assign(label="benign"),
    ], ignore_index=True)

    os.makedirs(os.path.join(args.out_dir, "BENIGNAS"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "MALIGNAS"), exist_ok=True)

    rows = []
    n_missing = 0
    for _, row in selected.iterrows():
        image_id = row["image_id"]
        src = find_image_file(image_id, search_dirs)
        if src is None:
            n_missing += 1
            continue

        dest_folder = "MALIGNAS" if row["label"] == "malignant" else "BENIGNAS"
        dest = os.path.join(args.out_dir, dest_folder, os.path.basename(src))
        shutil.copy2(src, dest)

        rows.append({
            "filepath": dest,
            "image_id": image_id,
            "lesion_id": row["lesion_id"],
            "dx_official": row["dx"],
            "label_from_folder": row["label"],
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.manifest_out, index=False)

    print(f"\nImagenes copiadas: {len(manifest)}  (faltantes/no encontradas: {n_missing})")
    print(manifest["label_from_folder"].value_counts())
    print(f"Lesiones base distintas: {manifest['lesion_id'].nunique()} "
          f"(de {len(manifest)} imagenes)")
    dup = manifest["lesion_id"].value_counts()
    dup = dup[dup > 1]
    print(f"Lesiones con mas de una imagen en el subset: {len(dup)}")
    print(f"\nManifest guardado en: {args.manifest_out}")
    print(f"Imagenes copiadas a: {args.out_dir}")


if __name__ == "__main__":
    main()
