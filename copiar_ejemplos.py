"""Copies the 12 example images listed in ejemplos_errores.csv (errors shared by the three
main pretrained pipelines on the test partition of the original split) to ejemplos_errores/."""
import argparse, os, shutil
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--ham_dirs", required=True); a = ap.parse_args()
want = set(pd.read_csv("ejemplos_errores.csv").image_id)
os.makedirs("ejemplos_errores", exist_ok=True); found = 0
for d in [x.strip() for x in a.ham_dirs.split(",") if x.strip()]:
    for root, _, files in os.walk(d):
        for f in files:
            if os.path.splitext(f)[0] in want:
                shutil.copy(os.path.join(root, f), "ejemplos_errores"); found += 1; want.discard(os.path.splitext(f)[0])
print(f"{found} images copied to ejemplos_errores/; missing: {sorted(want)}")
