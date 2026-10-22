"""
seleccionar_capsnet.py

Gate G6 of the revision protocol: validation-only selection of a revised
CapsNet-R3 configuration. Reads the grid runs trained with
entrenar_v3.py --no_test (split 2026, seed 2027) and applies the pre-specified
rule:

  1. Candidates: the six grid configurations (learning rate 1e-4, 3e-5, 1e-5
     x input normalization none / imagenet) whose learning rate passed the
     memorization gate G4 (diagnostico_capsnet/G4_lrs_aprobadas.json).
  2. Choose the highest validation AUC; ties (difference < 0.005) are broken
     by the lower best validation loss.
  3. The selected configuration is accepted as "functioning" only if its
     validation AUC is >= 0.60. Otherwise no revised configuration is carried
     forward and the CapsNet results are reported as a failure analysis.

The test partition is not used. Output: diagnostico_capsnet/capsnet_seleccion.json

Usage:  python seleccionar_capsnet.py
"""

import glob
import json
import os

import pandas as pd

GRID = [(lr, norm) for lr in ("1e-4", "3e-5", "1e-5") for norm in ("none", "imagenet")]
MIN_VAL_AUC = 0.60
TIE = 0.005


def main():
    with open(os.path.join("diagnostico_capsnet", "G4_lrs_aprobadas.json")) as f:
        g4 = {float(k): v for k, v in json.load(f).items()}
    rows = []
    for lr, norm in GRID:
        path = os.path.join("resultados_v3", "split_2026", f"resultados_capsnet_grid_lr{lr}_{norm}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"Missing grid run: {path}")
        r = pd.read_csv(path).iloc[0]
        rows.append({"lr": float(lr), "input_norm": norm, "val_auc": r.val_auc,
                     "best_val_loss": r.best_val_loss, "best_epoch": r.best_epoch,
                     "n_epochs_run": r.n_epochs_run, "passed_G4": bool(g4.get(float(lr), False))})
    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False))
    cand = grid[grid.passed_G4]
    result = {"selected": False, "rule": "max validation AUC among G4-passing lrs; tie<0.005 -> lower "
              "val loss; accept if val AUC >= 0.60", "grid": rows}
    if len(cand):
        top = cand.val_auc.max()
        best = cand[cand.val_auc >= top - TIE].sort_values("best_val_loss").iloc[0]
        result.update({"lr": best.lr, "input_norm": best.input_norm, "val_auc": float(best.val_auc)})
        result["selected"] = bool(best.val_auc >= MIN_VAL_AUC)
    with open(os.path.join("diagnostico_capsnet", "capsnet_seleccion.json"), "w") as f:
        json.dump(result, f, indent=1, default=float)
    if result["selected"]:
        print(f"\nSelected: lr={result['lr']:.0e}, input_norm={result['input_norm']} "
              f"(val AUC {result['val_auc']:.3f}). It will be trained as capsnet_r3_revised.")
    else:
        print("\nNo functioning revised configuration (no G4-passing lr, or val AUC < 0.60). "
              "capsnet_r3_revised will be skipped; report the failure analysis.")


if __name__ == "__main__":
    main()
