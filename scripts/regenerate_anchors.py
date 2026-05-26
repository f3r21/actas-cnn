"""Regenera fiducial_anchors.json como mediana de N actas con 15/15 detectados.

Los anchors actuales vienen de UNA acta snapshot (R2 de
docs/auditorias/fiducial-auditoria.md). Esto los hace idiosincraticos. Para
search-by-prior queremos posiciones representativas: la mediana sobre N
actas tomadas al azar (con detector zonal actual confirmado 15/15) es
robusta a outliers y captura el centro del patron real.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.detect_fiducials import detect_15


ROLES = ["TL", "T1", "T2", "T3", "TR",
         "L1", "L2", "L3", "L4",
         "R1", "R2", "R3", "R4",
         "BL", "BR"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-csv",
                    default=ROOT / "data/audit_fiducial_val.csv",
                    type=Path)
    ap.add_argument("--rendered-dir",
                    default=ROOT / "data/pdfs_train/rendered",
                    type=Path)
    ap.add_argument("--n", type=int, default=100,
                    help="cuantas actas usar (de las que tengan 15/15)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",
                    default=ROOT / "fiducial_anchors.json",
                    type=Path)
    ap.add_argument("--out-backup",
                    default=ROOT / "fiducial_anchors_v1.json",
                    type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.audit_csv)
    perfect = df[df["n_markers"] == 15]["archivo_id"].tolist()
    print(f"actas con 15/15 disponibles: {len(perfect)}")

    rng = np.random.RandomState(args.seed)
    sample = rng.choice(perfect, size=min(args.n, len(perfect)),
                        replace=False)
    print(f"muestreando {len(sample)} actas (seed={args.seed})")

    coords = {role: [] for role in ROLES}
    n_ok = 0
    for aid in sample:
        png = args.rendered_dir / f"{aid}_p0.png"
        if not png.exists():
            continue
        try:
            markers = detect_15(png)
        except Exception as e:
            print(f"  error en {aid}: {e}")
            continue
        if len(markers) != 15:
            continue
        for role in ROLES:
            coords[role].append(markers[role])
        n_ok += 1

    print(f"\\nactas procesadas con 15/15: {n_ok}")

    # Backup original
    if args.out.exists() and not args.out_backup.exists():
        args.out_backup.write_text(args.out.read_text())
        print(f"backup del v1: {args.out_backup.name}")

    # Mediana por rol
    new_anchors = {}
    for role in ROLES:
        if not coords[role]:
            print(f"  {role}: SIN DATOS")
            continue
        arr = np.array(coords[role])
        med_x = int(np.median(arr[:, 0]))
        med_y = int(np.median(arr[:, 1]))
        new_anchors[role] = [med_x, med_y]
        std_x = float(arr[:, 0].std())
        std_y = float(arr[:, 1].std())
        print(f"  {role}: ({med_x:4d}, {med_y:4d})  std ({std_x:5.1f}, {std_y:5.1f})")

    args.out.write_text(json.dumps(new_anchors, indent=2))
    print(f"\\nanchors -> {args.out}")


if __name__ == "__main__":
    main()
