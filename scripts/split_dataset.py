"""Split de archivoIds en train/val/test por ACTA, no por crop random.

Critico para evitar leak: si splits son por crop, crops de la misma acta caen
en train y val/test, lo que infla val accuracy. Splits por archivoId garantizan
que cada acta entera va a un solo split.

Defaults: 70/15/15 reproducible con seed=42.

Input: --ids-file con un archivoId por linea (lo que genera el sampling de
data/splits/sample_5000_ids.txt).
Output: data/splits/{train,val,test}_ids.txt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def split(ids: list[str], train: float, val: float, test: float, seed: int) -> dict[str, list[str]]:
    if abs(train + val + test - 1.0) > 1e-9:
        raise ValueError(f"fracciones deben sumar 1: {train}+{val}+{test}")
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val = int(n * val)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("data/splits"), type=Path)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ids = [line.strip() for line in args.ids_file.read_text().splitlines() if line.strip()]
    splits = split(ids, args.train, args.val, args.test, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        path = args.out_dir / f"{name}_ids.txt"
        path.write_text("\n".join(items) + "\n")
        print(f"  {name}: {len(items)} ids -> {path}")


if __name__ == "__main__":
    main()
