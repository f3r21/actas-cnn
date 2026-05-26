"""Construye el manifiesto del dataset y, opcional, lo sube con redundancia.

Espera una carpeta de recortes organizada como crops/<label>/<archivo>.png,
donde label es el digito 0-9. Genera manifest.csv (columnas path,label) y lo
sube a todos los backends disponibles.
"""
import argparse
import csv
from pathlib import Path

import storage


def build_manifest(crops_dir, out_csv):
    crops_dir = Path(crops_dir)
    rows = []
    for label_dir in sorted(crops_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for img in sorted(label_dir.glob("*.png")):
            rows.append((str(img.relative_to(crops_dir)), label))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True, help="carpeta crops/<label>/*.png")
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--push", action="store_true", help="subir a los backends")
    args = ap.parse_args()
    n = build_manifest(args.crops, args.out)
    print(f"{n} recortes en {args.out}")
    if args.push:
        storage.upload(args.out, "manifest.csv", kind="dataset")


if __name__ == "__main__":
    main()
