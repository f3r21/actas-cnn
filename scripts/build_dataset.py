"""CLI: construye el manifiesto del dataset (path,label) desde crops/<label>/.

Wrapper de actas_cnn.data.build_manifest. Opcional --push para subir con
redundancia (actas_cnn.storage).

Uso:
  python scripts/build_dataset.py --crops data/crops_train --out data/manifest_train.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actas_cnn import storage
from actas_cnn.data import build_manifest


def main() -> None:
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
