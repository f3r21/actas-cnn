#!/bin/bash
# Pipeline limpio: 5,000 manuscritas puras (sin STAE).
# 1. Split por archivoId 70/15/15 sobre manuscritas_full.txt
# 2. Build crops por split (filtra celdas vacias label-based)
# 3. Genera manifests
# 4. Smoke train DeepCNN 5 epochs
set -e
cd "$(dirname "$0")/.."

echo "=== split por archivoId sobre manuscritas_full.txt ==="
python3 scripts/split_dataset.py --ids-file data/splits/manuscritas_full.txt

echo ""
echo "=== build crops por split (manuscritas puras) ==="
# limpiar crops viejos para no mezclar STAE
rm -rf data/crops_train data/crops_val data/crops_test
for split in train val test; do
  echo "--- $split ---"
  python3 scripts/build_crops.py \
    --rendered-dir data/pdfs_train/rendered \
    --out-crops data/crops \
    --split "$split" \
    --ids-file "data/splits/${split}_ids.txt"
done

echo ""
echo "=== regenerar manifests ==="
for split in train val test; do
  python3 build_dataset.py \
    --crops "data/crops_${split}" \
    --out "data/manifest_${split}.csv"
done

echo ""
echo "=== distribucion por clase (post-clean) ==="
python3 - <<'PY'
import pandas as pd
for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data/manifest_{split}.csv")
    print(f"\n{split} (n={len(df):,}):")
    print(df["label"].value_counts().sort_index().to_string())
PY

echo ""
echo "=== smoke train DeepCNN 5 epochs (numeros honestos) ==="
python3 train.py --manifest data/manifest_train.csv --root data/crops_train --arch deep --epochs 5
