"""Audit del detector fiducial (`scripts/detect_fiducials.py`) sobre val.

Mide por cada PNG del split val:
  - cuantos markers de los 15 detecto
  - cuales roles encontro / cuales falto
  - si la deteccion supera el umbral de 4 (que es el minimo para aplicar
    afin en build_crops.py)
  - per-acta accuracy del CNN con el checkpoint actual

Salida:
  - data/audit_fiducial_val.csv (una fila por acta val)
  - histograma de markers detectados
  - grids visuales: 10 actas con <4 markers (donde alineacion no se aplica),
    con los markers detectados dibujados sobre la imagen, para diagnosticar
    si los markers fisicos existen pero el detector los pierde.

La pregunta clave: de las actas donde el detector falla, cuantas tienen
markers FISICAMENTE presentes en el scan? Si la mayoria si tiene markers,
el problema es el detector. Si no tienen markers visibles, el problema es
la imagen.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.detect_fiducials import detect_15, draw_overlay
from dataset import CropsDataset, default_transforms
from model import build_model
from env import torch_device


ROLES = ["TL", "T1", "T2", "T3", "TR",
         "L1", "L2", "L3", "L4",
         "R1", "R2", "R3", "R4",
         "BL", "BR"]


def per_acta_accuracy(manifest_csv: Path, crops_root: Path,
                       checkpoint: Path) -> dict[str, tuple[int, int]]:
    """Devuelve {archivo_id: (correct, total)} con el checkpoint actual."""
    import pandas as pd
    device = torch_device()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model("deep").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    df = pd.read_csv(manifest_csv)
    df["archivo_id"] = df["path"].apply(lambda p: p.split("/")[1].split("_")[0])

    ds = CropsDataset(manifest_csv, root=str(crops_root),
                       transform=default_transforms(32, train=False))
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    preds = np.zeros(len(df), dtype=np.int64)
    idx = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = model(x).argmax(1).cpu().numpy()
            preds[idx:idx + len(p)] = p
            idx += len(p)
    df["pred"] = preds
    df["correct"] = (df["pred"] == df["label"]).astype(int)
    agg = df.groupby("archivo_id")["correct"].agg(["sum", "count"]).reset_index()
    return {r["archivo_id"]: (int(r["sum"]), int(r["count"])) for _, r in agg.iterrows()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rendered-dir",
                    default=Path("data/pdfs_train/rendered"), type=Path)
    ap.add_argument("--manifest",
                    default=Path("data/manifest_val.csv"), type=Path)
    ap.add_argument("--crops-root", default=Path("data/crops_val"),
                    type=Path)
    ap.add_argument("--checkpoint", default=Path("checkpoints/deep_best.pt"),
                    type=Path)
    ap.add_argument("--ids-file", default=Path("data/splits/val_ids.txt"),
                    type=Path)
    ap.add_argument("--out-csv", default=Path("data/audit_fiducial_val.csv"),
                    type=Path)
    ap.add_argument("--viz-dir",
                    default=Path("data/visualizaciones/audit_fiducial"),
                    type=Path)
    args = ap.parse_args()

    ids = [l.strip() for l in args.ids_file.read_text().splitlines() if l.strip()]
    print(f"corriendo detector sobre {len(ids)} actas val...")

    # Per-acta accuracy
    print("calculando per-acta accuracy con checkpoint actual...")
    per_acc = per_acta_accuracy(args.manifest, args.crops_root, args.checkpoint)

    rows = []
    n_done = 0
    roles_missing = Counter()
    for aid in ids:
        png = args.rendered_dir / f"{aid}_p0.png"
        if not png.exists():
            continue
        markers = detect_15(png)
        n = len(markers)
        missing = [r for r in ROLES if r not in markers]
        for m in missing:
            roles_missing[m] += 1
        correct, total = per_acc.get(aid, (0, 0))
        acc = correct / total if total else None
        rows.append({
            "archivo_id": aid,
            "n_markers": n,
            "afin_aplicable": int(n >= 4),
            "missing_roles": "|".join(missing),
            "cnn_correct": correct,
            "cnn_total": total,
            "cnn_acc": f"{acc:.4f}" if acc is not None else "",
        })
        n_done += 1
        if n_done % 200 == 0:
            print(f"  {n_done}/{len(ids)}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV -> {args.out_csv}")

    print()
    print("=" * 60)
    print("RESUMEN DE DETECCION FIDUCIAL")
    print("=" * 60)

    counts = [r["n_markers"] for r in rows]
    accs = [float(r["cnn_acc"]) for r in rows if r["cnn_acc"]]
    afin_ok = sum(r["afin_aplicable"] for r in rows)

    print(f"  actas evaluadas       : {len(rows)}")
    print(f"  marker count distribution (de 15):")
    hist = Counter(counts)
    for k in sorted(hist):
        bar = "#" * int(hist[k] * 50 / len(rows))
        print(f"    {k:2d} markers: {hist[k]:4d}  {bar}")
    print(f"  >=4 markers (afin OK) : {afin_ok}/{len(rows)} = {afin_ok/len(rows):.1%}")
    print(f"  >=12 markers (alta calidad): {sum(1 for c in counts if c>=12)}/{len(rows)}")
    print(f"  ==15 markers (perfecto)    : {sum(1 for c in counts if c==15)}/{len(rows)}")
    print()
    print(f"  roles mas perdidos (de los que faltan):")
    for role, n in roles_missing.most_common():
        print(f"    {role}: missing en {n} actas ({n/len(rows):.1%})")

    # Correlacion: detectados vs accuracy
    print()
    print("CORRELACION marker_count vs cnn_acc:")
    buckets = {(0,3): [], (4,7): [], (8,11): [], (12,14): [], (15,15): []}
    for r in rows:
        if not r["cnn_acc"]:
            continue
        n = r["n_markers"]
        acc = float(r["cnn_acc"])
        for (lo, hi) in buckets:
            if lo <= n <= hi:
                buckets[(lo, hi)].append(acc)
                break
    for (lo, hi), accs in buckets.items():
        if accs:
            print(f"  markers [{lo:2d}-{hi:2d}]: n={len(accs):4d}  "
                  f"mean_acc={np.mean(accs):.3f}  median={np.median(accs):.3f}")
        else:
            print(f"  markers [{lo:2d}-{hi:2d}]: n=0")

    # Actas que fallan deteccion (n < 4) + ordenadas por marker count
    failed = sorted([r for r in rows if r["n_markers"] < 4],
                    key=lambda r: (r["n_markers"], r["archivo_id"]))[:10]
    success = sorted([r for r in rows if r["n_markers"] >= 12],
                     key=lambda r: -r["n_markers"])[:10]

    print()
    print(f"Actas con DETECCION FALLIDA (<4 markers): {sum(1 for r in rows if r['n_markers']<4)} totales")
    print(f"Generando overlays para las 10 peores en {args.viz_dir}/...")
    args.viz_dir.mkdir(parents=True, exist_ok=True)
    for r in failed:
        png = args.rendered_dir / f"{r['archivo_id']}_p0.png"
        markers = detect_15(png)
        out = args.viz_dir / f"fail_{r['n_markers']:02d}m_{r['archivo_id']}.png"
        draw_overlay(png, out, markers)
        acc = r["cnn_acc"] or "-"
        print(f"  {r['archivo_id']}  n={r['n_markers']:2d}  cnn_acc={acc}  -> {out.name}")

    print()
    print(f"Para comparar, 10 actas con DETECCION EXITOSA (>=12 markers):")
    for r in success:
        png = args.rendered_dir / f"{r['archivo_id']}_p0.png"
        markers = detect_15(png)
        out = args.viz_dir / f"ok_{r['n_markers']:02d}m_{r['archivo_id']}.png"
        draw_overlay(png, out, markers)
        acc = r["cnn_acc"] or "-"
        print(f"  {r['archivo_id']}  n={r['n_markers']:2d}  cnn_acc={acc}  -> {out.name}")


if __name__ == "__main__":
    main()
