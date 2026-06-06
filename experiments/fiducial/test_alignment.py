"""Aplica registro afin via fiducials a las worst-20 actas del audit.

Para cada acta:
  1. Detecta 15 markers
  2. Computa afin que mapea detectados -> canonicos
  3. Aplica afin INVERSA al template (mueve las 42 cajas)
  4. Genera crops con template transformado
  5. Compara accuracy vs ground truth

Compara la accuracy con/sin alineacion. Output: AUDIT_ALIGNMENT.md y
visualizaciones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]

from detect_fiducials import detect_15
from actas_cnn.data import default_transforms
from actas_cnn.env import torch_device
from actas_cnn.model import build_model
from actas_cnn.preprocess.crops import (crop_fields, es_celda_escrita,
                                        field_value_for, int_to_digits,
                                        load_templates, split_digits)


def transform_template(template: dict, src_markers: dict, dst_anchors: dict,
                        img_size: tuple[int, int]) -> dict:
    """Devuelve nuevo template con los boxes movidos via afin src->dst.

    src_markers: markers detectados en esta acta (dict role -> (x,y))
    dst_anchors: anchors canonicos (dict role -> (x,y))
    img_size: (w, h) de la acta nueva
    """
    common = sorted(set(src_markers) & set(dst_anchors))
    if len(common) < 4:
        return template  # sin suficientes puntos, no transformar
    src = np.array([src_markers[r] for r in common], dtype=np.float32)
    dst = np.array([dst_anchors[r] for r in common], dtype=np.float32)
    # afin que mapea src -> dst
    M, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                  ransacReprojThreshold=8.0)
    if M is None:
        return template
    # Aplicar M^-1 a los boxes (el template esta en espacio dst, lo movemos
    # a espacio src para que coincida con la acta nueva)
    M_inv = cv2.invertAffineTransform(M)

    w, h = img_size
    # Para acta canonica original (donde se calibro el template), usar sus
    # dimensiones de referencia
    ref_w, ref_h = template.get("image_size_reference", [2339, 3309])

    new_fields = []
    for f in template["fields"]:
        x0_f, y0_f, x1_f, y1_f = f["box"]
        # convert fraction -> pixel en espacio canonico
        x0 = x0_f * ref_w; y0 = y0_f * ref_h
        x1 = x1_f * ref_w; y1 = y1_f * ref_h
        # afin inversa: punto canonico -> punto en la acta nueva
        pts = np.array([[x0, y0], [x1, y1]], dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.transform(pts, M_inv).reshape(-1, 2)
        nx0, ny0 = warped[0]
        nx1, ny1 = warped[1]
        # convert pixel acta nueva -> fraccion
        new_fields.append({
            **f,
            "box": [nx0 / w, ny0 / h, nx1 / w, ny1 / h],
        })
    return {**template, "fields": new_fields}


def predict_acta(png_path: Path, idActa: int, template: dict,
                  archivos, votos, cabecera, model, device) -> tuple[int, int]:
    """Devuelve (correct, total) digits sobre esta acta con este template."""
    votos_acta = votos[votos["idActa"] == idActa]
    cab_row = cabecera[cabecera["idActa"] == idActa].iloc[0]
    total_emitidos_raw = cab_row["totalVotosEmitidos"]
    if pd.isna(total_emitidos_raw):
        return 0, 0
    total_emitidos = int(total_emitidos_raw)

    fields_crops = crop_fields(png_path, template)
    correct = 0
    total = 0
    tx = default_transforms(32, train=False)
    for field in template["fields"]:
        name = field["name"]
        n = field["n_digits"]
        value = field_value_for(name, votos_acta, total_emitidos)
        labels = int_to_digits(value, n)
        digits = split_digits(fields_crops[name], n)
        for pos, (lbl, dimg) in enumerate(zip(labels, digits)):
            if not es_celda_escrita(value, n, pos):
                continue
            x = tx(dimg).unsqueeze(0).to(device)
            pred = model(x).argmax(1).item()
            if pred == lbl:
                correct += 1
            total += 1
    return correct, total


def main():
    device = torch_device()
    ckpt = torch.load(ROOT / "checkpoints" / "deep_best.pt",
                       map_location=device, weights_only=False)
    model = build_model(ckpt.get("arch", "deep"), 1, 10).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    template = load_templates(ROOT / "templates.json")["presidencial"]
    anchors_raw = json.loads((ROOT / "fiducial_anchors.json").read_text())
    anchors = {k: tuple(v) for k, v in anchors_raw.items()}

    archivos = pd.read_parquet(ROOT / "data/labels/actas_archivos.parquet")
    votos = pd.read_parquet(ROOT / "data/labels/actas_votos.parquet")
    cabecera = pd.read_parquet(ROOT / "data/labels/actas_cabecera.parquet")

    # Las 20 peores del audit previo (archive/auditorias/template-generalizacion.md)
    worst_20 = [
        "69e22147d7b6147f63ec8db0", "69e47a45bbc459e6486a81f3",
        "69e291bdd7b6147f63ecbc8f", "69dc54cfd7b6147f63e5afd2",
        "69e2c744d7b6147f63ecf91d", "69dc7271d7b6147f63e5f6cd",
        "69e0aafcd7b6147f63eb1c0d", "69e029abd7b6147f63ea0ee0",
        "69df62efd7b6147f63e8e47f", "69e41cdfbd301579eab812d5",
        "69e045ded7b6147f63ea4b3d", "69e00779d7b6147f63e9cbb3",
        "69e04df3d7b6147f63ea5e9e", "69dfb3bad7b6147f63e971b4",
        "69e039a0d7b6147f63ea3086", "69e00a31d7b6147f63e9d2f3",
        "69e22266d7b6147f63ec8e00", "69e05094d7b6147f63ea649b",
        "69e10cc3d7b6147f63ebe7fb", "69e0ad37d7b6147f63eb2161",
    ]

    print(f"{'aid':<28} {'#markers':>10} {'acc_old':>9} {'acc_new':>9} delta")
    print("-" * 80)
    rendered_dir = ROOT / "data/pdfs_train/rendered"
    deltas = []
    for aid in worst_20:
        png = rendered_dir / f"{aid}_p0.png"
        if not png.exists():
            continue
        arc = archivos[archivos["archivoId"] == aid]
        if len(arc) == 0: continue
        idActa = int(arc.iloc[0]["idActa"])

        markers = detect_15(png)

        # baseline: sin alineacion
        c_old, t_old = predict_acta(png, idActa, template,
                                     archivos, votos, cabecera, model, device)
        # con alineacion
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        new_template = transform_template(template, markers, anchors,
                                            (img.shape[1], img.shape[0]))
        c_new, t_new = predict_acta(png, idActa, new_template,
                                     archivos, votos, cabecera, model, device)

        acc_old = c_old / max(t_old, 1)
        acc_new = c_new / max(t_new, 1)
        delta = acc_new - acc_old
        deltas.append(delta)
        print(f"{aid:<28} {len(markers):>10} {acc_old:>9.3f} {acc_new:>9.3f}  {delta:+.3f}")

    print(f"\nMejora promedia: {np.mean(deltas):+.3f}")
    print(f"Mejora mediana:  {np.median(deltas):+.3f}")
    print(f"Empeoraron:      {sum(1 for d in deltas if d < -0.01)}/{len(deltas)}")
    print(f"Igual (~0):      {sum(1 for d in deltas if abs(d) <= 0.01)}/{len(deltas)}")
    print(f"Mejoraron:       {sum(1 for d in deltas if d > 0.01)}/{len(deltas)}")


if __name__ == "__main__":
    main()
