"""CLI: visualizador de templates.json sobre una imagen de acta.

Dibuja las cajas y la division en n_digits para verificar que las coordenadas
calzan con el layout real. El dibujo vive en actas_cnn.viz.dibujar_overlay.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actas_cnn.viz import dibujar_overlay


def cargar_template(path: Path, key: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if key not in data:
        raise KeyError(f"clave {key!r} no esta en {path} (disponibles: {list(data)})")
    return data[key]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--templates", default=Path("templates.json"), type=Path)
    ap.add_argument("--key", required=True, help="clave de plantilla en templates.json")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    template = cargar_template(args.templates, args.key)
    img = Image.open(args.image)
    overlay = dibujar_overlay(img, template)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
