"""Visualizador de templates.json sobre una imagen de acta.

Dibuja las cajas y la division en n_digits para verificar que las coordenadas
calzan con el layout real. Sin emojis, sin prints superfluos.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def cargar_template(path: Path, key: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if key not in data:
        raise KeyError(f"clave {key!r} no esta en {path} (disponibles: {list(data)})")
    return data[key]


def _fuente(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", size)
    except OSError:
        return ImageFont.load_default()


def dibujar_overlay(img: Image.Image, template: dict) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    font = _fuente(14)

    for field in template["fields"]:
        x0_f, y0_f, x1_f, y1_f = field["box"]
        x0, y0 = int(x0_f * w), int(y0_f * h)
        x1, y1 = int(x1_f * w), int(y1_f * h)

        # caja del campo (rojo semitransparente)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=3)

        # divisiones por digito (verde)
        n = field.get("n_digits", 1)
        if n > 1:
            step = (x1 - x0) / n
            for i in range(1, n):
                gx = int(x0 + i * step)
                draw.line([(gx, y0), (gx, y1)], fill=(0, 200, 0, 200), width=1)

        # etiqueta del campo
        draw.text((x0 + 2, y0 - 16), field["name"], fill=(0, 0, 255, 255), font=font)
    return out


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
