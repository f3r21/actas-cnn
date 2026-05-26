"""Genera y visualiza los crops de digitos de una acta para validar el template.

Usa extract_crops.crop_fields + split_digits sobre un PNG de acta renderizada y
arma una grilla con todos los digitos individuales para inspeccion visual.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_crops import crop_fields, load_templates, split_digits


def _fuente(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", size)
    except OSError:
        return ImageFont.load_default()


def grilla_digitos(
    field_crops: dict[str, Image.Image],
    template: dict,
    cell_w: int = 80,
    cell_h: int = 120,
    pad: int = 6,
) -> Image.Image:
    """Arma una grilla: 1 fila por field, N columnas por digito."""
    fields = template["fields"]
    max_digits = max(f.get("n_digits", 1) for f in fields)
    n_rows = len(fields)
    label_w = 240  # ancho para el nombre del field a la izquierda

    grid_w = label_w + (cell_w + pad) * max_digits + pad
    grid_h = (cell_h + pad) * n_rows + pad

    out = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    font = _fuente(12)

    for r, field in enumerate(fields):
        name = field["name"]
        n = field.get("n_digits", 1)
        img = field_crops[name]
        digits = split_digits(img, n)

        y = pad + r * (cell_h + pad)
        draw.text((4, y + cell_h // 2 - 8), name, fill=(0, 0, 0), font=font)

        for d, dimg in enumerate(digits):
            x = label_w + d * (cell_w + pad)
            # encajar el digit crop dentro de cell_w x cell_h
            ar = dimg.size[0] / dimg.size[1] if dimg.size[1] > 0 else 1
            new_h = cell_h
            new_w = int(new_h * ar)
            if new_w > cell_w:
                new_w = cell_w
                new_h = int(new_w / ar) if ar > 0 else cell_h
            resized = dimg.resize((new_w, new_h))
            paste_x = x + (cell_w - new_w) // 2
            paste_y = y + (cell_h - new_h) // 2
            out.paste(resized, (paste_x, paste_y))
            draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(180, 180, 180), width=1)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--templates", default=Path("templates.json"), type=Path)
    ap.add_argument("--key", required=True, help="clave de plantilla")
    ap.add_argument("--out-grid", required=True, type=Path, help="PNG de la grilla")
    args = ap.parse_args()

    template = load_templates(args.templates)[args.key]
    crops = crop_fields(args.image, template)
    grid = grilla_digitos(crops, template)
    args.out_grid.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out_grid)
    print(args.out_grid)


if __name__ == "__main__":
    main()
