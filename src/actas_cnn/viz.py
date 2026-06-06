"""Visualizaciones del pipeline (overlays de plantilla, grillas de QA).

`dibujar_overlay` lo comparten el preview de plantilla y las auditorias de
generalizacion; vive en el paquete para no acoplar scripts entre si.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def _fuente(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", size)
    except OSError:
        return ImageFont.load_default()


def dibujar_overlay(img: Image.Image, template: dict) -> Image.Image:
    """Dibuja las cajas de los 42 campos y su division en n_digits sobre el acta.

    Rojo = caja del campo, verde = divisiones por digito, azul = nombre. Sirve
    para verificar visualmente que la plantilla calza con el layout real.
    """
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    font = _fuente(14)

    for field in template["fields"]:
        x0_f, y0_f, x1_f, y1_f = field["box"]
        x0, y0 = int(x0_f * w), int(y0_f * h)
        x1, y1 = int(x1_f * w), int(y1_f * h)

        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=3)

        n = field.get("n_digits", 1)
        if n > 1:
            step = (x1 - x0) / n
            for i in range(1, n):
                gx = int(x0 + i * step)
                draw.line([(gx, y0), (gx, y1)], fill=(0, 200, 0, 200), width=1)

        draw.text((x0 + 2, y0 - 16), field["name"], fill=(0, 0, 255, 255), font=font)
    return out
