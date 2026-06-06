"""Localizador zonal por plantilla calibrada (metodo OFICIAL del proyecto).

Recorta cada campo por las cajas relativas de la plantilla (`templates.json`)
y lo parte en `n_digits` celdas equiespaciadas. Es el camino con el que se
reportan las metricas oficiales (acta-level 90.33%).

Probamos alternativas (projection profile vertical en split_digits; alineacion
por marcadores fiduciales antes de recortar) y ninguna mejoro el acta-level: la
afin del template ya alinea bien en >98% de actas. El detector fiducial vive en
`experiments/fiducial/` como localizador alternativo (experimento negativo
documentado). Ver docs/05-backlog.md.

Este modulo es la implementacion por defecto de la interfaz `DigitLocalizer`.
Para cambiar "donde estan los digitos", escribe otro localizador con la misma
interfaz (ver `base.py`) y pasalo al pipeline de crops.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .crops import crop_fields, split_digits


class TemplateZonalLocalizer:
    """Implementa `DigitLocalizer` usando solo la plantilla (sin fiduciales)."""

    def localize(
        self, image_path: str | Path, template: dict
    ) -> dict[str, list[Image.Image]]:
        fields_crops = crop_fields(image_path, template)
        out: dict[str, list[Image.Image]] = {}
        for field in template["fields"]:
            name = field["name"]
            out[name] = split_digits(fields_crops[name], field["n_digits"])
        return out
