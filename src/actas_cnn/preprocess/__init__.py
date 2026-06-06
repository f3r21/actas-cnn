"""Preprocesamiento: de PNG de acta a recortes de digitos etiquetados.

*** Esta es la superficie que mas se itera: la forma de detectar donde estan
los digitos en el acta. *** El metodo oficial es zonal por plantilla
(`localize_digits`). Para cambiar la deteccion, reemplaza `localize_digits` o
pasa otro callable `(png, template) -> {campo: [celdas]}` a
`build_crops_for_acta`. El localizador fiducial alternativo vive en
`experiments/fiducial/`.
"""
from .crops import (
    build_crops_for_acta,
    crop_fields,
    es_celda_escrita,
    field_value_for,
    int_to_digits,
    load_templates,
    localize_digits,
    split_digits,
    tiene_tinta,
)

__all__ = [
    "localize_digits",
    "build_crops_for_acta",
    "crop_fields",
    "split_digits",
    "es_celda_escrita",
    "tiene_tinta",
    "int_to_digits",
    "field_value_for",
    "load_templates",
]
