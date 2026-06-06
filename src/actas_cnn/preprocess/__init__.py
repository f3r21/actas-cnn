"""Preprocesamiento: de PNG de acta a recortes de digitos etiquetados.

*** Esta es la superficie que mas se itera del proyecto: la forma de detectar
donde estan los digitos en el acta. ***

Para cambiar el metodo de deteccion, implementa la interfaz `DigitLocalizer`
(ver `base.py`) y pasa tu localizador a `build_crops_for_acta`. El default es
`TemplateZonalLocalizer` (zonal por plantilla calibrada), el metodo oficial.
"""
from .base import DigitLocalizer
from .crops import (
    build_crops_for_acta,
    crop_fields,
    es_celda_escrita,
    field_value_for,
    int_to_digits,
    load_templates,
    split_digits,
    tiene_tinta,
    to_array,
)
from .template_zonal import TemplateZonalLocalizer

__all__ = [
    "DigitLocalizer",
    "TemplateZonalLocalizer",
    "build_crops_for_acta",
    "crop_fields",
    "split_digits",
    "es_celda_escrita",
    "tiene_tinta",
    "to_array",
    "int_to_digits",
    "field_value_for",
    "load_templates",
]
