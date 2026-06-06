"""Interfaz de localizacion de digitos (la superficie que mas se itera).

La pregunta central del preprocesamiento es: *dado el PNG de un acta, donde
estan los digitos?* Esta interfaz aisla esa decision para poder cambiar el
metodo de deteccion (zonal por plantilla hoy; detectores aprendidos o por
contornos manana) sin tocar render / entrenamiento / evaluacion.

Un `DigitLocalizer` recibe la imagen de un acta y la plantilla del formato, y
devuelve, por cada campo, la lista ordenada de recortes de celda (uno por
digito potencial, izquierda->derecha). El resto del pipeline (labels,
filtrado de vacias, guardado de crops) es agnostico al metodo concreto.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image


@runtime_checkable
class DigitLocalizer(Protocol):
    """Localiza las celdas de digitos de cada campo numerico de un acta.

    Implementaciones conocidas:
      - `template_zonal.TemplateZonalLocalizer`: oficial. Recorta cada campo por
        las cajas de la plantilla calibrada y lo parte en n_digits celdas
        equiespaciadas.
      - `experiments/fiducial/`: alternativa que alinea por marcadores fiduciales
        antes de recortar (experimento negativo, -0.72pp acta-level).
    """

    def localize(
        self, image_path: str | Path, template: dict
    ) -> dict[str, list[Image.Image]]:
        """Devuelve {nombre_campo: [celda_0, celda_1, ...]} en escala de grises.

        El largo de la lista de cada campo debe ser su `n_digits`. Las celdas
        van ordenadas de izquierda a derecha (la convencion right-justified de
        ONPE se resuelve despues, con los labels).
        """
        ...
