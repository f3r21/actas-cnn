"""Extraccion de recortes de digitos desde imagenes de actas.

ESQUELETO: las coordenadas de los campos dependen de la plantilla real del
acta, asi que hay que calibrarlas con una muestra (ver TODO al final). Flujo:
  1. (opcional) alinear la pagina a una plantilla de referencia
  2. recortar cada campo numerico segun coordenadas relativas
  3. segmentar el campo en digitos individuales

Las coordenadas van en fraccion [0, 1] del ancho/alto para tolerar diferencias
de DPI y de escaneo entre actas.
"""
import json

import numpy as np
from PIL import Image


def load_templates(path="templates.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def crop_fields(image_path, template):
    """Recorta los campos numericos definidos en la plantilla.

    template["fields"]: lista de {name, box:[x0,y0,x1,y1], n_digits} con box en
    fraccion [0, 1]. Devuelve un dict name -> PIL.Image (escala de grises).
    """
    img = Image.open(image_path).convert("L")
    w, h = img.size
    crops = {}
    for field in template["fields"]:
        x0, y0, x1, y1 = field["box"]
        box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
        crops[field["name"]] = img.crop(box)
    return crops


def split_digits(field_img, n_digits):
    """Divide un campo en n_digits recortes iguales (segmentacion equiespaciada).

    Nota: probamos projection profile vertical (Sem 2 dia 2) para acomodar
    escritura irregular, pero la afin del template ya alinea bien los campos
    en >98% de actas y el equiespaciado da mejor accuracy downstream
    (acta-level 90.33% vs 89.61% con projection profile). Ver
    docs/05-backlog.md: experimento de mejora de preprocesamiento.
    """
    w, h = field_img.size
    step = w / n_digits
    return [field_img.crop((int(i * step), 0, int((i + 1) * step), h))
            for i in range(n_digits)]


def to_array(digit_img, size=32):
    """Normaliza un recorte de digito a un arreglo cuadrado uint8."""
    return np.asarray(digit_img.resize((size, size)), dtype=np.uint8)


def es_celda_escrita(value, n_cells, cell_position):
    """Indica si la celda deberia contener un digito escrito a mano.

    Convencion ONPE: numeros right-justified, leading zeros se dejan en blanco.
    - value=0     -> todas las celdas vacias (nadie escribe "000")
    - value=5     -> en 3 celdas: [vacio, vacio, "5"] -> cell 0,1 empty; cell 2 written
    - value=18    -> [vacio, "1", "8"]   -> cell 0 empty; cells 1,2 written
    - value=144   -> ["1", "4", "4"]      -> todas escritas
    - value=20    -> [vacio, "2", "0"]    -> cell 0 empty; cells 1,2 written (el "0" SI se escribe)

    Esta funcion es la fuente de verdad para filtrar vacias vs escritas en
    entrenamiento: las truly-empty no se guardan como crops; las escritas
    (incluyendo "0" escritos) si entran al training set.
    """
    if value == 0:
        return False
    num_digits = len(str(int(value)))
    first_written = n_cells - num_digits
    return cell_position >= first_written


def tiene_tinta(digit_img, fraccion_minima=0.02, intensidad_max=180):
    """Detector de tinta image-based. Util como sanity-check, no para filtro
    principal de training. La separacion empty vs escrito se hace mejor con
    es_celda_escrita() porque es deterministica desde el label.

    Cuenta la fraccion de pixeles oscuros (< intensidad_max).
    """
    arr = np.asarray(digit_img.convert("L"), dtype=np.uint8)
    oscuros = (arr < intensidad_max).sum()
    return oscuros / arr.size > fraccion_minima


# TODO calibracion (hacer una vez con datos reales):
#   1. Renderiza unas pocas actas con pdf_to_images.
#   2. Abre una imagen, ubica las cajas de los campos de conteo y anota sus
#      coordenadas relativas en templates.json (una entrada por plantilla).
#   3. Ajusta n_digits y, si hace falta, split_digits segun cada formato.
