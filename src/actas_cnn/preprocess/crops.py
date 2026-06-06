"""Extraccion y etiquetado de recortes de digitos desde imagenes de actas.

Recursos de bajo nivel del preprocesamiento:
  - crop_fields / split_digits   recorte por plantilla + division en celdas
  - localize_digits              "donde estan los digitos" (zonal, oficial)
  - es_celda_escrita / tiene_tinta   filtrado de celdas vacias (convencion ONPE)
  - int_to_digits / field_value_for  labels right-justified desde el ground truth
  - build_crops_for_acta             orquestador: PNG + labels -> crops en disco

Las coordenadas van en fraccion [0, 1] del ancho/alto para tolerar diferencias
de DPI y de escaneo entre actas. Para cambiar el metodo de deteccion, reemplaza
`localize_digits` o pasa otro callable a `build_crops_for_acta`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def localize_digits(image_path, template):
    """Localiza las celdas de digitos de cada campo: {campo: [celda_0, ...]}.

    Metodo OFICIAL (zonal): recorta cada campo por su caja de la plantilla y lo
    parte en n_digits celdas equiespaciadas. Es el punto unico de "donde estan
    los digitos": para cambiar la deteccion, reemplaza esta funcion (o pasa otra
    con la misma firma a build_crops_for_acta). El localizador fiducial
    alternativo (experimento negativo) vive en experiments/fiducial/.
    """
    fields = crop_fields(image_path, template)
    return {f["name"]: split_digits(fields[f["name"]], f["n_digits"])
            for f in template["fields"]}


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


# --- Labels desde el ground truth ONPE ---------------------------------------

N_PARTIDOS = 38
NPOSICION_BLANCO = 80
NPOSICION_NULOS = 81
NPOSICION_IMPUGNADOS = 82


def int_to_digits(value: int, n_cells: int) -> list[int]:
    """Convierte un entero a una lista de N digitos right-justified.

    Ejemplos:
        int_to_digits(18, 3) -> [0, 1, 8]
        int_to_digits(144, 4) -> [0, 1, 4, 4]
        int_to_digits(0, 3) -> [0, 0, 0]
    """
    if value < 0:
        raise ValueError(f"valor negativo: {value}")
    s = str(int(value)).zfill(n_cells)
    if len(s) > n_cells:
        raise ValueError(f"valor {value} no cabe en {n_cells} celdas")
    return [int(c) for c in s]


def field_value_for(name: str, votos_acta: "pd.DataFrame", total_emitidos: int) -> int:
    """Devuelve el entero del ground truth para un field name.

    - partido_NN          -> actas_votos.nvotos donde nposicion=NN
    - votos_blanco/nulos/impugnados -> nposicion 80/81/82
    - total_ciudadanos    -> actas_cabecera.totalVotosEmitidos
    """
    if name.startswith("partido_"):
        pos = int(name.split("_")[1])
        row = votos_acta[votos_acta["nposicion"] == pos]
        return int(row.iloc[0]["nvotos"]) if len(row) else 0
    mapping = {
        "votos_blanco": NPOSICION_BLANCO,
        "votos_nulos": NPOSICION_NULOS,
        "votos_impugnados": NPOSICION_IMPUGNADOS,
    }
    if name in mapping:
        row = votos_acta[votos_acta["nposicion"] == mapping[name]]
        return int(row.iloc[0]["nvotos"]) if len(row) else 0
    if name == "total_ciudadanos":
        return int(total_emitidos)
    raise ValueError(f"field desconocido: {name}")


# --- Orquestador: PNG + labels -> crops en disco -----------------------------

def build_crops_for_acta(
    png_path: "str | Path",
    archivo_id: str,
    id_acta: int,
    template: dict,
    votos: "pd.DataFrame",
    cabecera: "pd.DataFrame",
    crops_root: "str | Path",
    localizer=None,
    filtrar_vacias: bool = True,
) -> tuple[int, int]:
    """Procesa una acta y guarda sus crops. Devuelve (n_guardados, n_filtrados).

    `localizer` (callable png,template -> {campo:[celdas]}) decide donde estan
    los digitos; None usa el zonal `localize_digits` (oficial). Si filtrar_vacias=True las
    celdas sin digito escrito segun el label (es_celda_escrita) NO se guardan:
    resuelve el imbalance (76% de las celdas son vacias y dominarian el training).
    """
    crops_root = Path(crops_root)
    votos_acta = votos[votos["idActa"] == id_acta]
    cab_row = cabecera[cabecera["idActa"] == id_acta]
    if len(cab_row) == 0:
        return 0, 0
    total_raw = cab_row.iloc[0]["totalVotosEmitidos"]
    # Actas en estado "Para envio al JEE" / "Pendiente" tienen totales NaN: sin
    # ground truth para total_ciudadanos no entran al training set.
    if pd.isna(total_raw):
        return 0, 0
    total_emitidos = int(total_raw)

    localizer = localizer or localize_digits
    fields_cells = localizer(png_path, template)

    n_saved, n_filtered = 0, 0
    for field in template["fields"]:
        name = field["name"]
        n_cells = field["n_digits"]
        value = field_value_for(name, votos_acta, total_emitidos)
        labels = int_to_digits(value, n_cells)
        digit_imgs = fields_cells[name]
        for pos, (label, dimg) in enumerate(zip(labels, digit_imgs)):
            if filtrar_vacias and not es_celda_escrita(value, n_cells, pos):
                n_filtered += 1
                continue
            dest_dir = crops_root / str(label)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dimg.save(dest_dir / f"{archivo_id}_{name}_c{pos}.png")
            n_saved += 1
    return n_saved, n_filtered
