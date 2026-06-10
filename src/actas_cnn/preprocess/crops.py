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


def crop_fields(image, template):
    """Recorta los campos numericos definidos en la plantilla.

    template["fields"]: lista de {name, box:[x0,y0,x1,y1], n_digits} con box en
    fraccion [0, 1]. `image` puede ser una ruta a PNG o una PIL.Image ya
    rasterizada en memoria (render.rasterize_first_page). Devuelve un dict
    name -> PIL.Image (escala de grises).
    """
    img = image if isinstance(image, Image.Image) else Image.open(image).convert("L")
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


def localize_digits(image, template):
    """Localiza las celdas de digitos de cada campo: {campo: [celda_0, ...]}.

    Metodo OFICIAL (zonal): recorta cada campo por su caja de la plantilla y lo
    parte en n_digits celdas equiespaciadas. Es el punto unico de "donde estan
    los digitos": para cambiar la deteccion, reemplaza esta funcion (o pasa otra
    con la misma firma a build_crops_for_acta). El localizador fiducial
    alternativo (experimento negativo) vive en experiments/fiducial/.
    """
    fields = crop_fields(image, template)
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


# --- Deteccion de tinta por celda (etiquetado ink-aware) ----------------------
#
# Una minoria de actas (~3% en val) viola la convencion right-justified: el
# escribiente llena las cifras desde la primera celda o centradas. Con el
# etiquetado posicional esas actas quedan envenenadas (celdas vacias con label
# de digito, digitos con el label del vecino) y concentran el 82% de los
# errores de campo en eval. Estas funciones detectan DONDE cae la tinta para
# remapear los labels en esas actas. Validadas en
# experiments/justificacion/audit_justificacion.py (19/19 actas de la cola de
# eval clasifican como violadoras; 0 por desalineacion geometrica).

def ventana_central(cell_img, mx=0.25, my=0.15):
    """Ventana central de la celda como array uint8. La tinta de un digito
    escrito cae al centro; el sangrado de trazos del digito vecino y los
    bordes punteados se concentran en los margenes, asi que se descartan."""
    w, h = cell_img.size
    box = (int(w * mx), int(h * my), int(w * (1 - mx)), int(h * (1 - my)))
    return np.asarray(cell_img.crop(box).convert("L"), dtype=np.uint8)


def umbral_adaptativo(arrays, delta=55):
    """Umbral de oscuridad relativo al fondo del escaneo de ESTA acta.

    Un umbral fijo falla en escaneos grisaceos (fondo ~170: todo cuenta como
    tinta). La mediana de todos los pixeles de las celdas es fondo casi puro
    (la tinta es minoria), asi que fondo - delta separa trazos del papel en
    escaneos claros y oscuros por igual.
    """
    fondo = int(np.median(np.concatenate([a.ravel() for a in arrays])))
    return int(np.clip(fondo - delta, 40, 200))


def patron_de_tinta(fracs, piso=0.07, rel=0.55):
    """Clasifica donde cae la tinta de un campo. Devuelve (patron, run, inked).

    El corte por celda es relativo al maximo del campo (la fraccion absoluta
    de un "1" delgado empata con el sangrado del vecino, pero dentro del campo
    el digito escrito siempre domina).

      RIGHT    run contiguo anclado a la ultima celda (convencion asumida;
               incluye ceros a la izquierda escritos y digitos tenues)
      LEFT     run anclado a la primera celda sin llegar a la ultima
      MEDIO    run que no toca ningun extremo
      OTRO     mas de un run (tinta salteada)
      AMBIGUO  tinta demasiado tenue para decidir

    run es (a, b) con celdas inked en [a, b), o None si no hay run unico.
    """
    fmax = max(fracs)
    if fmax < piso:
        return "AMBIGUO", None, tuple(False for _ in fracs)
    corte = max(piso, rel * fmax)
    inked = tuple(f >= corte for f in fracs)
    runs = []
    i = 0
    while i < len(inked):
        if inked[i]:
            j = i
            while j < len(inked) and inked[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    if len(runs) != 1:
        return "OTRO", None, inked
    a, b = runs[0]
    if b == len(inked):
        return "RIGHT", (a, b), inked
    if a == 0:
        return "LEFT", (a, b), inked
    return "MEDIO", (a, b), inked


def remapeo_ink_aware(fields_cells, template, votos_acta, total_emitidos,
                      min_informativos=4, umbral_viola=0.5, piso=0.07):
    """Plan de remapeo de labels para una acta que viola la convencion.

    Devuelve {} si la acta cumple la convencion right-justified (lo normal:
    nada cambia). Si la mayoria de sus campos legibles esta corrida (LEFT o
    MEDIO), devuelve {field_name: {pos: label}} SOLO para los campos donde el
    remapeo es confiable: run de tinta del largo exacto del valor y celdas
    esperadas por la convencion sin tinta (si la celda esperada tambien tiene
    tinta, es un digito a caballo entre ventanas por offset del escaneo, no
    una violacion: 1 acta de val escribe asi y evalua perfecto). Todo lo demas
    conserva el etiquetado posicional de siempre: el fix solo toca lo que
    puede arreglar con confianza, nunca empeora el statu quo.
    """
    centros = {name: [ventana_central(c) for c in cells]
               for name, cells in fields_cells.items()}
    intensidad = umbral_adaptativo([a for cs in centros.values() for a in cs])

    analisis = {}
    for field in template["fields"]:
        name = field["name"]
        value = field_value_for(name, votos_acta, total_emitidos)
        if value <= 0:
            continue
        fracs = [float((a < intensidad).sum() / a.size) for a in centros[name]]
        analisis[name] = (value, fracs, *patron_de_tinta(fracs))

    conteo = {}
    for _, _, patron, _, _ in analisis.values():
        conteo[patron] = conteo.get(patron, 0) + 1
    informativos = sum(conteo.get(p, 0) for p in ("RIGHT", "LEFT", "MEDIO", "OTRO"))
    viola = conteo.get("LEFT", 0) + conteo.get("MEDIO", 0)
    if informativos < min_informativos or viola / informativos < umbral_viola:
        return {}

    candidatos = []
    for name, (value, fracs, patron, run, _) in analisis.items():
        if patron not in ("LEFT", "MEDIO") or run is None:
            continue
        digitos = [int(c) for c in str(value)]
        if run[1] - run[0] != len(digitos):
            continue  # tinta no calza con el numero de digitos: no tocar
        candidatos.append((name, run, digitos))
    if not candidatos:
        return {}

    # Distinguir violacion real de offset del escaneo: si el digito quedo "a
    # caballo" entre dos ventanas (acta corrida unos px), la celda que la
    # convencion espera escrita TAMBIEN tiene tinta — medida a celda completa,
    # porque pegada al borde la ventana central no la ve. En la violacion real
    # esa celda esta vacia. Se decide por acta (mediana sobre los campos
    # remapeables): el straddle es sistematico, el sangrado puntual no.
    evidencias = []
    for name, run, digitos in candidatos:
        n_cells = len(analisis[name][1])
        fuera = [p for p in range(n_cells - len(digitos), n_cells)
                 if not run[0] <= p < run[1]]
        if not fuera:
            continue
        fulls = [ventana_central(fields_cells[name][p], mx=0.05, my=0.05)
                 for p in fuera]
        evidencias.append(max(float((a < intensidad).sum() / a.size)
                              for a in fulls))
    if evidencias and float(np.median(evidencias)) >= piso:
        return {}  # offset, no violacion: el etiquetado posicional ya es correcto

    return {name: dict(zip(range(run[0], run[1]), digitos))
            for name, run, digitos in candidatos}


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
    image: "str | Path | Image.Image",
    archivo_id: str,
    id_acta: int,
    template: dict,
    votos: "pd.DataFrame",
    cabecera: "pd.DataFrame",
    crops_root: "str | Path",
    localizer=None,
    filtrar_vacias: bool = True,
    ink_aware: bool = True,
) -> tuple[int, int]:
    """Procesa una acta y guarda sus crops. Devuelve (n_guardados, n_filtrados).

    `image`: ruta a PNG o PIL.Image en memoria (render.rasterize_first_page).
    `localizer` (callable imagen,template -> {campo:[celdas]}) decide donde estan
    los digitos; None usa el zonal `localize_digits` (oficial). Si filtrar_vacias=True las
    celdas sin digito escrito segun el label (es_celda_escrita) NO se guardan:
    resuelve el imbalance (76% de las celdas son vacias y dominarian el training).
    Con ink_aware=True, las actas que violan la convencion right-justified
    (escritura corrida a la izquierda o centrada, ~3% en val) se detectan por
    tinta y sus labels se remapean a las celdas realmente escritas
    (remapeo_ink_aware); en el resto de actas no cambia nada.
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
    fields_cells = localizer(image, template)
    remap = (remapeo_ink_aware(fields_cells, template, votos_acta, total_emitidos)
             if ink_aware else {})

    n_saved, n_filtered = 0, 0
    for field in template["fields"]:
        name = field["name"]
        n_cells = field["n_digits"]
        value = field_value_for(name, votos_acta, total_emitidos)
        digit_imgs = fields_cells[name]
        if name in remap:
            etiqueta_en = remap[name]
        else:
            labels = int_to_digits(value, n_cells)
            etiqueta_en = {pos: labels[pos] for pos in range(n_cells)
                           if not filtrar_vacias or es_celda_escrita(value, n_cells, pos)}
        for pos, dimg in enumerate(digit_imgs):
            if pos not in etiqueta_en:
                n_filtered += 1
                continue
            dest_dir = crops_root / str(etiqueta_en[pos])
            dest_dir.mkdir(parents=True, exist_ok=True)
            dimg.save(dest_dir / f"{archivo_id}_{name}_c{pos}.png")
            n_saved += 1
    return n_saved, n_filtered
