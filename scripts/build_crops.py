"""Genera recortes de digitos con labels reales desde los parquets ONPE.

Pipeline:
1. Lee actas_archivos.parquet filtrando tipo=1 idEleccion=10 (Presidencial).
2. Para cada archivoId procesa la imagen renderizada (PNG ya producido por
   pdf_to_images.py) aplicando templates.json[presidencial].
3. Cada field se parte en n_digits celdas. El valor de cada celda (label 0-9)
   se obtiene del ground truth: right-justified con leading zeros.
4. Cada celda se guarda en crops/<label>/<archivoId>_<field>_c<pos>.png.

Mapeo field -> ground truth:
- partido_NN -> actas_votos.nvotos where nposicion=N (es_especial=False)
- votos_blanco/nulos/impugnados -> nposicion=80/81/82 (es_especial=True)
- total_ciudadanos -> actas_cabecera.totalVotosEmitidos

Convencion ONPE: cifras right-justified en sus celdas. Celdas vacias = 0.
Salida lista para alimentar build_dataset.py (manifest.csv).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_crops import crop_fields, es_celda_escrita, load_templates, split_digits
from scripts.detect_fiducials import detect_15, load_anchors, transform_template
from lakehouse.field_mapping import valor_oficial_para
from PIL import Image as _PIL_Image


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


def procesar_acta(
    png_path: Path,
    archivo_id: str,
    id_acta: int,
    template: dict,
    votos_acta: pd.DataFrame,
    cab_row: pd.DataFrame,
    crops_root: Path,
    filtrar_vacias: bool = True,
    anchors: dict | None = None,
) -> tuple[int, int]:
    """Procesa una acta. Devuelve (n_crops_guardados, n_celdas_vacias_filtradas).

    `votos_acta` y `cab_row` ya vienen filtrados por idActa (pre-group en main()),
    por eso aqui no se vuelve a escanear las tablas completas.

    Si filtrar_vacias=True, las celdas que no tienen digito escrito segun el
    label (es_celda_escrita) NO se guardan como crops. Esto resuelve el
    imbalance: 76% de las celdas son vacias y dominarian el training.
    """
    if len(cab_row) == 0:
        return 0, 0
    total_raw = cab_row.iloc[0]["totalVotosEmitidos"]
    # Algunas actas en estado "Para envio al JEE" / "Pendiente" tienen totales NaN.
    # Sin ground truth para total_ciudadanos no entran al training set.
    if pd.isna(total_raw):
        return 0, 0
    total_emitidos = int(total_raw)

    # Registracion afin via fiducial markers (Semana 1 cierre)
    aligned_template = template
    if anchors is not None:
        markers = detect_15(png_path)
        if len(markers) >= 4:
            img = _PIL_Image.open(png_path)
            aligned_template = transform_template(template, markers, anchors,
                                                    img.size)

    fields_crops = crop_fields(png_path, aligned_template)
    n_saved, n_filtered = 0, 0
    for field in template["fields"]:
        name = field["name"]
        n_cells = field["n_digits"]
        value = valor_oficial_para(name, votos_acta, total_emitidos)
        labels = int_to_digits(value, n_cells)
        digit_imgs = split_digits(fields_crops[name], n_cells)
        for pos, (label, dimg) in enumerate(zip(labels, digit_imgs)):
            if filtrar_vacias and not es_celda_escrita(value, n_cells, pos):
                n_filtered += 1
                continue
            dest_dir = crops_root / str(label)
            dest_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{archivo_id}_{name}_c{pos}.png"
            dimg.save(dest_dir / fname)
            n_saved += 1
    return n_saved, n_filtered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rendered-dir", required=True, type=Path,
                    help="carpeta con PNGs renderizados, formato <archivoId>_p0.png")
    ap.add_argument("--templates", default=Path("templates.json"), type=Path)
    ap.add_argument("--template-key", default="presidencial")
    ap.add_argument("--labels-dir", default=Path("data/labels"), type=Path,
                    help="carpeta con actas_archivos.parquet, actas_votos.parquet, actas_cabecera.parquet")
    ap.add_argument("--out-crops", default=Path("data/crops"), type=Path,
                    help="raiz de salida; si --split, se le aniade _<split>")
    ap.add_argument("--id-eleccion", type=int, default=10, help="filtro de idEleccion (default 10=Presidencial)")
    ap.add_argument("--limit", type=int, default=None, help="limite de actas a procesar (smoke)")
    ap.add_argument("--ids-file", type=Path, default=None,
                    help="lista de archivoIds (1 por linea) a procesar; restringe to_proc")
    ap.add_argument("--split", type=str, default=None,
                    help="nombre del split (train|val|test); cambia out_crops a <out_crops>_<split>")
    ap.add_argument("--no-filtrar-vacias", action="store_true",
                    help="por default, las celdas sin digito escrito (es_celda_escrita=False) NO se guardan. Pasa esto para incluirlas (no recomendado: domina label-0).")
    args = ap.parse_args()

    template = load_templates(args.templates)[args.template_key]
    archivos = pd.read_parquet(args.labels_dir / "actas_archivos.parquet")
    votos = pd.read_parquet(args.labels_dir / "actas_votos.parquet")
    cabecera = pd.read_parquet(args.labels_dir / "actas_cabecera.parquet")

    archivos_pres = archivos[(archivos["tipo"] == 1) & (archivos["idEleccion"] == args.id_eleccion)]
    pngs = {p.stem.removesuffix("_p0"): p for p in args.rendered_dir.glob("*_p0.png")}
    to_proc = archivos_pres[archivos_pres["archivoId"].isin(pngs.keys())]
    if args.ids_file:
        ids = {line.strip() for line in args.ids_file.read_text().splitlines() if line.strip()}
        to_proc = to_proc[to_proc["archivoId"].isin(ids)]
    if args.limit:
        to_proc = to_proc.head(args.limit)

    out_crops = args.out_crops
    if args.split:
        out_crops = args.out_crops.with_name(f"{args.out_crops.name}_{args.split}")
    out_crops.mkdir(parents=True, exist_ok=True)

    # Cargar anchors fiduciales para registracion afin (opcional)
    anchors = None
    anchors_path = Path("fiducial_anchors.json")
    if anchors_path.exists():
        anchors = load_anchors(anchors_path)
        print(f"fiducial anchors cargados de {anchors_path} ({len(anchors)} roles)")

    print(f"actas a procesar: {len(to_proc)} -> {out_crops}")
    filtrar = not args.no_filtrar_vacias

    # Pre-group una sola vez: en vez de escanear los 18.6M de votos (y la cabecera)
    # por cada acta dentro del loop, se agrupa por idActa una sola vez y luego se hace
    # lookup O(1). Convierte el join O(actas x votos) en O(votos) + O(actas). Se
    # restringe a las idActas a procesar para no materializar el universo entero.
    ids_acta = set(to_proc["idActa"].astype(int))
    votos_vacio = votos.iloc[0:0]
    cab_vacio = cabecera.iloc[0:0]
    votos_por_acta = {
        int(aid): g for aid, g in votos[votos["idActa"].isin(ids_acta)].groupby("idActa")
    }
    cab_por_acta = {
        int(aid): g for aid, g in cabecera[cabecera["idActa"].isin(ids_acta)].groupby("idActa")
    }

    n_actas_ok, n_crops_total, n_filtradas = 0, 0, 0
    for _, row in to_proc.iterrows():
        aid = row["archivoId"]
        id_acta = int(row["idActa"])
        n_saved, n_filt = procesar_acta(
            png_path=pngs[aid],
            archivo_id=aid,
            id_acta=id_acta,
            template=template,
            votos_acta=votos_por_acta.get(id_acta, votos_vacio),
            cab_row=cab_por_acta.get(id_acta, cab_vacio),
            crops_root=out_crops,
            filtrar_vacias=filtrar,
            anchors=anchors,
        )
        if n_saved > 0:
            n_actas_ok += 1
            n_crops_total += n_saved
            n_filtradas += n_filt
    print(f"procesadas {n_actas_ok}/{len(to_proc)} actas, {n_crops_total} crops guardados, {n_filtradas} celdas vacias filtradas")


if __name__ == "__main__":
    main()
