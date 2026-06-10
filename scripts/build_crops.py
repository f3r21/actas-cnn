"""CLI: genera recortes de digitos etiquetados desde PNGs renderizados + parquets.

Wrapper de actas_cnn.preprocess.build_crops_for_acta. Usa el localizador ZONAL
por plantilla (`localize_digits`, metodo OFICIAL). El localizador fiducial
(experimento negativo, -0.72pp acta-level) vive en experiments/fiducial/.

Mapeo field -> ground truth:
- partido_NN -> actas_votos.nvotos where nposicion=N
- votos_blanco/nulos/impugnados -> nposicion 80/81/82
- total_ciudadanos -> actas_cabecera.totalVotosEmitidos

Uso:
  python scripts/build_crops.py --rendered-dir data/pdfs_train/rendered \\
      --out-crops data/crops --split train --ids-file data/splits/train_ids.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actas_cnn.preprocess import build_crops_for_acta, load_templates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rendered-dir", required=True, type=Path,
                    help="carpeta con PNGs renderizados, formato <archivoId>_p0.png")
    ap.add_argument("--templates", default=Path("templates.json"), type=Path)
    ap.add_argument("--template-key", default="presidencial")
    ap.add_argument("--labels-dir", default=Path("data/labels"), type=Path,
                    help="carpeta con actas_archivos/votos/cabecera.parquet")
    ap.add_argument("--out-crops", default=Path("data/crops"), type=Path,
                    help="raiz de salida; si --split, se le aniade _<split>")
    ap.add_argument("--id-eleccion", type=int, default=10,
                    help="filtro de idEleccion (default 10=Presidencial)")
    ap.add_argument("--limit", type=int, default=None, help="limite de actas (smoke)")
    ap.add_argument("--ids-file", type=Path, default=None,
                    help="lista de archivoIds (1 por linea) a procesar")
    ap.add_argument("--split", type=str, default=None,
                    help="nombre del split (train|val|test); out_crops -> <out_crops>_<split>")
    ap.add_argument("--no-filtrar-vacias", action="store_true",
                    help="incluir celdas sin digito escrito (no recomendado: domina label-0)")
    args = ap.parse_args()

    template = load_templates(args.templates)[args.template_key]
    archivos = pd.read_parquet(args.labels_dir / "actas_archivos.parquet")
    votos = pd.read_parquet(args.labels_dir / "actas_votos.parquet")
    cabecera = pd.read_parquet(args.labels_dir / "actas_cabecera.parquet")

    archivos_pres = archivos[(archivos["tipo"] == 1) & (archivos["idEleccion"] == args.id_eleccion)]
    pngs = {p.stem.removesuffix("_p0"): p for p in args.rendered_dir.glob("*_p0.png")}
    to_proc = archivos_pres[archivos_pres["archivoId"].isin(pngs.keys())]
    if args.ids_file:
        ids = {ln.strip() for ln in args.ids_file.read_text().splitlines() if ln.strip()}
        to_proc = to_proc[to_proc["archivoId"].isin(ids)]
    if args.limit:
        to_proc = to_proc.head(args.limit)

    out_crops = args.out_crops
    if args.split:
        out_crops = args.out_crops.with_name(f"{args.out_crops.name}_{args.split}")
    out_crops.mkdir(parents=True, exist_ok=True)

    filtrar = not args.no_filtrar_vacias
    print(f"actas a procesar: {len(to_proc)} -> {out_crops} (localizador zonal)")
    n_actas_ok, n_crops_total, n_filtradas = 0, 0, 0
    for _, row in to_proc.iterrows():
        aid = row["archivoId"]
        n_saved, n_filt = build_crops_for_acta(
            image=pngs[aid],
            archivo_id=aid,
            id_acta=int(row["idActa"]),
            template=template,
            votos=votos,
            cabecera=cabecera,
            crops_root=out_crops,
            filtrar_vacias=filtrar,
        )
        if n_saved > 0:
            n_actas_ok += 1
            n_crops_total += n_saved
            n_filtradas += n_filt
    print(f"procesadas {n_actas_ok}/{len(to_proc)} actas, {n_crops_total} crops, "
          f"{n_filtradas} celdas vacias filtradas")


if __name__ == "__main__":
    main()
