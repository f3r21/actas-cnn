"""Regenera los crops de un split con etiquetado ink-aware (Fase 1, A/B).

NO toca los artefactos oficiales: escribe a data/crops_<split>_ink y
data/manifest_<split>_ink.csv. La comparacion es contra data/evaluate_<split>.csv
con el mismo checkpoint:

  python experiments/justificacion/build_val_ink.py --split val
  python scripts/evaluate.py --split val --checkpoint checkpoints/resnet18_best.pt \
      --manifest data/manifest_val_ink.csv --crops-root data/crops_val_ink \
      --out-csv data/evaluate_val_ink.csv
"""
from __future__ import annotations

import argparse
import shutil
import sys
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from actas_cnn.data import build_manifest  # noqa: E402
from actas_cnn.preprocess import build_crops_for_acta, load_templates  # noqa: E402
from actas_cnn.render import rasterize_first_page  # noqa: E402

_G: dict = {}


def procesa_acta(aid: str):
    try:
        pdf = _G["pdf_dir"] / f"{aid}.pdf"
        if not pdf.exists():
            return aid, 0, "pdf no encontrado"
        img = rasterize_first_page(pdf)
        ns, _ = build_crops_for_acta(
            img, aid, int(_G["aid_to_idacta"][aid]), _G["template"],
            _G["votos"], _G["cabecera"], _G["crops_root"],
            ink_aware=_G["ink_aware"])
        return aid, ns, None
    except Exception as e:  # noqa: BLE001
        return aid, 0, repr(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--nproc", type=int, default=7)
    ap.add_argument("--no-ink", action="store_true",
                    help="baseline: mismo render en memoria pero etiquetado "
                         "posicional clasico (aisla el efecto del etiquetado "
                         "del de la geometria del render)")
    args = ap.parse_args()

    ids = (REPO / f"data/splits/{args.split}_ids.txt").read_text().split()
    archivos = pd.read_parquet(REPO / "data/labels/actas_archivos.parquet")
    votos = pd.read_parquet(REPO / "data/labels/actas_votos.parquet")
    cabecera = pd.read_parquet(REPO / "data/labels/actas_cabecera.parquet")
    aid_to_idacta = dict(zip(archivos["archivoId"], archivos["idActa"]))
    sel = {int(aid_to_idacta[a]) for a in ids if a in aid_to_idacta}

    sufijo = "noink" if args.no_ink else "ink"
    crops_root = REPO / f"data/crops_{args.split}_{sufijo}"
    shutil.rmtree(crops_root, ignore_errors=True)
    crops_root.mkdir(parents=True)
    _G.update(
        ink_aware=not args.no_ink,
        pdf_dir=REPO / "data/pdfs_train",
        aid_to_idacta=aid_to_idacta,
        votos=votos[votos["idActa"].isin(sel)],
        cabecera=cabecera[cabecera["idActa"].isin(sel)],
        template=load_templates(REPO / "templates.json")["presidencial"],
        crops_root=crops_root,
    )

    saved, errores = 0, []
    with get_context("fork").Pool(args.nproc) as pool:
        for i, (aid, ns, err) in enumerate(
                pool.imap_unordered(procesa_acta, ids, chunksize=8), 1):
            saved += ns
            if err:
                errores.append((aid, err))
            if i % 100 == 0:
                print(f"  {i}/{len(ids)} actas")

    manifest = REPO / f"data/manifest_{args.split}_{sufijo}.csv"
    n_rows = build_manifest(crops_root, manifest)
    print(f"{args.split}: {saved} crops, manifest {n_rows} filas, "
          f"{len(errores)} errores -> {crops_root}")
    for aid, err in errores[:5]:
        print(f"  ERROR {aid}: {err}")


if __name__ == "__main__":
    main()
