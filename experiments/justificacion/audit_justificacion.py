"""Auditoria de justificacion de cifras (Fase 0 del fix ink-aware).

Hipotesis (verificada a mano en 3 actas de la cola de val): una minoria de
actas viola la convencion right-justified de ONPE — el escribiente llena las
cifras desde la primera celda. El etiquetado posicional (es_celda_escrita +
int_to_digits) las envenena: celdas vacias quedan etiquetadas con digito y
digitos quedan etiquetados con el del vecino.

Este script NO toca el pipeline: mide. Por cada acta del split rasteriza,
localiza celdas (zonal oficial) y clasifica cada campo con valor > 0 segun
donde cae la tinta (ventana central de cada celda, umbral de oscuridad
adaptativo al fondo del escaneo, corte relativo al maximo del campo):

  RIGHT    run contiguo de tinta anclado a la ultima celda (convencion
           asumida; incluye ceros a la izquierda escritos y digitos tenues)
  LEFT     run contiguo anclado a la primera celda sin llegar a la ultima
           (violacion: escritura desde la izquierda, incluso "apretada")
  MEDIO    run contiguo que no toca ningun extremo
  OTRO     tinta en mas de un run (salteada)
  AMBIGUO  tinta demasiado tenue para decidir

Agrega por acta, cruza con data/evaluate_<split>.csv (errores per-acta) e
imprime el resumen para el go/no-go. Con --geom ademas mide el offset
geometrico (detector fiducial de experiments/fiducial/) de las actas de la
cola que NO clasifican como violadoras, contra una cohorte de actas sanas.

Uso:
  python experiments/justificacion/audit_justificacion.py --split val
  python experiments/justificacion/audit_justificacion.py --ids <aid> ... --detalle
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from actas_cnn.preprocess.crops import (  # noqa: E402
    field_value_for,
    load_templates,
    localize_digits,
    patron_de_tinta,
    umbral_adaptativo,
    ventana_central,
)
from actas_cnn.render import rasterize_first_page  # noqa: E402

# Estado compartido con los workers (heredado via fork, sin picklear por tarea).
_G: dict = {}


def procesa_acta(aid: str):
    """Una acta: rasteriza, localiza celdas y clasifica sus campos escritos."""
    try:
        pdf = _G["pdf_dir"] / f"{aid}.pdf"
        if not pdf.exists():
            return aid, None, "pdf no encontrado"
        id_acta = int(_G["aid_to_idacta"][aid])
        votos_acta = _G["votos"][_G["votos"]["idActa"] == id_acta]
        cab = _G["cabecera"][_G["cabecera"]["idActa"] == id_acta]
        if not len(cab) or pd.isna(cab.iloc[0]["totalVotosEmitidos"]):
            return aid, None, "sin total (NaN), fuera del universo de eval"
        total = int(cab.iloc[0]["totalVotosEmitidos"])
        img = rasterize_first_page(pdf)
        celdas = localize_digits(img, _G["template"])
        centros = {name: [ventana_central(c, _G["mx"], _G["my"]) for c in cs]
                   for name, cs in celdas.items()}
        intensidad = (_G["intensidad"] if _G["intensidad"] is not None else
                      umbral_adaptativo([a for cs in centros.values() for a in cs],
                                        _G["delta"]))
        filas = []
        for f in _G["template"]["fields"]:
            value = field_value_for(f["name"], votos_acta, total)
            if value <= 0:
                continue
            k = len(str(value))
            fracs = [float((a < intensidad).sum() / a.size)
                     for a in centros[f["name"]]]
            patron, _, inked = patron_de_tinta(fracs, _G["piso"], _G["rel"])
            tinta = ("?" * len(fracs) if patron == "AMBIGUO" else
                     "".join("1" if b else "0" for b in inked))
            filas.append({
                "archivoId": aid, "field": f["name"], "value": value, "k": k,
                "tinta": tinta, "patron": patron,
                "frac_max": round(max(fracs), 4),
                "intensidad": intensidad,
            })
        return aid, filas, None
    except Exception as e:  # noqa: BLE001 - un PDF malo no tumba la auditoria
        return aid, None, repr(e)


def clasificar_acta(cnt: dict) -> str:
    """VIOLA si la mayoria de los campos legibles esta corrida; CUMPLE si
    ninguno lo esta (y el ruido OTRO es bajo); DUDOSA en el medio."""
    informativos = sum(cnt.get(p, 0) for p in ("RIGHT", "LEFT", "MEDIO", "OTRO"))
    viola = cnt.get("LEFT", 0) + cnt.get("MEDIO", 0)
    if informativos < 4:
        # Con 1-3 campos legibles (acta casi vacia o muy tenue) no hay base
        # para declarar violacion sistematica.
        return "SIN_SENAL"
    if viola / informativos >= 0.5:
        return "VIOLA"
    # Hasta ~20% de campos "corridos" es ruido de medicion (digitos tenues o
    # sangrado borde-linea del corte): las violadoras reales miden 85-100%.
    if viola / informativos <= 0.2 and cnt.get("OTRO", 0) / informativos <= 0.25:
        return "CUMPLE"
    return "DUDOSA"


def chequeo_geometrico(sospechosas: list[str], cohorte: list[str]) -> None:
    """Mide el offset de los markers fiduciales vs la mediana de actas sanas."""
    sys.path.insert(0, str(REPO / "experiments/fiducial"))
    from detect_fiducials import detect_15  # noqa: PLC0415

    def markers_de(aid: str, tmp: Path):
        png = tmp / f"{aid}.png"
        rasterize_first_page(_G["pdf_dir"] / f"{aid}.pdf").save(png)
        return detect_15(png)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        por_rol: dict[str, list] = {}
        for aid in cohorte:
            for rol, (x, y) in markers_de(aid, tmp).items():
                por_rol.setdefault(rol, []).append((x, y))
        mediana = {r: (float(np.median([p[0] for p in v])),
                       float(np.median([p[1] for p in v])))
                   for r, v in por_rol.items() if len(v) >= len(cohorte) * 0.6}
        print(f"\n--- chequeo geometrico (cohorte de {len(cohorte)} actas sanas, "
              f"{len(mediana)} roles de referencia) ---")
        for aid in sospechosas:
            m = markers_de(aid, tmp)
            comunes = [r for r in m if r in mediana]
            if not comunes:
                print(f"  {aid}: sin markers comparables ({len(m)} detectados)")
                continue
            dx = float(np.median([m[r][0] - mediana[r][0] for r in comunes]))
            dy = float(np.median([m[r][1] - mediana[r][1] for r in comunes]))
            print(f"  {aid}: {len(m)} markers, offset mediano vs sanas "
                  f"dx={dx:+.0f}px dy={dy:+.0f}px")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--ids", nargs="*", default=None,
                    help="auditar solo estos archivoIds (calibracion)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mx", type=float, default=0.25,
                    help="margen horizontal de la ventana central de la celda")
    ap.add_argument("--my", type=float, default=0.15,
                    help="margen vertical de la ventana central de la celda")
    ap.add_argument("--piso", type=float, default=0.07,
                    help="fraccion de tinta minima para considerar legible")
    ap.add_argument("--rel", type=float, default=0.55,
                    help="corte por celda = rel * fraccion maxima del campo")
    ap.add_argument("--intensidad", type=int, default=None,
                    help="umbral fijo de oscuridad; default: adaptativo por acta")
    ap.add_argument("--delta", type=int, default=55,
                    help="umbral adaptativo = fondo mediano de la acta - delta")
    ap.add_argument("--nproc", type=int, default=None)
    ap.add_argument("--detalle", action="store_true",
                    help="imprime las filas por campo (para calibrar)")
    ap.add_argument("--geom", action="store_true",
                    help="mide offset fiducial de las actas de la cola no-VIOLA")
    args = ap.parse_args()

    archivos = pd.read_parquet(REPO / "data/labels/actas_archivos.parquet")
    votos = pd.read_parquet(REPO / "data/labels/actas_votos.parquet")
    cabecera = pd.read_parquet(REPO / "data/labels/actas_cabecera.parquet")

    if args.ids:
        ids = args.ids
    else:
        ids = (REPO / f"data/splits/{args.split}_ids.txt").read_text().split()
    if args.limit:
        ids = ids[:args.limit]

    aid_to_idacta = dict(zip(archivos["archivoId"], archivos["idActa"]))
    sel = {int(aid_to_idacta[a]) for a in ids if a in aid_to_idacta}
    _G.update(
        pdf_dir=REPO / "data/pdfs_train",
        aid_to_idacta=aid_to_idacta,
        votos=votos[votos["idActa"].isin(sel)],
        cabecera=cabecera[cabecera["idActa"].isin(sel)],
        template=load_templates(REPO / "templates.json")["presidencial"],
        mx=args.mx, my=args.my, piso=args.piso, rel=args.rel,
        intensidad=args.intensidad, delta=args.delta,
    )

    nproc = args.nproc or max(1, (len(ids) > 8) * 7) or 1
    filas, errores = [], []
    if nproc > 1:
        with get_context("fork").Pool(nproc) as pool:
            resultados = pool.imap_unordered(procesa_acta, ids, chunksize=8)
            for i, (aid, fs, err) in enumerate(resultados, 1):
                if err:
                    errores.append((aid, err))
                else:
                    filas.extend(fs)
                if i % 100 == 0:
                    print(f"  {i}/{len(ids)} actas auditadas")
    else:
        for aid in ids:
            aid, fs, err = procesa_acta(aid)
            errores.append((aid, err)) if err else filas.extend(fs)

    campos = pd.DataFrame(filas)
    out_dir = REPO / "experiments/justificacion"
    sufijo = args.split if not args.ids else "ids"
    campos.to_csv(out_dir / f"justificacion_{sufijo}_campos.csv", index=False)

    if args.detalle:
        print(campos.to_string(index=False))

    # --- Agregado por acta ---------------------------------------------------
    pivot = campos.pivot_table(index="archivoId", columns="patron",
                               values="field", aggfunc="count",
                               fill_value=0).reset_index()
    for col in ("RIGHT", "LEFT", "MEDIO", "OTRO", "AMBIGUO"):
        if col not in pivot:
            pivot[col] = 0
    pivot["clase"] = pivot.apply(
        lambda r: clasificar_acta({c: r[c] for c in ("RIGHT", "LEFT", "MEDIO", "OTRO")}),
        axis=1)

    eval_csv = REPO / f"data/evaluate_{args.split}.csv"
    if eval_csv.exists():
        ev = pd.read_csv(eval_csv)
        mal = ev[~ev.correct].groupby("archivoId").size().rename("campos_mal")
        pivot = pivot.merge(mal, on="archivoId", how="left")
        pivot["campos_mal"] = pivot["campos_mal"].fillna(0).astype(int)
    out_csv = out_dir / f"justificacion_{sufijo}.csv"
    pivot.to_csv(out_csv, index=False)

    # --- Resumen para el go/no-go --------------------------------------------
    print(f"\nactas auditadas: {pivot.archivoId.nunique()}, errores: {len(errores)}")
    for aid, err in errores[:5]:
        print(f"  ERROR {aid}: {err}")
    print("\npatrones a nivel campo:")
    print(campos.patron.value_counts().to_string())
    print("\nclases a nivel acta:")
    print(pivot.clase.value_counts().to_string())

    if "campos_mal" in pivot:
        cola = pivot[pivot.campos_mal >= 5].sort_values("campos_mal", ascending=False)
        print(f"\ncola de eval (>=5 campos mal): {len(cola)} actas")
        cols = ["archivoId", "campos_mal", "clase", "RIGHT", "LEFT", "MEDIO", "OTRO", "AMBIGUO"]
        print(cola[cols].to_string(index=False))
        viola_sin_cola = pivot[(pivot.clase == "VIOLA") & (pivot.campos_mal < 5)]
        print(f"\nactas VIOLA fuera de la cola (eval casi limpio): {len(viola_sin_cola)}")
        if len(viola_sin_cola):
            print(viola_sin_cola[cols].to_string(index=False))

        if args.geom:
            sospechosas = list(cola[cola.clase != "VIOLA"].archivoId)
            cohorte = list(pivot[(pivot.clase == "CUMPLE")
                                 & (pivot.campos_mal == 0)].archivoId[:30])
            if sospechosas and cohorte:
                chequeo_geometrico(sospechosas, cohorte)
            else:
                print("\nchequeo geometrico: nada que medir "
                      f"({len(sospechosas)} sospechosas, {len(cohorte)} en cohorte)")

    print(f"\nCSV por acta:  {out_csv}")
    print(f"CSV por campo: {out_dir / f'justificacion_{sufijo}_campos.csv'}")


if __name__ == "__main__":
    main()
