"""Capa silver: tablas conformadas a partir de bronze.

- silver_votos_oficial: el join unico archivos|votos|cabecera a grano (idActa,
  nposicion) para todas las actas presidenciales de escrutinio. Es la expresion
  lakehouse del fix del join de build_crops y la fuente de los facts gold.
- silver_predicciones: evaluate_val enriquecido con idActa (via archivos) y
  nposicion (via field_mapping), tipado y dedup.
- silver_digitos_confianza: logits por digito + prob_max/entropia/confianza_baja.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from lakehouse import io_delta
from lakehouse.field_mapping import field_a_nposicion

# Filtro de proyecto: acta presidencial de escrutinio.
TIPO_ESCRUTINIO = 1
ID_ELECCION_PRESIDENCIAL = 10


def _votos_oficial(con: duckdb.DuckDBPyConnection, destino: str | None) -> int:
    con.register("archivos", io_delta.leer_arrow("bronze", "actas_archivos"))
    con.register("votos", io_delta.leer_arrow("bronze", "actas_votos"))
    con.register("cabecera", io_delta.leer_arrow("bronze", "actas_cabecera"))
    sql = f"""
        SELECT
            v.idActa,
            v.idEleccion,
            v.nposicion,
            v.nvotos                              AS nvotos_oficial,
            v.es_especial,
            v.descripcion                         AS descripcion_partido,
            v.nagrupacionPolitica                 AS nagrupacion_politica,
            a.archivoId,
            TRY_CAST(c.totalVotosEmitidos AS BIGINT) AS total_votos_emitidos,
            c.estadoActa                          AS estado_acta,
            c.descripcionEstadoActa               AS descripcion_estado_acta,
            c.ubigeoDepartamento                  AS ubigeo_departamento,
            c.ubigeoProvincia                     AS ubigeo_provincia,
            c.ubigeoDistrito                      AS ubigeo_distrito,
            c.nombreDistrito                      AS nombre_distrito,
            c.codigoMesa                          AS codigo_mesa
        FROM archivos a
        JOIN votos v
          ON v.idActa = a.idActa AND v.idEleccion = a.idEleccion
        LEFT JOIN cabecera c
          ON c.idActa = a.idActa
        WHERE a.tipo = {TIPO_ESCRUTINIO} AND a.idEleccion = {ID_ELECCION_PRESIDENCIAL}
    """
    tbl = con.execute(sql).fetch_arrow_table()
    targets = io_delta.escribir_delta(tbl, "silver", "silver_votos_oficial", destino=destino)
    print(f"[silver] silver_votos_oficial    {tbl.num_rows:>10,} filas -> {targets[0]}")
    return tbl.num_rows


def _predicciones(destino: str | None) -> int:
    pred = io_delta.leer_arrow("bronze", "pred_evaluate_val").to_pandas()
    arch = io_delta.leer_arrow("bronze", "actas_archivos").to_pandas()
    arch_p = arch[(arch["tipo"] == TIPO_ESCRUTINIO)
                  & (arch["idEleccion"] == ID_ELECCION_PRESIDENCIAL)]
    arch_p = arch_p[["archivoId", "idActa"]].drop_duplicates("archivoId")

    m = pred.merge(arch_p, on="archivoId", how="left")
    sin_idacta = int(m["idActa"].isna().sum())
    m = m[m["idActa"].notna()].copy()

    out = pd.DataFrame({
        "archivoId": m["archivoId"],
        "idActa": m["idActa"].astype("int64"),
        "field": m["field"],
        "nposicion": m["field"].map(field_a_nposicion).astype("Int64"),
        "es_total_ciudadanos": m["field"].eq("total_ciudadanos"),
        "n_cells": m["n_cells"].astype("int32"),
        "n_pred_cells": m["n_pred_cells"].astype("int32"),
        "nvotos_predicho": m["pred"].astype("int64"),
        "nvotos_oficial_eval": m["real"].astype("int64"),
        "correcto": m["correct"].astype(bool),
        "error": m["error"].astype("int64"),
    }).drop_duplicates(["archivoId", "field"]).reset_index(drop=True)

    targets = io_delta.escribir_delta(out, "silver", "silver_predicciones", destino=destino)
    print(f"[silver] silver_predicciones     {len(out):>10,} filas -> {targets[0]} "
          f"(descartadas sin idActa: {sin_idacta})")
    return len(out)


def _digitos_confianza(destino: str | None) -> int:
    lg = io_delta.leer_arrow("bronze", "pred_eval_logits_val").to_pandas()
    cols_lp = [f"lp_{c}" for c in range(10)]
    logp = lg[cols_lp].to_numpy()              # log-probabilidades (log_softmax)
    prob = np.exp(logp)
    prob_max = prob.max(axis=1)
    entropia = -(prob * logp).sum(axis=1)      # -sum p*log(p)

    out = lg.copy()
    out["prob_max"] = prob_max.astype("float32")
    out["entropia"] = entropia.astype("float32")
    out["confianza_baja"] = (prob_max < 0.90)
    targets = io_delta.escribir_delta(out, "silver", "silver_digitos_confianza", destino=destino)
    print(f"[silver] silver_digitos_confianza {len(out):>9,} filas -> {targets[0]}")
    return len(out)


def construir(destino: str | None = None) -> dict[str, int]:
    print("== SILVER ==")
    con = duckdb.connect()
    conteos = {
        "silver_votos_oficial": _votos_oficial(con, destino),
        "silver_predicciones": _predicciones(destino),
        "silver_digitos_confianza": _digitos_confianza(destino),
    }
    con.close()
    return conteos
