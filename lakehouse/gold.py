"""Capa gold: star schema (4 dims + 2 facts) + marts agregados.

Dims:  dim_organizacion_politica, dim_ubicacion, dim_acta, dim_eleccion.
Facts: fact_resultados_oficiales (padron oficial completo, ~3.46M filas),
       fact_qa_modelo (reconciliacion predicho-vs-oficial, ~29k filas).
Marts: mart_kpis_globales, mart_calidad_por_departamento, mart_calidad_por_partido,
       mart_resultados_por_departamento, mart_peores_actas, mart_error_hist.

Se construye con DuckDB sobre las tablas silver/bronze leidas como Arrow. Cada
tabla gold se registra en la misma conexion para que los marts la referencien.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import duckdb
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from lakehouse import io_delta

ID_ELECCION = 10
CHECKPOINT_OFICIAL = "resnet18_best.pt"


def _wr(con, nombre: str, sql: str, destino: str | None, registrar: bool = True) -> int:
    """Ejecuta SQL, registra el resultado en la conexion y lo escribe a Delta gold."""
    tbl = con.execute(sql).fetch_arrow_table()
    if registrar:
        con.register(nombre, tbl)
    targets = io_delta.escribir_delta(tbl, "gold", nombre, destino=destino)
    print(f"[gold] {nombre:32s} {tbl.num_rows:>9,} filas -> {targets[0]}")
    return tbl.num_rows


def construir(destino: str | None = None) -> dict[str, int]:
    print("== GOLD ==")
    con = duckdb.connect()
    con.register("svo", io_delta.leer_arrow("silver", "silver_votos_oficial"))
    con.register("sp", io_delta.leer_arrow("silver", "silver_predicciones"))
    con.register("sdc", io_delta.leer_arrow("silver", "silver_digitos_confianza"))
    con.register("departamentos", io_delta.leer_arrow("bronze", "departamentos"))
    con.register("worst", io_delta.leer_arrow("bronze", "pred_worst20_val"))

    conteos: dict[str, int] = {}

    # --- Dimensiones ---
    conteos["dim_organizacion_politica"] = _wr(con, "dim_organizacion_politica", """
        SELECT nposicion,
               max(descripcion_partido)   AS nombre_organizacion,
               max(nagrupacion_politica)   AS nagrupacion_politica,
               (nposicion IN (80, 81, 82))       AS es_especial,
               CASE nposicion WHEN 80 THEN 'blanco' WHEN 81 THEN 'nulo'
                              WHEN 82 THEN 'impugnado' ELSE 'partido' END AS tipo_posicion
        FROM svo GROUP BY nposicion ORDER BY nposicion
    """, destino)

    conteos["dim_ubicacion"] = _wr(con, "dim_ubicacion", """
        SELECT u.ubigeo_distrito,
               max(u.ubigeo_provincia)     AS ubigeo_provincia,
               max(u.ubigeo_departamento)  AS ubigeo_departamento,
               max(u.nombre_distrito)      AS nombre_distrito,
               max(d.nombre)               AS nombre_departamento,
               max(d.idAmbitoGeografico)   AS id_ambito_geografico
        FROM svo u LEFT JOIN departamentos d ON d.ubigeo = u.ubigeo_departamento
        GROUP BY u.ubigeo_distrito
    """, destino)

    conteos["dim_acta"] = _wr(con, "dim_acta", """
        SELECT idActa,
               max(archivoId)              AS archivoId,
               max(codigo_mesa)            AS codigo_mesa,
               max(estado_acta)            AS estado_acta,
               max(total_votos_emitidos)   AS total_votos_emitidos,
               max(ubigeo_distrito)        AS ubigeo_distrito,
               (idActa IN (SELECT DISTINCT idActa FROM sp)) AS fue_evaluada_modelo
        FROM svo GROUP BY idActa
    """, destino)

    conteos["dim_eleccion"] = _wr(con, "dim_eleccion", f"""
        SELECT {ID_ELECCION} AS idEleccion,
               'Elecciones Generales 2026 - Presidencial' AS nombre_eleccion,
               'Escrutinio' AS tipo_acta
    """, destino)

    # --- Facts ---
    conteos["fact_resultados_oficiales"] = _wr(con, "fact_resultados_oficiales", """
        SELECT idActa, nposicion, ubigeo_distrito, idEleccion, nvotos_oficial
        FROM svo
    """, destino)

    conteos["fact_qa_modelo"] = _wr(con, "fact_qa_modelo", """
        WITH conf AS (
            SELECT archivoId, field, bool_or(confianza_baja) AS confianza_baja
            FROM sdc GROUP BY archivoId, field
        )
        SELECT p.idActa, p.nposicion, p.field, p.archivoId,
               da.ubigeo_distrito,
               p.nvotos_oficial_eval             AS nvotos_oficial,
               p.nvotos_predicho,
               p.error,
               abs(p.error)                      AS abs_error,
               p.correcto                        AS match_flag,
               COALESCE(c.confianza_baja, FALSE) AS confianza_baja
        FROM sp p
        LEFT JOIN dim_acta da ON da.idActa = p.idActa
        LEFT JOIN conf c ON c.archivoId = p.archivoId AND c.field = p.field
    """, destino)

    # Intermedia (no se escribe): agregado por acta para KPIs/marts de calidad.
    con.execute("""
        CREATE TEMP TABLE qa_acta AS
        WITH agg AS (
            SELECT idActa,
                   SUM(CASE WHEN field <> 'total_ciudadanos' THEN nvotos_predicho ELSE 0 END) AS sum_pred,
                   SUM(CASE WHEN field <> 'total_ciudadanos' THEN nvotos_oficial  ELSE 0 END) AS sum_real,
                   BOOL_AND(match_flag) AS acta_correcta
            FROM fact_qa_modelo GROUP BY idActa
        )
        SELECT idActa, sum_pred, sum_real,
               (sum_pred - sum_real)      AS err_acta,
               abs(sum_pred - sum_real)   AS abs_err_acta,
               acta_correcta
        FROM agg
    """)

    # --- Marts ---
    kpis = {
        "digit_acc": con.execute("SELECT avg((pred=label)::int) FROM sdc").fetchone()[0],
        "field_acc": con.execute("SELECT avg(match_flag::int) FROM fact_qa_modelo").fetchone()[0],
        "acta_acc": con.execute("SELECT avg(acta_correcta::int) FROM qa_acta").fetchone()[0],
        "mae_total_agregado": con.execute("SELECT avg(abs_err_acta) FROM qa_acta").fetchone()[0],
        "mediana_abs_error": con.execute("SELECT median(abs_err_acta) FROM qa_acta").fetchone()[0],
        "pct_reconstruccion_exacta": con.execute("SELECT avg((err_acta=0)::int) FROM qa_acta").fetchone()[0],
        "pct_error_le_5": con.execute("SELECT avg((abs_err_acta<=5)::int) FROM qa_acta").fetchone()[0],
        "n_actas_evaluadas": con.execute("SELECT count(*) FROM qa_acta").fetchone()[0],
        "n_campos_evaluados": con.execute("SELECT count(*) FROM fact_qa_modelo").fetchone()[0],
        "checkpoint": CHECKPOINT_OFICIAL,
        "id_corrida": uuid.uuid4().hex,
    }
    kpis_df = pd.DataFrame([kpis])
    io_delta.escribir_delta(kpis_df, "gold", "mart_kpis_globales", destino=destino)
    conteos["mart_kpis_globales"] = 1
    print(f"[gold] mart_kpis_globales            digit={kpis['digit_acc']:.4f} "
          f"field={kpis['field_acc']:.4f} acta={kpis['acta_acc']:.4f} MAE={kpis['mae_total_agregado']:.2f}")

    conteos["mart_calidad_por_departamento"] = _wr(con, "mart_calidad_por_departamento", """
        WITH fa AS (
            SELECT du.nombre_departamento AS depto, avg(f.match_flag::int) AS field_acc
            FROM fact_qa_modelo f
            LEFT JOIN dim_acta da ON da.idActa = f.idActa
            LEFT JOIN dim_ubicacion du ON du.ubigeo_distrito = da.ubigeo_distrito
            GROUP BY 1
        ),
        aa AS (
            SELECT du.nombre_departamento AS depto, count(*) AS n_actas,
                   avg((q.err_acta=0)::int) AS pct_reconstruccion_exacta,
                   avg(q.abs_err_acta) AS mae
            FROM qa_acta q
            LEFT JOIN dim_acta da ON da.idActa = q.idActa
            LEFT JOIN dim_ubicacion du ON du.ubigeo_distrito = da.ubigeo_distrito
            GROUP BY 1
        )
        SELECT aa.depto AS nombre_departamento, aa.n_actas,
               aa.pct_reconstruccion_exacta, aa.mae, fa.field_acc
        FROM aa LEFT JOIN fa ON fa.depto = aa.depto
        ORDER BY aa.n_actas DESC
    """, destino, registrar=False)

    conteos["mart_calidad_por_partido"] = _wr(con, "mart_calidad_por_partido", """
        SELECT o.nposicion, o.nombre_organizacion, count(*) AS n_campos,
               avg(f.match_flag::int) AS field_acc, avg(f.abs_error) AS mae_campo
        FROM fact_qa_modelo f
        JOIN dim_organizacion_politica o ON o.nposicion = f.nposicion
        WHERE f.nposicion IS NOT NULL
        GROUP BY 1, 2 ORDER BY o.nposicion
    """, destino, registrar=False)

    conteos["mart_resultados_por_departamento"] = _wr(con, "mart_resultados_por_departamento", """
        SELECT du.nombre_departamento, o.nombre_organizacion, o.nposicion,
               sum(fr.nvotos_oficial) AS votos_totales
        FROM fact_resultados_oficiales fr
        LEFT JOIN dim_ubicacion du ON du.ubigeo_distrito = fr.ubigeo_distrito
        LEFT JOIN dim_organizacion_politica o ON o.nposicion = fr.nposicion
        GROUP BY 1, 2, 3
    """, destino, registrar=False)

    conteos["mart_peores_actas"] = _wr(con, "mart_peores_actas", """
        SELECT w.archivoId, w.total_pred, w.total_real, w.error_total,
               w.n_fields_wrong, w.fields_wrong,
               da.estado_acta, du.nombre_departamento
        FROM worst w
        LEFT JOIN dim_acta da ON da.archivoId = w.archivoId
        LEFT JOIN dim_ubicacion du ON du.ubigeo_distrito = da.ubigeo_distrito
        ORDER BY abs(w.error_total) DESC
    """, destino, registrar=False)

    conteos["mart_error_hist"] = _wr(con, "mart_error_hist", """
        SELECT bin_idx, bin,
               count(*) FILTER (WHERE in_bin) AS n_actas
        FROM (
            SELECT idActa,
                   CASE WHEN abs_err_acta=0 THEN 0 WHEN abs_err_acta<=1 THEN 1
                        WHEN abs_err_acta<=2 THEN 2 WHEN abs_err_acta<=3 THEN 3
                        WHEN abs_err_acta<=5 THEN 4 WHEN abs_err_acta<=10 THEN 5
                        WHEN abs_err_acta<=20 THEN 6 WHEN abs_err_acta<=50 THEN 7
                        WHEN abs_err_acta<=100 THEN 8 ELSE 9 END AS bin_idx,
                   CASE WHEN abs_err_acta=0 THEN '0' WHEN abs_err_acta<=1 THEN '1'
                        WHEN abs_err_acta<=2 THEN '2' WHEN abs_err_acta<=3 THEN '3'
                        WHEN abs_err_acta<=5 THEN '4-5' WHEN abs_err_acta<=10 THEN '6-10'
                        WHEN abs_err_acta<=20 THEN '11-20' WHEN abs_err_acta<=50 THEN '21-50'
                        WHEN abs_err_acta<=100 THEN '51-100' ELSE '100+' END AS bin,
                   TRUE AS in_bin
            FROM qa_acta
        ) GROUP BY bin_idx, bin ORDER BY bin_idx
    """, destino, registrar=False)

    con.close()
    return conteos
