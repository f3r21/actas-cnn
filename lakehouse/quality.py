"""Capa de calidad: expectativas sobre silver/gold. Output: QUALITY_REPORT.md.

Reusa el patron CheckResult de scripts/audit.py (aqui ChequeoCalidad, mismo
contrato n/claim/metodo/evidencia/resultado/notas) pero desacoplado del pipeline
de imagen: estas expectativas viven sobre las tablas Delta, no sobre PNGs/crops.

8 expectativas (cada una un query DuckDB sobre las tablas locales):
  1. Conteo silver vs fuente (reconciliacion del join).
  2. Integridad de FK en el star schema gold.
  3. Dominio de nposicion en silver.
  4. Sin nulls en nvotos_oficial.
  5. avg(match_flag) de fact_qa_modelo ~= field_acc de evaluate (bronze).
  6. MAE del total agregado gold == MAE de evaluate (bronze).  [assert clave]
  7. Sin leak: actas evaluadas en val, no en train/test.
  8. Grano unico en fact_resultados_oficiales y silver_predicciones.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from env import base_dir
from lakehouse import io_delta

REPORT_PATH = _REPO / "QUALITY_REPORT.md"
TOL_PP = 0.01      # tolerancia para field_acc (1 punto porcentual)
TOL_MAE = 0.01     # tolerancia para MAE
NPOSICIONES_VALIDAS = set(range(1, 39)) | {80, 81, 82}


@dataclass
class ChequeoCalidad:
    n: int
    titulo: str
    claim: str
    metodo: str
    evidencia: str
    resultado: str  # PASS | FAIL | WARNING
    notas: str = ""


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("svo", io_delta.leer_arrow("silver", "silver_votos_oficial"))
    con.register("sp", io_delta.leer_arrow("silver", "silver_predicciones"))
    con.register("pred", io_delta.leer_arrow("bronze", "pred_evaluate_val"))
    con.register("arch", io_delta.leer_arrow("bronze", "actas_archivos"))
    con.register("votos", io_delta.leer_arrow("bronze", "actas_votos"))
    con.register("dim_acta", io_delta.leer_arrow("gold", "dim_acta"))
    con.register("dim_org", io_delta.leer_arrow("gold", "dim_organizacion_politica"))
    con.register("dim_ubic", io_delta.leer_arrow("gold", "dim_ubicacion"))
    con.register("fqa", io_delta.leer_arrow("gold", "fact_qa_modelo"))
    con.register("fro", io_delta.leer_arrow("gold", "fact_resultados_oficiales"))
    return con


def _q(con, sql: str):
    return con.execute(sql).fetchone()[0]


def chk1_conteo_silver(con) -> ChequeoCalidad:
    svo = _q(con, "SELECT count(*) FROM svo")
    ref = _q(con, """SELECT count(*) FROM votos v
                     WHERE v.idEleccion=10 AND v.idActa IN
                       (SELECT idActa FROM arch WHERE tipo=1 AND idEleccion=10)""")
    ok = svo == ref
    return ChequeoCalidad(1, "Conteo silver vs fuente",
        "silver_votos_oficial = votos presidenciales de actas de escrutinio (sin fan-out ni perdida).",
        "count(silver_votos_oficial) vs count(votos idEleccion=10 de archivos tipo=1).",
        f"svo={svo:,} ref={ref:,}", "PASS" if ok else "FAIL",
        "" if ok else "El join silver no reconcilia con la fuente.")


def chk2_fk_gold(con) -> ChequeoCalidad:
    fqa_acta = _q(con, "SELECT count(*) FROM fqa f LEFT JOIN dim_acta d ON d.idActa=f.idActa WHERE d.idActa IS NULL")
    fqa_npos = _q(con, "SELECT count(*) FROM fqa f WHERE f.nposicion IS NOT NULL AND f.nposicion NOT IN (SELECT nposicion FROM dim_org)")
    fro_acta = _q(con, "SELECT count(*) FROM fro f LEFT JOIN dim_acta d ON d.idActa=f.idActa WHERE d.idActa IS NULL")
    fro_npos = _q(con, "SELECT count(*) FROM fro f WHERE f.nposicion NOT IN (SELECT nposicion FROM dim_org)")
    fro_ubic = _q(con, "SELECT count(*) FROM fro f WHERE f.ubigeo_distrito IS NOT NULL AND f.ubigeo_distrito NOT IN (SELECT ubigeo_distrito FROM dim_ubic)")
    huerfanos = fqa_acta + fqa_npos + fro_acta + fro_npos + fro_ubic
    ok = huerfanos == 0
    return ChequeoCalidad(2, "Integridad de FK en gold",
        "Toda FK de fact_qa_modelo y fact_resultados_oficiales existe en su dimension.",
        "Anti-joins fact -> dim_acta / dim_organizacion_politica / dim_ubicacion.",
        f"huerfanos: fqa_acta={fqa_acta}, fqa_npos={fqa_npos}, fro_acta={fro_acta}, fro_npos={fro_npos}, fro_ubic={fro_ubic}",
        "PASS" if ok else "FAIL", "" if ok else "Hay claves foraneas sin dimension.")


def chk3_dominio_nposicion(con) -> ChequeoCalidad:
    validas = ",".join(str(x) for x in sorted(NPOSICIONES_VALIDAS))
    fuera = _q(con, f"SELECT count(*) FROM svo WHERE nposicion NOT IN ({validas})")
    ok = fuera == 0
    return ChequeoCalidad(3, "Dominio de nposicion",
        "nposicion en {1..38, 80, 81, 82}.",
        "count de filas con nposicion fuera del dominio.",
        f"fuera_de_dominio={fuera}", "PASS" if ok else "FAIL",
        "" if ok else "nposicion inesperada en silver_votos_oficial.")


def chk4_sin_nulls_nvotos(con) -> ChequeoCalidad:
    n_svo = _q(con, "SELECT count(*) FROM svo WHERE nvotos_oficial IS NULL")
    n_fro = _q(con, "SELECT count(*) FROM fro WHERE nvotos_oficial IS NULL")
    ok = (n_svo == 0 and n_fro == 0)
    return ChequeoCalidad(4, "Sin nulls en nvotos_oficial",
        "nvotos_oficial nunca es null (los votos existen aunque el total del acta sea NaN).",
        "count de nulls en silver_votos_oficial y fact_resultados_oficiales.",
        f"nulls_svo={n_svo}, nulls_fro={n_fro}", "PASS" if ok else "FAIL",
        "" if ok else "Hay nvotos_oficial nulos.")


def chk5_match_flag_vs_field_acc(con) -> ChequeoCalidad:
    gold = _q(con, "SELECT avg(match_flag::int) FROM fqa")
    evalu = _q(con, "SELECT avg(correct::int) FROM pred")
    diff = abs(gold - evalu)
    ok = diff <= TOL_PP
    return ChequeoCalidad(5, "match_flag de gold ~= field_acc de evaluate",
        "avg(match_flag) de fact_qa_modelo coincide con field_acc de evaluate (bronze).",
        "avg(match_flag) gold vs avg(correct) de pred_evaluate_val.",
        f"gold={gold:.4f} evaluate={evalu:.4f} diff={diff:.4f} (tol={TOL_PP})",
        "PASS" if ok else "FAIL", "" if ok else "field-level accuracy no reconcilia: join de prediccion roto.")


def chk6_mae_reconciliacion(con) -> ChequeoCalidad:
    sql_mae = """
        WITH agg AS (
            SELECT {id} AS k,
                   SUM(CASE WHEN {fld} <> 'total_ciudadanos' THEN {p} ELSE 0 END) AS sp,
                   SUM(CASE WHEN {fld} <> 'total_ciudadanos' THEN {r} ELSE 0 END) AS sr
            FROM {t} GROUP BY {id}
        ) SELECT avg(abs(sp-sr)) FROM agg
    """
    mae_gold = _q(con, sql_mae.format(id="idActa", fld="field", p="nvotos_predicho", r="nvotos_oficial", t="fqa"))
    mae_eval = _q(con, sql_mae.format(id="archivoId", fld="field", p="pred", r="real", t="pred"))
    diff = abs(mae_gold - mae_eval)
    ok = diff <= TOL_MAE
    return ChequeoCalidad(6, "MAE del total agregado gold == evaluate",
        "El MAE del total agregado calculado sobre gold coincide con evaluate (bronze).",
        "MAE = avg(|sum_pred - sum_real|) por acta (sin total_ciudadanos) en gold vs bronze.",
        f"mae_gold={mae_gold:.4f} mae_evaluate={mae_eval:.4f} diff={diff:.4f} (tol={TOL_MAE})",
        "PASS" if ok else "FAIL", "" if ok else "El MAE de gold no coincide con el reporte del modelo.")


def chk7_sin_leak(con) -> ChequeoCalidad:
    def _ids(nombre):
        p = base_dir() / "data" / "splits" / f"{nombre}_ids.txt"
        return {l.strip() for l in p.read_text().splitlines() if l.strip()}
    val, train, test = _ids("val"), _ids("train"), _ids("test")
    aids = {r[0] for r in con.execute("SELECT DISTINCT archivoId FROM fqa").fetchall()}
    fuera_val = aids - val
    en_train = aids & train
    en_test = aids & test
    ok = not fuera_val and not en_train and not en_test
    return ChequeoCalidad(7, "Sin leak entre splits",
        "Las actas evaluadas (fact_qa_modelo) estan todas en val y en ningun train/test.",
        "Set diff de archivoId de fact_qa_modelo vs val/train/test_ids.txt.",
        f"evaluadas={len(aids)} fuera_de_val={len(fuera_val)} en_train={len(en_train)} en_test={len(en_test)}",
        "PASS" if ok else "FAIL", "" if ok else "Hay leak de split en las actas evaluadas.")


def chk8_grano_unico(con) -> ChequeoCalidad:
    dup_fro = _q(con, "SELECT count(*) FROM (SELECT idActa,nposicion FROM fro GROUP BY 1,2 HAVING count(*)>1)")
    dup_sp = _q(con, "SELECT count(*) FROM (SELECT archivoId,field FROM sp GROUP BY 1,2 HAVING count(*)>1)")
    n_sp = _q(con, "SELECT count(*) FROM sp")
    ok = (dup_fro == 0 and dup_sp == 0 and n_sp == 29106)
    return ChequeoCalidad(8, "Grano unico",
        "(idActa,nposicion) unico en fact_resultados_oficiales; (archivoId,field) unico en silver_predicciones (29,106 filas).",
        "Conteo de grupos duplicados por grano + conteo de filas.",
        f"dup_fro={dup_fro} dup_sp={dup_sp} filas_sp={n_sp:,}",
        "PASS" if ok else "FAIL", "" if ok else "Grano duplicado o conteo inesperado.")


CHEQUEOS = [chk1_conteo_silver, chk2_fk_gold, chk3_dominio_nposicion, chk4_sin_nulls_nvotos,
            chk5_match_flag_vs_field_acc, chk6_mae_reconciliacion, chk7_sin_leak, chk8_grano_unico]


def render_md(resultados: list[ChequeoCalidad]) -> str:
    counts = {"PASS": 0, "FAIL": 0, "WARNING": 0}
    for r in resultados:
        counts[r.resultado] = counts.get(r.resultado, 0) + 1
    lines = ["# QUALITY REPORT -- Capa lakehouse (silver + gold)", "",
             "## Resumen", "",
             f"- **PASS**: {counts.get('PASS', 0)}",
             f"- **WARNING**: {counts.get('WARNING', 0)}",
             f"- **FAIL**: {counts.get('FAIL', 0)}", "",
             "## Detalle por expectativa", ""]
    for r in resultados:
        lines.append(f"### [EXP {r.n}] {r.titulo} -- **{r.resultado}**")
        lines.append("")
        lines.append(f"- **Claim:** {r.claim}")
        lines.append(f"- **Metodo:** {r.metodo}")
        lines.append(f"- **Evidencia:** {r.evidencia}")
        if r.notas:
            lines.append(f"- **Notas:** {r.notas}")
        lines.append("")
    return "\n".join(lines)


def construir(destino: str | None = None) -> dict[str, int]:
    print("== CALIDAD ==")
    con = _con()
    resultados = []
    for fn in CHEQUEOS:
        try:
            r = fn(con)
        except Exception as exc:
            r = ChequeoCalidad(0, fn.__name__, "-", "-", f"excepcion: {exc}", "FAIL", str(exc))
        print(f"[calidad] EXP {r.n}: {r.resultado}  {r.evidencia}")
        resultados.append(r)
    con.close()
    REPORT_PATH.write_text(render_md(resultados))
    counts = {"PASS": 0, "FAIL": 0, "WARNING": 0}
    for r in resultados:
        counts[r.resultado] = counts.get(r.resultado, 0) + 1
    print(f"[calidad] QUALITY_REPORT -> {REPORT_PATH}  ({counts})")
    return counts


if __name__ == "__main__":
    construir()
