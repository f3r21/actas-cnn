"""Escritura/lectura de tablas Delta (delta-rs, sin Spark) en local y/o ADLS.

La copia local es siempre la fuente de trabajo: las capas silver/gold leen el
bronze/silver local. El destino controla si ademas se espeja a ADLS:
- local : solo local.
- adls / ambos : local + ADLS (local es necesaria como copia de trabajo).

El selector de destino sigue el idioma de storage.py (variable de entorno
LAKEHOUSE_DESTINO), de modo que ADLS es opt-in y un run local nunca necesita la
access key de Azure.
"""
from __future__ import annotations

import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from lakehouse import paths


def _a_arrow(data) -> pa.Table:
    """Acepta pyarrow.Table o pandas.DataFrame y devuelve un pyarrow.Table saneado.

    delta-rs no soporta el tipo Arrow `null` (columnas enteramente NaN, p.ej. varias
    columnas de mesas.parquet). Se castean a string (siguen siendo all-null pero con
    tipo valido para Delta Lake). Los tipos dictionary (categoricals de bronze) si los
    soporta delta-rs >= 1.x y se dejan pasar; otros tipos no estandar (duration,
    large_list) necesitarian saneo adicional aqui si aparecieran.
    """
    tbl = data if isinstance(data, pa.Table) else pa.Table.from_pandas(data, preserve_index=False)
    for i in range(tbl.num_columns):
        if pa.types.is_null(tbl.field(i).type):
            tbl = tbl.set_column(i, tbl.field(i).with_type(pa.string()),
                                 tbl.column(i).cast(pa.string()))
    return tbl


def escribir_delta(data, capa: str, tabla: str, *,
                   partition_by: str | list[str] | None = None,
                   destino: str | None = None) -> list[str]:
    """Escribe un Delta (overwrite idempotente). Devuelve los targets escritos."""
    destino = (destino or paths.destino_default()).lower()
    tbl = _a_arrow(data)

    targets: list[str] = []
    local = paths.local_dir(capa, tabla)
    local.mkdir(parents=True, exist_ok=True)
    write_deltalake(str(local), tbl, mode="overwrite", schema_mode="overwrite",
                    partition_by=partition_by)
    targets.append(str(local))

    if destino in ("adls", "ambos"):
        uri = paths.adls_uri(capa, tabla)
        write_deltalake(uri, tbl, mode="overwrite", schema_mode="overwrite",
                        partition_by=partition_by, storage_options=paths.storage_options())
        targets.append(uri)
    return targets


def leer_arrow(capa: str, tabla: str) -> pa.Table:
    """Lee la copia local de un Delta como pyarrow.Table (para registrar en DuckDB)."""
    return DeltaTable(str(paths.local_dir(capa, tabla))).to_pyarrow_table()


def existe(capa: str, tabla: str) -> bool:
    return (paths.local_dir(capa, tabla) / "_delta_log").exists()
