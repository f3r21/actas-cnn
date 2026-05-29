"""Capa bronze: ingesta cruda de las fuentes a Delta + metadata de ingesta.

Fuentes (relativas a data/):
- 5 parquets ONPE: actas_archivos, actas_cabecera, actas_votos, mesas, departamentos.
- 3 salidas del modelo: evaluate_val.csv, eval_logits_val.parquet, evaluate_worst20_val.csv.

A cada tabla se le agregan 4 columnas de metadata: _ingesta_ts, _fuente,
_archivo_origen_hash (sha256 del archivo fuente) e _id_corrida (uuid por corrida).
Las columnas de string constantes se guardan como categorical para no inflar la
memoria en la tabla grande (actas_votos, 18.6M filas).
"""
from __future__ import annotations

import hashlib
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from env import base_dir
from lakehouse import io_delta

_META_CATEGORICAS = ("_fuente", "_archivo_origen_hash", "_id_corrida")


@dataclass(frozen=True)
class FuenteBronze:
    tabla: str
    ruta_rel: str            # relativo a data/
    kind: str                # 'parquet' | 'csv'
    partition_by: str | None = None


FUENTES: tuple[FuenteBronze, ...] = (
    FuenteBronze("actas_archivos", "labels/actas_archivos.parquet", "parquet"),
    FuenteBronze("actas_cabecera", "labels/actas_cabecera.parquet", "parquet"),
    FuenteBronze("actas_votos", "labels/actas_votos.parquet", "parquet", "idEleccion"),
    FuenteBronze("mesas", "labels/mesas.parquet", "parquet"),
    FuenteBronze("departamentos", "labels/departamentos.parquet", "parquet"),
    FuenteBronze("pred_evaluate_val", "evaluate_val.csv", "csv"),
    FuenteBronze("pred_eval_logits_val", "eval_logits_val.parquet", "parquet"),
    FuenteBronze("pred_worst20_val", "evaluate_worst20_val.csv", "csv"),
)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(chunk), b""):
            h.update(bloque)
    return h.hexdigest()


def _leer(fuente: FuenteBronze, ruta: Path) -> pd.DataFrame:
    return pd.read_parquet(ruta) if fuente.kind == "parquet" else pd.read_csv(ruta)


def ingestar(fuente: FuenteBronze, *, id_corrida: str, ts: datetime,
             destino: str | None) -> int | None:
    ruta = base_dir() / "data" / fuente.ruta_rel
    if not ruta.exists():
        print(f"[bronze] FALTA {ruta} -- se omite {fuente.tabla}")
        return None
    df = _leer(fuente, ruta)
    df["_ingesta_ts"] = ts
    df["_fuente"] = str(Path("data") / fuente.ruta_rel)
    df["_archivo_origen_hash"] = _sha256(ruta)
    df["_id_corrida"] = id_corrida
    for col in _META_CATEGORICAS:
        df[col] = df[col].astype("category")
    targets = io_delta.escribir_delta(df, "bronze", fuente.tabla,
                                      partition_by=fuente.partition_by, destino=destino)
    print(f"[bronze] {fuente.tabla:24s} {len(df):>10,} filas -> {targets[0]}")
    return len(df)


def construir(destino: str | None = None) -> dict[str, int]:
    id_corrida = uuid.uuid4().hex
    ts = datetime.now(timezone.utc)
    print(f"== BRONZE (id_corrida={id_corrida}) ==")
    conteos: dict[str, int] = {}
    for fuente in FUENTES:
        n = ingestar(fuente, id_corrida=id_corrida, ts=ts, destino=destino)
        if n is not None:
            conteos[fuente.tabla] = n
    return conteos
