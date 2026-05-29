"""Fuente unica del mapeo field <-> nposicion (ONPE) y del valor oficial por field.

Antes esta logica estaba duplicada en `scripts/build_crops.py` (field_value_for) y
`scripts/evaluate.py` (real_value_for). Aqui se centraliza para que el pipeline de
crops, la evaluacion y la capa silver del lakehouse no puedan divergir.

Convencion del template Presidencial (42 fields):
- partido_NN      -> nposicion = NN          (1..38, partidos)
- votos_blanco    -> nposicion = 80          (especial)
- votos_nulos     -> nposicion = 81          (especial)
- votos_impugnados-> nposicion = 82          (especial)
- total_ciudadanos-> sin nposicion; su valor sale de cabecera.totalVotosEmitidos
"""
from __future__ import annotations

import pandas as pd

N_PARTIDOS = 38
NPOSICION_BLANCO = 80
NPOSICION_NULOS = 81
NPOSICION_IMPUGNADOS = 82
FIELD_TOTAL = "total_ciudadanos"

# field name -> nposicion para los especiales (no partido, no total).
_ESPECIALES = {
    "votos_blanco": NPOSICION_BLANCO,
    "votos_nulos": NPOSICION_NULOS,
    "votos_impugnados": NPOSICION_IMPUGNADOS,
}


def field_a_nposicion(name: str) -> int | None:
    """Devuelve la nposicion ONPE de un field, o None para total_ciudadanos.

    Levanta ValueError si el field no pertenece al template conocido.
    """
    if name.startswith("partido_"):
        return int(name.split("_")[1])
    if name in _ESPECIALES:
        return _ESPECIALES[name]
    if name == FIELD_TOTAL:
        return None
    raise ValueError(f"field desconocido: {name}")


def valor_oficial_para(name: str, votos_acta: pd.DataFrame, total_emitidos: int) -> int:
    """Entero del ground truth oficial para un field.

    `votos_acta` debe ser el subconjunto de actas_votos de UNA acta (ya filtrado por
    idActa). Mantiene exactamente el comportamiento de las antiguas field_value_for /
    real_value_for: si la posicion no existe en votos_acta, devuelve 0.
    """
    npos = field_a_nposicion(name)
    if npos is None:  # total_ciudadanos
        return int(total_emitidos)
    row = votos_acta[votos_acta["nposicion"] == npos]
    return int(row.iloc[0]["nvotos"]) if len(row) else 0
