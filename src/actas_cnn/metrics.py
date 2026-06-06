"""Metricas y tablas para el informe (Cap. 4): confusion, P/R/F1, ablations.

Funciones puras (numpy/pandas), sin estado ni I/O salvo leer CSVs de evaluacion.
Las comparte el paquete (`evaluate.py`) y el notebook entregable para no duplicar
la logica de reporte.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def confusion_matrix(labels, preds, n_classes: int = 10) -> np.ndarray:
    """Matriz de confusion n_classes x n_classes (filas=real, columnas=predicho)."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(labels), np.asarray(preds)):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(confusion: np.ndarray) -> pd.DataFrame:
    """precision / recall / F1 / soporte por clase a partir de la confusion."""
    n = confusion.shape[0]
    recall = np.array([confusion[i, i] / max(confusion[i].sum(), 1) for i in range(n)])
    precision = np.array([confusion[i, i] / max(confusion[:, i].sum(), 1) for i in range(n)])
    f1 = np.array([2 * p * r / max(p + r, 1e-9) for p, r in zip(precision, recall)])
    support = confusion.sum(axis=1)
    return pd.DataFrame(
        {"clase": np.arange(n), "n": support,
         "precision": precision, "recall": recall, "f1": f1}
    )


def metrics_from_eval_csv(eval_csv: "str | Path") -> dict:
    """Field/acta-level + reconstruccion del total desde un CSV de evaluate.py.

    El CSV trae una fila por (archivoId, field) con columnas pred/real/correct.
    Nota: el digit-level (cell-level) NO esta en este CSV; se obtiene corriendo
    evaluate.py, que lo imprime aparte.
    """
    res = pd.read_csv(eval_csv)
    field_acc = float(res["correct"].mean())
    acta_acc = float(res.groupby("archivoId")["correct"].all().mean())
    no_total = res[res["field"] != "total_ciudadanos"]
    sum_pred = no_total.groupby("archivoId")["pred"].sum()
    sum_real = no_total.groupby("archivoId")["real"].sum()
    err = (sum_pred - sum_real).abs()
    return {
        "n_actas": int(res["archivoId"].nunique()),
        "n_fields": int(len(res)),
        "field_acc": field_acc,
        "acta_acc": acta_acc,
        "total_mae": float(err.mean()),
        "total_exact_pct": float((err == 0).mean() * 100.0),
    }


def ablations_table(csv_map: dict[str, "str | Path"]) -> pd.DataFrame:
    """Tabla comparativa de ablations.

    `csv_map`: {nombre_legible: ruta_al_evaluate_val_*.csv}. Devuelve un DataFrame
    con field-level, acta-level, MAE del total y % de reconstruccion exacta por
    variante. Para el Cap. 4 del informe (base vs ls_ra vs ls_ra_mu_cos).
    """
    rows = []
    for nombre, csv in csv_map.items():
        m = metrics_from_eval_csv(csv)
        rows.append({"variante": nombre, **m})
    return pd.DataFrame(rows).set_index("variante")
