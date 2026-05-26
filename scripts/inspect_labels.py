"""Inspecciona los parquets de labels descargados en data/labels/.

Para cada archivo imprime: numero de filas, columnas con sus dtypes y
una muestra de las primeras 3 filas. La salida es texto plano apto para
redirigirse a un archivo de documentacion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# Carpeta donde viven los parquets ya descargados desde GCS.
LABELS_DIR = Path(__file__).resolve().parent.parent / "data" / "labels"

# Orden explicito para que el reporte sea estable entre ejecuciones.
PARQUET_FILES: tuple[str, ...] = (
    "actas_archivos.parquet",
    "actas_cabecera.parquet",
    "actas_votos.parquet",
    "mesas.parquet",
    "departamentos.parquet",
)


def inspeccionar_parquet(ruta: Path) -> None:
    # Carga el parquet en memoria y reporta su forma y esquema.
    df = pd.read_parquet(ruta)

    print(f"## {ruta.name}")
    print()
    print(f"- filas: {len(df):,}")
    print(f"- columnas: {len(df.columns)}")
    print()

    print("### dtypes")
    print("```")
    for nombre, dtype in df.dtypes.items():
        print(f"{nombre}: {dtype}")
    print("```")
    print()

    print("### muestra (3 filas)")
    print("```")
    # to_string evita truncar columnas y mantiene salida en texto plano.
    print(df.head(3).to_string(index=False))
    print("```")
    print()


def main() -> int:
    if not LABELS_DIR.exists():
        print(f"ERROR: no existe el directorio {LABELS_DIR}", file=sys.stderr)
        return 1

    for nombre in PARQUET_FILES:
        ruta = LABELS_DIR / nombre
        if not ruta.exists():
            print(f"ERROR: falta el archivo {ruta}", file=sys.stderr)
            return 1
        inspeccionar_parquet(ruta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
