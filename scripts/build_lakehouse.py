"""Orquestador del lakehouse medallion: bronze -> silver -> gold -> calidad -> export.

Uso:
  python scripts/build_lakehouse.py --capa todo --destino local
  python scripts/build_lakehouse.py --capa silver
  python scripts/build_lakehouse.py --capa bronce --destino ambos   # local + ADLS

--destino: local | adls | ambos. Si se omite, usa LAKEHOUSE_DESTINO del .env
(default local). La copia local siempre se escribe (es la fuente de trabajo de las
capas siguientes); adls/ambos ademas espejan a ADLS.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CAPAS_ORDEN = ["bronce", "silver", "gold", "calidad", "export"]
# Capas que aun pueden no estar implementadas durante el desarrollo incremental,
# con el modulo exacto que las implementa (se compara contra exc.name, no substring,
# para no tragarse un ImportError real dentro de un modulo que si existe).
MODULO_DE_CAPA = {"gold": "lakehouse.gold", "calidad": "lakehouse.quality",
                  "export": "lakehouse.export_sheets"}


def _cargar_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO / ".env")
    except Exception:
        pass


def _correr(capa: str, destino: str | None):
    if capa == "bronce":
        from lakehouse import bronze
        return bronze.construir(destino)
    if capa == "silver":
        from lakehouse import silver
        return silver.construir(destino)
    if capa == "gold":
        from lakehouse import gold
        return gold.construir(destino)
    if capa == "calidad":
        from lakehouse import quality
        return quality.construir(destino)
    if capa == "export":
        from lakehouse import export_sheets
        return export_sheets.construir(destino)
    raise ValueError(f"capa desconocida: {capa}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capa", choices=["bronce", "silver", "gold", "calidad", "export", "todo"],
                    default="todo")
    ap.add_argument("--destino", choices=["local", "adls", "ambos"], default=None)
    args = ap.parse_args()
    _cargar_env()

    plan = CAPAS_ORDEN if args.capa == "todo" else [args.capa]
    ejecutadas = []
    for capa in plan:
        try:
            _correr(capa, args.destino)
            ejecutadas.append(capa)
        except ModuleNotFoundError as exc:
            if exc.name == MODULO_DE_CAPA.get(capa):
                print(f"[build_lakehouse] {capa}: modulo aun no disponible, se omite")
                continue
            raise
    print(f"\nlakehouse: capas ejecutadas {ejecutadas} (destino={args.destino or 'env/default'})")


if __name__ == "__main__":
    main()
