"""Export de los marts gold para el dashboard (Looker Studio / Databricks SQL).

- CSV (siempre): un archivo por mart en data/lakehouse/gold_export/. Sirve como
  fuente para Looker Studio (subida de CSV o carga a Google Sheets) y para
  re-subir a un volumen administrado de Databricks Free Edition.
- Google Sheets (opcional): si GOOGLE_SHEETS_SA_JSON y GOOGLE_SHEETS_SPREADSHEET_ID
  estan definidos y gspread esta instalado, escribe un tab por mart. Si no, se
  omite con un mensaje (el CSV es suficiente para el dashboard).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from env import base_dir
from lakehouse import io_delta

MARTS = (
    "mart_kpis_globales",
    "mart_calidad_por_departamento",
    "mart_calidad_por_partido",
    "mart_resultados_por_departamento",
    "mart_peores_actas",
    "mart_error_hist",
)


def _export_csv() -> Path:
    out_dir = base_dir() / "data" / "lakehouse" / "gold_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    for mart in MARTS:
        df = io_delta.leer_arrow("gold", mart).to_pandas()
        destino = out_dir / f"{mart}.csv"
        df.to_csv(destino, index=False)
        print(f"[export] {mart:32s} {len(df):>7,} filas -> {destino}")
    return out_dir


def _export_sheets() -> bool:
    sa_json = os.environ.get("GOOGLE_SHEETS_SA_JSON")
    spreadsheet_id = os.environ.get("GOOGLE_SHEETS_SPREADSHEET_ID")
    if not sa_json or not spreadsheet_id:
        print("[export] Google Sheets omitido (define GOOGLE_SHEETS_SA_JSON y "
              "GOOGLE_SHEETS_SPREADSHEET_ID para activarlo).")
        return False
    try:
        import gspread
        from gspread_dataframe import set_with_dataframe
    except ImportError:
        print("[export] Google Sheets omitido (instala gspread y gspread-dataframe).")
        return False

    gc = gspread.service_account(filename=sa_json)
    sh = gc.open_by_key(spreadsheet_id)
    for mart in MARTS:
        df = io_delta.leer_arrow("gold", mart).to_pandas()
        try:
            ws = sh.worksheet(mart)
            ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=mart, rows=max(len(df) + 1, 10), cols=max(len(df.columns), 4))
        set_with_dataframe(ws, df)
        print(f"[export] Sheets tab '{mart}' actualizado ({len(df)} filas)")
    return True


def construir(destino: str | None = None) -> dict[str, str]:
    print("== EXPORT ==")
    out_dir = _export_csv()
    sheets_ok = _export_sheets()
    return {"csv_dir": str(out_dir), "sheets": "ok" if sheets_ok else "omitido"}


if __name__ == "__main__":
    construir()
