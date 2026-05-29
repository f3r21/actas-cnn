"""Rutas locales y URIs ADLS Gen2 para las capas del lakehouse.

Local: <repo>/data/lakehouse/<capa>/<tabla> (Delta dir; gitignored via data/).
ADLS:  abfss://lakehouse@actaslake25067.dfs.core.windows.net/<capa>/<tabla>

Reusa env.base_dir() para portabilidad (local / Kaggle / Colab).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from env import base_dir

ACCOUNT = "actaslake25067"
FILESYSTEM = "lakehouse"
CAPAS = ("bronze", "silver", "gold")


def adls_uri(capa: str, tabla: str) -> str:
    """URI abfss canonico (object_store). No usar az://."""
    return f"abfss://{FILESYSTEM}@{ACCOUNT}.dfs.core.windows.net/{capa}/{tabla}"


def local_dir(capa: str, tabla: str) -> Path:
    return base_dir() / "data" / "lakehouse" / capa / tabla


def storage_options() -> dict:
    """Credenciales para delta-rs hacia ADLS. Lanza si falta la access key."""
    key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
    if not key:
        raise RuntimeError(
            "AZURE_STORAGE_ACCOUNT_KEY no esta en el entorno. "
            "Definela en .env para escribir a ADLS (LAKEHOUSE_DESTINO=adls|ambos)."
        )
    return {"account_name": ACCOUNT, "account_key": key}


def destino_default() -> str:
    """local | adls | ambos. Default local (ADLS es opt-in via .env)."""
    return os.environ.get("LAKEHOUSE_DESTINO", "local").lower()
