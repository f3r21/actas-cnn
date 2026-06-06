"""Almacenamiento en Hugging Face Hub (dataset y modelo).

Sube y descarga artefactos a HF. Subir requiere `HF_TOKEN` con permiso de
escritura; descargar de un repo publico no necesita token. Si falta el token o
la libreria, `upload` no rompe el flujo: avisa y deja el artefacto en local.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .config import REMOTE


def _hf_available() -> bool:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("HF_TOKEN"))


def _repo_for(kind: str) -> tuple[str, str]:
    if kind == "model":
        return REMOTE.hf_model_repo, "model"
    return REMOTE.hf_dataset_repo, "dataset"


def upload(local_path, remote_name, kind="dataset") -> bool:
    """Sube un archivo a HF (kind='dataset'|'model'). Devuelve True si subio."""
    if not _hf_available():
        print("[storage] HF no disponible (falta huggingface_hub o HF_TOKEN); "
              "el artefacto queda solo en local")
        return False
    from huggingface_hub import HfApi
    repo_id, repo_type = _repo_for(kind)
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=str(local_path), path_in_repo=remote_name,
                    repo_id=repo_id, repo_type=repo_type)
    return True


def download(remote_name, dest, kind="dataset") -> str:
    """Descarga un archivo de HF a `dest`."""
    from huggingface_hub import hf_hub_download
    repo_id, repo_type = _repo_for(kind)
    cached = hf_hub_download(repo_id=repo_id, filename=remote_name,
                            repo_type=repo_type, token=os.environ.get("HF_TOKEN"))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cached, dest)
    return "hf"
