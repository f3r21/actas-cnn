"""Capa de almacenamiento redundante.

Sube y descarga artefactos (dataset, checkpoints) a varios servicios gratis:
Hugging Face Hub, Weights & Biases y Cloudflare R2 (S3). Cada backend es
opcional: si faltan credenciales o la libreria, se omite sin romper el flujo.

Idea: no depender de un solo proveedor. Se sube a todos los disponibles y se
descarga probando en orden hasta que uno responda.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from config import REMOTE

# Orden de preferencia para descargar.
DOWNLOAD_ORDER = ("hf", "r2", "wandb")


def _hf_available() -> bool:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("HF_TOKEN"))


def _r2_available() -> bool:
    try:
        import boto3  # noqa: F401
    except ImportError:
        return False
    has_keys = all(os.environ.get(k) for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"))
    return has_keys and bool(REMOTE.r2_endpoint)


def _wandb_available() -> bool:
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("WANDB_API_KEY"))


# ---- Hugging Face -----------------------------------------------------------
def _hf_upload(local_path, remote_name, repo_id, repo_type):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=str(local_path), path_in_repo=remote_name,
                    repo_id=repo_id, repo_type=repo_type)


def _hf_download(remote_name, dest, repo_id, repo_type):
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(repo_id=repo_id, filename=remote_name,
                            repo_type=repo_type, token=os.environ.get("HF_TOKEN"))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cached, dest)


# ---- Cloudflare R2 (S3) -----------------------------------------------------
def _r2_client():
    import boto3
    return boto3.client(
        "s3", endpoint_url=REMOTE.r2_endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )


def _r2_upload(local_path, remote_name):
    client = _r2_client()
    client.upload_file(str(local_path), REMOTE.r2_bucket, remote_name)


def _r2_download(remote_name, dest):
    client = _r2_client()
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    client.download_file(REMOTE.r2_bucket, remote_name, str(dest))


# ---- Weights & Biases -------------------------------------------------------
def _wandb_upload(local_path, remote_name, kind):
    import wandb
    run = wandb.init(project=REMOTE.wandb_project, entity=REMOTE.wandb_entity or None,
                     job_type="upload", reinit=True)
    art = wandb.Artifact(name=remote_name.replace("/", "_"), type=kind)
    art.add_file(str(local_path))
    run.log_artifact(art)
    run.finish()


def _wandb_download(remote_name, dest, kind):
    import wandb
    api = wandb.Api()
    name = remote_name.replace("/", "_")
    entity = REMOTE.wandb_entity or api.default_entity
    art = api.artifact(f"{entity}/{REMOTE.wandb_project}/{name}:latest", type=kind)
    folder = art.download()
    src = Path(folder) / Path(remote_name).name
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)


# ---- API publica ------------------------------------------------------------
def _repo_for(kind):
    if kind == "model":
        return REMOTE.hf_model_repo, "model"
    return REMOTE.hf_dataset_repo, "dataset"


def upload(local_path, remote_name, kind="dataset"):
    """Sube a todos los backends disponibles. Devuelve la lista de los que funcionaron."""
    local_path = Path(local_path)
    repo_id, repo_type = _repo_for(kind)
    done = []
    if _hf_available():
        try:
            _hf_upload(local_path, remote_name, repo_id, repo_type)
            done.append("hf")
        except Exception as exc:
            print(f"[storage] HF upload fallo: {exc}")
    if _r2_available():
        try:
            _r2_upload(local_path, remote_name)
            done.append("r2")
        except Exception as exc:
            print(f"[storage] R2 upload fallo: {exc}")
    if _wandb_available():
        try:
            _wandb_upload(local_path, remote_name, kind)
            done.append("wandb")
        except Exception as exc:
            print(f"[storage] W&B upload fallo: {exc}")
    if not done:
        print("[storage] ningun backend disponible; el artefacto queda solo en local")
    return done


def download(remote_name, dest, kind="dataset"):
    """Descarga probando backends en orden hasta que uno funcione."""
    repo_id, repo_type = _repo_for(kind)
    for backend in DOWNLOAD_ORDER:
        try:
            if backend == "hf" and _hf_available():
                _hf_download(remote_name, dest, repo_id, repo_type)
                return backend
            if backend == "r2" and _r2_available():
                _r2_download(remote_name, dest)
                return backend
            if backend == "wandb" and _wandb_available():
                _wandb_download(remote_name, dest, kind)
                return backend
        except Exception as exc:
            print(f"[storage] {backend} download fallo: {exc}")
    raise RuntimeError(f"No se pudo descargar {remote_name} de ningun backend")
