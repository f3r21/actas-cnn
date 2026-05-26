"""Deteccion del entorno de ejecucion y rutas portables.

Permite que el mismo codigo corra en Kaggle, Colab o local (M2) sin cambios.
"""
import os
from pathlib import Path


def detect_env() -> str:
    if os.path.isdir("/kaggle"):
        return "kaggle"
    if "COLAB_GPU" in os.environ or os.path.isdir("/content"):
        return "colab"
    return "local"


def base_dir() -> Path:
    env = detect_env()
    if env == "kaggle":
        return Path("/kaggle/working")
    if env == "colab":
        return Path("/content")
    return Path(__file__).resolve().parent


def data_dir() -> Path:
    d = base_dir() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoints_dir() -> Path:
    d = base_dir() / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def torch_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
