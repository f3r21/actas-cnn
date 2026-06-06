"""Configuracion central del proyecto actas-cnn.

Define identificadores de repos remotos (Hugging Face para dataset y
modelo) y los hiperparametros de entrenamiento. Los repos por defecto
apuntan a `f3r21/actas-cnn-{dataset,model}`; ajusta `RemoteConfig` si
forkeas el proyecto.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteConfig:
    # Hugging Face: "usuario/nombre-repo"
    hf_dataset_repo: str = "f3r21/actas-cnn-dataset"
    hf_model_repo: str = "f3r21/actas-cnn-model"
    # Weights & Biases
    wandb_project: str = "actas-cnn"
    wandb_entity: str = ""  # vacio = entidad por defecto de tu cuenta
    # Cloudflare R2 (S3 compatible); credenciales por variables de entorno
    r2_bucket: str = "actas-cnn"
    r2_endpoint: str = ""  # https://<accountid>.r2.cloudflarestorage.com


@dataclass(frozen=True)
class TrainConfig:
    num_classes: int = 10
    in_channels: int = 1   # 1 = escala de grises
    image_size: int = 32
    batch_size: int = 128
    epochs: int = 20
    lr: float = 5e-4
    seed: int = 42
    arch: str = "resnet18"  # modelo del proyecto; "lenet"/"deep" = baselines Sem 1


REMOTE = RemoteConfig()
TRAIN = TrainConfig()
