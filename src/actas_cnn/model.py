"""Arquitecturas CNN en PyTorch para clasificacion de digitos (0-9).

ResNet18CIFAR: modelo del proyecto. ResNet-18 (He et al., 2015) adaptada
al estilo CIFAR (stem 3x3 stride 1, sin MaxPool inicial) para preservar
resolucion en entradas chicas.
LeNetCNN y DeepCNN: lineas de referencia metodologicas de Semana 1.
"""
import torch.nn as nn
from torchvision.models import resnet18 as _torchvision_resnet18


class LeNetCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        # Pool 3x3 por MPS: input 32x32 -> features 6x6 -> 6/3 divisible.
        # Con LeNet en MPS, output sizes no-divisibles rompen (pytorch issue 96056).
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 3 * 3, 120), nn.ReLU(inplace=True),
            nn.Linear(120, 84), nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.1, inplace=True),
    )


class DeepCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, p_drop=0.5):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_channels, 32), _block(32, 32), nn.MaxPool2d(2),
            _block(32, 64), _block(64, 64), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True), nn.Dropout(p_drop),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def resnet18_cifar(in_channels=1, num_classes=10):
    """ResNet-18 adaptada al estilo CIFAR.

    Parches sobre torchvision.models.resnet18:
    - conv1: 3x3 stride 1 (en vez de 7x7 stride 2) para no perder
      resolucion en entradas 32x32.
    - maxpool: Identity (skipped) porque la imagen ya es chica.
    - in_channels=1 para grayscale.
    Mantiene los 4 etapas residuales (2 bloques cada una, canales
    64->128->256->512), GAP final y Linear(512, num_classes).
    """
    m = _torchvision_resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                        padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def build_model(arch="resnet18", in_channels=1, num_classes=10):
    if arch == "lenet":
        return LeNetCNN(in_channels, num_classes)
    if arch == "deep":
        return DeepCNN(in_channels, num_classes)
    if arch == "resnet18":
        return resnet18_cifar(in_channels, num_classes)
    raise ValueError(f"arch desconocida: {arch}")
