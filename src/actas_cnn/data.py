"""Dataset PyTorch de recortes de digitos + construccion del manifiesto.

El manifiesto tiene columnas: path,label. Asi es indiferente si los recortes
vienen de local, de Hugging Face o de un mirror: solo cambia la raiz.
"""
import csv
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_manifest(crops_dir, out_csv) -> int:
    """Escanea crops/<label>/*.png y escribe un manifiesto CSV (path,label).

    Espera una carpeta organizada como crops/<label>/<archivo>.png donde label
    es el digito 0-9. Las rutas se guardan relativas a `crops_dir`. Devuelve el
    numero de filas escritas.
    """
    crops_dir = Path(crops_dir)
    rows = []
    for label_dir in sorted(crops_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for img in sorted(label_dir.glob("*.png")):
            rows.append((str(img.relative_to(crops_dir)), label))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)
    return len(rows)


def default_transforms(image_size=32, train=True, randaugment=False):
    """Pipeline de transforms para crops de digitos.

    train=False: solo resize + normalize (eval).
    train=True: agrega RandomAffine como augmentation suave.
    randaugment=True: ademas de lo anterior, suma RandAugment (torchvision)
        que aplica una secuencia aleatoria de transformaciones de
        intensidad/geometricas mas agresivas. Util para empujar el modelo
        cuando el RandomAffine solo no basta.
    """
    ops = [transforms.Grayscale(num_output_channels=1),
           transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomAffine(degrees=8, translate=(0.1, 0.1),
                                           scale=(0.9, 1.1)))
        if randaugment:
            # num_ops=2, magnitude=9 son los defaults del paper original.
            ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops += [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    return transforms.Compose(ops)


class CropsDataset(Dataset):
    def __init__(self, manifest_csv, root=".", transform=None):
        self.df = pd.read_csv(manifest_csv)
        self.root = Path(root)
        self.transform = transform or default_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row["path"])
        x = self.transform(img)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y
