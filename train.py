"""Entrenamiento portable (Kaggle / Colab / M2 local) con checkpoints redundantes.

- Detecta el dispositivo (CUDA, MPS o CPU).
- Lee el dataset desde un manifiesto local (bajado antes con storage.download).
- Guarda el mejor checkpoint y lo sube a todos los backends disponibles, para
  poder reanudar desde otra plataforma de GPU.
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import storage
from config import TRAIN
from dataset import CropsDataset, default_transforms
from env import checkpoints_dir, torch_device
from model import build_model


def _mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Convex combination de pares aleatorios del batch (Zhang et al., 2017).
    Devuelve (x_mixed, y_a, y_b, lam). Loss en training: lam*CE(out,y_a) +
    (1-lam)*CE(out,y_b).
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    return x_mixed, y, y[idx], lam


def run_epoch(model, loader, device, criterion, optimizer=None, mixup_alpha=0.0):
    is_train = optimizer is not None
    model.train(is_train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train and mixup_alpha > 0:
                x_m, y_a, y_b, lam = _mixup_batch(x, y, mixup_alpha)
                out = model(x_m)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                # Para tracking de accuracy en train con mixup, usamos y_a
                # (target dominante). Es una aproximacion; el numero real
                # comparable es val_acc.
                correct += (out.argmax(1) == y_a).sum().item()
            else:
                out = model(x)
                loss = criterion(out, y)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                correct += (out.argmax(1) == y).sum().item()
            if is_train and mixup_alpha > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=".", help="raiz de las rutas del manifiesto")
    ap.add_argument("--arch", default=TRAIN.arch, choices=["lenet", "deep", "resnet18"])
    ap.add_argument("--epochs", type=int, default=TRAIN.epochs)
    ap.add_argument("--push", action="store_true", help="subir checkpoints a los backends")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="epsilon de label smoothing en CrossEntropy (typical 0.05-0.1)")
    ap.add_argument("--randaugment", action="store_true",
                    help="agregar RandAugment al pipeline de train")
    ap.add_argument("--mixup", type=float, default=0.0,
                    help="alpha del Beta para mixup (Zhang et al. 2017); 0=off, typical 0.2")
    ap.add_argument("--cosine-lr", action="store_true",
                    help="usar CosineAnnealingLR durante el entrenamiento")
    ap.add_argument("--suffix", default="",
                    help="sufijo para el nombre del checkpoint (e.g. 'ls_ra')")
    args = ap.parse_args()

    torch.manual_seed(TRAIN.seed)
    device = torch_device()

    full = CropsDataset(args.manifest, root=args.root,
                        transform=default_transforms(TRAIN.image_size, train=True,
                                                     randaugment=args.randaugment))
    n_val = max(1, int(0.2 * len(full)))
    train_set, val_set = random_split(full, [len(full) - n_val, n_val])

    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=TRAIN.batch_size, shuffle=True,
                              num_workers=2, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=TRAIN.batch_size, shuffle=False,
                            num_workers=2, pin_memory=pin)

    model = build_model(args.arch, TRAIN.in_channels, TRAIN.num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN.lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.cosine_lr else None)

    best_acc = 0.0
    ckpt_name = f"{args.arch}{('_' + args.suffix) if args.suffix else ''}_best.pt"
    ckpt = checkpoints_dir() / ckpt_name
    if ckpt.exists():
        try:
            prev = torch.load(ckpt, map_location="cpu")
            best_acc = float(prev.get("acc", 0.0))
            print(f"checkpoint previo: val_acc {best_acc:.4f} (no se sobrescribe a menos que mejore)")
        except Exception as e:
            print(f"checkpoint previo ilegible ({e}), se reemplaza si entrena bien")
    for epoch in range(1, args.epochs + 1):
        _, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer,
                              mixup_alpha=args.mixup)
        _, va_acc = run_epoch(model, val_loader, device, criterion)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:02d}  train_acc {tr_acc:.4f}  val_acc {va_acc:.4f}  lr {lr_now:.6f}")
        if scheduler is not None:
            scheduler.step()
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "acc": best_acc, "arch": args.arch}, ckpt)
            if args.push:
                storage.upload(ckpt, ckpt.name, kind="model")

    print(f"mejor val_acc: {best_acc:.4f} ({device})")


if __name__ == "__main__":
    main()
