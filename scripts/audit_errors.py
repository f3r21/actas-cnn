"""Error analysis sobre el modelo del proyecto.

Identifica los top-N crops del val set donde el modelo se equivoca con
mas confianza (mayor cross-entropy loss). Guarda CSV con metadatos y
genera un grid PNG para inspeccion visual y categorizacion manual del
modo de fallo dominante (segmentacion / ruido / lineas / ambiguedad /
label mal).

Uso:
  python scripts/audit_errors.py --top-n 200

Salidas:
  data/audit_errors_top.csv   (filas ordenadas por loss desc)
  data/visualizaciones/errors_top.png   (grid 10 cols por pagina)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn.functional import cross_entropy, softmax
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset import CropsDataset, default_transforms
from env import torch_device
from model import build_model


CKPT_CANDIDATES = ["resnet18_best.pt", "deep_best.pt", "lenet_best.pt"]


def _find_checkpoint() -> Path:
    for name in CKPT_CANDIDATES:
        p = ROOT / "checkpoints" / name
        if p.exists():
            return p
    raise FileNotFoundError(f"ninguno de {CKPT_CANDIDATES} en {ROOT/'checkpoints'}")


def _parse_path(rel_path: str) -> dict:
    """Parsea 'label/archivoId_field_pos.png' -> dict con metadatos."""
    p = Path(rel_path)
    parts = p.stem.split("_")
    archivo_id = parts[0] if parts else ""
    pos = parts[-1] if len(parts) >= 3 else ""
    field = "_".join(parts[1:-1]) if len(parts) >= 3 else ""
    return {"archivoId": archivo_id, "field": field, "pos": pos}


def _grid(crops: list[tuple[Path, int, int, float]], out_path: Path,
          cols: int = 10, thumb: int = 96) -> None:
    """Construye grid PNG. cada celda muestra el crop + label real, prediccion, confianza."""
    if not crops:
        return
    rows = (len(crops) + cols - 1) // cols
    header = 24
    cell_h = thumb + 28
    canvas = Image.new("RGB",
                       (cols * thumb, rows * cell_h + header),
                       "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 4), f"Top {len(crops)} errores (val) — real / pred (conf)",
              fill="black", font=font)

    for i, (img_path, true_lbl, pred_lbl, conf) in enumerate(crops):
        r, c = divmod(i, cols)
        x0, y0 = c * thumb, header + r * cell_h
        try:
            img = Image.open(img_path).convert("RGB").resize((thumb, thumb))
            canvas.paste(img, (x0, y0))
        except Exception:
            draw.rectangle([x0, y0, x0 + thumb, y0 + thumb], outline="red")
        caption = f"{true_lbl}/{pred_lbl} ({conf:.2f})"
        draw.text((x0 + 2, y0 + thumb + 2), caption, fill="red", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=ROOT / "data/manifest_val.csv",
                    type=Path)
    ap.add_argument("--root", default=ROOT / "data/crops_val", type=Path)
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--out-csv", default=ROOT / "data/audit_errors_top.csv",
                    type=Path)
    ap.add_argument("--out-png",
                    default=ROOT / "data/visualizaciones/errors_top.png",
                    type=Path)
    args = ap.parse_args()

    device = torch_device()
    ckpt_path = _find_checkpoint()
    print(f"checkpoint: {ckpt_path.name}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "resnet18")
    model = build_model(arch, 1, 10).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = CropsDataset(args.manifest, root=args.root,
                      transform=default_transforms(32, train=False))
    df_manifest = ds.df.reset_index(drop=True)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    all_losses = []
    all_preds = []
    all_confs = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = softmax(logits, dim=1)
            losses = cross_entropy(logits, y, reduction="none")
            preds = logits.argmax(1)
            confs = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
            all_losses.append(losses.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_confs.append(confs.cpu().numpy())

    losses = np.concatenate(all_losses)
    preds = np.concatenate(all_preds)
    confs = np.concatenate(all_confs)

    df = df_manifest.copy()
    df["loss"] = losses
    df["pred"] = preds
    df["pred_conf"] = confs
    df["error"] = df["pred"] != df["label"]

    # Solo errores, ordenados por loss desc
    errs = df[df["error"]].copy().sort_values("loss", ascending=False)
    top = errs.head(args.top_n).reset_index(drop=True)

    # Enriquecer con archivoId / field / pos
    meta = top["path"].apply(_parse_path).apply(pd.Series)
    top = pd.concat([top, meta], axis=1)

    cols_out = ["path", "label", "pred", "pred_conf", "loss",
                "archivoId", "field", "pos"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    top[cols_out].to_csv(args.out_csv, index=False)
    print(f"CSV: {args.out_csv}  ({len(top)} filas)")

    # Generar grid PNG
    crops = [(args.root / r["path"], int(r["label"]), int(r["pred"]),
              float(r["pred_conf"])) for _, r in top.iterrows()]
    _grid(crops, args.out_png)
    print(f"PNG: {args.out_png}")

    # Resumen rapido
    print(f"\\nTotal val: {len(df)}  errores: {int(df['error'].sum())} "
          f"({100*df['error'].mean():.2f}%)")
    print("Errores por (label, pred) — top 10 pares:")
    pairs = top.groupby(["label", "pred"]).size().sort_values(ascending=False)
    print(pairs.head(10).to_string())


if __name__ == "__main__":
    main()
