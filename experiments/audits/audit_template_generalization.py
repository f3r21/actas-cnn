"""Auditoria de generalizacion del template Presidencial.

Mide cuan bien las 42 cajas del template calzan en actas que NO fueron
calibradas. Estrategia:

1. Carga checkpoint deep_best.pt.
2. Predice sobre val + test (~46k crops).
3. Agrupa accuracy por archivoId.
4. Histograma de accuracy por acta.
5. Identifica los 20 peores actas.
6. Genera overlay del template sobre los 20 peores.
7. Escribe archive/auditorias/template-generalizacion.md con conclusion cuantitativa.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT / "src"))

from actas_cnn.data import CropsDataset, default_transforms
from actas_cnn.env import torch_device
from actas_cnn.model import build_model
from actas_cnn.viz import dibujar_overlay
from actas_cnn.preprocess.crops import load_templates

VIS = ROOT / "data" / "visualizaciones"
VIS.mkdir(exist_ok=True)


def infer(model, loader, device):
    """Devuelve (paths, labels, preds) en el orden del dataset."""
    preds_all, labels_all = [], []
    paths_all = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            p = model(x).argmax(1).cpu().numpy()
            preds_all.append(p)
            labels_all.append(y.numpy())
    return np.concatenate(labels_all), np.concatenate(preds_all)


def per_acta_accuracy(manifest_path: Path, crops_root: Path, model, device):
    """Calcula accuracy por archivoId."""
    df = pd.read_csv(manifest_path)
    df["archivoId"] = df["path"].str.extract(r"(?:.*/)?([a-f0-9]{24})")

    ds = CropsDataset(manifest_path, root=crops_root,
                      transform=default_transforms(32, train=False))
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    y_true, y_pred = infer(model, loader, device)

    df = df.iloc[:len(y_true)].copy()
    df["correct"] = (y_true == y_pred)
    per_acta = df.groupby("archivoId").agg(n=("correct", "size"),
                                            ok=("correct", "sum"))
    per_acta["acc"] = per_acta["ok"] / per_acta["n"]
    return per_acta.sort_values("acc")


def histograma_png(per_acta_df: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(per_acta_df["acc"], bins=40, color="#4a90e2", edgecolor="black")
    ax.set_xlabel("accuracy por acta")
    ax.set_ylabel("# actas")
    ax.set_title(f"Distribucion accuracy por acta (n={len(per_acta_df)} actas)")
    ax.axvline(per_acta_df["acc"].median(), color="red", linestyle="--",
               label=f"mediana {per_acta_df['acc'].median():.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def worst_overlays(per_acta_df: pd.DataFrame, k: int, rendered_dir: Path, out: Path) -> None:
    """Grid de los k peores actas con overlay del template."""
    template = load_templates(ROOT / "templates.json")["presidencial"]
    worst_ids = per_acta_df.head(k).index.tolist()

    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    cols, rows = 5, (k + 4) // 5
    THUMB_W = 500
    THUMB_H = int(500 * 3309 / 2339)
    PAD = 10
    HEADER = 30
    canvas_w = cols * (THUMB_W + PAD) + PAD
    canvas_h = rows * (THUMB_H + PAD + HEADER) + PAD
    canvas = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)

    for i, aid in enumerate(worst_ids):
        r, c = divmod(i, cols)
        png_path = rendered_dir / f"{aid}_p0.png"
        if not png_path.exists():
            # Buscar en sample_pdfs/presidencial/rendered
            alt = ROOT / "data/sample_pdfs/presidencial/rendered" / f"{aid}_p0.png"
            if alt.exists(): png_path = alt
            else: continue
        img = Image.open(png_path)
        overlay = dibujar_overlay(img, template).resize((THUMB_W, THUMB_H))
        x = PAD + c * (THUMB_W + PAD)
        y = PAD + r * (THUMB_H + PAD + HEADER)
        canvas.paste(overlay, (x, y + HEADER))
        acc = per_acta_df.loc[aid, "acc"]
        n = per_acta_df.loc[aid, "n"]
        draw.text((x + 4, y + 4), f"{aid[:16]}  acc={acc:.2f}  n={n}",
                  fill=(0, 0, 0), font=font)
    canvas.save(out)


def main() -> None:
    device = torch_device()
    ckpt_path = ROOT / "checkpoints" / "deep_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt.get("arch", "deep"), 1, 10).to(device)
    model.load_state_dict(ckpt["model"])

    print("Calculando per-acta accuracy en val + test...")
    val_acc = per_acta_accuracy(ROOT / "data/manifest_val.csv",
                                ROOT / "data/crops_val", model, device)
    test_acc = per_acta_accuracy(ROOT / "data/manifest_test.csv",
                                 ROOT / "data/crops_test", model, device)
    all_acc = pd.concat([val_acc, test_acc])
    print(f"  actas analizadas: {len(all_acc)}")
    print(f"  mediana: {all_acc['acc'].median():.4f}")
    print(f"  P10:     {all_acc['acc'].quantile(0.10):.4f}")
    print(f"  P25:     {all_acc['acc'].quantile(0.25):.4f}")
    print(f"  P75:     {all_acc['acc'].quantile(0.75):.4f}")
    print(f"  P90:     {all_acc['acc'].quantile(0.90):.4f}")
    print(f"  min:     {all_acc['acc'].min():.4f}")

    # Histograma
    histograma_png(all_acc, VIS / "audit_template_histogram.png")
    print(f"  histograma -> {VIS / 'audit_template_histogram.png'}")

    # Overlays de los peores
    print("Generando overlays de los 20 peores actas...")
    worst_overlays(all_acc, 20, ROOT / "data/pdfs_train/rendered",
                   VIS / "audit_template_worst_20.png")
    print(f"  overlays -> {VIS / 'audit_template_worst_20.png'}")

    # Reporte
    report_path = ROOT / "archive" / "auditorias" / "template-generalizacion.md"
    lines = ["# AUDIT — Generalizacion del template Presidencial", ""]
    lines.append(f"Auditoria sobre {len(all_acc)} actas (val + test).")
    lines.append(f"Modelo: DeepCNN 5 epochs val_acc 95.5%.")
    lines.append("")
    lines.append("## Distribucion de accuracy por acta")
    lines.append("")
    lines.append(f"- **mediana**: {all_acc['acc'].median():.4f}")
    lines.append(f"- **media**:   {all_acc['acc'].mean():.4f}")
    lines.append(f"- **min**:     {all_acc['acc'].min():.4f}")
    lines.append(f"- **P10**:     {all_acc['acc'].quantile(0.10):.4f}")
    lines.append(f"- **P25**:     {all_acc['acc'].quantile(0.25):.4f}")
    lines.append(f"- **P75**:     {all_acc['acc'].quantile(0.75):.4f}")
    lines.append("")
    lines.append(f"![histograma](data/visualizaciones/audit_template_histogram.png)")
    lines.append("")
    lines.append("## Lectura cuantitativa")
    lines.append("")
    n_99 = (all_acc["acc"] >= 0.99).sum()
    n_95 = ((all_acc["acc"] >= 0.95) & (all_acc["acc"] < 0.99)).sum()
    n_80 = ((all_acc["acc"] >= 0.80) & (all_acc["acc"] < 0.95)).sum()
    n_50 = ((all_acc["acc"] >= 0.50) & (all_acc["acc"] < 0.80)).sum()
    n_bad = (all_acc["acc"] < 0.50).sum()
    lines.append(f"- {n_99} actas con acc >= 0.99")
    lines.append(f"- {n_95} actas con acc 0.95-0.99")
    lines.append(f"- {n_80} actas con acc 0.80-0.95")
    lines.append(f"- {n_50} actas con acc 0.50-0.80")
    lines.append(f"- {n_bad} actas con acc < 0.50 (template probablemente roto)")
    lines.append("")
    lines.append("## Los 20 peores actas")
    lines.append("")
    lines.append(f"![worst overlays](data/visualizaciones/audit_template_worst_20.png)")
    lines.append("")
    for aid, row in all_acc.head(20).iterrows():
        lines.append(f"- `{aid}`  acc={row['acc']:.3f}  n_crops={int(row['n'])}")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    if n_bad <= 5 and all_acc["acc"].median() >= 0.95:
        verdict = "PASS — template generaliza bien"
        notes = ("La mediana esta en >=0.95, los outliers son <=5 actas. "
                 "El template Presidencial es suficientemente general.")
    elif n_bad <= 20:
        verdict = "WARNING — template generaliza con outliers manejables"
        notes = (f"Hay {n_bad} actas donde el template falla mas de la mitad "
                 "de las celdas. Inspeccion visual del grid de worst 20 "
                 "puede revelar la causa: rotacion, escala distinta, o "
                 "sub-templates.")
    else:
        verdict = "FAIL — template NO generaliza"
        notes = (f"Hay {n_bad} actas con accuracy <50%. El template "
                 "necesita revision sistematica.")
    lines.append(f"**Veredicto**: {verdict}")
    lines.append("")
    lines.append(notes)
    report_path.write_text("\n".join(lines))
    print(f"\nReporte: {report_path}")


if __name__ == "__main__":
    main()
