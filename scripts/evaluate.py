"""Evaluacion downstream del modelo del proyecto.

Mide lo que realmente importa segun el alcance del proyecto:
  - Digit-level accuracy (ya en audit CHECK 7).
  - Field-level accuracy: el entero del campo reconstruido coincide con el
    ground truth (cifra del partido, votos blanco/nulos/impugnados, total).
  - Acta-level accuracy: los 42 campos correctos en simultaneo.
  - Reconstruccion del total agregado: suma(partidos + blanco + nulos +
    impugnados) predicha vs `totalVotosEmitidos` oficial; |error| MAE,
    mediana, distribucion.
  - Per-class metrics + matriz de confusion (digit-level).

Reusa el filtro right-justified de `extract_crops.es_celda_escrita`: las
posiciones sin crop en el manifest se interpretan como leading zeros (no
escritas), exactamente la convencion del entrenamiento.

Uso:
  python scripts/evaluate.py --split val
  python scripts/evaluate.py --split test --out data/evaluate_test.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


def parse_crop_path(rel: str) -> tuple[str, str, int]:
    """'<label>/<aid>_<field>_c<pos>.png' -> (aid, field, pos)."""
    stem = Path(rel).stem
    parts = stem.split("_")
    aid = parts[0]
    pos_str = parts[-1]  # 'c1', 'c2', ...
    field = "_".join(parts[1:-1])
    pos = int(pos_str[1:])
    return aid, field, pos


def reconstruct_value(preds_by_pos: dict[int, int], n_cells: int) -> int:
    """Combina digitos predichos en entero right-justified. Posiciones sin
    crop en el manifest se interpretan como leading zeros (0).
    """
    digits = [preds_by_pos.get(p, 0) for p in range(n_cells)]
    return int("".join(str(d) for d in digits))


def real_value_for(name: str, votos_acta: pd.DataFrame, total_emitidos: int) -> int:
    if name.startswith("partido_"):
        pos = int(name.split("_")[1])
        row = votos_acta[votos_acta["nposicion"] == pos]
        return int(row.iloc[0]["nvotos"]) if len(row) else 0
    mapping = {"votos_blanco": 80, "votos_nulos": 81, "votos_impugnados": 82}
    if name in mapping:
        row = votos_acta[votos_acta["nposicion"] == mapping[name]]
        return int(row.iloc[0]["nvotos"]) if len(row) else 0
    if name == "total_ciudadanos":
        return int(total_emitidos)
    raise ValueError(f"field desconocido: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--manifest", default=None, type=Path,
                    help="override; default: data/manifest_<split>.csv")
    ap.add_argument("--crops-root", default=None, type=Path,
                    help="override; default: data/crops_<split>")
    ap.add_argument("--out-csv", default=None, type=Path,
                    help="CSV con una fila por (archivoId, field); default: data/evaluate_<split>.csv")
    ap.add_argument("--checkpoint", default=None, type=Path,
                    help="path explicito al .pt; default: busca en checkpoints/")
    ap.add_argument("--save-logits", action="store_true",
                    help="guarda log_softmax por crop en data/eval_logits_<split>.parquet "
                         "(necesario para el checksum solver downstream)")
    args = ap.parse_args()

    manifest = args.manifest or (ROOT / f"data/manifest_{args.split}.csv")
    crops_root = args.crops_root or (ROOT / f"data/crops_{args.split}")
    out_csv = args.out_csv or (ROOT / f"data/evaluate_{args.split}.csv")
    logits_path = ROOT / f"data/eval_logits_{args.split}.parquet"

    # Cargar parquets
    archivos = pd.read_parquet(ROOT / "data/labels/actas_archivos.parquet")
    votos = pd.read_parquet(ROOT / "data/labels/actas_votos.parquet")
    cabecera = pd.read_parquet(ROOT / "data/labels/actas_cabecera.parquet")
    aid_to_idacta = dict(zip(archivos["archivoId"], archivos["idActa"]))

    # Cargar template para n_cells por field
    template = json.loads((ROOT / "templates.json").read_text())["presidencial"]
    field_specs = {f["name"]: f["n_digits"] for f in template["fields"]}

    # Modelo
    device = torch_device()
    ckpt_path = args.checkpoint if args.checkpoint else _find_checkpoint()
    print(f"checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "resnet18")
    model = build_model(arch, 1, 10).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Predict sobre todos los crops del manifest
    ds = CropsDataset(manifest, root=crops_root,
                      transform=default_transforms(32, train=False))
    df = ds.df.reset_index(drop=True)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    all_preds = []
    all_logprobs: list[np.ndarray] = [] if args.save_logits else []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            all_preds.append(logits.argmax(1).cpu().numpy())
            if args.save_logits:
                # log_softmax mas estable numericamente que log(softmax)
                lp = torch.nn.functional.log_softmax(logits, dim=1)
                all_logprobs.append(lp.cpu().numpy())
    df["pred"] = np.concatenate(all_preds)

    # Extraer (archivoId, field, pos) de cada path
    parsed = df["path"].apply(parse_crop_path).apply(pd.Series)
    parsed.columns = ["archivoId", "field", "pos"]
    df = pd.concat([df, parsed], axis=1)

    # Guardar log_softmax por crop si se pidio (consumido por checksum_solver)
    if args.save_logits:
        lp_arr = np.concatenate(all_logprobs, axis=0)  # (N, 10)
        logits_df = df[["archivoId", "field", "pos", "label", "pred"]].copy()
        for c in range(10):
            logits_df[f"lp_{c}"] = lp_arr[:, c].astype(np.float32)
        logits_df.to_parquet(logits_path, index=False)
        print(f"log_softmax por crop -> {logits_path} ({len(logits_df)} filas)")

    # Construir tabla por (archivoId, field): valor predicho vs real
    rows = []
    for aid, df_acta in df.groupby("archivoId"):
        if aid not in aid_to_idacta:
            continue
        id_acta = int(aid_to_idacta[aid])
        cab = cabecera[cabecera["idActa"] == id_acta]
        if len(cab) == 0 or pd.isna(cab.iloc[0]["totalVotosEmitidos"]):
            continue
        total_real = int(cab.iloc[0]["totalVotosEmitidos"])
        votos_acta = votos[votos["idActa"] == id_acta]
        for fname, n_cells in field_specs.items():
            crops_field = df_acta[df_acta["field"] == fname]
            preds_by_pos = dict(zip(crops_field["pos"], crops_field["pred"]))
            pred_value = reconstruct_value(preds_by_pos, n_cells)
            real_value = real_value_for(fname, votos_acta, total_real)
            rows.append({
                "archivoId": aid, "field": fname,
                "n_cells": n_cells, "n_pred_cells": len(preds_by_pos),
                "pred": pred_value, "real": real_value,
                "correct": pred_value == real_value,
                "error": pred_value - real_value,
            })

    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    print(f"resultados por field guardados en {out_csv} ({len(res)} filas)")

    # --- Reporte ---
    n_actas = res["archivoId"].nunique()
    n_fields = len(res)
    print(f"\\nactas evaluadas: {n_actas}")
    print(f"campos evaluados: {n_fields}")

    # Digit-level (cell-level)
    digit_acc = float(df["pred"].eq(df["label"]).mean())
    print(f"\\ndigit-level accuracy: {digit_acc:.4f}  (n={len(df)})")

    # Field-level
    field_acc = float(res["correct"].mean())
    print(f"field-level accuracy: {field_acc:.4f}")

    # Acta-level
    actas_correct = res.groupby("archivoId")["correct"].all()
    acta_acc = float(actas_correct.mean())
    print(f"acta-level accuracy: {acta_acc:.4f}  ({actas_correct.sum()}/{len(actas_correct)} actas)")

    # Per-field-type
    print("\\nfield-level accuracy por tipo de campo:")
    res["field_type"] = res["field"].apply(
        lambda f: "partido" if f.startswith("partido_") else f)
    by_type = res.groupby("field_type").agg(
        n=("correct", "size"), acc=("correct", "mean"))
    for typ, row in by_type.iterrows():
        print(f"  {typ:18s}  n={int(row['n']):5d}  acc={row['acc']:.4f}")

    # Reconstruccion del total agregado
    print("\\nreconstruccion del total agregado (sum partidos + blanco + nulos + impugnados):")
    sum_pred = res[res["field"] != "total_ciudadanos"].groupby("archivoId")["pred"].sum()
    sum_real = res[res["field"] != "total_ciudadanos"].groupby("archivoId")["real"].sum()
    err = sum_pred - sum_real
    abs_err = err.abs()
    print(f"  MAE: {abs_err.mean():.2f}")
    print(f"  mediana |error|: {int(abs_err.median())}")
    print(f"  max |error|: {int(abs_err.max())}")
    print(f"  actas con error == 0: {(err == 0).sum()}/{len(err)} ({100*(err==0).mean():.2f}%)")
    print(f"  actas con |error| <= 1: {(abs_err <= 1).sum()}/{len(err)} ({100*(abs_err<=1).mean():.2f}%)")
    print(f"  actas con |error| <= 5: {(abs_err <= 5).sum()}/{len(err)} ({100*(abs_err<=5).mean():.2f}%)")
    print(f"  actas con |error| <= 20: {(abs_err <= 20).sum()}/{len(err)} ({100*(abs_err<=20).mean():.2f}%)")

    # total_ciudadanos comparison aparte (un solo numero por acta)
    res_total = res[res["field"] == "total_ciudadanos"].copy()
    print(f"\\nreconstruccion del total_ciudadanos (campo de 4 digitos):")
    err_tot = res_total["error"]
    abs_err_tot = err_tot.abs()
    print(f"  MAE: {abs_err_tot.mean():.2f}")
    print(f"  mediana |error|: {int(abs_err_tot.median())}")
    print(f"  actas con error == 0: {(err_tot == 0).sum()}/{len(res_total)} ({100*(err_tot==0).mean():.2f}%)")

    # --- Analisis profundo ---
    viz_dir = ROOT / "data" / "visualizaciones"
    viz_dir.mkdir(parents=True, exist_ok=True)
    split = args.split

    # Matriz de confusion 10x10 + per-class precision/recall/F1 (digit-level)
    confusion = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(df["label"].values, df["pred"].values):
        confusion[int(t), int(p)] += 1
    per_class_recall = np.array([confusion[i, i] / max(confusion[i].sum(), 1) for i in range(10)])
    per_class_precision = np.array([confusion[i, i] / max(confusion[:, i].sum(), 1) for i in range(10)])
    per_class_f1 = np.array([
        2 * p * r / max(p + r, 1e-9) for p, r in zip(per_class_precision, per_class_recall)
    ])
    n_per_class = confusion.sum(axis=1)

    print("\\nper-class digit-level metrics:")
    print(f"  {'clase':6s} {'n':>6s} {'prec':>8s} {'recall':>8s} {'F1':>8s}")
    for c in range(10):
        print(f"  {c:6d} {int(n_per_class[c]):>6d} "
              f"{per_class_precision[c]:>8.4f} {per_class_recall[c]:>8.4f} {per_class_f1[c]:>8.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusion ({split}, acc={digit_acc:.4f}, n={len(df)})")
    vmax = confusion.max()
    for i in range(10):
        for j in range(10):
            ax.text(j, i, confusion[i, j], ha="center", va="center",
                    color="white" if confusion[i, j] > vmax / 2 else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    confusion_png = viz_dir / f"evaluate_confusion_{split}.png"
    fig.savefig(confusion_png, dpi=120)
    plt.close(fig)
    print(f"matriz de confusion -> {confusion_png}")

    # Histograma de |error| del total agregado por acta
    fig, ax = plt.subplots(figsize=(8, 5))
    # Bins exponenciales para captura buena de cola larga
    max_e = int(abs_err.max())
    bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, max(101, max_e + 1)]
    ax.hist(abs_err.values, bins=bins, edgecolor="black", color="steelblue")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(bins)
    ax.set_xticklabels([str(b) for b in bins], rotation=45)
    ax.set_xlabel("|error| del total reconstruido (votos)")
    ax.set_ylabel("cantidad de actas")
    ax.set_title(f"Distribucion de error del total agregado ({split}, n={len(abs_err)})")
    ax.grid(axis="y", alpha=0.3)
    # Anotaciones de cuantiles
    for q, label in [(0.50, "P50"), (0.90, "P90"), (0.95, "P95"), (0.99, "P99")]:
        v = abs_err.quantile(q)
        ax.axvline(v, color="red", linestyle="--", alpha=0.5)
        ax.text(v, ax.get_ylim()[1] * 0.95, f"{label}={int(v)}",
                rotation=90, va="top", ha="right", color="red", fontsize=8)
    fig.tight_layout()
    hist_png = viz_dir / f"evaluate_error_hist_{split}.png"
    fig.savefig(hist_png, dpi=120)
    plt.close(fig)
    print(f"histograma de errores -> {hist_png}")

    # Ranking de las 20 peores actas
    worst = (err.abs().sort_values(ascending=False).head(20).index.tolist())
    worst_rows = []
    for aid in worst:
        df_aid = res[res["archivoId"] == aid]
        n_fields_err = (~df_aid["correct"]).sum()
        fields_wrong = df_aid[~df_aid["correct"]]["field"].tolist()
        worst_rows.append({
            "archivoId": aid,
            "total_pred": int(sum_pred.loc[aid]),
            "total_real": int(sum_real.loc[aid]),
            "error_total": int(err.loc[aid]),
            "n_fields_wrong": int(n_fields_err),
            "fields_wrong": ",".join(fields_wrong[:10]),
        })
    worst_df = pd.DataFrame(worst_rows)
    worst_csv = ROOT / f"data/evaluate_worst20_{split}.csv"
    worst_df.to_csv(worst_csv, index=False)
    print(f"ranking 20 peores actas -> {worst_csv}")
    print("\\nTop 5 peores:")
    print(worst_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
