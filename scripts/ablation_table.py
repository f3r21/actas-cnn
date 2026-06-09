"""Consolida la tabla comparativa de ablations (Semana 2).

Lee los logs y CSVs per-campo que genera `scripts/evaluate.py` para cada
checkpoint (base, ls_ra, ls_ra_mu_cos) y arma una tabla unica con las
metricas headline: digit-level, field-level, acta-level, reconstruccion
exacta del total agregado y MAE.

Las metricas digit-level y de reconstruccion salen del log (se calculan
durante la inferencia); field-level y acta-level se recomputan desde el
CSV y se validan contra el log para detectar inconsistencias.

Salidas:
  - data/ablations_summary.csv
  - tabla markdown por stdout (para pegar en docs/04 y README)
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = [
    ("base", "resnet18_best.pt", "evaluate_val.log", "evaluate_val.csv",
     "sin augmentation"),
    ("ls_ra", "resnet18_ls_ra_best.pt", "evaluate_val_ls_ra.log",
     "evaluate_val_ls_ra.csv", "label smoothing + RandAugment"),
    ("ls_ra_mu_cos", "resnet18_ls_ra_mu_cos_best.pt",
     "evaluate_val_ls_ra_mu_cos.log", "evaluate_val_ls_ra_mu_cos.csv",
     "+ mixup + cosine LR"),
]


def parse_log(path: Path) -> dict:
    """Extrae las metricas headline del stdout de evaluate.py."""
    text = path.read_text()
    metrics = {}

    m = re.search(r"digit-level accuracy: ([\d.]+)", text)
    metrics["digit_acc"] = float(m.group(1))
    m = re.search(r"field-level accuracy: ([\d.]+)", text)
    metrics["field_acc"] = float(m.group(1))
    m = re.search(r"acta-level accuracy: ([\d.]+)\s+\((\d+)/(\d+)", text)
    metrics["acta_acc"] = float(m.group(1))
    metrics["actas_ok"] = int(m.group(2))
    metrics["actas_n"] = int(m.group(3))

    # Solo la seccion del total agregado (la de total_ciudadanos viene despues)
    seccion = text.split("reconstruccion del total agregado")[1]
    seccion = seccion.split("reconstruccion del total_ciudadanos")[0]
    m = re.search(r"MAE: ([\d.]+)", seccion)
    metrics["mae_total"] = float(m.group(1))
    m = re.search(r"actas con error == 0: (\d+)/(\d+) \(([\d.]+)%\)", seccion)
    metrics["recon_ok"] = int(m.group(1))
    metrics["recon_pct"] = float(m.group(3))
    return metrics


def validate_against_csv(path: Path, metrics: dict) -> None:
    """Recomputa field/acta-level desde el CSV y exige igualdad con el log."""
    res = pd.read_csv(path)
    field_acc = float(res["correct"].mean())
    acta_acc = float(res.groupby("archivoId")["correct"].all().mean())
    for nombre, csv_val, log_val in [("field", field_acc, metrics["field_acc"]),
                                     ("acta", acta_acc, metrics["acta_acc"])]:
        if abs(csv_val - log_val) > 5e-5:
            sys.exit(f"ERROR: {nombre}-level no cuadra en {path.name}: "
                     f"csv={csv_val:.4f} vs log={log_val:.4f}")


def main() -> None:
    rows = []
    for variant, ckpt, log_name, csv_name, desc in VARIANTS:
        log_path = ROOT / "data" / log_name
        csv_path = ROOT / "data" / csv_name
        if not log_path.exists() or not csv_path.exists():
            sys.exit(f"ERROR: falta {log_name} o {csv_name}; correr "
                     f"scripts/evaluate.py --checkpoint checkpoints/{ckpt}")
        metrics = parse_log(log_path)
        validate_against_csv(csv_path, metrics)
        rows.append({"variant": variant, "checkpoint": ckpt, "config": desc,
                     **metrics})

    df = pd.DataFrame(rows)
    out = ROOT / "data" / "ablations_summary.csv"
    df.to_csv(out, index=False)
    print(f"resumen guardado en {out}\n")

    print("| Variante | Config | Digit | Field | Acta | Recon. exacta | MAE |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['variant']} | {r['config']} "
              f"| {r['digit_acc'] * 100:.2f}% "
              f"| {r['field_acc'] * 100:.2f}% "
              f"| {r['acta_acc'] * 100:.2f}% ({r['actas_ok']}/{r['actas_n']}) "
              f"| {r['recon_pct']:.2f}% ({r['recon_ok']}/{r['actas_n']}) "
              f"| {r['mae_total']:.2f} |")


if __name__ == "__main__":
    main()
