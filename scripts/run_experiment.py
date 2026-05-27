"""Runner unico de experimentos para el loop de optimizacion de hiperparametros.

Lee una config JSON, ejecuta `train.py` con los flags derivados, despues corre
`scripts/evaluate.py --split val`, parsea las tres metricas del stdout, computa
la metrica compuesta `0.5 * digit_acc + 0.5 * acta_acc` y agrega una fila a
`results.tsv` en la raiz del repo.

El espacio de busqueda se limita a lo que `train.py` ya expone via CLI:
`arch`, `epochs`, `label_smoothing`, `randaugment`, `mixup`, `cosine_lr`,
`suffix`. Hiperparametros como `lr`, `batch_size` y `seed` viven en
`config.py` (`TRAIN`) y no se varian en este loop.

Uso:
    python scripts/run_experiment.py --config configs/run_001.json
    python scripts/run_experiment.py --config configs/run_001.json --smoke

Formato del JSON de config:
    {
      "run_id": "run_001",
      "arch": "resnet18",
      "label_smoothing": 0.1,
      "randaugment": true,
      "mixup": 0.2,
      "cosine_lr": true,
      "epochs": 20,
      "notes": "label smoothing + RandAugment + mixup + cosine LR"
    }

Con `--smoke`: fuerza epochs=2 y agrega "smoke" al notes; util para
verificar que la config no rompe nada antes de un run completo.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_TSV = ROOT / "results.tsv"

TSV_COLUMNS = [
    "timestamp", "run_id", "arch", "label_smoothing", "randaugment", "mixup",
    "cosine_lr", "epochs", "digit_acc", "field_acc", "acta_acc", "composite",
    "kept", "platform", "wall_seconds", "notes",
]

# Regex para parsear las tres metricas del stdout de evaluate.py.
# Formato en evaluate.py lineas 194-203 (estable):
#     digit-level accuracy: 0.9812  (n=29106)
#     field-level accuracy: 0.9887
#     acta-level accuracy: 0.9033  (626/693 actas)
RE_DIGIT = re.compile(r"digit-level accuracy:\s*([0-9.]+)")
RE_FIELD = re.compile(r"field-level accuracy:\s*([0-9.]+)")
RE_ACTA = re.compile(r"acta-level accuracy:\s*([0-9.]+)")


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    arch: str = "resnet18"
    label_smoothing: float = 0.0
    randaugment: bool = False
    mixup: float = 0.0
    cosine_lr: bool = False
    epochs: int = 20
    notes: str = ""

    @classmethod
    def from_json(cls, path: Path) -> "RunConfig":
        raw = json.loads(path.read_text())
        if "run_id" not in raw:
            raise ValueError(f"falta run_id en {path}")
        valid = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"campos desconocidos en {path}: {sorted(unknown)}")
        return cls(**raw)


def build_train_cmd(cfg: RunConfig, manifest: Path, root_dir: Path) -> list[str]:
    """Construye el comando para invocar train.py con los flags soportados."""
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--manifest", str(manifest),
        "--root", str(root_dir),
        "--arch", cfg.arch,
        "--epochs", str(cfg.epochs),
        "--label-smoothing", str(cfg.label_smoothing),
        "--mixup", str(cfg.mixup),
        "--suffix", cfg.run_id,
    ]
    if cfg.randaugment:
        cmd.append("--randaugment")
    if cfg.cosine_lr:
        cmd.append("--cosine-lr")
    return cmd


def parse_eval_stdout(text: str) -> tuple[float, float, float]:
    """Extrae digit/field/acta accuracy del stdout de evaluate.py.

    Levanta ValueError si alguna metrica falta (proteccion contra cambios
    silenciosos en el formato de salida de evaluate.py).
    """
    m_digit = RE_DIGIT.search(text)
    m_field = RE_FIELD.search(text)
    m_acta = RE_ACTA.search(text)
    if not (m_digit and m_field and m_acta):
        raise ValueError(
            "No se pudieron parsear las metricas del stdout de evaluate.py. "
            "Revisa que el formato de impresion no haya cambiado."
        )
    return float(m_digit.group(1)), float(m_field.group(1)), float(m_acta.group(1))


def detect_platform() -> str:
    """Heuristica simple para etiquetar la plataforma en el TSV."""
    if Path("/kaggle").exists():
        return "kaggle"
    if Path("/content").exists():
        return "colab"
    return "local"


def best_composite_so_far() -> float:
    """Lee el mejor composite previo de results.tsv. Si el archivo no
    existe o solo tiene header, devuelve 0.0."""
    if not RESULTS_TSV.exists():
        return 0.0
    with RESULTS_TSV.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        best = 0.0
        for row in reader:
            try:
                v = float(row.get("composite", "") or 0.0)
                best = max(best, v)
            except ValueError:
                continue
        return best


def append_row(row: dict[str, object]) -> None:
    """Append a results.tsv. Crea el archivo con header si no existe."""
    write_header = not RESULTS_TSV.exists()
    with RESULTS_TSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in TSV_COLUMNS})


def base_row(cfg: RunConfig, platform: str, t0: float) -> dict[str, object]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": cfg.run_id,
        "arch": cfg.arch,
        "label_smoothing": cfg.label_smoothing,
        "randaugment": int(cfg.randaugment),
        "mixup": cfg.mixup,
        "cosine_lr": int(cfg.cosine_lr),
        "epochs": cfg.epochs,
        "platform": platform,
        "wall_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True,
                    help="JSON con la config del run")
    ap.add_argument("--smoke", action="store_true",
                    help="fuerza epochs=2 y marca el run como 'smoke' en notes")
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "data" / "manifest_train.csv",
                    help="manifest para train.py (default data/manifest_train.csv)")
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "crops_train",
                    help="root de los crops (default data/crops_train)")
    args = ap.parse_args()

    cfg = RunConfig.from_json(args.config)
    if args.smoke:
        cfg = RunConfig(**{**cfg.__dict__, "epochs": 2,
                           "notes": (cfg.notes + " [smoke]").strip()})

    platform = detect_platform()
    print(f"[run_experiment] run_id={cfg.run_id} platform={platform} smoke={args.smoke}")
    print(f"[run_experiment] config: {cfg}")

    t0 = time.time()

    # 1) Entrenar.
    train_cmd = build_train_cmd(cfg, args.manifest, args.root)
    print(f"[run_experiment] train cmd: {' '.join(train_cmd)}")
    train_res = subprocess.run(train_cmd, cwd=ROOT)
    if train_res.returncode != 0:
        print(f"[run_experiment] train.py fallo (exit {train_res.returncode})")
        row = base_row(cfg, platform, t0)
        row.update({
            "digit_acc": "", "field_acc": "", "acta_acc": "",
            "composite": "", "kept": "ERR",
            "notes": (cfg.notes + " train.py crash").strip(),
        })
        append_row(row)
        return train_res.returncode

    ckpt = ROOT / "checkpoints" / f"{cfg.arch}_{cfg.run_id}_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"train.py no genero {ckpt}. Revisa logs arriba.")

    # 2) Evaluar sobre val.
    eval_cmd = [
        sys.executable, str(ROOT / "scripts" / "evaluate.py"),
        "--split", "val", "--checkpoint", str(ckpt),
    ]
    print(f"[run_experiment] eval cmd: {' '.join(eval_cmd)}")
    eval_res = subprocess.run(eval_cmd, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(eval_res.stdout)
    sys.stderr.write(eval_res.stderr)
    if eval_res.returncode != 0:
        print(f"[run_experiment] evaluate.py fallo (exit {eval_res.returncode})")
        row = base_row(cfg, platform, t0)
        row.update({
            "digit_acc": "", "field_acc": "", "acta_acc": "",
            "composite": "", "kept": "ERR",
            "notes": (cfg.notes + " evaluate.py crash").strip(),
        })
        append_row(row)
        return eval_res.returncode

    digit_acc, field_acc, acta_acc = parse_eval_stdout(eval_res.stdout)
    composite = 0.5 * digit_acc + 0.5 * acta_acc

    best_prev = best_composite_so_far()
    # Umbral 0.001 para evitar contar ruido de seed como mejora.
    kept = "Y" if composite > best_prev + 1e-3 else "N"

    print(f"[run_experiment] digit={digit_acc:.4f} field={field_acc:.4f} "
          f"acta={acta_acc:.4f} composite={composite:.4f} "
          f"best_prev={best_prev:.4f} kept={kept}")

    row = base_row(cfg, platform, t0)
    row.update({
        "digit_acc": f"{digit_acc:.4f}",
        "field_acc": f"{field_acc:.4f}",
        "acta_acc": f"{acta_acc:.4f}",
        "composite": f"{composite:.4f}",
        "kept": kept,
        "notes": cfg.notes,
    })
    append_row(row)

    # Limpieza en smoke runs (no contaminar checkpoints/ con modelos de 2 epochs).
    if args.smoke:
        try:
            ckpt.unlink()
            print(f"[run_experiment] smoke run: checkpoint eliminado")
        except OSError as e:
            print(f"[run_experiment] no se pudo limpiar smoke checkpoint: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
