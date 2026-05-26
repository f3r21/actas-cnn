"""Auditoria del dataset y modelo. Read-only. Output: AUDIT_REPORT.md.

Ejecuta 6 chequeos independientes contra el estado real en disco para
validar que los numeros reportados son ciertos:

  2 5,000 PDFs descargados, todos Presidencial escrutinio
  3 5,000 PNGs renderizados 1:1 con PDFs
  4 Splits sin leak entre train/val/test
  5 Labels coinciden con imagen para 30 crops random
  6 Templates funcionan en actas random (no solo las 3 calibradas)
  7 val_acc del checkpoint no es solo majority class

Cada chequeo es independiente — si uno falla, los demas siguen. La
numeracion 2-7 se conserva por compatibilidad con AUDIT_REPORT
historicos; el ex-CHECK 1 (gap de descarga) y ex-CHECK 8 (snapshot
scraper) se eliminaron porque eran de la fase pre-Semana-1.
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_crops import crop_fields, load_templates, split_digits
from scripts.build_crops import int_to_digits


ROOT = Path(__file__).resolve().parent.parent
VIS = ROOT / "data" / "visualizaciones"
VIS.mkdir(exist_ok=True)
REPORT_PATH = ROOT / "AUDIT_REPORT.md"


@dataclass
class CheckResult:
    n: int
    title: str
    claim: str
    method: str
    evidence: str
    result: str  # PASS | FAIL | WARNING
    notes: str = ""


# ------------------------- CHECK 2: 5k PDFs validos ----------------------

def check_2_pdfs_5k() -> CheckResult:
    pdf_dir = ROOT / "data" / "pdfs_train"
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    # manuscritas_full.txt es el universo real en disco (3478 manuscritas
    # originales + 1522 extras tras filtrar STAE Lima/Callao). sample_5000_ids.txt
    # es el listado planeado anterior y quedo desactualizado.
    sample_ids = {l.strip() for l in (ROOT / "data/splits/manuscritas_full.txt").read_text().splitlines() if l.strip()}
    stems = {p.stem for p in pdfs}
    zero_size = [p for p in pdfs if p.stat().st_size == 0]

    archivos = pd.read_parquet(ROOT / "data/labels/actas_archivos.parquet")
    pres = set(archivos[(archivos["tipo"] == 1) & (archivos["idEleccion"] == 10)]["archivoId"])
    outside_universe = stems - pres
    not_in_sample = stems - sample_ids
    missing_from_disk = sample_ids - stems

    ok = (len(pdfs) == 5000 and not zero_size and not outside_universe
          and not not_in_sample and not missing_from_disk)
    result = "PASS" if ok else "FAIL"
    return CheckResult(
        n=2, title="5,000 PDFs descargados (universo manuscritas Presidencial)",
        claim="5,000 PDFs en data/pdfs_train/, todos Presidencial tipo=1 idEleccion=10, alineados con manuscritas_full.txt (3,478 manuscritas + 1,522 extras tras filtrar STAE).",
        method="ls + cross-check con manuscritas_full.txt + cross-check con actas_archivos.parquet filtrado.",
        evidence=f"count={len(pdfs)}, zero_size={len(zero_size)}, outside_universe={len(outside_universe)}, missing_from_disk={len(missing_from_disk)}",
        result=result,
        notes="" if ok else f"Discrepancia(s): zero={len(zero_size)}, outside={len(outside_universe)}, missing={len(missing_from_disk)}",
    )


# ------------------------- CHECK 3: render 1:1 ---------------------------

def check_3_render_one_to_one() -> CheckResult:
    pdf_stems = {p.stem for p in (ROOT / "data/pdfs_train").glob("*.pdf")}
    pngs = list((ROOT / "data/pdfs_train/rendered").glob("*_p0.png"))
    png_stems = {p.stem.removesuffix("_p0") for p in pngs}

    diff_pdfs_no_pngs = pdf_stems - png_stems
    diff_pngs_no_pdfs = png_stems - pdf_stems

    # Sample 30 random para verificar dimensiones
    random.seed(0)
    sample = random.sample(pngs, min(30, len(pngs)))
    dims = {Image.open(p).size for p in sample}
    expected_dim = (2339, 3309)
    dims_ok = dims == {expected_dim}

    ok = (len(pngs) == 5000 and not diff_pdfs_no_pngs and not diff_pngs_no_pdfs and dims_ok)
    result = "PASS" if ok else "FAIL"
    return CheckResult(
        n=3, title="Render 1:1 con dimension consistente",
        claim="5,000 PNGs renderizados 1:1 con los PDFs, todos 2339×3309 px.",
        method="set diff entre PDF stems y PNG stems + sample 30 random verificar dims.",
        evidence=f"png_count={len(pngs)}, pdf_diff={len(diff_pdfs_no_pngs)}, png_diff={len(diff_pngs_no_pdfs)}, sample_dims={dims}",
        result=result,
        notes="" if ok else "Discrepancia entre PDFs y PNGs.",
    )


# ------------------------- CHECK 4: splits sin leak ----------------------

def check_4_splits_no_leak() -> CheckResult:
    splits = {}
    for s in ("train", "val", "test"):
        ids = {l.strip() for l in (ROOT / f"data/splits/{s}_ids.txt").read_text().splitlines() if l.strip()}
        splits[s] = ids

    inter_tv = splits["train"] & splits["val"]
    inter_tt = splits["train"] & splits["test"]
    inter_vt = splits["val"] & splits["test"]

    sample = {l.strip() for l in (ROOT / "data/splits/manuscritas_full.txt").read_text().splitlines() if l.strip()}
    union = splits["train"] | splits["val"] | splits["test"]
    union_matches = union == sample

    sizes = {k: len(v) for k, v in splits.items()}
    sizes_ok = sizes["train"] == 3500 and sizes["val"] == 750 and sizes["test"] == 750

    # Bonus: verificar 200 crops random no caen en split incorrecto
    crops_train_sample = random.sample(list((ROOT / "data/crops_train").rglob("*.png")), 200)
    crops_aids = {p.name.split("_")[0] for p in crops_train_sample}
    crops_misplaced = crops_aids - splits["train"]

    no_leak = not (inter_tv or inter_tt or inter_vt) and union_matches
    ok = no_leak and sizes_ok and not crops_misplaced
    if no_leak and sizes_ok and not crops_misplaced:
        result = "PASS"
        notes = "Sin interseccion entre splits; union cubre el sample original; crops respetan los splits."
    elif no_leak and not sizes_ok:
        result = "WARNING"
        notes = f"Sin leak pero tamanos {sizes} difieren de 3500/750/750."
    else:
        result = "FAIL"
        notes = f"Leak detected: tv={len(inter_tv)}, tt={len(inter_tt)}, vt={len(inter_vt)}, crops_misplaced={len(crops_misplaced)}"

    return CheckResult(
        n=4, title="Sin leak entre splits",
        claim="train/val/test particion por archivoId, sin overlap, union == manuscritas_full.",
        method="Set intersect + verificacion de tamanos + cross-check de 200 crops_train.",
        evidence=f"sizes={sizes}, intersect_tv={len(inter_tv)}, tt={len(inter_tt)}, vt={len(inter_vt)}, union_matches={union_matches}, crops_misplaced={len(crops_misplaced)}",
        result=result, notes=notes,
    )


# ------------------------- CHECK 5: labels vs imagen ---------------------

def _expected_label(archivo_id: str, field: str, pos: int,
                    archivos: pd.DataFrame, votos: pd.DataFrame, cab: pd.DataFrame,
                    template: dict) -> tuple[int, int, int]:
    """Devuelve (value_int, n_cells, expected_digit). raises if not found."""
    arc = archivos[archivos["archivoId"] == archivo_id].iloc[0]
    idActa = int(arc["idActa"])
    f = next(f for f in template["fields"] if f["name"] == field)
    n_cells = f["n_digits"]
    if field.startswith("partido_"):
        npos = int(field.split("_")[1])
        rows = votos[(votos["idActa"] == idActa) & (votos["nposicion"] == npos)]
        value = int(rows.iloc[0]["nvotos"]) if len(rows) else 0
    elif field == "votos_blanco":
        rows = votos[(votos["idActa"] == idActa) & (votos["nposicion"] == 80)]
        value = int(rows.iloc[0]["nvotos"]) if len(rows) else 0
    elif field == "votos_nulos":
        rows = votos[(votos["idActa"] == idActa) & (votos["nposicion"] == 81)]
        value = int(rows.iloc[0]["nvotos"]) if len(rows) else 0
    elif field == "votos_impugnados":
        rows = votos[(votos["idActa"] == idActa) & (votos["nposicion"] == 82)]
        value = int(rows.iloc[0]["nvotos"]) if len(rows) else 0
    elif field == "total_ciudadanos":
        value = int(cab[cab["idActa"] == idActa].iloc[0]["totalVotosEmitidos"])
    else:
        raise KeyError(field)
    return value, n_cells, int_to_digits(value, n_cells)[pos]


def check_5_labels_match_image() -> CheckResult:
    archivos = pd.read_parquet(ROOT / "data/labels/actas_archivos.parquet")
    votos = pd.read_parquet(ROOT / "data/labels/actas_votos.parquet")
    cab = pd.read_parquet(ROOT / "data/labels/actas_cabecera.parquet")
    template = load_templates(ROOT / "templates.json")["presidencial"]

    rng = random.Random(123)
    samples = []
    for label in range(10):
        files = list((ROOT / f"data/crops_train/{label}").glob("*.png"))
        rng.shuffle(files)
        for f in files[:3]:
            samples.append((label, f))

    rows = []
    mismatches = 0
    for label, fpath in samples:
        # parse: <archivoId>_<field>_c<pos>.png
        stem = fpath.stem
        aid = stem[:24]
        rest = stem[25:]  # quitar archivoId + _
        # field es todo menos los ultimos 3 chars _cN
        field = rest.rsplit("_c", 1)[0]
        pos = int(rest.rsplit("_c", 1)[1])
        try:
            value, n_cells, expected = _expected_label(aid, field, pos, archivos, votos, cab, template)
        except Exception as exc:
            rows.append((label, fpath.name, None, None, None, f"ERROR {exc}"))
            mismatches += 1
            continue
        match = (label == expected)
        if not match:
            mismatches += 1
        rows.append((label, fpath.name, value, n_cells, expected, "match" if match else "MISMATCH"))

    # Grid visual
    CELL_W, CELL_H, PAD, HEADER = 120, 100, 10, 30
    n = len(samples)
    cols = 5
    rows_g = (n + cols - 1) // cols
    canvas_w = cols * (CELL_W + PAD) + PAD
    canvas_h = rows_g * (CELL_H + HEADER + PAD) + PAD
    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 14)
    except OSError: font = ImageFont.load_default()
    for i, ((label, fpath), (lbl, name, value, n_cells, expected, status)) in enumerate(zip(samples, rows)):
        r, c = divmod(i, cols)
        x = PAD + c * (CELL_W + PAD)
        y = PAD + r * (CELL_H + HEADER + PAD)
        # encajar img
        img = Image.open(fpath)
        ar = img.size[0] / max(img.size[1], 1)
        new_h = CELL_H
        new_w = min(int(new_h * ar), CELL_W)
        new_h = int(new_w / ar) if ar > 0 else CELL_H
        thumb = img.resize((new_w, new_h))
        canvas.paste(thumb, (x + (CELL_W - new_w) // 2, y + HEADER + (CELL_H - new_h) // 2))
        color = (0, 130, 0) if status == "match" else (200, 0, 0)
        text = f"file={label} got={expected} v={value}"
        draw.text((x, y), text, fill=color, font=font)
        draw.rectangle([x, y + HEADER, x + CELL_W, y + HEADER + CELL_H], outline=(180, 180, 180), width=1)
    canvas.save(VIS / "audit_labels_30.png")

    if mismatches == 0:
        result, notes = "PASS", f"30/30 crops random tienen label correcto vs ground truth."
    elif mismatches <= 2:
        result, notes = "WARNING", f"{30-mismatches}/30 ok; {mismatches} mismatch(es) — revisar audit_labels_30.png"
    else:
        result, notes = "FAIL", f"Solo {30-mismatches}/30 ok; {mismatches} mismatches. Hay bug en el pipeline o labels."
    return CheckResult(
        n=5, title="Labels coinciden con imagen (30 crops random)",
        claim="Cada crop en data/crops_<split>/<label>/ tiene un digito que coincide con el label segun ground truth.",
        method="30 crops random (3 por clase) -> parse archivoId/field/pos -> lookup nvotos -> int_to_digits -> compare con label de carpeta.",
        evidence=f"matches={30-mismatches}/30; visual en {VIS / 'audit_labels_30.png'}",
        result=result, notes=notes,
    )


# ------------------------- CHECK 6: templates random ---------------------

def check_6_templates_random() -> CheckResult:
    from scripts.preview_template import dibujar_overlay
    template = load_templates(ROOT / "templates.json")["presidencial"]

    pngs = list((ROOT / "data/pdfs_train/rendered").glob("*_p0.png"))
    rng = random.Random(777)
    sample = rng.sample(pngs, 20)

    cols, rows_g = 5, 4
    THUMB_W = 500
    THUMB_H = int(500 * 3309 / 2339)
    PAD = 10
    HEADER = 24
    canvas_w = cols * (THUMB_W + PAD) + PAD
    canvas_h = rows_g * (THUMB_H + PAD + HEADER) + PAD
    canvas = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 16)
    except OSError: font = ImageFont.load_default()

    for i, p in enumerate(sample):
        r, c = divmod(i, cols)
        img = Image.open(p)
        overlay = dibujar_overlay(img, template).resize((THUMB_W, THUMB_H))
        x = PAD + c * (THUMB_W + PAD)
        y = PAD + r * (THUMB_H + PAD + HEADER)
        canvas.paste(overlay, (x, y + HEADER))
        draw.text((x + 4, y + 4), p.stem.removesuffix("_p0")[:16], fill=(0, 0, 0), font=font)

    canvas.save(VIS / "audit_overlays_20.png")

    # No tenemos forma de medir overlap automaticamente sin OCR; reporto WARNING
    # neutral indicando que la inspeccion es visual.
    return CheckResult(
        n=6, title="Templates en 20 actas random",
        claim="El template Presidencial calza en cualquier acta random, no solo en las 3 calibradas.",
        method="Overlay del template sobre 20 PNGs random, inspeccion visual del grid generado.",
        evidence=f"20 actas con overlay en {VIS / 'audit_overlays_20.png'}",
        result="MANUAL",
        notes="Requiere inspeccion visual (no se puede medir overlap sin OCR ground-truth). Si las cajas calzan en >=18/20 -> PASS; si 10-17 -> WARNING; si <10 -> FAIL.",
    )


# ------------------------- CHECK 7: val_acc no trivial -------------------

def check_7_val_acc_real() -> CheckResult:
    import torch
    from torch.utils.data import DataLoader
    from dataset import CropsDataset, default_transforms
    from env import torch_device
    from model import build_model

    device = torch_device()
    # Prioriza el modelo del proyecto (ResNet-18). Cae a las lineas de
    # referencia metodologicas si aun no se entreno el ResNet.
    ckpt_candidates = ["resnet18_best.pt", "deep_best.pt", "lenet_best.pt"]
    ckpt_path = None
    for name in ckpt_candidates:
        candidate = ROOT / "checkpoints" / name
        if candidate.exists():
            ckpt_path = candidate
            break
    if ckpt_path is None:
        return CheckResult(
            n=7, title="val_acc no trivial",
            claim="val_acc no es solo majority class.",
            method="-",
            evidence=f"ningun checkpoint encontrado en {ROOT / 'checkpoints'}",
            result="FAIL", notes="No hay checkpoint para evaluar.",
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt.get("arch", "deep"), 1, 10).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Sin augmentation en eval
    ds = CropsDataset(ROOT / "data/manifest_val.csv", root=ROOT / "data/crops_val",
                      transform=default_transforms(32, train=False))
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    confusion = np.zeros((10, 10), dtype=np.int64)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            yy = y.numpy()
            for t, p in zip(yy, preds):
                confusion[t][p] += 1
            correct += (preds == yy).sum()
            total += len(yy)
    acc = correct / total

    # Per-class recall (= diagonal / row sum)
    per_class_recall = np.array([confusion[i, i] / max(confusion[i].sum(), 1) for i in range(10)])
    per_class_n = confusion.sum(axis=1)

    # Render confusion matrix as figure (CPU-only matplotlib)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(f"Confusion matrix val (acc {acc:.4f}, n={total})")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, confusion[i, j], ha="center", va="center",
                    color="white" if confusion[i, j] > confusion.max() / 2 else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(VIS / "audit_confusion_matrix.png", dpi=120)
    plt.close(fig)

    classes_with_real_recall = (per_class_recall > 0.30).sum()
    notes_lines = [f"acc global {acc:.4f}",
                   f"clases con recall > 0.30: {classes_with_real_recall}/10",
                   "por clase:"]
    for c in range(10):
        notes_lines.append(f"  {c}: recall={per_class_recall[c]:.3f}  (n={per_class_n[c]})")

    # Mejor heuristica: si el modelo solo predice clase mayoritaria, recall del resto = 0
    if (per_class_recall == 0).sum() >= 7:
        result = "FAIL"
    elif classes_with_real_recall >= 8:
        result = "PASS"
    elif classes_with_real_recall >= 5:
        result = "WARNING"
    else:
        result = "FAIL"

    return CheckResult(
        n=7, title="val_acc no es trivial",
        claim="El val_acc reportado 75.5% refleja aprendizaje real, no solo clase mayoritaria.",
        method=f"Inferencia con checkpoint {ckpt_path.name} sobre val set sin augmentation; per-class recall + matriz de confusion.",
        evidence=f"acc {acc:.4f}; clases_con_recall>0.30: {classes_with_real_recall}/10; matriz en {VIS / 'audit_confusion_matrix.png'}",
        result=result, notes="\n  ".join(notes_lines),
    )


# ------------------------- main ------------------------------------------

def render_md(results: list[CheckResult]) -> str:
    lines = ["# AUDIT REPORT — Verificacion de Semana 1", ""]
    counts = {"PASS": 0, "FAIL": 0, "WARNING": 0, "MANUAL": 0}
    for r in results:
        counts[r.result] = counts.get(r.result, 0) + 1
    lines.append("## Resumen")
    lines.append("")
    lines.append(f"- **PASS**: {counts.get('PASS', 0)}")
    lines.append(f"- **WARNING**: {counts.get('WARNING', 0)}")
    lines.append(f"- **FAIL**: {counts.get('FAIL', 0)}")
    lines.append(f"- **MANUAL** (require inspeccion visual): {counts.get('MANUAL', 0)}")
    lines.append("")
    lines.append("## Detalle por chequeo")
    lines.append("")
    for r in results:
        lines.append(f"### [CHECK {r.n}] {r.title} — **{r.result}**")
        lines.append("")
        lines.append(f"- **Claim:** {r.claim}")
        lines.append(f"- **Metodo:** {r.method}")
        lines.append(f"- **Evidencia:** {r.evidence}")
        if r.notes:
            lines.append(f"- **Notas:**")
            for ln in r.notes.splitlines():
                lines.append(f"    {ln}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    checks = [
        check_2_pdfs_5k,
        check_3_render_one_to_one,
        check_4_splits_no_leak,
        check_5_labels_match_image,
        check_6_templates_random,
        check_7_val_acc_real,
    ]
    results = []
    for fn in checks:
        print(f"== running check {fn.__name__} ==")
        try:
            r = fn()
        except Exception as exc:
            r = CheckResult(n=int(fn.__name__.split("_")[1]), title=fn.__name__,
                            claim="-", method="-", evidence=f"exception: {exc}",
                            result="FAIL", notes=f"Excepcion durante chequeo: {exc}")
        print(f"  -> {r.result}")
        results.append(r)
    REPORT_PATH.write_text(render_md(results))
    print(f"\nAUDIT_REPORT escrito en: {REPORT_PATH}")


if __name__ == "__main__":
    main()
