"""Genera los notebooks Colab entregables desde los bloques de `_inline_code`.

Fuente de verdad de los notebooks: este script + tools/_inline_code.py. Re-correr
tras cambiar el preprocesamiento (PREPROCESS) o cualquier bloque inline:

    python tools/build_notebooks.py

Produce:
  notebooks/01_preprocesamiento_colab.ipynb  (PDFs HF -> crops -> sube a HF)
  notebooks/02_entregable_colab.ipynb         (crops -> train -> eval -> metricas)
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

import _inline_code as C

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = json.loads((ROOT / "templates.json").read_text())
HF_DATASET_REPO = "f3r21/actas-cnn-dataset"
HF_MODEL_REPO = "f3r21/actas-cnn-model"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(src: str):
    return nbf.v4.new_code_cell(src)


# --- celdas con inyeccion (template + config) --------------------------------

CELL_TEMPLATE = ("# Plantilla Presidencial: 42 campos (38 partidos + blanco/nulos/"
                 "impugnados + total).\n# Cajas en fraccion [0,1]; embebida para que el "
                 "notebook sea autonomo.\nTEMPLATE = "
                 + json.dumps(TEMPLATE["presidencial"], ensure_ascii=False, indent=0)
                 .replace("\n", " "))


def config_cell(modo_block: str) -> str:
    return f'''# === Config + entorno ===
import os, tarfile, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def torch_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")
DEVICE = torch_device(); print("device:", DEVICE)

HF_DATASET_REPO = "{HF_DATASET_REPO}"   # PDFs + labels (+ crops_bundle si 01 ya corrio)
HF_MODEL_REPO   = "{HF_MODEL_REPO}"     # checkpoints

WORK = Path("/content") if Path("/content").exists() else Path(".").resolve()
DATA = WORK / "data"; DATA.mkdir(parents=True, exist_ok=True)
{modo_block}'''


# === Notebook 01: preprocesamiento ==========================================

def build_preprocesamiento() -> nbf.NotebookNode:
    cells = [
        md("# actas-cnn — Preprocesamiento (Colab)\n\n"
           "**Que hace:** baja las actas (PDFs) + labels de Hugging Face, las renderiza, "
           "detecta los digitos, recorta y etiqueta, arma los manifests y **publica el "
           "bundle de crops en HF** para que el notebook entregable lo consuma.\n\n"
           "**Esta es la superficie que mas se itera:** para cambiar *como se detectan los "
           "digitos*, edita la celda marcada `PREPROCESAMIENTO` y vuelve a correr todo.\n\n"
           "Requiere un `HF_TOKEN` con permiso de escritura (panel de secretos de Colab) "
           "para la subida final."),
        md("## 0. Setup"),
        code(C.INSTALL),
        code(config_cell('N_ACTAS = 5000   # cuantas actas preprocesar\n'
                         'SUBIR_A_HF = True  # publica crops_bundle.tar.gz en HF (requiere HF_TOKEN)')),
        code(CELL_TEMPLATE),
        md("## 1. PREPROCESAMIENTO — deteccion de digitos (editar aqui)"),
        code(C.PREPROCESS),
        code(C.LABELS_BUILD),
        md("## 2. Descargar PDFs + labels de Hugging Face"),
        code('''from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download(HF_DATASET_REPO, repo_type="dataset", allow_patterns="labels/*",
                  local_dir=str(DATA))
archivos = pd.read_parquet(DATA / "labels/actas_archivos.parquet")
votos    = pd.read_parquet(DATA / "labels/actas_votos.parquet")
cabecera = pd.read_parquet(DATA / "labels/actas_cabecera.parquet")
pres = archivos[(archivos["tipo"] == 1) & (archivos["idEleccion"] == 10)]
ids = pres["archivoId"].tolist()
random.Random(42).shuffle(ids)
ids = ids[:N_ACTAS]
n = len(ids); ntr = int(n * 0.70); nva = int(n * 0.15)
splits = {"train": ids[:ntr], "val": ids[ntr:ntr + nva], "test": ids[ntr + nva:]}
print({k: len(v) for k, v in splits.items()})

pdf_dir = WORK / "pdfs"; pdf_dir.mkdir(exist_ok=True)
for i, aid in enumerate(ids):
    hf_hub_download(HF_DATASET_REPO, f"{aid}.pdf", repo_type="dataset", local_dir=str(pdf_dir))
    if (i + 1) % 200 == 0: print(f"  {i + 1}/{len(ids)} PDFs")
print("PDFs descargados")'''),
        md("## 3. Render + recorte + manifests por split"),
        code('''aid_to_idacta = dict(zip(archivos["archivoId"], archivos["idActa"]))
# Restringir labels a las actas elegidas: acelera los joins por idActa del recorte.
sel_idactas = {int(aid_to_idacta[a]) for a in ids if a in aid_to_idacta}
votos    = votos[votos["idActa"].isin(sel_idactas)]
cabecera = cabecera[cabecera["idActa"].isin(sel_idactas)]
rendered = WORK / "rendered"
for split, sids in splits.items():
    croot = DATA / f"crops_{split}"
    saved = 0
    for aid in sids:
        pdf = pdf_dir / f"{aid}.pdf"
        if not pdf.exists(): continue
        png = render_acta(pdf, rendered)
        ns, _ = build_crops_for_acta(png, aid, int(aid_to_idacta[aid]),
                                     TEMPLATE, votos, cabecera, croot)
        saved += ns
    n_rows = build_manifest(croot, DATA / f"manifest_{split}.csv")
    print(f"{split}: {saved} crops, manifest {n_rows} filas")'''),
        md("## 4. Empaquetar y publicar el bundle de crops en HF\n\n"
           "El entregable (`02_*`) baja este `crops_bundle.tar.gz` en `MODO=\"cache\"`."),
        code('''bundle = WORK / "crops_bundle.tar.gz"
with tarfile.open(bundle, "w:gz") as t:
    for split in ("train", "val", "test"):
        t.add(DATA / f"crops_{split}", arcname=f"data/crops_{split}")
        t.add(DATA / f"manifest_{split}.csv", arcname=f"data/manifest_{split}.csv")
print("bundle:", bundle, round(bundle.stat().st_size / 1e6, 1), "MB")

if SUBIR_A_HF:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_file(path_or_fileobj=str(bundle), path_in_repo="crops_bundle.tar.gz",
                    repo_id=HF_DATASET_REPO, repo_type="dataset")
    print("subido a", HF_DATASET_REPO)
else:
    print("SUBIR_A_HF=False: el bundle quedo local. Ponlo en True (con HF_TOKEN) para publicar.")'''),
        md("---\nListo. Con el bundle publicado, abre **`02_entregable_colab.ipynb`** "
           "(`MODO=\"cache\"`) para entrenar y obtener las metricas finales."),
    ]
    return _nb(cells)


# === Notebook 02: entregable ================================================

def build_entregable() -> nbf.NotebookNode:
    modo = '''# --- Modo de datos ---
# "cache": baja crops ya preprocesados de HF (~12 min). Requiere haber corrido
#          01_preprocesamiento_colab.ipynb una vez (publica crops_bundle.tar.gz).
# "full" : procesa N_ACTAS actas desde PDF, inline (sin prerequisitos, mas lento).
MODO = "cache"
N_ACTAS = 500              # solo MODO="full"
CARGAR_CHECKPOINT = False  # True = baja el checkpoint oficial de HF (numeros exactos)
EPOCHS = 20'''
    cells = [
        md("# actas-cnn — Entregable end-to-end (Colab)\n\n"
           "CNN que reconoce las cifras manuscritas de conteo de votos en actas "
           "electorales (ONPE, Elecciones Generales del Peru 2026). Va **desde las actas "
           "(PDFs) hasta las metricas finales**.\n\n"
           "**Como correr:** Runtime -> Change runtime type -> **T4 GPU**, luego "
           "Runtime -> Run all.\n\n"
           "| metrica (val, 693 actas) | esperado |\n|---|---|\n"
           "| digit-level | ~98.1% |\n| field-level | ~98.9% |\n"
           "| acta-level (42 campos) | ~90.3% |\n| reconstruccion del total (MAE) | ~2.4 votos |\n\n"
           "Modos de datos (celda de config): `cache` (baja crops preprocesados de HF, "
           "rapido) o `full` (procesa `N_ACTAS` desde PDF, inline). Un train fresco varia "
           "+-0.5pp por el seed; `CARGAR_CHECKPOINT=True` reproduce los numeros exactos."),
        md("## 0. Setup"),
        code(C.INSTALL),
        code(config_cell(modo)),
        code(CELL_TEMPLATE),
        md("## 1. Codigo del pipeline (inline)\n\n"
           "Todo el pipeline vive en estas celdas; el notebook es autonomo. La deteccion "
           "de digitos esta marcada como superficie de iteracion."),
        code(C.PREPROCESS),
        code(C.LABELS_BUILD),
        code(C.MODEL),
        code(C.DATASET),
        code(C.TRAIN),
        code(C.EVAL),
        code(C.METRICS),
        md("## 2. Datos: cache (crops de HF) o full (desde PDF)"),
        code('''from huggingface_hub import hf_hub_download, snapshot_download

snapshot_download(HF_DATASET_REPO, repo_type="dataset", allow_patterns="labels/*",
                  local_dir=str(DATA))
archivos = pd.read_parquet(DATA / "labels/actas_archivos.parquet")
votos    = pd.read_parquet(DATA / "labels/actas_votos.parquet")
cabecera = pd.read_parquet(DATA / "labels/actas_cabecera.parquet")

if MODO == "cache":
    try:
        bundle = hf_hub_download(HF_DATASET_REPO, "crops_bundle.tar.gz", repo_type="dataset")
    except Exception as e:
        raise RuntimeError("No hay crops_bundle.tar.gz en HF. Corre 01_preprocesamiento "
                           "primero, o usa MODO='full'.") from e
    with tarfile.open(bundle) as t: t.extractall(WORK)
    print("crops_bundle extraido")
elif MODO == "full":
    pres = archivos[(archivos["tipo"] == 1) & (archivos["idEleccion"] == 10)]
    ids = pres["archivoId"].tolist(); random.Random(42).shuffle(ids); ids = ids[:N_ACTAS]
    n = len(ids); ntr = int(n * 0.70); nva = int(n * 0.15)
    splits = {"train": ids[:ntr], "val": ids[ntr:ntr + nva], "test": ids[ntr + nva:]}
    aid_to_idacta = dict(zip(archivos["archivoId"], archivos["idActa"]))
    pdf_dir = WORK / "pdfs"; pdf_dir.mkdir(exist_ok=True)
    for split, sids in splits.items():
        croot = DATA / f"crops_{split}"
        for aid in sids:
            hf_hub_download(HF_DATASET_REPO, f"{aid}.pdf", repo_type="dataset", local_dir=str(pdf_dir))
            png = render_acta(pdf_dir / f"{aid}.pdf", WORK / "rendered")
            build_crops_for_acta(png, aid, int(aid_to_idacta[aid]), TEMPLATE, votos, cabecera, croot)
        build_manifest(croot, DATA / f"manifest_{split}.csv")
        print(f"{split} listo")
else:
    raise ValueError(MODO)'''),
        md("## 3. Demostracion: desde una acta hasta los digitos"),
        code('''import matplotlib.pyplot as plt
from PIL import ImageDraw

pres_ids = archivos[(archivos["tipo"] == 1) & (archivos["idEleccion"] == 10)]["archivoId"].tolist()
demo_aid = pres_ids[0]
hf_hub_download(HF_DATASET_REPO, f"{demo_aid}.pdf", repo_type="dataset", local_dir=str(WORK / "demo"))
demo_png = render_acta(WORK / "demo" / f"{demo_aid}.pdf", WORK / "demo")

img = Image.open(demo_png).convert("RGB"); draw = ImageDraw.Draw(img); w, h = img.size
for f in TEMPLATE["fields"]:
    x0, y0, x1, y1 = f["box"]
    draw.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=(255, 0, 0), width=3)
plt.figure(figsize=(7, 10)); plt.imshow(img); plt.axis("off")
plt.title(f"42 campos detectados — acta {demo_aid[:10]}"); plt.show()

cells = localize_digits(demo_png, TEMPLATE)
fig, axs = plt.subplots(1, 3, figsize=(6, 2))
for ax, c in zip(axs, cells["partido_01"]):
    ax.imshow(c, cmap="gray"); ax.axis("off")
fig.suptitle("partido_01 -> 3 celdas (right-justified)"); plt.show()'''),
        md("## 4. Entrenamiento (o cargar el checkpoint oficial)"),
        code('''if CARGAR_CHECKPOINT:
    try:
        ckpt = hf_hub_download(HF_MODEL_REPO, "resnet18_best.pt", repo_type="model")
    except Exception as e:
        raise RuntimeError("Checkpoint no publicado en HF todavia. Usa "
                           "CARGAR_CHECKPOINT=False para entrenar, o sube el .pt primero.") from e
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model = resnet18_cifar(1, 10).to(DEVICE); model.load_state_dict(state["model"])
    print("checkpoint oficial cargado (val_acc", round(float(state.get("acc", 0)), 4), ")")
else:
    model = train_model(DATA / "manifest_train.csv", DATA / "crops_train", DEVICE, epochs=EPOCHS)'''),
        md("## 5. Evaluacion + metricas finales"),
        code('''df, res = evaluate_split(model, DATA / "manifest_val.csv", DATA / "crops_val",
                         TEMPLATE, archivos, votos, cabecera, DEVICE)
metrics = report_metrics(df, res)'''),
        md("## 6. Visualizaciones + tabla de ablations"),
        code('''cm, prf = confusion_and_prf(df)
plot_confusion(cm, metrics["digit"])
print("\\nprecision / recall / F1 por clase:")
print(prf.round(4).to_string(index=False))

# Tabla de ablations (si estan los CSV de evaluate por variante en data/).
csv_map = {n: DATA / f for n, f in [("base", "evaluate_val.csv"),
           ("ls_ra", "evaluate_val_ls_ra.csv"),
           ("ls_ra_mu_cos", "evaluate_val_ls_ra_mu_cos.csv")] if (DATA / f).exists()}
if csv_map:
    print("\\nablations:"); print(ablations_table(csv_map).round(4))
else:
    print("\\n(ablations: corre scripts/evaluate.py por variante en local para la tabla del informe)")'''),
        md("---\n**Cierre.** Pipeline completo desde el PDF del acta hasta las metricas de "
           "reconstruccion de votos. Modelo: ResNet-18 estilo CIFAR. El detalle "
           "metodologico y los experimentos (preprocesamiento alternativo, solver) viven "
           "en el repo (`docs/`, `experiments/`)."),
    ]
    return _nb(cells)


def _nb(cells) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    return nb


def main() -> None:
    out = ROOT / "notebooks"
    nbf.write(build_preprocesamiento(), str(out / "01_preprocesamiento_colab.ipynb"))
    nbf.write(build_entregable(), str(out / "02_entregable_colab.ipynb"))
    print("escritos:")
    print("  notebooks/01_preprocesamiento_colab.ipynb")
    print("  notebooks/02_entregable_colab.ipynb")


if __name__ == "__main__":
    main()
