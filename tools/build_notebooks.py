"""Genera los notebooks Colab desde los bloques de `_inline_code`.

Fuente de verdad de los notebooks: este script + tools/_inline_code.py. Re-correr
tras editar cualquier bloque inline:

    python tools/build_notebooks.py

Produce (cada uno consume sus propios bloques; se iteran por separado):
  notebooks/01_preprocesamiento_colab.ipynb  bloques PREPROCESS + LABELS_BUILD
                                             (PDFs HF -> crops -> publica bundle en HF)
  notebooks/02_modelo_colab.ipynb            bloques MODEL + DATASET + TRAIN + EVAL + METRICS
                                             (baja crops del bundle -> train -> eval -> metricas)

Lo unico que acopla 01 y 02 son los datos, no el codigo: si cambias el bloque
PREPROCESS, re-corre 01 en Colab para republicar crops_bundle.tar.gz antes de que
02 lo refleje. Al final este script valida que ambos notebooks parsean.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat as nbf

import _inline_code as C

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = json.loads((ROOT / "templates.json").read_text())
HF_DATASET_REPO = "f3r21/actas-cnn-dataset"


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
import os, tarfile
from pathlib import Path
import pandas as pd
import torch

def torch_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")
DEVICE = torch_device(); print("device:", DEVICE)
if DEVICE.type != "cuda":
    print("AVISO: sin GPU CUDA. En Colab: Runtime -> Change runtime type -> T4 GPU. "
          "Sin GPU el entrenamiento es MUY lento.")

HF_DATASET_REPO = "{HF_DATASET_REPO}"   # PDFs + labels (+ crops_bundle si 01 ya corrio)

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
        md("## 2. Labels, universo y descarga de PDFs\n\n"
           "Descarga **en una sola pasada paralela** (`snapshot_download`, una barra de "
           "progreso) en vez de 5000 descargas con 5000 barras, que congelan el front-end "
           "de Colab. Con `HF_TOKEN` puesto va mas rapido (sin rate-limit de anonimo)."),
        code('''import random, logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)  # 1 aviso, no 5000
from huggingface_hub import snapshot_download, list_repo_files
snapshot_download(HF_DATASET_REPO, repo_type="dataset", allow_patterns="labels/*",
                  local_dir=str(DATA))
archivos = pd.read_parquet(DATA / "labels/actas_archivos.parquet")
votos    = pd.read_parquet(DATA / "labels/actas_votos.parquet")
cabecera = pd.read_parquet(DATA / "labels/actas_cabecera.parquet")
# Universo = PDFs realmente publicados en el dataset. El parquet actas tiene ~84k
# presidenciales, pero solo se subieron las 5000 manuscritas; seleccionar del
# parquet pediria PDFs inexistentes (404). list_repo_files da exactamente lo subido.
con_label = set(archivos["archivoId"])
ids = sorted(f[:-4] for f in list_repo_files(HF_DATASET_REPO, repo_type="dataset")
             if f.endswith(".pdf") and f[:-4] in con_label)
random.Random(42).shuffle(ids)
ids = ids[:N_ACTAS]
n = len(ids); ntr = int(n * 0.70); nva = int(n * 0.15)
splits = {"train": ids[:ntr], "val": ids[ntr:ntr + nva], "test": ids[ntr + nva:]}
print({k: len(v) for k, v in splits.items()})

pdf_dir = WORK / "pdfs"; pdf_dir.mkdir(exist_ok=True)
snapshot_download(HF_DATASET_REPO, repo_type="dataset",
                  allow_patterns=[f"{a}.pdf" for a in ids], local_dir=str(pdf_dir))
print(f"{len(ids)} PDFs descargados")'''),
        md("## 3. Render + recorte + manifests (streaming por acta)\n\n"
           "Por cada acta: renderiza, recorta los digitos y **borra el PNG**. Acumular los "
           "~73GB de PNGs intermedios agota el disco de Colab; asi solo crecen los crops "
           "(~600MB). Los PDFs (~14GB) caben y se reusan para la demo."),
        code('''aid_to_idacta = dict(zip(archivos["archivoId"], archivos["idActa"]))
# Restringir labels a las actas elegidas: acelera los joins por idActa del recorte.
sel_idactas = {int(aid_to_idacta[a]) for a in ids if a in aid_to_idacta}
votos    = votos[votos["idActa"].isin(sel_idactas)]
cabecera = cabecera[cabecera["idActa"].isin(sel_idactas)]
rendered = WORK / "rendered"; rendered.mkdir(exist_ok=True)
for split, sids in splits.items():
    croot = DATA / f"crops_{split}"; saved = 0
    for i, aid in enumerate(sids):
        pdf = pdf_dir / f"{aid}.pdf"
        if not pdf.exists(): continue
        png = render_acta(pdf, rendered)
        ns, _ = build_crops_for_acta(png, aid, int(aid_to_idacta[aid]),
                                     TEMPLATE, votos, cabecera, croot)
        saved += ns
        png.unlink(missing_ok=True)   # no acumular los ~73GB de PNG
        if (i + 1) % 500 == 0: print(f"  {split}: {i + 1}/{len(sids)}")
    n_rows = build_manifest(croot, DATA / f"manifest_{split}.csv")
    print(f"{split}: {saved} crops, manifest {n_rows} filas")'''),
        md("## 4. Demostracion: desde una acta hasta los digitos\n\n"
           "Visualiza el resultado de la deteccion sobre una acta: los 42 campos "
           "localizados y, dentro de uno, las celdas de cada digito."),
        code('''import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

demo_aid = ids[0]
demo_png = render_acta(pdf_dir / f"{demo_aid}.pdf", rendered)
img = Image.open(demo_png).convert("RGB"); draw = ImageDraw.Draw(img); w, h = img.size
for f in TEMPLATE["fields"]:
    x0, y0, x1, y1 = f["box"]
    draw.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=(255, 0, 0), width=3)
plt.figure(figsize=(7, 10)); plt.imshow(img); plt.axis("off")
plt.title(f"42 campos detectados — acta {demo_aid[:10]}"); plt.show()

celdas = localize_digits(demo_png, TEMPLATE)
fig, axs = plt.subplots(1, 3, figsize=(6, 2))
for ax, c in zip(axs, celdas["partido_01"]):
    ax.imshow(c, cmap="gray"); ax.axis("off")
fig.suptitle("partido_01 -> 3 celdas (right-justified)"); plt.show()'''),
        md("## 5. Empaquetar y publicar el bundle de crops en HF\n\n"
           "El notebook del modelo (`02_modelo_colab.ipynb`) baja este "
           "`crops_bundle.tar.gz`."),
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
        md("---\nListo. Con el bundle publicado, abre **`02_modelo_colab.ipynb`** "
           "para entrenar y obtener las metricas finales."),
    ]
    return _nb(cells)


# === Notebook 02: modelo ====================================================

def build_modelo() -> nbf.NotebookNode:
    config = '''# Epochs de entrenamiento (~5-8 min en T4 con 20).
EPOCHS = 20'''
    cells = [
        md("# actas-cnn — Modelo + evaluacion (Colab)\n\n"
           "CNN que reconoce las cifras manuscritas de conteo de votos en actas "
           "electorales (ONPE, Elecciones Generales del Peru 2026). Parte de los **crops "
           "preprocesados** (publicados por `01_preprocesamiento_colab.ipynb`) y va hasta "
           "las **metricas finales**.\n\n"
           "**Prerequisito:** corre antes `01_preprocesamiento_colab.ipynb` una vez "
           "(publica `crops_bundle.tar.gz` en HF). Aqui el bundle se baja en segundos.\n\n"
           "**Como correr:** Runtime -> Change runtime type -> **T4 GPU**, luego "
           "Runtime -> Run all.\n\n"
           "| metrica (val, 693 actas) | esperado |\n|---|---|\n"
           "| digit-level | ~98.1% |\n| field-level | ~98.9% |\n"
           "| acta-level (42 campos) | ~90.3% |\n| reconstruccion del total (MAE) | ~2.4 votos |\n\n"
           "Un train fresco varia +-0.5pp por el seed; los resultados de cada corrida "
           "quedan en las salidas del notebook."),
        md("## 0. Setup"),
        code(C.INSTALL),
        code(config_cell(config)),
        code(CELL_TEMPLATE),
        md("## 1. Codigo del modelo (inline)\n\n"
           "Modelo, dataset, entrenamiento y evaluacion. El preprocesamiento (deteccion "
           "de digitos) vive en `01_preprocesamiento_colab.ipynb`."),
        code(C.MODEL),
        code(C.DATASET),
        code(C.TRAIN),
        code(C.EVAL),
        code(C.METRICS),
        md("## 2. Datos: crops preprocesados (cache de HF)"),
        code('''from huggingface_hub import hf_hub_download, snapshot_download

snapshot_download(HF_DATASET_REPO, repo_type="dataset", allow_patterns="labels/*",
                  local_dir=str(DATA))
archivos = pd.read_parquet(DATA / "labels/actas_archivos.parquet")
votos    = pd.read_parquet(DATA / "labels/actas_votos.parquet")
cabecera = pd.read_parquet(DATA / "labels/actas_cabecera.parquet")

# Crops preprocesados publicados por 01_preprocesamiento_colab.ipynb.
try:
    bundle = hf_hub_download(HF_DATASET_REPO, "crops_bundle.tar.gz", repo_type="dataset")
except Exception as e:
    raise RuntimeError("No hay crops_bundle.tar.gz en HF. Corre "
                       "01_preprocesamiento_colab.ipynb primero.") from e
with tarfile.open(bundle) as t: t.extractall(WORK)
print("crops_bundle extraido")'''),
        md("## 3. Entrenamiento"),
        code('''model = train_model(DATA / "manifest_train.csv", DATA / "crops_train", DEVICE, epochs=EPOCHS)'''),
        md("## 4. Evaluacion + metricas finales"),
        code('''df, res = evaluate_split(model, DATA / "manifest_val.csv", DATA / "crops_val",
                         TEMPLATE, archivos, votos, cabecera, DEVICE)
metrics = report_metrics(df, res)'''),
        md("## 5. Visualizaciones + tabla de ablations"),
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
    # IDs deterministas por indice: regenerar no produce churn de git.
    for i, c in enumerate(cells):
        c.id = f"c{i:02d}"
    nb.cells = cells
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    return nb


def _validate(path: Path) -> None:
    """Re-lee el notebook escrito, valida nbformat y parsea cada celda de codigo.

    Las magics de Colab (lineas que empiezan con % o !) no son Python valido; se
    omiten antes de parsear. Lanza si algo no valida/parsea para no publicar un
    notebook roto.
    """
    nb = nbf.read(str(path), as_version=4)
    nbf.validate(nb)
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        src = "\n".join(l for l in c.source.split("\n")
                        if not l.lstrip().startswith(("%", "!")))
        try:
            ast.parse(src)
        except SyntaxError as e:
            raise SyntaxError(f"{path.name} celda {i}: {e}") from e


def main() -> None:
    out = ROOT / "notebooks"
    outputs = {
        out / "01_preprocesamiento_colab.ipynb": build_preprocesamiento(),
        out / "02_modelo_colab.ipynb": build_modelo(),
    }
    print("escritos:")
    for path, nb in outputs.items():
        nbf.write(nb, str(path))
        _validate(path)
        print(f"  notebooks/{path.name}  (valida + parsea)")


if __name__ == "__main__":
    main()
