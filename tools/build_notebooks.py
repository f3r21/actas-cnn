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


def badge(notebook_name: str):
    """Celda-badge 'Open in Colab', identica a la que inserta Colab al guardar
    en GitHub: generarla aqui evita que cada save de Colab la re-agregue (churn)."""
    cell = md(f'<a href="https://colab.research.google.com/github/f3r21/actas-cnn/'
              f'blob/main/notebooks/{notebook_name}" target="_parent">'
              f'<img src="https://colab.research.google.com/assets/colab-badge.svg" '
              f'alt="Open In Colab"/></a>')
    cell.metadata = {"id": "view-in-github", "colab_type": "text"}
    return cell


# --- celdas con inyeccion (template + config) --------------------------------

CELL_TEMPLATE = ("# Plantilla Presidencial: 42 campos (38 partidos + blanco/nulos/"
                 "impugnados + total).\n# Cajas en fraccion [0,1]; embebida para que el "
                 "notebook sea autonomo.\nTEMPLATE = "
                 + json.dumps(TEMPLATE["presidencial"], ensure_ascii=False, indent=0)
                 .replace("\n", " "))


def config_cell(modo_block: str, con_gpu: bool = True) -> str:
    if con_gpu:
        device_block = '''import torch

def torch_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")
DEVICE = torch_device(); print("device:", DEVICE)
if DEVICE.type != "cuda":
    print("AVISO: sin GPU CUDA. En Colab: Runtime -> Change runtime type -> T4 GPU. "
          "Sin GPU el entrenamiento es MUY lento.")'''
    else:
        device_block = '''# Este notebook es CPU-only (PyMuPDF/PIL/pandas, la GPU jamas se usa).
# NO pidas runtime GPU: Colab desconecta runtimes con GPU ociosa a mitad de la
# celda larga de render ("Runtime disconnected"). Usa el runtime CPU normal.'''
    return f'''# === Config + entorno ===
import os, tarfile
from pathlib import Path
import pandas as pd
{device_block}

HF_DATASET_REPO = "{HF_DATASET_REPO}"   # PDFs + labels (+ crops_bundle si 01 ya corrio)

WORK = Path("/content") if Path("/content").exists() else Path(".").resolve()
DATA = WORK / "data"; DATA.mkdir(parents=True, exist_ok=True)
{modo_block}'''


# === Notebook 01: preprocesamiento ==========================================

def build_preprocesamiento() -> nbf.NotebookNode:
    cells = [
        badge("01_preprocesamiento_colab.ipynb"),
        md("# actas-cnn — Preprocesamiento (Colab)\n\n"
           "**Que hace:** baja las actas (PDFs) + labels de Hugging Face, las renderiza, "
           "detecta los digitos, recorta y etiqueta, arma los manifests y **publica el "
           "bundle de crops en HF** para que el notebook entregable lo consuma.\n\n"
           "**Esta es la superficie que mas se itera:** para cambiar *como se detectan "
           "los digitos*, edita la celda marcada `PREPROCESAMIENTO`, pon "
           "`REHACER_DESDE_CERO = True` en la config y vuelve a correr todo (sin el "
           "flag, las actas ya procesadas se saltan y publicarias crops del metodo "
           "viejo).\n\n"
           "**Como correr:** runtime **CPU normal** (NO actives GPU: este notebook no la "
           "usa y Colab desconecta runtimes con GPU ociosa a mitad del procesamiento). "
           "Pon el `HF_TOKEN` (permiso de escritura) en el panel de secretos de Colab "
           "**antes** de correr: la subida final lo necesita y la celda de config frena "
           "si falta (se configura una sola vez por cuenta; nunca pegues el token en una "
           "celda: el repo es publico y HF lo revocaria). Luego Run all. Si la "
           "sesion se corta pero la VM sigue viva, re-correr todo continua donde quedo; "
           "si Colab recicla la VM, `/content` se pierde y la corrida empieza de cero."),
        md("## 0. Setup"),
        code(C.INSTALL),
        code(config_cell('''N_ACTAS = 5000   # cuantas actas preprocesar
SUBIR_A_HF = True  # publica crops_bundle.tar.gz en HF (requiere HF_TOKEN)
# True si editaste la deteccion de digitos: borra crops y progreso y reprocesa
# todo. Sin esto, re-correr salta las actas ya hechas (publicarias crops viejos).
REHACER_DESDE_CERO = False
if SUBIR_A_HF:
    import os
    if not os.environ.get("HF_TOKEN"):
        # En Colab, leer el secreto directo y exportarlo como env var. No usar
        # get_token() para esto: cachea por sesion, y si la primera consulta fue
        # antes de dar acceso al secreto, devuelve None para siempre.
        try:
            from google.colab import userdata
            os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        except Exception:  # fuera de Colab, o secreto ausente / sin acceso
            pass
    if not os.environ.get("HF_TOKEN"):
        from huggingface_hub import get_token  # local: .env o token cacheado
        if get_token() is None:
            # Frenar AHORA y no tras ~40 min, cuando fallaria la subida final.
            raise RuntimeError(
                "No hay HF_TOKEN y SUBIR_A_HF=True: la subida final fallaria al "
                "terminar. Agrega HF_TOKEN en el panel de secretos de Colab (icono "
                "de la llave; permiso de escritura; ACTIVA el acceso para este "
                "notebook) y re-corre esta celda, o pon SUBIR_A_HF=False. Se "
                "configura UNA sola vez por cuenta.")''', con_gpu=False)),
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
        md("## 3. Render + recorte + manifests (en memoria, paralelo, reanudable)\n\n"
           "Por cada acta: rasteriza el PDF **directo a imagen en memoria** (el PNG "
           "intermedio costaba ~3/4 del tiempo por acta en encode/decode — medido "
           "0.75s de 0.97s en M2 — y se borraba un segundo despues; los pixeles son "
           "identicos) y guarda solo los crops (~600MB). El etiquetado es "
           "**ink-aware**: ~3% de las actas escribe las cifras corridas a la "
           "izquierda o centradas (viola la convencion right-justified de ONPE); se "
           "detecta donde cae la tinta y los labels se remapean a las celdas "
           "realmente escritas solo cuando es confiable. Corre en paralelo con todos "
           "los nucleos y es **reanudable dentro de la misma VM**: las actas ya "
           "procesadas (`data/procesadas_<split>.txt`) se saltan al re-correr la celda "
           "(si Colab recicla la VM, `/content` se pierde y se empieza de cero). Si "
           "editaste la deteccion de digitos, pon `REHACER_DESDE_CERO = True` en la "
           "config para no reusar crops del metodo viejo."),
        code('''import shutil
from functools import partial
from multiprocessing import get_context
from tqdm.auto import tqdm

if REHACER_DESDE_CERO:
    for split in splits:
        shutil.rmtree(DATA / f"crops_{split}", ignore_errors=True)
        (DATA / f"procesadas_{split}.txt").unlink(missing_ok=True)
    print("REHACER_DESDE_CERO: crops y progreso borrados, se reprocesa todo")

aid_to_idacta = dict(zip(archivos["archivoId"], archivos["idActa"]))
# Restringir labels a las actas elegidas: acelera los joins por idActa del recorte.
sel_idactas = {int(aid_to_idacta[a]) for a in ids if a in aid_to_idacta}
votos    = votos[votos["idActa"].isin(sel_idactas)]
cabecera = cabecera[cabecera["idActa"].isin(sel_idactas)]

def procesa_acta(aid, croot):
    """Una acta end-to-end: rasteriza en memoria, recorta y guarda sus crops."""
    try:
        pdf = pdf_dir / f"{aid}.pdf"
        if not pdf.exists():
            return aid, 0, "pdf no descargado"
        img = rasterize_acta(pdf)
        ns, _ = build_crops_for_acta(img, aid, int(aid_to_idacta[aid]),
                                     TEMPLATE, votos, cabecera, croot)
        return aid, ns, None
    except Exception as e:  # un PDF malo no debe tumbar la corrida entera
        return aid, 0, repr(e)

NPROC = os.cpu_count() or 2
for split, sids in splits.items():
    croot = DATA / f"crops_{split}"
    done_file = DATA / f"procesadas_{split}.txt"
    hechas = set(done_file.read_text().split()) if done_file.exists() else set()
    pend = [a for a in sids if a not in hechas]
    saved, errores = 0, []
    if pend:
        with get_context("fork").Pool(NPROC) as pool, open(done_file, "a") as marca:
            tareas = pool.imap_unordered(partial(procesa_acta, croot=croot), pend)
            for aid, ns, err in tqdm(tareas, total=len(pend), desc=split):
                saved += ns
                if err:
                    errores.append((aid, err))
                else:
                    marca.write(aid + "\\n"); marca.flush()
    n_rows = build_manifest(croot, DATA / f"manifest_{split}.csv")
    print(f"{split}: +{saved} crops ({len(hechas)} actas ya estaban hechas), "
          f"manifest {n_rows} filas, {len(errores)} errores")
    for aid, err in errores[:5]:
        print(f"  ERROR {aid}: {err}")'''),
        md("## 4. Demostracion: desde una acta hasta los digitos\n\n"
           "Visualiza el resultado de la deteccion sobre una acta: los 42 campos "
           "localizados y, dentro de uno, las celdas de cada digito."),
        code('''import matplotlib.pyplot as plt
from PIL import ImageDraw

# Acta de demo: la primera (en el orden del shuffle) con el total escrito.
# Ojo: ids[0] puede ser una acta totalmente en blanco (todos los votos en 0 y
# total NaN en los labels de ONPE); el filtro es_celda_escrita las descarta
# enteras (0 crops), asi que como demo no mostraria ningun digito.
con_total = set(cabecera.loc[cabecera["totalVotosEmitidos"].notna(), "idActa"])
demo_aid = next(a for a in ids if aid_to_idacta[a] in con_total)
demo_total = int(cabecera.loc[cabecera["idActa"] == aid_to_idacta[demo_aid],
                              "totalVotosEmitidos"].iloc[0])

gris = rasterize_acta(pdf_dir / f"{demo_aid}.pdf")
img = gris.convert("RGB"); draw = ImageDraw.Draw(img); w, h = img.size
for f in TEMPLATE["fields"]:
    x0, y0, x1, y1 = f["box"]
    draw.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=(255, 0, 0), width=3)
plt.figure(figsize=(7, 10)); plt.imshow(img); plt.axis("off")
plt.title(f"42 campos detectados — acta {demo_aid[:10]}"); plt.show()

celdas = localize_digits(gris, TEMPLATE)
fig, axs = plt.subplots(1, 4, figsize=(8, 2))
for ax, c in zip(axs, celdas["total_ciudadanos"]):
    ax.imshow(c, cmap="gray"); ax.axis("off")
fig.suptitle(f"total_ciudadanos = {demo_total} -> 4 celdas (right-justified)")
plt.show()'''),
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
    api = HfApi()  # token: HF_TOKEN del entorno o del panel de secretos de Colab
    api.upload_file(path_or_fileobj=str(bundle), path_in_repo="crops_bundle.tar.gz",
                    repo_id=HF_DATASET_REPO, repo_type="dataset")
    print("subido a", HF_DATASET_REPO)
else:
    print("SUBIR_A_HF=False: el bundle quedo local. Ponlo en True (con HF_TOKEN) para publicar.")'''),
        md("---\nListo. Con el bundle publicado, abre **`02_modelo_colab.ipynb`** "
           "para entrenar y obtener las metricas finales."),
    ]
    return _nb(cells, gpu=False)


# === Notebook 02: modelo ====================================================

def build_modelo() -> nbf.NotebookNode:
    config = '''# Epochs de entrenamiento (~5-8 min en T4 con 20).
EPOCHS = 20'''
    cells = [
        badge("02_modelo_colab.ipynb"),
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


def _nb(cells, gpu: bool = True) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    # IDs deterministas por indice: regenerar no produce churn de git.
    for i, c in enumerate(cells):
        c.id = f"c{i:02d}"
    nb.cells = cells
    nb.metadata = {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    # Solo el notebook del modelo pide GPU. El de preprocesamiento es CPU-only:
    # pedir GPU hace que Colab desconecte el runtime por GPU ociosa a mitad
    # del render de 5000 actas.
    if gpu:
        nb.metadata["accelerator"] = "GPU"
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
