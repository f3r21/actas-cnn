# actas-cnn

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/f3r21/actas-cnn/blob/main/notebooks/02_entregable_colab.ipynb)

Reconocimiento automatico y agregacion de votos manuscritos en actas
electorales presidenciales del Peru (Elecciones Generales 2026) con
redes neuronales convolucionales en PyTorch. Proyecto del curso
Topicos en Inteligencia Artificial (CCOMP9-1).

## Resultados oficiales (val set, 693 actas, 29,106 campos)

| Metrica | Valor |
|---|---|
| Digit-level accuracy | **98.12%** |
| Field-level accuracy | **98.87%** |
| **Acta-level accuracy** (los 42 campos correctos) | **90.33%** |
| **Reconstruccion exacta del total agregado** | **93.80%** (650/693) |
| MAE del total agregado | 2.40 votos |
| Mediana abs error | 0 |
| Total ciudadanos correcto (campo 4 digitos) | 95.96% |

Modelo: **ResNet-18 estilo CIFAR** (He et al., 2015), 11.17M params,
adaptada a entrada 32×32 px en escala de grises.

Detalle completo del proyecto y del pipeline en
[`CLAUDE.md`](CLAUDE.md) y en [`docs/`](docs/).

## Entregable

El entregable son **dos notebooks de Colab** que separan el preprocesamiento
(lo que mas se itera) del modelo, comunicados por el bundle de crops en HF:

- **[`notebooks/01_preprocesamiento_colab.ipynb`](notebooks/01_preprocesamiento_colab.ipynb)** —
  preprocesa las 5,000 actas (render → deteccion de digitos → crops → manifests)
  y **publica el bundle de crops en HF**. Es la superficie que mas se itera:
  para cambiar *como se detectan los digitos* se edita aca y se re-publica; el
  notebook del modelo consume la ultima version.

- **[`notebooks/02_entregable_colab.ipynb`](notebooks/02_entregable_colab.ipynb)** —
  modelo + evaluacion: abrir en Colab, `Runtime → Change runtime type → T4 GPU`,
  `Run all`. Es autonomo (lleva el codigo del modelo inline; no clona el repo) y
  baja los crops preprocesados del HF dataset publico (prerequisito: `01` corrio
  una vez). Entrena (~5-8 min en T4) y reporta las metricas finales;
  `CARGAR_CHECKPOINT=True` reproduce los numeros oficiales exactos.

## Reproduccion local (paquete)

```bash
git clone https://github.com/f3r21/actas-cnn.git
cd actas-cnn
pip install -e .          # instala el paquete actas_cnn (layout src/)

# Con los crops en data/ (bundle de HF o generados por scripts/build_crops.py):
python scripts/train.py    --manifest data/manifest_train.csv --root data/crops_train \
                           --arch resnet18 --epochs 20 \
                           --label-smoothing 0.1 --randaugment --mixup 0.2 --cosine-lr \
                           --suffix ls_ra_mu_cos
python scripts/evaluate.py --split val --checkpoint checkpoints/resnet18_ls_ra_mu_cos_best.pt
python scripts/audit.py    # genera AUDIT_REPORT.md
```

Los scripts son wrappers CLI delgados del paquete `actas_cnn`. Los notebooks se
generan desde el paquete con `python tools/build_notebooks.py`.

## Pipeline (de PDF a votos por partido)

```
PDF (ONPE)
   └→ actas_cnn.render           renderiza a PNG 2339x3309 (auto-rota landscape)
       └→ actas_cnn.preprocess   recorta 42 campos por plantilla calibrada;
                                  parte cada campo en 3-4 celdas; filtra vacias
                                  via convencion right-justified (es_celda_escrita)
                                  *** superficie de iteracion: deteccion de digitos ***
           └→ actas_cnn.data     build_manifest (path,label) + CropsDataset
               └→ actas_cnn.training   ResNet-18 CIFAR sobre MPS / CUDA / CPU
                   └→ actas_cnn.evaluate   reconstruye enteros y los suma por
                                           organizacion politica; compara contra
                                           los parquets oficiales
```

Wrappers CLI: `scripts/build_crops.py`, `scripts/build_dataset.py`,
`scripts/train.py`, `scripts/evaluate.py`.

42 campos: 38 organizaciones politicas + votos en blanco + votos
nulos + votos impugnados + total de ciudadanos votantes.

## Datos

- **Fuente**: bucket publico de la Oficina Nacional de Procesos
  Electorales (ONPE), Elecciones Generales 2026.
- **Universo**: 84,449 actas presidenciales de escrutinio
  (`idEleccion=10`, `tipo=1`), ~10% del total del bucket (871,001
  PDFs).
- **Muestra de trabajo**: 5,000 actas manuscritas con semilla fija.
  Se excluyen las actas STAE de Lima y Callao (digitales, 2 paginas);
  manuscritas tienen 1 pagina.
- **Etiquetas**: cruce determinístico con manifiestos curados de
  ONPE (`actas_archivos`, `actas_cabecera`, `actas_votos`). Sin
  anotacion manual.
- **Partición**: 70/15/15 por `archivoId` sin leak. 106,123 / 22,876 /
  22,955 crops de 32×32 grayscale.
- **Validacion del manifest**: 469/469 actas (con `totalVotosEmitidos`
  no-nulo) tienen `sum(votos) == total`. Internamente consistente.

En Hugging Face, [`f3r21/actas-cnn-dataset`](https://huggingface.co/datasets/f3r21/actas-cnn-dataset)
aloja los **5,000 PDFs fuente** (raiz) y los **labels** (`labels/`, parquets
ONPE). El bundle de crops ya preprocesados (`crops_bundle.tar.gz`) lo genera y
publica `notebooks/01_preprocesamiento_colab.ipynb`; el entregable lo consume en
`MODO="cache"`.

## Modelo

- **ResNet-18 estilo CIFAR** (He et al., 2015): adaptada para entrada
  1×32×32. Stem `Conv2d(1, 64, 3, stride=1, padding=1)` sin MaxPool
  inicial para preservar resolucion en imagenes chicas. 4 etapas
  residuales (64 → 128 → 256 → 512), Global Average Pool, Linear(512,
  10). 11.17M parametros.
- **Entrenamiento**: Adam lr=5e-4, batch 128, 20 epochs. Sin
  augmentation explicita = baseline. La combinacion ganadora de
  ablations agrega label smoothing 0.1, RandAugment, mixup α=0.2 y
  cosine LR schedule.
- **Restriccion MPS**: `AdaptiveAvgPool2d((1,1))` (GAP) evita el bug
  pytorch#96056 con divisores no divisibles.

`model.py` tambien expone `LeNetCNN` y `DeepCNN` como lineas de
referencia metodologicas de Semana 1 (CNN custom alcanzo 97.77%
val_acc digit-level).

## Estructura del repo

```
src/actas_cnn/            paquete (fuente de verdad del pipeline)
  render.py               PDF -> PNG
  preprocess/             *** deteccion de digitos (enchufable: localize_digits) ***
    crops.py              localize_digits, crop_fields, split_digits, es_celda_escrita, labels, build_crops_for_acta
  data.py                 CropsDataset, transforms, build_manifest
  model.py                resnet18_cifar + LeNetCNN + DeepCNN
  training.py             entrenamiento con flags de ablation
  evaluate.py             field/acta-level + reconstruccion de totales
  viz.py                  overlays del template
  config.py / env.py / storage.py   transversales (storage = Hugging Face)

scripts/                  wrappers CLI delgados (repro / testing)
  build_crops.py, build_dataset.py, split_dataset.py, train.py, evaluate.py
  audit.py                6 chequeos de integridad (genera AUDIT_REPORT.md)
  preview_template.py, preview_crops.py, inspect_labels.py   QA visual
  run_week1_clean_pipeline.sh

notebooks/
  01_preprocesamiento_colab.ipynb   PDFs HF -> crops -> publica bundle en HF
  02_entregable_colab.ipynb         *** ENTREGABLE *** crops -> train -> eval -> metricas

experiments/              fuera del hot path (pruebas / experimentos)
  fiducial/               localizador alternativo por marcadores (exp. negativo)
  audits/                 auditorias exploratorias
  solver/                 post-procesamiento con restriccion (codigo a recuperar)

tools/build_notebooks.py  genera los notebooks desde el paquete

docs/                     00-contexto .. 07-presentacion-outline (narrativa)
archive/                  side-projects archivados (migracion + auditorias historicas)
```

## Validacion del modelo

`python scripts/audit.py` corre 6 chequeos sobre el estado actual del
disco y del checkpoint. Resultado al ultimo commit:

- 5 PASS (conteo de PDFs, render 1:1, splits sin leak, labels-imagen
  30/30, val_acc 10/10 clases con recall > 0.95).
- 0 WARNING / 0 FAIL.
- 1 MANUAL (overlay del template, requiere inspeccion visual del
  grid generado).

Detalle en [`AUDIT_REPORT.md`](AUDIT_REPORT.md).

## Creditos

- **Datos**: Oficina Nacional de Procesos Electorales (ONPE), Peru.
  Bucket publico `gs://onpe-eg2026-pdfs-v2/` y manifiestos curados.
- **Arquitectura**: He, K., Zhang, X., Ren, S., & Sun, J. (2015).
  *Deep Residual Learning for Image Recognition.* CVPR 2016.
  https://arxiv.org/abs/1512.03385
- **Curso**: Topicos en Inteligencia Artificial (CCOMP9-1), 2026.
