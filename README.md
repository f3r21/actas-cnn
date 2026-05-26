# actas-cnn

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

## Quickstart

### Opcion A — Local

```bash
git clone <repo-url>
cd actas-cnn
pip install -r requirements.txt

# Bajar el bundle de datos (~460 MB) desde HF
python -c "
from huggingface_hub import hf_hub_download
import os
ckpt = hf_hub_download(repo_id='f3r21/actas-cnn-dataset',
                       filename='data_bundle.tar.gz',
                       repo_type='dataset',
                       token=os.environ['HF_TOKEN'])
os.system(f'tar -xzf {ckpt} -C .')
"

# Entrenar (con la combinacion ganadora de ablations: LS + RA + mixup + cosine LR)
python train.py --manifest data/manifest_train.csv \
                --root data/crops_train \
                --arch resnet18 --epochs 20 \
                --label-smoothing 0.1 --randaugment \
                --mixup 0.2 --cosine-lr \
                --suffix ls_ra_mu_cos

# Evaluar (digit / field / acta-level + reconstruccion totales)
python scripts/evaluate.py --split val \
    --checkpoint checkpoints/resnet18_ls_ra_mu_cos_best.pt
```

### Opcion B — Colab/Kaggle (GPU gratis)

Abrir [`notebooks/train_portable.ipynb`](notebooks/train_portable.ipynb)
en Colab y seguir los pasos. Guia paso a paso en
[`docs/08-setup-colab.md`](docs/08-setup-colab.md). Tiempo total
≈ 20-25 min por ablation en T4 gratuita.

## Pipeline (de PDF a votos por partido)

```
PDF (ONPE)
   └→ pdf_to_images.py        renderiza a PNG 200 dpi (auto-rota landscape)
       └→ scripts/build_crops.py    recorta 42 campos por plantilla calibrada;
                                    parte cada campo en 3-4 celdas; filtra vacias
                                    via convencion right-justified (es_celda_escrita)
           └→ build_dataset.py     genera manifest CSV (path, label)
               └→ dataset.py       CropsDataset PyTorch
                   └→ train.py     ResNet-18 CIFAR sobre MPS / CUDA / CPU
                       └→ scripts/evaluate.py    reconstruye enteros y los suma
                                                 por organizacion politica;
                                                 compara contra parquets oficiales
```

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

El bundle empaquetado (crops + manifests + parquets + templates +
anchors, ~460 MB) vive en
[`f3r21/actas-cnn-dataset`](https://huggingface.co/datasets/f3r21/actas-cnn-dataset)
en Hugging Face.

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
config.py             repos remotos + hiperparametros
env.py                deteccion de device (CUDA/MPS/CPU)
storage.py            capa redundante HF / R2 / W&B (opcional)
pdf_to_images.py      PDF -> PNG batch
extract_crops.py      crop_fields, split_digits, es_celda_escrita
dataset.py            CropsDataset PyTorch
model.py              ResNet18CIFAR + LeNetCNN + DeepCNN
train.py              entrenamiento con flags de ablation
build_dataset.py      crops -> manifest CSV

scripts/
  build_crops.py            genera crops etiquetados desde parquets
  split_dataset.py          70/15/15 split por archivoId
  audit.py                  6 chequeos de integridad sobre dataset + modelo
  evaluate.py               field/acta-level + reconstruccion totales
  detect_fiducials.py       detector zonal de 15 markers ONPE
  audit_errors.py           top-N crops mal clasificados
  preview_template.py       overlay del template sobre PNGs
  preview_crops.py          grilla de digit crops para QA
  audit_*.py                auditorias especificas (fiducial, template, render)
  run_week1_clean_pipeline.sh   regenera Semana 1 end-to-end

notebooks/
  train_portable.ipynb      Colab / Kaggle entry point

docs/
  00-contexto.md            00..08 narrativa numerada
  ...
  08-setup-colab.md
  auditorias/               docs internas de auditorias

archive/
  migracion/                side-project GCS -> HF/IA (no critico)
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
