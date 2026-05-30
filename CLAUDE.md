# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Lee tambien `docs/` en orden para el contexto completo (00-contexto, 01-decisiones,
02-migracion, 03-pipeline-datos, 04-modelo-entrenamiento, 05-backlog,
06-definicion-proyecto).

## Que es esto

Proyecto II del curso Topicos en Inteligencia Artificial (CCOMP9-1): una CNN en
PyTorch que reconoce las cifras manuscritas de conteo de votos en actas
electorales de las Elecciones Generales del Peru 2026.

- Entregable de definicion (PDF): generado, vence 24/05/2026. Ver
  `docs/06-definicion-proyecto.md`.
- Presentacion final: 18/06/2026.

## Estado actual (2026-05-24, Semana 1 cerrada + audit reconciliado)

- **Pipeline operativo end-to-end**: PDF → PNG → field crops → digit
  crops con labels desde parquets → manifest CSV → CNN entrenada.
- **Dataset en disco**: 5,000 actas Presidenciales = 3,478 manuscritas
  originales + 1,522 extras manuscritas (las STAE de Lima/Callao se
  reemplazaron por extras Presidenciales antes del entrenamiento). 106k
  train + 23k val + 23k test crops, split por archivoId 3500/750/750 sin
  leak. Universo canonico: `data/splits/manuscritas_full.txt`.
- **Modelo del proyecto**: **ResNet-18 estilo CIFAR (He et al., 2015)**,
  11.17M params, stem 3x3 stride 1 sin MaxPool, entrada 1x32x32.
  Entrenada 20 epochs sin augmentation explicito. Metricas oficiales
  via `scripts/evaluate.py --split val` (693 actas, 29,106 campos):
    - digit-level: **98.12%**
    - field-level (campo entero correcto): **98.87%**
    - **acta-level (los 42 campos correctos): 90.33%** (626/693)
    - reconstruccion exacta del total agregado: **93.80%** (650/693)
    - MAE del total agregado: 2.40 votos
- **Linea de referencia**: la CNN custom de Semana 1 (Conv+BN+
  LeakyReLU+Dropout) alcanzo 97.77% val_acc digit-level; ResNet-18
  mejora +0.35pp solo cambiando la arquitectura.
- **Experimento Sem 2 dia 2 (negativo, documentado)**: probamos
  mejorar el preprocesamiento (detector fiducial search-by-prior, std
  de roles TOP bajo 22-27x; projection profile en split_digits; filtro
  image-based con `tiene_tinta`). Tecnicamente correcto pero el
  pipeline nuevo dio -0.72pp en acta-level vs el zonal viejo. El
  techo del ~98% digit-level no esta en el preprocesamiento sino en
  el modelo / labels / ambiguedad inherente. Pipeline zonal viejo es
  el oficial. Backups del experimento: `data/crops_*_v3/`,
  `data/manifest_*_v3.csv`, `checkpoints/resnet18_best_new_pipeline.pt`.
- **Validacion auditada** (`AUDIT_REPORT.md`, 2026-05-24): 5 PASS / 2
  WARNING / 0 FAIL / 1 MANUAL. PASS en conteo de PDFs, render 1:1,
  splits sin leak, labels 30/30 correctos, val_acc no trivial (10/10
  clases recall > 93%). WARNINGs en CHECK 1 (gap pre-Semana-1, path
  externo) y CHECK 8 (snapshot scraper, ya no corre) — historicos, no
  bloqueantes. MANUAL: inspeccion visual del template (overlay grid en
  `data/visualizaciones/audit_overlays_20.png`).
- **Templates calibradas**: Presidencial 42 campos, auto-detectado via
  proyeccion horizontal con OpenCV.
- **train.py protegido contra overwrite**: carga `best_acc` del
  checkpoint existente antes de entrenar; solo sobrescribe si el nuevo
  run supera al anterior. Evita degradar deep_best.pt por smoke tests.
- **Hallazgo importante**: 30% del bucket eran STAE (Lima/Callao
  digital, no manuscrito). Filtrados; pipeline solo trabaja con
  manuscritos puros.

## Prioridad actual (Semana 2-4)

1. **Semana 2**: implementar ResNet-18 estilo CIFAR en `model.py`
   (adaptar stem a 1 canal, 32x32) y entrenar 20 epochs. Ablations:
   augmentation (RandAugment), mixup, label smoothing, profundidad
   (ResNet-18 vs 34). Tracking en W&B.
2. **Semana 2-3**: `scripts/evaluate.py` con metricas field-level,
   acta-level, y reconstruccion del total.
3. **Semana 3-4**: redactar informe (capitulos 1-5) y slides.
4. **Semana 4**: pulido, reproducibilidad, presentacion 18-jun.

El backlog detallado esta en `docs/05-backlog.md`.

## Restricciones (cumplir siempre)

- Maquina del usuario: MacBook Air M2, 24GB. Entrenar local en MPS o en
  Kaggle/Colab; nunca asumir CUDA local.
- Bug MPS: AdaptiveAvgPool con dimensiones no divisibles falla
  (pytorch#96056). Las arquitecturas del repo evitan esa configuracion
  (ResNet-18 CIFAR usa GAP `(1,1)`, divisor seguro).
- Sin emojis en codigo. Comentarios y mensajes en espanol.
- Prints solo cuando sumen (metricas, errores, estado de subidas).
- Independencia de proveedores: el diseno no depende de GCP. Los datos
  son publicos (ONPE).
- Deep Learning dentro del temario (clasificacion con CNN). El recorte
  por plantilla es preprocesamiento clasico, no un modelo de deteccion.

## Arquitectura

El repo tiene tres pistas:

1. **Migracion one-shot** (`archive/migracion/`): saca originales de GCS a
   HF + IA. No critico para el curso, side-project.
2. **Pipeline principal** (raiz + `scripts/`): PDF → crops → entrenamiento.
3. **Capa lakehouse** (`lakehouse/` + `scripts/build_lakehouse.py`): data
   engineering ALREDEDOR del modelo congelado (medallion bronze/silver/gold +
   star schema + calidad + dashboard) sobre ADLS Gen2. No toca el modelo. Ver
   `docs/08-lakehouse.md`. Reporte de calidad en `QUALITY_REPORT.md`.

Las auditorias exploratorias (fiduciales y generalizacion de template)
viven en `docs/auditorias/`; `AUDIT_REPORT.md` en raiz es el resumen
generado por `scripts/audit.py`.

### Pipeline de datos (lineal)

```
pdf_to_images.py            PDFs -> PNGs (PyMuPDF, tamano fijo 2339x3309, auto-rota landscape, --first-page-only)
scripts/split_dataset.py    archivoIds -> splits train/val/test 70/15/15
scripts/build_crops.py      PNGs + parquets + templates -> crops/<label>/*.png
                            (filtra celdas vacias via es_celda_escrita, no umbral)
build_dataset.py            crops -> manifest_<split>.csv
dataset.py                  CropsDataset lee manifest durante entrenamiento
train.py                    ResNet-18 CIFAR (modelo del proyecto) sobre MPS/CUDA/CPU
```

### Modulos transversales

- **`config.py`**: `RemoteConfig` (nombres de repos HF/R2/W&B) y
  `TrainConfig` (hiperparams).
- **`env.py`**: detecta entorno (kaggle/colab/local) y device.
- **`storage.py`**: capa de redundancia HF / R2 / W&B. Opcional.
- **`extract_crops.py`**: `crop_fields`, `split_digits`,
  `es_celda_escrita`, `tiene_tinta` (sanity check image-based).

### Scripts utilitarios

- `scripts/preview_template.py`: visualizador de templates sobre PNG.
- `scripts/preview_crops.py`: grilla de digit crops para QA.
- `scripts/audit.py`: auditoria de claims del dataset/modelo.
- `scripts/run_week1_clean_pipeline.sh`: regenera todo Semana 1.

### Backends de storage

`storage.py` detecta por variable de entorno:
- **HF**: `HF_TOKEN`
- **R2**: `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`
- **W&B**: `WANDB_API_KEY`

Tokens locales en `.env` (ver `.env.example`); en Kaggle/Colab via
paneles de secretos. Nunca hardcodear. `config.py` tiene TODOs para
rellenar antes de Semana 4.

### Modelos

`model.py` expone via `build_model(arch, ...)`:
- `ResNet18CIFAR`: modelo del proyecto. ResNet-18 estilo CIFAR
  (He et al., 2015) con stem `3x3 stride 1` adaptado a entrada 1×32×32,
  4 etapas residuales, GAP final. **Pendiente de implementar en Sem 2.**
- `LeNetCNN` y `DeepCNN`: lineas de referencia metodologicas de
  Semana 1, conservadas para reproducibilidad. La CNN custom alcanzo
  97.77% val_acc; es el piso a superar con ResNet-18.

## Como trabajar

- No hay acceso a GCP desde aqui. Los parquets curados ya estan locales en
  `data/labels/`. Los PDFs de entrenamiento ya estan en `data/pdfs_train/`.
- Antes de tocar el pipeline de datos, recordar: STAE de Lima/Callao se
  filtra por 2 paginas; manuscritas son 1 pagina.
- Antes de declarar resultados, correr `scripts/audit.py` para validar.

## Comandos

```
pip install -r requirements.txt

# Pipeline completo (regenerar Semana 1):
bash scripts/run_week1_clean_pipeline.sh

# Solo entrenar (asumiendo manifests listos):
python train.py --manifest data/manifest_train.csv --root data/crops_train \
                --arch deep --epochs 20

# Auditoria de progreso:
python scripts/audit.py   # genera AUDIT_REPORT.md
```

Entrenamiento en GPU gratis: abrir `notebooks/train_portable.ipynb` en
Kaggle o Colab.

## Datos en disco (al cerrar Semana 1)

```
data/
├── labels/             5 parquets de ONPE + SCHEMAS.md + COVERAGE.md (122 MB)
├── pdfs_train/         5,000 PDFs Presidenciales manuscritas (10.7 GB)
│   └── rendered/       6,522 PNGs renderizados a 200 DPI (~65 GB)
├── splits/             listas de archivoIds por split + manuscritas_full.txt
├── crops_train/        106,123 crops PNG por label/ (415 MB)
├── crops_val/          22,876 crops (89 MB)
├── crops_test/         22,955 crops (90 MB)
├── manifest_<split>.csv per-split manifests
├── sample_pdfs/        3 actas calibracion + 5 muestras varias idEleccion
└── visualizaciones/    audit grids, thumbnails, confusion matrix
```

Total: ~89 GB. Los PNGs renderizados se pueden regenerar en ~30 min.
