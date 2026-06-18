# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Lee tambien `docs/` en orden para el contexto completo (00-contexto, 01-decisiones,
03-pipeline-datos, 04-modelo-entrenamiento, 05-backlog, 06-definicion-proyecto,
07-presentacion-outline). La doc del side-project de migracion se archivo en
`archive/migracion/02-migracion.md`.

## Que es esto

Proyecto II del curso Topicos en Inteligencia Artificial (CCOMP9-1): una CNN en
PyTorch que reconoce las cifras manuscritas de conteo de votos en actas
electorales de las Elecciones Generales del Peru 2026.

- Entregable de definicion (PDF): generado, vence 24/05/2026. Ver
  `docs/06-definicion-proyecto.md`.
- Presentacion final: 18/06/2026.

## Estado actual (2026-06-18, modelo oficial ink-aware re-entrenado)

- **Hallazgo y fix de etiquetado (2026-06-10)**: ~3% de las actas viola
  la convencion right-justified de ONPE (escribe las cifras corridas a
  la izquierda o centradas). El etiquetado posicional las envenenaba y
  **concentraban el 82% de los errores de campo** en val (19 de las 19
  actas de la cola con >=5 campos mal; 0 eran desalineacion geometrica
  del escaneo — auditado en `experiments/justificacion/`). El
  preprocesamiento ahora es **ink-aware**: `remapeo_ink_aware`
  (`src/actas_cnn/preprocess/crops.py`) detecta donde cae la tinta y
  remapea labels solo en las actas corridas, con guard contra digitos a
  caballo por offset; `evaluate.reconstruct_value` reconstruye por
  concatenacion de celdas presentes (equivalencia exacta verificada
  sobre los crops de mayo). **Delta del lado de evaluacion** (mismo
  `resnet18_best.pt`, sin re-entrenar, sustituyendo las 20 actas
  violadoras por crops ink-aware): field 98.87 -> **99.45%**, acta-level
  90.33 -> **90.62%**, MAE total 2.40 -> **1.58**, 330 -> 159 campos mal
  (-52%), **0 regresiones** en las otras 673 actas. El fix esta
  propagado a los notebooks Colab y al `crops_bundle.tar.gz` republicado
  en HF (2026-06-10).
- **Modelo oficial ink-aware: ENTRENADO en Colab (2026-06-18)**. Se
  corrieron los tres notebooks sobre el bundle ink-aware. La receta
  ganadora `ls_ra_mu_cos` (label smoothing + RandAugment + mixup +
  cosine LR) es ahora la oficial; metricas del run en Colab
  (`02_modelo_colab.ipynb`), **primera evaluacion sobre test incluida**:
    - val:  digit 98.83%, field 99.34%, acta-level 90.62%, recon exacta 93.80%, MAE 1.77
    - test: digit 98.31%, field 99.00%, acta-level 88.84%, recon exacta 91.95%, MAE 2.12
  Numeros del run publicado (commit `704b3b2`; train fresco sin semilla,
  ±0.5pp corrida-a-corrida). El **checkpoint `resnet18_best.pt`
  (ls_ra_mu_cos ink-aware) ya esta publicado en HF** (`f3r21/actas-cnn-model`,
  2026-06-18), sobrescribiendo el base de mayo. Crops locales `data/crops_<split>` son
  ink-aware; los de mayo quedan como respaldo `data/crops_<split>_mayo`.
- **Ablacion limpia sobre ink-aware (2026-06-18, `03_ablaciones_colab.ipynb`)**:
  el ranking `ls_ra_mu_cos > ls_ra > base` se sostiene en val Y test, de
  forma monotona (tablas en `README.md` y `docs/04`). base/ls_ra salen de
  03; `ls_ra_mu_cos` de 02 (la corrida de 03 se corto en epoch 16 de la
  3a variante; falta re-correrla completa para auto-generar la tabla/CSV).
  La brecha val->test es modesta (~0.6pp digit, ~2pp acta): generaliza
  sin overfit.

## Estado heredado (2026-06-09, comparativa de ablations cerrada)

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
- **Ablations: comparativa cerrada (2026-06-09)**. Las 3 variantes
  re-evaluadas con `scripts/evaluate.py --split val` y consolidadas con
  `scripts/ablation_table.py` (tabla completa en
  `docs/04-modelo-entrenamiento.md`, resumen en
  `data/ablations_summary.csv`, logs en `data/evaluate_val*.log`):
  `ls_ra_mu_cos` (label smoothing + RandAugment + mixup + cosine LR)
  domina en todas las metricas — digit 98.21%, field 98.93%, acta-level
  92.21% (+1.88pp vs base), reconstruccion exacta 95.24%, MAE 2.18.
  `resnet18_best.pt` en HF era el base mayo; el 2026-06-18 se promovio
  `ls_ra_mu_cos` ink-aware y se publico sobre ese mismo nombre en HF
  (ver "Estado actual" arriba).
- **Evaluacion extendida ya generada**: matriz de confusion 10x10,
  per-class precision/recall/F1, histograma de errores y ranking
  worst-20 (`data/visualizaciones/evaluate_confusion_val.png`,
  `evaluate_error_hist_val.png`, `data/evaluate_worst20_val.csv`).
- **Experimento de solver (post-procesamiento, sin codigo en el repo)**:
  hay artefactos `data/eval_with_solver_val_K*_tol*.csv` (2026-05-27,
  columnas `baseline` vs `solver` por campo, barriendo top-K y
  tolerancia) que corrigen las predicciones de campo con una restriccion
  (probablemente el total reportado). El script/notebook que los genera
  NO esta en el repo (ningun .py/.ipynb menciona `solver`); recuperar o
  reimplementar antes de citarlo en el informe.
- **Experimento Sem 2 dia 2 (negativo, documentado)**: probamos
  mejorar el preprocesamiento (detector fiducial search-by-prior, std
  de roles TOP bajo 22-27x; projection profile en split_digits; filtro
  image-based con `tiene_tinta`). Tecnicamente correcto pero el
  pipeline nuevo dio -0.72pp en acta-level vs el zonal viejo. El techo
  no estaba en la *geometria* del preprocesamiento (alinear mejor no
  ayudo) sino en los **labels**: el etiquetado posicional asumia la
  convencion right-justified que ~3% de las actas viola (ver el hallazgo
  ink-aware de 2026-06-10 al inicio). Pipeline zonal viejo es el
  oficial. Backup del experimento conservado:
  `checkpoints/resnet18_best_new_pipeline.pt` (los `data/crops_*_v3/` y
  `data/manifest_*_v3.csv` se borraron en la limpieza de disco del 2026-06-01).
- **Validacion auditada** (`AUDIT_REPORT.md`, via `scripts/audit.py`):
  5 PASS / 0 WARNING / 0 FAIL / 1 MANUAL. PASS en conteo de PDFs
  (CHECK 2), render 1:1 (CHECK 3), splits sin leak (CHECK 4), labels
  30/30 correctos (CHECK 5) y val_acc no trivial (CHECK 7: 98.12%,
  10/10 clases con recall > 0.95). MANUAL: inspeccion visual del
  template en 20 actas (CHECK 6); el overlay vivia en
  `data/visualizaciones/` (borrado el 2026-06-01, regenerable con
  `scripts/audit.py`).
- **Templates calibradas**: Presidencial 42 campos, auto-detectado via
  proyeccion horizontal con OpenCV.
- **train.py protegido contra overwrite**: carga `best_acc` del
  checkpoint existente antes de entrenar; solo sobrescribe si el nuevo
  run supera al anterior. Evita degradar deep_best.pt por smoke tests.
- **Hallazgo importante**: 30% del bucket eran STAE (Lima/Callao
  digital, no manuscrito). Filtrados; pipeline solo trabaja con
  manuscritos puros.

## Prioridad actual (cierre)

**No hay informe ni slides que entregar** (decision del 2026-06-09). Lo
que queda:

**Cerrado el 2026-06-18**: el checkpoint oficial `ls_ra_mu_cos` ink-aware
se entreno en Colab y se publico en HF (`f3r21/actas-cnn-model`,
`resnet18_best.pt`), con val+test documentados. Lo que queda es opcional:

1. Opcional: **re-correr `03_ablaciones_colab.ipynb` completo** (la
   ultima corrida se corto en epoch 16 de `ls_ra_mu_cos`) para
   auto-generar la tabla y `data/ablations_ink_summary.csv` en una sola
   pasada, en vez de tomar la fila de `ls_ra_mu_cos` del 02.
2. Opcional: refrescar `AUDIT_REPORT.md` (CHECK 3/6 necesitan
   `data/pdfs_train/rendered/`, borrado el 06-01).

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

**El entregable son dos notebooks de Colab** que separan el preprocesamiento del
modelo, comunicados por el bundle de crops en HF:
`notebooks/01_preprocesamiento_colab.ipynb` (actas PDFs -> crops -> publica
bundle en HF; superficie que mas se itera; **runtime CPU, nunca GPU**: no la
usa, y la combinacion de GPU ociosa + corrida secuencial de horas es la causa
probable del "Runtime disconnected" que mataba la corrida a mitad del render) y
`notebooks/02_modelo_colab.ipynb` (baja los crops -> entrena -> evalua ->
metricas; runtime T4). Ambos autonomos (codigo inline, no clonan el repo).
Desde 2026-06-09 el loop de render+recorte de 01 es en memoria (sin PNG
intermedio; pixeles identicos, verificado byte a byte sobre 469 crops),
paralelo (multiprocessing fork) y reanudable dentro de la misma VM
(`data/procesadas_<split>.txt`; si Colab recicla la VM se empieza de cero).
Para iterar la deteccion de digitos hay que poner `REHACER_DESDE_CERO=True`
(si no, el skip de reanudacion publicaria crops del metodo viejo).
El resto del repo es el "laboratorio" que los respalda. Tres capas:

1. **`src/actas_cnn/`** — paquete, fuente de verdad del pipeline. Los notebooks
   inline-an su logica (DRY via `tools/build_notebooks.py`); `scripts/` son
   wrappers CLI delgados.
2. **`experiments/`** — fuera del hot path (localizador fiducial = experimento
   negativo, auditorias exploratorias, solver). Para pruebas, no para metricas
   oficiales.
3. **`archive/`** — side-projects archivados (migracion GCS→HF, auditorias
   historicas). `AUDIT_REPORT.md` en raiz lo genera `scripts/audit.py`.

### Pipeline de datos (lineal)

```
actas_cnn.render            PDFs -> PNGs (PyMuPDF, tamano fijo 2339x3309, auto-rota landscape)
                            rasterize_first_page: mismo raster directo a PIL en memoria
                            (pixeles identicos; evita el PNG intermedio, ~3/4 del tiempo
                            por acta medido en M2)
actas_cnn.preprocess        imagen (PIL o PNG) + plantilla -> crops/<label>/*.png  *** deteccion de digitos (enchufable) ***
                            (zonal por plantilla = oficial; filtra vacias via es_celda_escrita)
actas_cnn.data              build_manifest (path,label) + CropsDataset
actas_cnn.training          ResNet-18 CIFAR (modelo del proyecto) sobre MPS/CUDA/CPU
actas_cnn.evaluate          reconstruye enteros, suma por partido, compara vs parquets
scripts/split_dataset.py    archivoIds -> splits train/val/test 70/15/15
```

`actas_cnn.preprocess` aisla *donde estan los digitos* en la funcion
`localize_digits` (zonal por plantilla, oficial). Es la superficie que mas se
itera: para cambiar la deteccion, reemplaza `localize_digits` o pasa otro
callable a `build_crops_for_acta` (o edita el bloque PREPROCESS de
`tools/_inline_code.py` para los notebooks). El localizador fiducial vive en
`experiments/fiducial/`.

### Modulos transversales

- **`actas_cnn.config`**: `RemoteConfig` (repos HF) y `TrainConfig`.
- **`actas_cnn.env`**: detecta entorno (kaggle/colab/local) y device; `base_dir()`.
- **`actas_cnn.storage`**: subida/descarga a Hugging Face. Opcional (requiere `HF_TOKEN`).
- **`actas_cnn.viz`**: overlays del template (compartido por previews y auditorias).

### Scripts utilitarios

- `scripts/preview_template.py`, `scripts/preview_crops.py`: QA visual.
- `scripts/audit.py`: auditoria de claims del dataset/modelo.
- `scripts/ablation_table.py`: consolida la tabla comparativa de
  ablations desde los logs/CSVs de `scripts/evaluate.py`.
- `scripts/run_week1_clean_pipeline.sh`: regenera todo Semana 1.
- `tools/build_notebooks.py`: genera los notebooks Colab desde el paquete.

### Storage (Hugging Face)

`storage.py` sube/baja a HF con `HF_TOKEN` (escritura para subir; descargas de
repos publicos sin token). Token local en `.env` (ver `.env.example`); en
Kaggle/Colab via paneles de secretos. Nunca hardcodear. Los repos ya estan en
`actas_cnn.config` (`f3r21/actas-cnn-{dataset,model}`).

### Modelos

`actas_cnn.model` expone via `build_model(arch, ...)`:
- `resnet18_cifar()` (`arch="resnet18"`): modelo del proyecto, ya
  implementado. Parchea `torchvision.models.resnet18`: stem `3x3
  stride 1`, `maxpool=Identity`, `in_channels=1` para entrada 1×32×32;
  mantiene las 4 etapas residuales y el GAP `(1,1)` final.
- `LeNetCNN` y `DeepCNN` (`arch="lenet"` / `"deep"`): lineas de
  referencia metodologicas de Semana 1, conservadas para
  reproducibilidad. La CNN custom (deep) alcanzo 97.77% val_acc; es el
  piso que ResNet-18 supera (+0.35pp).

## Como trabajar

- No hay acceso a GCP desde aqui. Los parquets curados ya estan locales en
  `data/labels/`. Los PDFs de entrenamiento ya estan en `data/pdfs_train/`.
- Antes de tocar el pipeline de datos, recordar: STAE de Lima/Callao se
  filtra por 2 paginas; manuscritas son 1 pagina.
- Antes de declarar resultados, correr `scripts/audit.py` para validar.

## Comandos

```
pip install -e .          # instala el paquete actas_cnn (layout src/)

# Pipeline completo (regenerar Semana 1):
bash scripts/run_week1_clean_pipeline.sh

# Solo entrenar el modelo del proyecto (asumiendo manifests listos):
python scripts/train.py --manifest data/manifest_train.csv --root data/crops_train \
                        --arch resnet18 --epochs 20

# Evaluacion + auditoria:
python scripts/evaluate.py --split val --checkpoint checkpoints/resnet18_best.pt
python scripts/audit.py   # genera AUDIT_REPORT.md

# Regenerar los notebooks Colab desde el paquete:
python tools/build_notebooks.py
```

Entrenamiento en GPU gratis: abrir `notebooks/02_modelo_colab.ipynb`
en Colab (T4, baja los crops preprocesados del dataset HF publico y entrena;
prerequisito: `01_preprocesamiento_colab.ipynb` publico el bundle una vez).

## Datos en disco (tras limpieza de disco 2026-06-01)

```
data/
├── labels/             5 parquets de ONPE + SCHEMAS.md + COVERAGE.md (114 MB)
├── pdfs_train/         5,000 PDFs Presidenciales manuscritas (~14 GB)
│                       rendered/ BORRADO (regenerable con actas_cnn.render)
├── splits/             listas de archivoIds por split + manuscritas_full.txt
├── crops_train/        106,123 crops PNG por label/ (415 MB)
├── crops_val/          22,876 crops (89 MB)
├── crops_test/         22,955 crops (90 MB)
├── manifest_<split>.csv per-split manifests
└── sample_pdfs/        3 actas calibracion + 5 muestras varias idEleccion
```

Total: **~15 GB** (antes ~89 GB). El 2026-06-01 se borraron derivados
regenerables para liberar disco: `pdfs_train/rendered/` (73 GB de PNG),
`visualizaciones/` (1.8 GB), los backups del experimento negativo
`crops_*_v3/` + `manifest_*_v3.csv`, y `data_bundle.tar.gz`. El
entrenamiento lee de `data/crops_*` (no de rendered/), asi que se entrena
sin regenerar nada. Todo lo borrado es gitignored (`data/`, `checkpoints/`).

> Nota (2026-06-10): `data/crops_<split>` locales ahora son **ink-aware**
> (render de tamano fijo + remapeo de actas corridas). Los anteriores
> right-justified (render de mayo) quedaron como respaldo
> `data/crops_<split>_mayo` — necesarios para reproducir exactamente las
> metricas de `resnet18_best.pt` (el checkpoint de mayo es sensible a la
> geometria del render: con crops de geometria nueva su digit-level baja
> ~2pp por mismatch, no por el etiquetado).

**Respaldo remoto** (nuevo): los 5,000 PDFs fuente y `labels/` estan
subidos al dataset HF publico `f3r21/actas-cnn-dataset` (PDFs en raiz,
labels en `labels/`). Es la primera copia fuera del disco local;
descargable con `huggingface_hub` si se pierde el local.

> Nota de nomenclatura: el `data_bundle.tar.gz` borrado arriba era el bundle
> viejo (crops + manifests + parquets). El bundle nuevo que consume el
> entregable se llama `crops_bundle.tar.gz` y lo publica
> `notebooks/01_preprocesamiento_colab.ipynb` (publicado en HF el
> 2026-06-09; **republicado ink-aware el 2026-06-10**; en
> `f3r21/actas-cnn-dataset`).
