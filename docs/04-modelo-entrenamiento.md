# 04 - Modelo y entrenamiento

> **Nota (reorganizacion):** el codigo se movio al paquete `src/actas_cnn/`:
> `model.py`→`actas_cnn.model` (`resnet18_cifar`), `train.py`→`actas_cnn.training`,
> `scripts/evaluate.py`→`actas_cnn.evaluate` (con wrapper `scripts/evaluate.py`).

## Arquitectura del proyecto

**`ResNet18CIFAR`** (He et al., 2015): variante de ResNet-18 adaptada
a entrada 32×32 px en escala de grises. Aporte arquitectural central:
las **skip connections** dentro de cada bloque residual permiten
entrenar redes profundas sin que el gradiente se desvanezca, problema
que limitaba las CNN apiladas convencionales. Adaptaciones respecto al
ResNet de ImageNet:

- Stem: `Conv2d(1, 64, kernel_size=3, stride=1, padding=1)` (no
  `7x7 stride 2` ni MaxPool inicial; preserva resolucion).
- Cuatro etapas con 2 bloques cada una (canales 64 → 128 → 256 → 512).
- Global Average Pool al final, luego Linear a 10 clases.

Modelo a entrenar en Semana 2. Se conserva la antigua CNN custom
(Conv+BN+LeakyReLU+Dropout) en `actas_cnn.model` como linea de referencia
metodologica (alcanzo 97.77% val_acc en 5 epochs), pero el modelo del
entregable y del informe es ResNet-18 CIFAR.

## Linea de referencia (Semana 1)

La CNN custom entrenada en Semana 1 establece el piso a superar.
Convergio a **97.77% val_acc** en 5 epochs sobre 106k crops de
entrenamiento, medido por `scripts/audit.py` CHECK 7 sobre el manifest
de validacion completo. Detalle reproducible en `AUDIT_REPORT.md`.

## Entrenamiento (actas_cnn.training)

- Detecta device: CUDA (Kaggle/Colab), MPS (M2) o CPU.
- Split train/val 80/20 random sobre el manifest pasado (en Semana 2 se
  cambia a usar manifest_val.csv explicito sin random_split).
- Adam, CrossEntropy, lr=5e-4, batch_size=128, epochs=20.
- Guarda el mejor checkpoint y, con `--push`, lo sube a backends
  (`storage.upload`, kind="model") para reanudar desde otra GPU.

## Notas sobre MPS

Bug pytorch#96056: `AdaptiveAvgPool2d` con dimensiones no divisibles
en MPS lanza error. Las arquitecturas del proyecto evitan
`AdaptiveAvgPool` con tamanos no compatibles; ResNet-18 CIFAR usa
`AdaptiveAvgPool2d((1, 1))` (Global Average Pool), donde el divisor 1
nunca da problema.

## Portabilidad (redundancia de GPU)

- `notebooks/02_modelo_colab.ipynb` (modelo + evaluacion) corre en Colab:
  es autonomo (codigo del modelo inline), baja los crops preprocesados del HF
  dataset publico (sin token) y entrena/evalua. El preprocesamiento de las 5,000
  actas vive aparte en `notebooks/01_preprocesamiento_colab.ipynb`, que publica
  el bundle de crops que este consume.
- Tokens via secretos de Kaggle/Colab; nunca hardcodear.

## Metricas y evaluacion

**Implementadas en `actas_cnn.training`**:
- Exactitud por digito (digit-level accuracy).

**Pendientes para `scripts/evaluate.py`** (Semana 2-3):
- Exactitud por campo (3 digitos juntos = un numero correcto).
- Exactitud por acta (todos los 42 campos correctos).
- Reconstruccion del total: suma(partidos) + blanco + nulos +
  impugnados vs `actas_cabecera.totalVotosEmitidos`. |Error| medio,
  histograma, % actas con error 0.
- Matriz de confusion 10×10.
- Per-class precision/recall/F1.

## Pendientes de mejora (Semana 2-3)

- Implementar `ResNet18CIFAR` en `model.py` y exponerlo via
  `build_model("resnet18", ...)`.
- Entrenar 20 epochs con augmentation moderada (RandAugment) y label
  smoothing.
- Ablations: con/sin residual (para mostrar el aporte de skip
  connections), profundidad (ResNet-18 vs ResNet-34), augmentation
  policy.
- Class weighting o focal loss (label 1 representa 34%, sesga
  predicciones).
- Early stopping y scheduler de LR (cosine annealing).
- Pre-entreno opcional en MNIST/SVHN (transfer learning) — ablation
  contra el modelo from-scratch.
- Tracking en Weights & Biases con tags por configuracion.
- Empaquetar los recortes (parquet/webdataset/tar) para carga rapida
  en la nube si fuera necesario escalar.
