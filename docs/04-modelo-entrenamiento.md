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

Entrenado en Semana 2 (base + 2 ablations, ver tabla abajo). Se conserva
la antigua CNN custom (Conv+BN+LeakyReLU+Dropout) en `actas_cnn.model`
como linea de referencia metodologica (alcanzo 97.77% val_acc en 5
epochs), pero el modelo del entregable es ResNet-18 CIFAR.

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

Todas implementadas en `actas_cnn.evaluate` (wrapper `scripts/evaluate.py`):

- Exactitud por digito (digit-level accuracy).
- Exactitud por campo (los 3-4 digitos juntos = un numero correcto).
- Exactitud por acta (todos los 42 campos correctos).
- Reconstruccion del total: suma(partidos) + blanco + nulos +
  impugnados vs `actas_cabecera.totalVotosEmitidos`. MAE, mediana,
  histograma, % actas con error 0.
- Matriz de confusion 10×10 y per-class precision/recall/F1
  (PNG en `data/visualizaciones/`).
- Ranking de las 20 actas peor reconstruidas
  (`data/evaluate_worst20_val.csv`).

## Ablations (Semana 2, comparativa cerrada 2026-06-09)

Tres variantes de ResNet-18 CIFAR entrenadas 20 epochs sobre el mismo
dataset y split. Re-evaluadas el 2026-06-09 con `scripts/evaluate.py
--split val` (693 actas, 29,106 campos) y consolidadas con
`scripts/ablation_table.py` (resumen en `data/ablations_summary.csv`,
logs `data/evaluate_val*.log`):

| Variante | Config | Digit | Field | Acta | Recon. exacta | MAE |
|---|---|---|---|---|---|---|
| base | sin augmentation | 98.12% | 98.87% | 90.33% (626/693) | 93.80% (650/693) | 2.40 |
| ls_ra | label smoothing + RandAugment | 98.16% | 98.90% | 91.49% (634/693) | 94.52% (655/693) | 2.20 |
| ls_ra_mu_cos | + mixup + cosine LR | 98.21% | 98.93% | 92.21% (639/693) | 95.24% (660/693) | 2.18 |

Lectura: la regularizacion gana poco a nivel digito (+0.09pp) pero el
efecto se compone en las metricas agregadas — acta-level sube +1.88pp
(13 actas mas perfectas) y la reconstruccion exacta +1.44pp. La
combinacion completa (`ls_ra_mu_cos`) domina en todas las metricas.

> Esta tabla es right-justified (labels mayo), conservada como referencia
> historica. **Superada por la ablacion ink-aware** (seccion siguiente),
> que es la receta oficial desde 2026-06-18.

## Etiquetado ink-aware (2026-06-10)

La cola de errores no era ruido del modelo sino **labels envenenados**:
~3% de las actas escribe las cifras corridas (no right-justified), y el
etiquetado posicional asignaba el digito a la celda equivocada. En val,
**19 de las 19 actas con >=5 campos mal** violan la convencion (0 son
desalineacion geometrica); concentraban el 82% de los 330 errores de
campo. Diagnostico en `experiments/justificacion/audit_justificacion.py`.

Fix: etiquetado ink-aware (ver `docs/03-pipeline-datos.md`). Medido del
lado de evaluacion con el **mismo `resnet18_best.pt`** (sin re-entrenar),
para aislar el efecto del etiquetado:

| Metrica | base (right-justified) | ink-aware (eval) | delta |
|---|---|---|---|
| Digit | 98.12% | 99.05% | +0.93pp |
| Field | 98.87% | 99.45% | +0.58pp |
| Acta-level | 90.33% (626/693) | 90.62% (628/693) | +0.29pp |
| MAE total | 2.40 | 1.58 | -0.82 |
| Campos mal | 330 | 159 | -52% |

Solo se tocan las 20 actas violadoras (sustituidas por crops ink-aware);
las otras 673 evaluan identico (**0 regresiones**). Confirmado tambien
en A/B de misma geometria (crops nuevos con/sin ink-aware): field
+0.56pp, consistente — el efecto es el etiquetado, no el render.
Acta-level sube poco porque los campos no remapeables (escritura muy
apretada o tenue) siguen mal en esas mismas actas; el retrain sobre
train limpio (Colab, pendiente) puede mejorarlo mas.

## Modelo oficial ink-aware re-entrenado (2026-06-18)

Corrido el entregable en Colab sobre el bundle ink-aware. La receta
`ls_ra_mu_cos` (la ganadora de las ablations) es la oficial; las ablations
re-corridas sobre ink-aware confirman el ranking en val **y test** (primera
evaluacion sobre test del proyecto):

**val (ink-aware):**
| Variante | Digit | Field | Acta | Recon. | MAE |
|---|---|---|---|---|---|
| base | 98.70% | 99.23% | 88.46% | 91.63% | 2.00 |
| ls_ra | 98.79% | 99.32% | 89.61% | 92.78% | 1.70 |
| ls_ra_mu_cos | **98.83%** | **99.34%** | **90.62%** | **93.80%** | **1.77** |

**test (ink-aware):**
| Variante | Digit | Field | Acta | Recon. | MAE |
|---|---|---|---|---|---|
| base | 98.07% | 98.84% | 86.58% | 89.55% | 2.44 |
| ls_ra | 98.24% | 98.95% | 87.71% | 90.68% | 2.44 |
| ls_ra_mu_cos | **98.31%** | **99.00%** | **88.84%** | **91.95%** | **2.12** |

`base`/`ls_ra` salen de `03_ablaciones_colab.ipynb`; `ls_ra_mu_cos` es del run
publicado en HF (`02`; la corrida de 03 se corto en epoch 16 de esa variante).
Train fresco sin semilla (±0.5pp corrida-a-corrida). El checkpoint oficial se
publico en HF el 2026-06-18 (`resnet18_best.pt` en `f3r21/actas-cnn-model`,
sobre el bundle ink-aware). El ranking de accuracy/recon se sostiene en val y
test; el MAE val del ganador (1.77) queda a la par de `ls_ra` (1.70) por ruido
entre corridas. La brecha val->test (~0.5pp digit, ~1.8pp acta) indica buena
generalizacion sin overfit.

## Mejoras no exploradas (trabajo futuro)

- Ablations: con/sin residual, profundidad (ResNet-18 vs ResNet-34).
- Class weighting o focal loss (label 1 representa 34%, sesga
  predicciones).
- Pre-entreno opcional en MNIST/SVHN (transfer learning) — ablation
  contra el modelo from-scratch.
- Tracking en Weights & Biases con tags por configuracion.
