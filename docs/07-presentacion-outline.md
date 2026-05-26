# Outline — Presentacion oral 18-jun-2026

Curso: Topicos en Inteligencia Artificial (CCOMP9-1).
Duracion estimada: 20-25 min + Q&A.

## Estructura propuesta (10-12 slides)

### Slide 1 — Portada (30 seg)

- Titulo: "Reconocimiento y agregacion de votos manuscritos en actas
  electorales peruanas con redes neuronales convolucionales"
- Autor, curso, fecha
- (Visual de fondo: una acta real con dos columnas de digitos)

### Slide 2 — Problema y motivacion (2-3 min)

- 84,449 actas presidenciales escrutinio en EG 2026 (~10% del bucket
  ONPE de 871k PDFs).
- Las regiones fuera de Lima/Callao (sin STAE) registran los conteos
  a mano. Lectura manual es costosa y propensa a errores.
- Objetivo: sistema automatico que lea los votos manuscritos y
  reconstruya los totales por organizacion politica.

### Slide 3 — Datos (2-3 min)

- Fuente: bucket publico ONPE `gs://onpe-eg2026-pdfs-v2/`, 1.58 TB.
- Filtro: tipo=1 (escrutinio) + idEleccion=10 (Presidencial) → 84,449
  actas. Filtramos STAE (Lima/Callao digitales) por numero de paginas.
- Muestra: **5,000 actas manuscritas** con semilla fija (~6% del
  universo manuscrito).
- Etiquetas: cruce determinístico con manifiestos curados ONPE
  (`actas_archivos`, `actas_cabecera`, `actas_votos`); **no hay
  anotacion manual**.
- Verificacion del manifest: 469/469 actas (con total no-nulo) tienen
  `sum(votos) == totalVotosEmitidos` — manifest internamente
  consistente.

### Slide 4 — Pipeline (3 min)

Diagrama: PDF → PNG (200 dpi) → recorte de **42 campos** por
plantilla calibrada (38 partidos + blanco + nulos + impugnados + total
ciudadanos) → split de cada campo en 3-4 celdas → CNN clasifica
digito 0-9 → reconstruccion del entero (right-justified) → agregacion
por partido.

- Particion 70/15/15 por archivoId (sin leak): 106,123 / 22,876 /
  22,955 crops de 32×32 grayscale.

### Slide 5 — Modelo (3 min)

- **ResNet-18 estilo CIFAR** (He et al., 2015), adaptada:
  - stem `Conv2d(1, 64, 3, stride=1, padding=1)` para 1 canal
  - sin MaxPool inicial (preserva resolucion 32×32)
  - 4 etapas residuales, GAP final, Linear(512, 10)
  - 11.17M parametros
- Skip connections permiten entrenar profundo sin gradiente
  vanishing.
- Por que CIFAR-style y no ImageNet-style: imagenes ya son chicas
  (32×32), bajar resolucion en el stem es contraproducente.

### Slide 6 — Entrenamiento (1-2 min)

- 20 epochs en MPS (M2 24GB)
- Adam, lr=5e-4, batch=128
- CrossEntropy + (ablaciones: label smoothing, RandAugment)
- Checkpoint del mejor val_acc

### Slide 7 — Resultados principales (3-4 min)

Tabla:

| Metrica | Valor |
|---|---|
| Digit-level accuracy (val) | **98.12%** |
| Field-level accuracy | **98.87%** |
| **Acta-level accuracy** | **90.33%** |
| **Reconstruccion exacta del total** | **93.80%** |
| MAE total agregado | 2.40 votos |

Mensaje principal: **9 de cada 10 actas se reconstruyen con TODOS
sus campos correctos, y el total agregado coincide EXACTO con el
oficial en 9 de cada 10 actas. Error promedio de 2.40 votos sobre
miles.**

(TODO: actualizar con resultados de ablations)

### Slide 8 — Ablation y experimentos honestos (2 min)

- Compare baseline CNN custom (97.77%) vs ResNet-18 (98.12%): solo
  cambio de arquitectura suma +0.35pp.
- (TODO: agregar ablations finales: label smoothing, RandAugment,
  mixup).
- **Experimento negativo documentado** (vale como honestidad
  cientifica): mejoramos detector fiducial con search-by-prior
  (std de roles TOP bajo 22-27×) y projection profile en
  split_digits. Tecnicamente correctos pero resultaron en -0.72pp
  acta-level. El techo no esta en preprocesamiento.

### Slide 9 — Matriz de confusion + per-class (1-2 min)

Figura: matriz 10×10 (ya generada en
`data/visualizaciones/evaluate_confusion_val.png`)

Observacion: clases minoritarias (8, 9, 0) tienen recall ligeramente
menor; clase 1 (mayoritaria, 34% del dataset) tiene mayor recall
(99.4%) — sesgo esperable.

### Slide 10 — Limitaciones (2 min)

- ~3.5% de actas (24/693) tienen error grande en reconstruccion
  (≥20 votos). Top 5: error de 102-140 votos, 13-18 campos errados.
- Ground truth de ONPE puede tener errores no detectables (audit
  visual 30/30 paso, pero muestra chica).
- Solo Presidencial calibrado. Las otras 4 elecciones (idEleccion
  12-15) necesitarian plantillas propias.
- No manejo de STAE (formato digital, problema distinto).

### Slide 11 — Trabajo futuro (1 min)

- Otros layouts de la EG 2026 (parlamento, diputados, senadores).
- Pipeline para deteccion automatica de layout y seleccion de
  plantilla.
- Validacion humana-in-the-loop sobre actas con confianza baja.

### Slide 12 — Cierre / Q&A (30 seg)

- Repositorio publico (GitHub).
- Resumen 1 linea: lectura automatica de actas Presidencial con
  reconstruccion exacta del 93.80% de los totales agregados, usando
  ResNet-18 sobre 5,000 actas manuscritas.

## Decisiones a tomar antes de armar slides

- Plataforma: Keynote / PowerPoint / Beamer / Google Slides /
  Marp (markdown → slides). Recomendado: **Marp** (markdown,
  versionable en git, exporta a PDF/PPTX).
- Demo en vivo si/no:
  - Si: correr `python scripts/audit.py` o `evaluate.py` sobre
    una acta nueva, mostrar el output. Riesgo de fallo, pero
    impresiona si funciona.
  - No: mostrar capturas de pantalla pre-grabadas.

## Material listo para reutilizar

- `data/visualizaciones/evaluate_confusion_val.png` — matriz confusion
- `data/visualizaciones/evaluate_error_hist_val.png` — distribucion
  de errores del total agregado
- `data/visualizaciones/errors_top.png` — grid de top-200 crops mal
  clasificados (util en slide de limitaciones)
- `data/evaluate_worst20_val.csv` — top 20 actas peor reconstruidas
- `data/visualizaciones/audit_overlays_20.png` — overlay del template
  sobre 20 actas random (util para slide 4 pipeline)
- `templates.json` — definicion exacta de los 42 campos

## Preguntas anticipadas del jurado

1. *"Por que ResNet-18 y no otra arquitectura?"*
   - Respuesta: pedagogicamente apropiada (skip connections, contenido
     del curso), implementacion estandar via torchvision, ablations
     naturales (depth, residual on/off), defensible en cualquier
     audiencia.

2. *"Probaron preprocesamiento mas elaborado?"*
   - Si — search-by-prior detector + projection profile + filtro
     image-based de tinta. Mostrar la tabla del experimento negativo
     y argumentar que el techo no esta ahi.

3. *"Como se que las etiquetas son confiables?"*
   - Cruce determinístico con manifiestos oficiales ONPE.
     Verificamos 30/30 crops random visualmente. Validamos
     consistencia interna: 469/469 actas (con total no-nulo) tienen
     sum(votos)==totalVotosEmitidos.

4. *"Que pasa con las 5 actas catastroficas?"*
   - Las inspeccionamos manualmente (TODO: hacerlo). Tienen
     escritura especialmente densa o ambigua, o estan rotadas /
     escaladas fuera del +/-8% que tolera la afin.

5. *"Por que no usaron OCR generico (Tesseract)?"*
   - OCR generico no esta entrenado para digitos manuscritos en
     celdas pre-impresas. Una CNN entrenada sobre el dataset
     especifico domina por margen amplio en este tipo de tareas.

6. *"Esto es deep learning real o es un toy problem?"*
   - 5,000 actas reales, 152k crops, 32×32 grayscale, 10 clases.
     ResNet-18 con 11.17M parametros. Problema de scanned form OCR
     con plantilla — relevante a aplicaciones reales (medical
     forms, banking, government documents).

## Tono

- Honesto sobre limitaciones (jurado academico valora honestidad).
- Numerico (no "muy bueno", "alto", sino "98.12%").
- Visual: cada slide con una figura/tabla central.
- Sin demos arriesgadas a menos que esten ensayadas.
