# 05 - Backlog priorizado

Orden de trabajo. Estado actualizado al 2026-05-22 (Semana 1 cerrada).

> **Nota (reorganizacion):** el codigo se movio al paquete `src/actas_cnn/`
> (`model.py`→`actas_cnn.model`, `train.py`→`actas_cnn.training`,
> `scripts/evaluate.py`→`actas_cnn.evaluate`). El entregable es
> `notebooks/02_entregable_colab.ipynb`. Los items historicos abajo mantienen
> los nombres de la epoca.

## P0 - Dataset (COMPLETADO)

- [x] Conseguir actas reales del bucket ONPE.
- [x] Calibrar `templates.json` con coordenadas reales para Presidencial
      (38 partidos + 3 especiales + total = 42 campos).
- [x] Resolver labels: cruce con parquets curados de ONPE
      (`actas_archivos`, `actas_cabecera`, `actas_votos`).
- [x] Filtrar STAE (Lima/Callao digital): solo manuscritas.
- [x] Generar dataset: 5,000 manuscritas → 106k/22k/22k crops train/val/test.
- [x] Pipeline reproducible (`scripts/run_week1_clean_pipeline.sh`).

## P1 - ResNet-18 CIFAR y ablations (Semana 2)

- [x] Implementar `ResNet18CIFAR` en `model.py` (stem `3x3 stride 1`,
      sin MaxPool inicial, 4 etapas con 2 bloques cada una, GAP final).
      Registrarlo en `build_model("resnet18", ...)` y aceptar `--arch
      resnet18` en `train.py`.
- [x] Entrenar ResNet-18 20 epochs sobre dataset limpio. **Resultado:
      val_acc 98.12% (digit-level), 90.33% acta-level, 93.80%
      reconstruccion exacta del total.**
- [x] Error analysis sobre val: top-200 errores categorizados. Patron:
      57% predicciones a clase 1, 66% en posicion c2. Hipotesis:
      crops casi-vacios + segmentacion equiespaciada.
- [x] **Experimento Sem 2 dia 2 (negativo)**: mejorar preprocesamiento.
      Detector fiducial search-by-prior (std roles TOP bajo 22-27x),
      projection profile en split_digits, filtro image-based via
      `tiene_tinta`. Resultado: pipeline tecnicamente correcto pero
      acta-level bajo -0.72pp vs zonal viejo. Rollback completo. El
      techo no esta en preprocesamiento. Util para Cap 4 del informe
      como ablation honesta.
- [ ] Ablations de regularizacion sobre ResNet-18: RandAugment, label
      smoothing, mixup, cosine LR schedule. Esperar +0.3-0.5pp digit.
- [ ] Ablation: con/sin residual (para evidenciar el aporte de skip
      connections).
- [ ] Ablation: profundidad ResNet-18 vs ResNet-34.
- [ ] Ablation: augmentation policy (sin, RandomAffine, RandAugment).
- [ ] Ablation: con/sin label smoothing.
- [ ] Pre-entreno en MNIST opcional: pretrained vs from-scratch.
- [ ] Class weighting o focal loss para reducir bias hacia label 1.
- [ ] Tracking en Weights & Biases por configuracion.

## P2 - Metricas downstream (Semana 2-3)

- [x] `scripts/evaluate.py` v1:
  - [x] Field-level accuracy (3-4 digitos juntos por campo).
  - [x] Acta-level accuracy (todos los 42 campos correctos).
  - [x] Reconstruccion total: suma vs `totalVotosEmitidos`.
- [ ] Extender evaluate.py: matriz de confusion 10x10, per-class
      precision/recall/F1, histograma de errores por acta, ranking
      de actas peor reconstruidas.

## P3 - Informe y presentacion (Semana 3-4, 18/06/2026)

- [ ] Redactar Cap 1-3 del informe (intro + trabajos relacionados +
      metodologia). Usar skill `paper-structure-cs` y
      `abstract-methods-results-cs`.
- [ ] Cap 4 (Resultados): tabla principal de ablations sobre ResNet-18
      (residual on/off, depth, augmentation, label smoothing), matriz
      de confusion, curvas train/val por epoch.
- [ ] Cap 5 (Discusion): limitaciones, comparacion contra SOTA en MNIST,
      trabajo futuro (otros layouts, STAE multi-page, segmentacion
      mejorada).
- [ ] Figuras del pipeline (mermaid o tikz).
- [ ] Slides (20 min) con `paper-structure-cs` adaptado a presentacion.
- [ ] Auditar informe con skill `academic-final-review-cs` y
      `defending-empirical-methods-cs`.

## P4 - Reproducibilidad y publicacion (Semana 4)

- [ ] README de "como reproducir": setup, comandos, link a dataset HF
      y checkpoints.
- [ ] Subir dataset publico a Hugging Face (manuscritas + manifest).
- [ ] Subir mejor checkpoint a HF + W&B.
- [ ] Verificar que `notebooks/02_entregable_colab.ipynb` corre end-to-end
      en Colab sin cambios (publicar `crops_bundle.tar.gz` + checkpoint en HF).

## Config a rellenar antes de Semana 4

- `config.py`: `hf_dataset_repo`, `hf_model_repo`, `wandb_project`,
  `wandb_entity`, `r2_bucket`, `r2_endpoint`. Tokens en `.env` (HF, W&B,
  R2).

## Fuera de alcance (trabajo futuro documentable)

- Layouts de las otras 4 elecciones simultaneas (Parlamento Andino,
  Diputados, Senadores DE Multiple, Senadores DE Unico landscape).
- Manejo de actas STAE multi-pagina (problema distinto: OCR de impreso).
- Migracion del bucket de 1.58 TB a Hugging Face / Internet Archive
  (irrelevante para el curso).
- Reconocimiento end-to-end con object detection (overkill cuando hay
  plantilla fija).
