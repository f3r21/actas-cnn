# AUDIT REPORT — Verificacion de Semana 1

## Resumen

- **PASS**: 5
- **WARNING**: 0
- **FAIL**: 0
- **MANUAL** (require inspeccion visual): 1

## Detalle por chequeo

### [CHECK 2] 5,000 PDFs descargados (universo manuscritas Presidencial) — **PASS**

- **Claim:** 5,000 PDFs en data/pdfs_train/, todos Presidencial tipo=1 idEleccion=10, alineados con manuscritas_full.txt (3,478 manuscritas + 1,522 extras tras filtrar STAE).
- **Metodo:** ls + cross-check con manuscritas_full.txt + cross-check con actas_archivos.parquet filtrado.
- **Evidencia:** count=5000, zero_size=0, outside_universe=0, missing_from_disk=0

### [CHECK 3] Render 1:1 con dimension consistente — **PASS**

- **Claim:** 5,000 PNGs renderizados 1:1 con los PDFs, todos 2339×3309 px.
- **Metodo:** set diff entre PDF stems y PNG stems + sample 30 random verificar dims.
- **Evidencia:** png_count=5000, pdf_diff=0, png_diff=0, sample_dims={(2339, 3309)}

### [CHECK 4] Sin leak entre splits — **PASS**

- **Claim:** train/val/test particion por archivoId, sin overlap, union == manuscritas_full.
- **Metodo:** Set intersect + verificacion de tamanos + cross-check de 200 crops_train.
- **Evidencia:** sizes={'train': 3500, 'val': 750, 'test': 750}, intersect_tv=0, tt=0, vt=0, union_matches=True, crops_misplaced=0
- **Notas:**
    Sin interseccion entre splits; union cubre el sample original; crops respetan los splits.

### [CHECK 5] Labels coinciden con imagen (30 crops random) — **PASS**

- **Claim:** Cada crop en data/crops_<split>/<label>/ tiene un digito que coincide con el label segun ground truth.
- **Metodo:** 30 crops random (3 por clase) -> parse archivoId/field/pos -> lookup nvotos -> int_to_digits -> compare con label de carpeta.
- **Evidencia:** matches=30/30; visual en /Users/99/2026-1/Tópicos en Inteligencia Artificial/Proyecto II - Actas Electorales CNN/data/visualizaciones/audit_labels_30.png
- **Notas:**
    30/30 crops random tienen label correcto vs ground truth.

### [CHECK 6] Templates en 20 actas random — **MANUAL**

- **Claim:** El template Presidencial calza en cualquier acta random, no solo en las 3 calibradas.
- **Metodo:** Overlay del template sobre 20 PNGs random, inspeccion visual del grid generado.
- **Evidencia:** 20 actas con overlay en /Users/99/2026-1/Tópicos en Inteligencia Artificial/Proyecto II - Actas Electorales CNN/data/visualizaciones/audit_overlays_20.png
- **Notas:**
    Requiere inspeccion visual (no se puede medir overlap sin OCR ground-truth). Si las cajas calzan en >=18/20 -> PASS; si 10-17 -> WARNING; si <10 -> FAIL.

### [CHECK 7] val_acc no es trivial — **PASS**

- **Claim:** El val_acc reportado 75.5% refleja aprendizaje real, no solo clase mayoritaria.
- **Metodo:** Inferencia con checkpoint resnet18_best.pt sobre val set sin augmentation; per-class recall + matriz de confusion.
- **Evidencia:** acc 0.9812; clases_con_recall>0.30: 10/10; matriz en /Users/99/2026-1/Tópicos en Inteligencia Artificial/Proyecto II - Actas Electorales CNN/data/visualizaciones/audit_confusion_matrix.png
- **Notas:**
    acc global 0.9812
      clases con recall > 0.30: 10/10
      por clase:
        0: recall=0.970  (n=810)
        1: recall=0.994  (n=7731)
        2: recall=0.977  (n=4617)
        3: recall=0.975  (n=2620)
        4: recall=0.980  (n=1821)
        5: recall=0.974  (n=1363)
        6: recall=0.974  (n=1173)
        7: recall=0.974  (n=983)
        8: recall=0.973  (n=895)
        9: recall=0.957  (n=863)
