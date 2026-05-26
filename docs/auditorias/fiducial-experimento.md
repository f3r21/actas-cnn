# Experimento: registro por fiducial markers (refinado)

## Hipotesis

El usuario observo un patron consistente de 15 fiducial markers en cada
acta Presidencial:
- 4 corners (TL, TR, BL, BR)
- 3 en margen superior (T1, T2, T3)
- 4 en cada lateral (L1-L4, R1-R4)
- 0 en margen inferior

Si los detectamos y derivamos una transformacion afin contra anchors
canonicos, podemos corregir desalineamiento por acta.

## Resultado: **GO** — integrar como preprocesamiento

### H1 confirmada: deteccion estable

Detector calibrado en `scripts/detect_fiducials.py`:
- threshold < 150
- area 350-800 px
- aspect ratio 0.6-1.5
- regiones por zona: TOP y<60, LEFT x<70, RIGHT x>w-70, BOT y>h-150

Resultado en 3 actas calibradas: **15/15 markers detectados** en cada
una con asignacion correcta de roles.

### H2 confirmada: posicion estable

Variacion entre actas:
- dx maximo: 9-22 px
- dy maximo: 7-13 px
- Shifts CONSISTENTES (toda la imagen offset, no ruido) — exactamente el
  problema que la afin transform corrige.

### H3 parcialmente confirmada: mejora en actas afectadas, no harm en el resto

Validacion sobre las 20 peores actas del audit anterior
(`template-generalizacion.md`):

| | |
|---|---|
| Mejora promedia | +7.3 pp |
| Mejora mediana | 0.0 pp |
| Actas mejoraron | 3/20 (15%) |
| Actas iguales | 17/20 (85%) |
| Actas empeoraron | **0/20** |

Tres actas pasaron de ~50% a **100%** de accuracy por digito:
- `69e22266`: 50.0% → 100.0%
- `69e10cc3`: 51.4% → 100.0%
- `69e0ad37`: 52.4% → 100.0%

Las 17 actas que no cambiaron caen en dos grupos:
1. **Detector fallo** (markers 0-6/15): scans pobres, baja contrast. El
   template queda sin transformar.
2. **15 markers OK pero acc baja**: template ya estaba alineado, la
   transform ≈ identidad. Problema en CNN, no en template.

## Decision: integrar al pipeline

Apply registration as a preprocessing step in `scripts/build_crops.py`.

Plan de integracion:

```python
# en procesar_acta() de build_crops.py
markers = detect_15(png_path)
if len(markers) >= 4:
    template_aligned = transform_template(template, markers, anchors, img_size)
else:
    template_aligned = template  # sin alineacion, template original
fields_crops = crop_fields(png_path, template_aligned)
```

Costo: ~1-2 segundos extra por acta. Beneficio: las actas con
desalineamiento sutil pasan a 100%; el resto no se afecta.

## Lo que queda fuera

- Actas con scans muy pobres donde el detector ve 0-6 markers (< umbral
  minimo de 4): la afin no se aplica, queda como antes. Para esas, la
  solucion es robustez de CNN (augmentation + class weighting) en
  Semana 2.

## Archivos producidos

- `scripts/detect_fiducials.py` — detector de los 15 markers con etiquetado
- `scripts/test_alignment.py` — validacion contra worst-20
- `fiducial_anchors.json` — 15 (x, y) canonicos de la acta referencia
- `data/visualizaciones/` — overlays para inspeccion visual

## Proximos pasos

1. Modificar `scripts/build_crops.py` para aplicar registracion afin
   como preprocesamiento.
2. Regenerar crops del val + test set con alineamiento.
3. Re-evaluar val_acc del DeepCNN baseline (esperable subida marginal
   global, +0.5 a +1.5 pp).
4. Despues integrar a Semana 2 oficialmente.
