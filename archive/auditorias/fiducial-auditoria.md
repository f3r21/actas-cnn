# Auditoria del metodo de coordinacion por fiduciales

Fecha: 2026-05-23
Alcance: revisar el metodo que usa el pipeline para localizar los 42
campos de cada acta Presidencial. Hipotesis a verificar: el sistema se
apoya en los "cuadraditos grises" del borde del acta. Sin cambios de
codigo, solo evidencia y juicio.

Convencion: `archivo:linea` referencia codigo verificable en el repo.

---

## 1. Resumen ejecutivo

**Veredicto:** el metodo funciona dentro de su regimen nominal (~93.6%
de detecciones perfectas 15/15 sobre val_750), pero **(a) su impacto
end-to-end no esta medido en val completo**, **(b) la ancla canonica
proviene de una sola acta**, y **(c) existe un modo de fallo no
documentado en ~1.9% del val donde la afin se aplica con puntos casi
colineales y la accuracy CNN cae al 43%**. Es defendible mantenerlo
para Semana 2-4 atendiendo R1 y R7, pero la sustentacion mejora si
ademas se atiende R-NEW.

Hallazgos:

| # | Tipo | Resumen |
|---|------|---------|
| F1 | Fortaleza | Detector zonal CLAHE + connected components: 99.73% de actas con >=4 markers |
| F2 | Fortaleza | Fallback de tres capas evita corrupciones graves (0 actas peores en el experimento worst-20) |
| F3 | Fortaleza | Afin 6 DOF es la complejidad correcta (no homografia) |
| F4 | Fortaleza | Render canonico 2339x3309 + auto-rotacion landscape elimina ambiguedad |
| R-NEW | **Riesgo critico** | Bucket [4-7] markers (n=14) tiene accuracy media 0.43: la afin se aplica con puntos casi colineales |
| R1 | Riesgo | Impacto end-to-end del alignment no esta medido en val completo |
| R2 | Riesgo | `fiducial_anchors.json` proviene de una unica acta de referencia |
| R4 | Riesgo | Fallback global por una sola caja degenerada (rechaza 41 cajas buenas por 1 mala) |
| R5 | Riesgo | Umbrales hardcodeados acoplados a resolucion 2339x3309 |
| R6 | Riesgo | Sin telemetria runtime: nadie sabe que actas se alinearon en la corrida de Sem 1 |
| R7 | Stale-doc | `fiducial-experimento.md` describe parametros que ya no estan en el codigo |
| R8 | Riesgo | Suposicion no validada de patron de 15 fiduciales identico en todas las Presidenciales |

---

## 2. Flujo verificado del metodo

```
PDF -> [pdf_to_images.py:21,49-50]
   render 2339x3309, auto-rotacion landscape->portrait
PNG canonico
   |
   v
[build_crops.py:173-176]
   carga fiducial_anchors.json si existe
   |
   v
[build_crops.py:103-110] procesar_acta()
   detect_15(png_path)                       # detect_fiducials.py:72-117
      |
      v
   if len(markers) >= 4:
      transform_template(template, markers,  # detect_fiducials.py:145-187
                         anchors, img.size)
         |
         v
      estimateAffine2D(src, dst, RANSAC)     # :157
      _affine_is_sane(M)  y  _affine_is_sane(M_inv)  # :159,163
      M_inv = invertAffineTransform(M)
      reproyecta 42 cajas: fraccion -> px canonical -> M_inv -> fraccion en acta nueva
      if cualquier caja < 10px: return template original  # :181
   else: aligned_template = template
   |
   v
[extract_crops.py:23-36] crop_fields()
   recorta los 42 campos de la imagen
   |
   v
[extract_crops.py:39-48] split_digits()
   divide cada campo en N celdas digit
   |
   v
crops/<label>/<archivoId>_<field>_c<pos>.png
```

---

## 3. Fortalezas (con evidencia)

### F1. Deteccion zonal estable

Datos reales de `data/audit_fiducial_val.csv` (750 actas val):

```
n_markers  actas
2              2
5              3
7             11
10             1
11            30
13             1
15           702
```

- 702/750 (93.60%) con deteccion perfecta 15/15.
- 748/750 (99.73%) con >=4 markers (umbral de aplicacion de afin).

Mecanismo: CLAHE local (`detect_fiducials.py:47-50`) + threshold
estatico 150 + `connectedComponentsWithStats` filtrado por area
(200-1200 px) y aspect ratio (0.6-1.5) (`detect_fiducials.py:38-69`).
Cuatro zonas hardcodeadas evitan el costo de buscar en toda la imagen
(`:88-101`).

### F2. Fallback de tres capas

1. `build_crops.py:107`: si `len(markers) < 4` no se aplica afin.
2. `detect_fiducials.py:159,163`: `_affine_is_sane(M)` y
   `_affine_is_sane(M_inv)` rechazan transformaciones con escala fuera
   de 0.92-1.08, angulo > 5 grados o traslacion > 100 px.
3. `detect_fiducials.py:181`: si cualquier caja warpea a < 10 px de
   ancho/alto, retorna el template original entero.

Cualquier fallo silencioso preserva el comportamiento sin alineamiento,
que era el del baseline previo. El experimento worst-20 confirma 0/20
actas empeoraron por la integracion (`fiducial-experimento.md:47`).
Esa afirmacion vale solo para esas 20 actas; ver R-NEW.

### F3. Afin (6 DOF), no homografia (8 DOF)

Decision correcta: las actas escaneadas pueden tener traslacion,
rotacion plana y leve escala isotropica/anisotropica, pero no
perspectiva extrema (la camara/scanner es ortogonal). Una homografia
seria sobreparametrizada y mas inestable con pocos puntos. Verificado
en `detect_fiducials.py:157`.

### F4. Lienzo canonico forzado

`pdf_to_images.py:21` fija `TARGET_W, TARGET_H = 2339, 3309`.
`pdf_to_images.py:49-50` rota a portrait si la pagina es landscape.
Resultado: todos los PNGs entran al detector con el mismo sistema de
coordenadas, lo que justifica que los umbrales y zonas sean
literales en pixeles.

---

## 4. Riesgos (con evidencia)

### R-NEW. Modo de fallo no documentado: alineacion con puntos colineales

**Hallazgo nuevo de esta auditoria** (no aparece en
`fiducial-experimento.md`).

De las 14 actas con `n_markers in [4,7]` (1.87% del val), **todas
pierden los 8 marcadores laterales (L1-L4 y R1-R4)**:

```
69e0aafc n=7 acc=0.226 missing=L1|L2|L3|L4|R1|R2|R3|R4
69dff5c4 n=7 acc=0.645 missing=L1|L2|L3|L4|R1|R2|R3|R4
69e2c744 n=7 acc=0.310 missing=L1|L2|L3|L4|R1|R2|R3|R4
69e22147 n=5 acc=0.258 missing=L1|L2|L3|L4|R1|R2|R3|R4|BL|BR
69dd1f67 n=7 acc=0.647 missing=L1|L2|L3|L4|R1|R2|R3|R4
... (resto del bucket, mismo patron)
```

Datos provienen de `data/audit_fiducial_val.csv`.

Consecuencia geometrica: cuando solo quedan los 5 markers del TOP y a
veces BL/BR, los puntos quedan **casi colineales en Y** (todos los TOP
estan en y~45, BL/BR estan en y~3270). RANSAC encuentra una afin
geometricamente valida pero **el eje Y queda mal condicionado**, y la
afin "estira" mal el template vertical.

Resultado empirico:

| Bucket markers | n actas | mean cnn_acc | median |
|----------------|---------|--------------|--------|
| [0-3]          | 2       | 0.808        | 0.808  |
| **[4-7]**      | **14**  | **0.434**    | **0.364** |
| [8-11]         | 31      | 0.943        | 1.000  |
| [12-14]        | 1       | 1.000        | 1.000  |
| [15-15]        | 702     | 0.979        | 1.000  |

El bucket que **aplica afin** ([4-7]) tiene **peor accuracy que el que
no la aplica** ([0-3]). `_affine_is_sane` no detecta el problema
porque mide escala/rotacion/traslacion globales, no condicionamiento
de la nube de puntos.

Mitigacion documental sugerida: subir el umbral de `len(markers) >= 4`
a algo como `>= 8 con cobertura en >=3 zonas`, o bien verificar que
`np.std(src_y)` no sea < threshold antes de aceptar la afin.

### R1. Impacto end-to-end no medido

El experimento original (`fiducial-experimento.md:38-47`) midio
before/after en **worst-20 actas** del audit anterior: 3 mejoraron,
17 igual, 0 peor.

El audit posterior (`scripts/audit_fiducial_detector.py`) corre sobre
val_750 pero **solo correlaciona `marker_count` con `cnn_acc`; no
compara con-afin vs sin-afin sobre el mismo CNN**.

Hoy: nadie sabe si la alineacion contribuye 0.0 pp, +0.5 pp, o
+1.5 pp al val_acc 95.53% reportado en `AUDIT_REPORT.md`. Para la
sustentacion del informe, el numero importa.

### R2. Anchor canonico proviene de una sola acta

`fiducial_anchors.json` tiene 15 pares (x,y) generados via
`scripts/detect_fiducials.py --save-anchors` (`detect_fiducials.py:226-228`).
Ese comando guarda **el output de un solo `detect_15` sobre una sola
imagen**. No hay agregacion ni mediana sobre N actas.

Implicacion: si la acta de referencia tenia, por ejemplo, T1 con un
ligero offset interno, todas las afines de las 5000 actas se calculan
contra ese offset. RANSAC absorbe ruido pero no offset sistematico
de la referencia.

Mejora barata: regenerar el JSON como mediana de `detect_15` sobre
>=20 actas con 15/15.

### R4. Fallback global por una sola caja degenerada

`detect_fiducials.py:181-182`:

```python
if nx1 - nx0 < 10 or ny1 - ny0 < 10:
    return template  # fallback completo
```

Si **una sola** de las 42 cajas warpea a < 10 px de ancho/alto, **el
template entero vuelve al original**. Demasiado conservador: para una
afin sana puede ocurrir que el `votos_blanco` (caja pequena) caiga al
borde y se degenere, descartando la alineacion correcta para los 41
campos restantes.

Mejor: dejar la caja problematica con el valor original, no abortar el
template.

### R5. Umbrales hardcodeados acoplados a la resolucion

`detect_fiducials.py:38-42`:

```python
THRESHOLD = 150       # gris absoluto
AREA_MIN = 200        # px**2
AREA_MAX = 1200       # px**2
ASPECT_MIN = 0.6
ASPECT_MAX = 1.5
```

`detect_fiducials.py:88-101`:

```python
img[0:100, :]          # TOP zone (px absolutos)
img[200:h-100, 0:70]   # LEFT zone
img[200:h-100, w-70:w] # RIGHT zone
img[h-150:h, :]        # BOTTOM zone
```

Calibrados sobre 2339x3309. Si se cambia el DPI de
`pdf_to_images.py:21`, **todo rompe silenciosamente** sin warning ni
fallback (las zonas se quedan vacias y `detect_15` regresa < 4
markers).

Mitigacion: expresar zonas y areas como fraccion de las dimensiones de
la imagen.

### R6. Sin telemetria runtime

`procesar_acta` (`build_crops.py:75-129`) **no registra** si la afin
se aplico o no. El unico registro empirico es
`audit_fiducial_val.csv`, generado por un script separado
(`audit_fiducial_detector.py`).

En la corrida real que produjo `data/crops_train/` no quedo traza
per-acta de aligned vs skipped. Imposible saber post-mortem cuantos
de los 106k crops train usaron template alineado.

Mitigacion: anadir un contador y un CSV simple en
`build_crops.py:main()`.

### R7. Documentacion obsoleta

`fiducial-experimento.md` declara parametros que ya no estan en el
codigo:

| Afirmacion (fiducial-experimento.md) | Codigo real (detect_fiducials.py) |
|-------------------------------------|------------------------------------|
| `area 350-800 px` (linea 21)        | `AREA_MIN=200, AREA_MAX=1200` (linea 39-40) |
| `TOP y<60` (linea 23)               | `img[0:100, :]` (linea 88) |
| `anchors canonicos` (linea 90)      | snapshot de una unica acta (linea 226-228) |
| `bootstrap iterativo` (docstring `detect_15`) | no implementado, `anchors` arg ignorado (linea 75-77) |
| `+7.3 pp mejora promedia` (linea 43) | solo en worst-20, no representativo del val |

El doc esta en la raiz del repo y se lee como fuente de verdad por
defecto. Para un lector externo (jurado del informe) la inconsistencia
es perceptible.

### R8. Suposicion no validada del patron unico de 15 fiduciales

El proyecto asume una sola plantilla Presidencial. Si ONPE imprimio
variantes (tamano de hoja, posicion del bloque), no hay deteccion. El
audit existente sugiere consistencia (702/750 actas con 15/15 perfecto)
pero **roles laterales se pierden en patron sospechoso** (R1-R4
fallan 41 veces, L1-L4 fallan 21 veces, casi siempre los 4 a la vez).

Eso puede significar:
- (a) un porcentaje de actas tienen un margen lateral diferente (variante de plantilla), o
- (b) el scanner recorta el margen lateral en algunas actas, o
- (c) la zona `x<70` es muy estrecha para algunas actas mal centradas.

Hoy no hay forma de distinguir. Vale generar overlays para las 30
actas con `n_markers < 12` (10 ya se generan en
`data/visualizaciones/audit_fiducial/` segun
`audit_fiducial_detector.py:189-197`).

---

## 5. Discrepancias entre afirmaciones y codigo

| Donde se afirma | Que afirma | Codigo real | Linea |
|-----------------|------------|-------------|-------|
| `fiducial-experimento.md:21` | "area 350-800 px" | `AREA_MIN=200; AREA_MAX=1200` | `detect_fiducials.py:39-40` |
| `fiducial-experimento.md:23` | "TOP y<60" | `img[0:100, :]` | `detect_fiducials.py:88` |
| `fiducial-experimento.md:43` | "+7.3 pp mejora promedia" | medida sobre 20 actas, mediana del bucket es 0.0 pp; no se ha medido sobre val_750 | `fiducial-experimento.md:38-47` |
| `detect_fiducials.py:72-77` docstring | acepta `anchors` para bootstrap iterativo | "El parametro `anchors` se acepta por retro-compatibilidad pero no se usa" | mismo docstring |
| `fiducial-experimento.md:60` | "GO: integrar" | integrado en `build_crops.py:103-110` pero sin telemetria | OK |

---

## 6. Validacion empirica reproducible

Tres comandos para reproducir los numeros clave sin pedir nada:

```bash
# (a) Distribucion de n_markers y per-bucket accuracy sobre val_750
python3 -c "
import pandas as pd
df = pd.read_csv('data/audit_fiducial_val.csv')
print(df['n_markers'].value_counts().sort_index())
for lo,hi in [(0,3),(4,7),(8,11),(12,14),(15,15)]:
    sub = df[(df['n_markers']>=lo)&(df['n_markers']<=hi)]
    accs = pd.to_numeric(sub['cnn_acc'], errors='coerce').dropna()
    if len(accs):
        print(f'[{lo}-{hi}]: n={len(sub)} mean_acc={accs.mean():.4f}')
"

# (b) Confirmar que el anchor JSON viene de una sola corrida CLI
grep -n "save-anchors\|save_anchors" scripts/detect_fiducials.py
# espera: linea 211 (--save-anchors flag) y linea 226-228 (json.dump del output de un solo detect_15)

# (c) Diff de parametros codigo vs documentacion vieja
diff <(grep -E "^(THRESHOLD|AREA_|ASPECT_)|img\[" scripts/detect_fiducials.py) \
     <(grep -iE "threshold|area|y<|x<|y>" fiducial-experimento.md)
```

---

## 7. Recomendaciones (ordenadas por ROI antes del cierre Semana 2)

| # | Accion | Costo | Impacto | Resuelve |
|---|--------|-------|---------|----------|
| 1 | **Audit before/after en val completo**: un script que corra el pipeline dos veces (con/sin anchors) y compare val_acc CNN end-to-end. | 1 dia | Alto: da el numero defendible para el informe. | R1 |
| 2 | **Endurecer condicion de aplicacion** de la afin: requerir `>=8 markers con cobertura en >=3 zonas`, o medir `std(src_y)` antes de aceptar. | 5 LOC + medicion | Alto: salva 14 actas (1.9%) de potencial degradacion. | R-NEW |
| 3 | Sincronizar `fiducial-experimento.md` con codigo (o sustituirlo por este doc). | 15 min | Alto en credibilidad de la sustentacion. | R7 |
| 4 | Regenerar `fiducial_anchors.json` como mediana de >=20 actas con 15/15 detectado. | ½ dia | Marginal pero defendible. | R2 |
| 5 | Telemetria per-acta en `build_crops.py`: contadores aligned/skipped + CSV. | 10 LOC | Habilita post-mortem. | R6 |
| 6 | Generar overlays de las 30 actas con `n_markers < 12` para resolver R8. | 1h (script ya existe) | Cierra la duda de variante de plantilla. | R8 |
| 7 | Fallback por caja (R4) y parametros relativos (R5). | 1h cada uno | Cosmetico. | R4, R5 |

---

## 8. Veredicto

**Mantener el metodo actual** para Semana 2-4 atendiendo recomendaciones
1, 2 y 3 antes del informe. R-NEW es el unico hallazgo que puede
filtrarse en defensa oral si no se aborda; las otras 14 actas
explican parte de la cola de errores del CNN y vale la pena saberlo
publicamente. Reemplazar el metodo no se justifica: F1-F4 muestran
que el diseno es solido en su regimen nominal.
