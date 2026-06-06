# 03 - Pipeline de datos

De actas en PDF a un dataset de recortes de digitos etiquetados.

> **Nota (reorganizacion):** el codigo se movio al paquete `src/actas_cnn/`. Mapeo
> de los nombres usados abajo: `pdf_to_images.py`→`actas_cnn.render`,
> `extract_crops.py`→`actas_cnn.preprocess.crops`, `dataset.py` y
> `build_dataset.py`→`actas_cnn.data`. La deteccion de digitos quedo enchufable
> en `actas_cnn.preprocess`. Los wrappers CLI siguen en `scripts/`.

## Flujo

1. `pdf_to_images.py`: renderiza cada PDF a imagen de pagina (PyMuPDF, sin
   poppler). DPI 200, primera pagina.
2. `extract_crops.py` + `templates.json`: alinea por plantilla, recorta los
   campos numericos y segmenta cada campo en celdas. Coordenadas en
   fraccion [0,1].
3. `scripts/build_crops.py`: aplica template, deriva labels desde parquets
   curados, filtra celdas vacias (no las que tienen "0" escrito, solo las
   que jamas iban a tener tinta), guarda `data/crops_<split>/<label>/*.png`.
4. `build_dataset.py`: genera `manifest_<split>.csv` (columnas path, label).
5. `dataset.py`: `CropsDataset` lee el manifiesto para entrenar.

## Universo de actas en el bucket

El bucket `gs://onpe-eg2026-pdfs-v2/` (1.58 TB, 871k PDFs) tiene cinco
tipos de documentos. Solo `tipo=1` aporta cifras manuscritas:

| tipo | descripcion                      | filas    | Para entrenar |
|------|----------------------------------|---------:|---------------|
| 1    | ACTA DE ESCRUTINIO               | 346,117  | Si |
| 2    | ACTA DE INSTALACIÓN Y SUFRAGIO   | 226,401  | No |
| 3    | ACTA DE INSTALACIÓN              | 119,725  | No |
| 4    | ACTA DE SUFRAGIO                 | 119,717  | No |
| 5    | RESOLUCIÓN 1                     |      24  | No |

Mas ~59k PDFs (memos administrativos, resoluciones JNE) que **no aparecen**
en `actas_archivos.parquet`. El filtro `tipo == 1` los descarta.

**Universo total tipo=1**: 346,117 actas. **Filtrando por idEleccion=10
(Presidencial)**: 84,449 actas.

## Hallazgo critico: STAE vs Manuscritas

No todas las actas de escrutinio son manuscritas. Lima y Callao usan el
**Sistema de Transmision de Actas Electorales (STAE)** que genera PDFs
digitales con cifras impresas, mientras provincias y exterior siguen el
flujo manual de paper-and-pencil.

| Departamento | Manuscritas (1 pag) | STAE (2 pags) | % STAE |
|--------------|--------------------:|--------------:|-------:|
| 14 (Lima)    | 282                 | 1,363         | 82.9%  |
| 24 (Callao)  | 5                   | 159           | 97.0%  |
| Demas        | ~3,200              | 0             | 0%     |

**Discriminacion empirica**: las STAE tienen siempre 2 paginas, las
manuscritas 1. Las STAE pesan tipicamente < 1.5 MB, las manuscritas > 1.5 MB.

**Decision**: el proyecto se enfoca exclusivamente en lectura de
**manuscritos** (provincias y exterior). Las STAE quedan **fuera del
alcance** porque:
- Sus cifras son texto impreso (problema trivial de OCR de fuente
  computacional, no manuscrito).
- Solo renderizando pagina 1 perderiamos los partidos 23-38 (estan en pag 2).
- Cualquier suma reconstructiva fallaria por datos incompletos.

El sistema entrenado complementa al STAE para regiones que aun dependen
del flujo manual.

## Layouts por idEleccion

Las cinco elecciones simultaneas usan layouts distintos:

| idEleccion | Eleccion                            | Layout                                  | Actas tipo=1 |
|-----------:|-------------------------------------|-----------------------------------------|-------------:|
| 10         | Presidencial                        | Vertical, 38 partidos × TOTAL solo       | 84,449       |
| 12         | Parlamento Andino                   | Vertical, ~25 partidos × (TOTAL + pref 1-15) | 65,227 |
| 13         | Diputados                           | Vertical, ~25 partidos × (TOTAL + pref 1-15) | 65,845 |
| 14         | Senadores DE Multiple               | Vertical, ~20 partidos × TOTAL solo      | 66,279       |
| 15         | Senadores DE Unico (Nacional)       | **Landscape**, ~30 partidos × pref 1-30  | 64,317       |

**Alcance del proyecto: solo idEleccion=10 (Presidencial)**. Layout mas
simple, 84k actas disponibles, ~5,000 muestreadas para esta entrega. Los
otros 4 layouts quedan como trabajo futuro documentado.

## Etiquetas (resueltas)

`data/labels/` contiene los parquets curados de ONPE (descargados de
`gs://onpe-eg2026-pdfs-v2/data/curated/`):

- **`actas_archivos.parquet`** (811k filas): manifiesto `archivoId -> idActa`,
  con `tipo`, `idEleccion`, `ubigeoDistrito`.
- **`actas_cabecera.parquet`** (463k filas): metadatos por acta, incluyendo
  `totalVotosEmitidos`, `totalVotosValidos`, `totalElectoresHabiles`,
  `descripcionEstadoActa`. PK `idActa`.
- **`actas_votos.parquet`** (18.6M filas): la columna **`nvotos` (int64) es
  la cifra manuscrita por organizacion politica**. PK compuesta
  `(idActa, nposicion)`. `nposicion` 1..38 son partidos (`es_especial=False`);
  80, 81, 82 son votos blanco, nulos, impugnados respectivamente.
- **`mesas.parquet`** y **`departamentos.parquet`**: dimensiones geograficas.

El join se hace en `scripts/build_crops.py`:

```
archivoId -> (actas_archivos) idActa -> (actas_votos) nposicion -> nvotos
```

Para `total_ciudadanos`: `actas_cabecera.totalVotosEmitidos`.

## Convencion ONPE para escritura de cifras

Las cifras se escriben **right-justified** en las celdas. Para un campo
de 3 celdas:
- valor 5     -> `[vacio, vacio, "5"]`
- valor 18    -> `[vacio, "1", "8"]`
- valor 144   -> `["1", "4", "4"]`
- valor 0     -> `[vacio, vacio, vacio]` (nadie escribe "000")
- valor 20    -> `[vacio, "2", "0"]` (el "0" trailing si se escribe)

Esta convencion define la funcion `extract_crops.es_celda_escrita(value,
n_cells, cell_position)` que filtra celdas vacias del training set sin
tener que medir tinta en la imagen.

## Templates calibradas

### Presidencial (idEleccion=10, tipo=1)

Resuelto en sesion 2026-05-21. `templates.json` clave `presidencial`.
Referencia: A4 portrait a 200 DPI = 2339×3309 px (algunos a 1654×2339,
mismo aspect ratio).

- **38 organizaciones politicas** en filas numeradas, columna TOTAL DE
  VOTOS con 3 sub-celdas.
- **4 filas especiales** debajo: `votos_blanco`, `votos_nulos`,
  `votos_impugnados` (3 digitos), y `total_ciudadanos` (4 digitos).
- Total: **42 campos**, ~127 digit cells por acta.

Coordenadas claves (fraccion [0,1]):
- Columna 3-digitos partidos: `x0=0.462, x1=0.539` (~0.026 por celda).
- Fila 1 top: `y0=0.2149`.
- Row height: `0.01662` (auto-detectado via proyeccion horizontal con
  OpenCV — picos cada 55 px en el original).
- `total_ciudadanos`: `x0=0.4363`, `y` justo despues de impugnados.

**Calibracion**: scripts/preview_template.py (overlay rojo) +
scripts/preview_crops.py (grilla de digit splits). Validado contra ground
truth en 30/30 crops random del audit.

### Otros layouts (pendientes — fuera del alcance)

- idEleccion=12 Parlamento Andino: muestra renderizada disponible.
- idEleccion=13 Diputados: requiere muestreo y calibracion.
- idEleccion=14 Senadores DE Multiple: muestra renderizada disponible.
- idEleccion=15 Senadores DE Unico: **LANDSCAPE**, requiere rotacion y
  template propio.

## Pendientes de mejora

- `extract_crops.py` actual hace `split_digits` ingenuo (divide en N
  partes iguales). En la mayoria de actas las celdas son uniformes y
  funciona; en escaneos muy mal alineados algunos digitos pueden
  cortarse. Mejorar con proyeccion vertical o contornos cuando el
  baseline este corriendo.
- Detectar y rotar automaticamente actas landscape (raras dentro de
  Presidencial, mas relevantes para idEleccion=15).
- Reportar confianza por digito (margen del top-2 logit) para flaggear
  ilegibles y compararlos vs los oficiales.

## Notas de calidad (ML)

- La muestra de 5,000 cubre 1,000+ ubigeos distintos de Peru y mesas del
  extranjero — buena diversidad regional dentro del problema manuscrito.
- Imbalance natural: label 1 domina con ~34% (la gente vota mayormente
  numeros chicos por mesa); labels 7-9 son escasos.
- Comparar totales reconstruidos vs `actas_cabecera.totalVotosEmitidos`
  por mesa para validar end-to-end del sistema.
