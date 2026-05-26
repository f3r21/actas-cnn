# 00 - Contexto

## Curso y entregables

- Curso: Topicos en Inteligencia Artificial, CCOMP9-1 (pregrado).
- Unidad: Redes Neuronales Convolucionales - Proyecto Deep Learning.
- Entregable de definicion: un PDF con titulo, resumen (indicando el modelo de DL)
  y descripcion de datos. Generado; vence 24/05/2026.
- Presentacion final del proyecto: 18/06/2026.
- Ventana de construccion: ~3.5 semanas desde la definicion.

## El proyecto

Reconocer las cifras manuscritas de conteo de votos en actas electorales de las
Elecciones Generales del Peru 2026, con una CNN en PyTorch. Es clasificacion de
imagenes de digitos (0-9); opcionalmente, reconocimiento de numeros multi-digito
por campo (estilo SVHN).

## Relacion con el Proyecto I

El Proyecto I (prediccion de electos con ML, datos de candidatos de otorongo.club)
ya esta hecho. Este Proyecto II trabaja sobre las actas, que son la fuente cruda
de los votos; por eso son complementarios y dan una buena narrativa para el
informe.

## Datos (confirmados)

- 1.58 TB de actas en PDF escaneado, en
  `gs://onpe-eg2026-pdfs-v2/` (us-central1). 871k PDFs totales.
- 5 tipos de documentos; solo `tipo=1` ACTA DE ESCRUTINIO sirve para
  entrenar (346k actas en total, 84,449 Presidenciales).
- 4-5 layouts segun `idEleccion`. **Alcance del proyecto: solo
  Presidencial (idEleccion=10)**.
- Las actas de Lima/Callao son STAE (digitales, impresas, 2 paginas) y se
  excluyen — el proyecto se enfoca en manuscritos de provincias y
  exterior.
- Son registros publicos de ONPE.
- **Etiquetas resueltas**: existen en los manifiestos curados de ONPE
  (`actas_archivos.parquet`, `actas_cabecera.parquet`,
  `actas_votos.parquet`) en `gs://onpe-eg2026-pdfs-v2/data/curated/`.
  Cruce determinista `archivoId -> nposicion -> nvotos`. Ya descargados
  en `data/labels/`.
