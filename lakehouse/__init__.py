"""Capa de data engineering (lakehouse medallion) alrededor del CNN congelado.

Envuelve el proyecto de vision con una arquitectura medallion (bronze/silver/gold),
un modelo dimensional (star schema), una capa de calidad de datos y exports para
dashboard. No toca el modelo ni el pipeline de imagen: solo lee los parquets ONPE y
las salidas de evaluacion del modelo para construir tablas Delta analiticas.

Modulos:
- field_mapping: fuente unica del mapeo field <-> nposicion (reusada por build_crops,
  evaluate y silver).
- paths: rutas locales y URIs ADLS (abfss) + storage_options.
- io_delta: escribir/leer Delta en local o ADLS (backend por variable de entorno).
- bronze / silver / gold: las tres capas medallion.
- quality: expectativas de calidad sobre silver/gold (patron CheckResult).
- export_sheets: export de los marts gold a Sheets/CSV para Looker Studio.
"""
