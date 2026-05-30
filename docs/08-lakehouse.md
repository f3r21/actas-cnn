# 08 - Capa de Data Engineering (lakehouse medallion)

Capa de ingenieria de datos construida ALREDEDOR del CNN congelado. No mejora ni
toca el modelo (pesos, `templates.json`, `extract_crops.py`, splits, checkpoints):
envuelve el proyecto de vision con un lakehouse medallion, un modelo dimensional
(star schema), una capa de calidad de datos y exports para dashboard. Objetivo:
practica preprofesional en d-Una (BI / Databricks), hablando el idioma de un equipo
de datos.

## Arquitectura medallion

```
data/labels/*.parquet  +  data/evaluate_val.csv / eval_logits_val.parquet / evaluate_worst20_val.csv
        |   deltalake (delta-rs, sin Spark) + duckdb + pandas
        v
BRONZE  crudo + metadata de ingesta            8 tablas Delta
        |   DuckDB SQL: filtro tipo=1 & idEleccion=10, joins, tipado, dedup
        v
SILVER  conformado                              silver_votos_oficial / silver_predicciones / silver_digitos_confianza
        |   DuckDB SQL: dims (claves naturales), facts, marts
        v
GOLD    star schema + marts                     4 dims + 2 facts + 6 marts
        |
        +--> ADLS Gen2 (abfss)  -> Databricks SQL (Free Edition)
        +--> data/lakehouse/gold_export/*.csv  -> Looker Studio
```

Las tablas locales viven en `data/lakehouse/<capa>/<tabla>` (Delta dirs, gitignored
via `data/`). El espejo en ADLS vive en
`abfss://lakehouse@actaslake25067.dfs.core.windows.net/<capa>/<tabla>`.

## Decisiones

- **Stack hibrido sin Spark**: `deltalake` (delta-rs, Rust, corre en el M2),
  `duckdb` (SQL en proceso, lee tablas Arrow zero-copy) y `pandas`. Azure Databricks
  no arranca en sub de estudiante (cuota vCPU); Databricks Free Edition queda como
  showcase aditivo, no como sistema de registro.
- **No se toca el modelo**: el join O(actas x votos) de `build_crops.py` se arreglo
  con un pre-group comportamiento-preservante (crops byte-identicos, ver Fase A); el
  lakehouse solo lee parquets ONPE y las salidas de evaluacion.
- **Dos facts**: `fact_resultados_oficiales` (padron oficial nacional, analitica
  electoral real) y `fact_qa_modelo` (reconciliacion predicho-vs-oficial, monitoreo
  de calidad de extraccion). Separados para no contaminar 3.46M filas con columnas de
  prediccion casi-todas-null.

## Tablas

### Bronze (8) -- crudo + `_ingesta_ts`, `_fuente`, `_archivo_origen_hash`, `_id_corrida`
`actas_archivos` (811,984), `actas_cabecera` (463,830), `actas_votos` (18,612,565,
particionada por `idEleccion`), `mesas` (92,766), `departamentos` (30),
`pred_evaluate_val` (29,106), `pred_eval_logits_val` (22,876), `pred_worst20_val` (20).

### Silver (3)
- `silver_votos_oficial` (3,461,671) -- join unico archivos|votos|cabecera a grano
  `(idActa, nposicion)` para todas las actas presidenciales de escrutinio.
- `silver_predicciones` (29,106) -- `evaluate_val` + `idActa` + `nposicion` (via
  `lakehouse/field_mapping.py`), tipado y dedup.
- `silver_digitos_confianza` (22,876) -- logits por digito + `prob_max`, `entropia`,
  `confianza_baja`.

### Gold (star schema)
- Dims: `dim_organizacion_politica` (41), `dim_ubicacion` (2,065), `dim_acta`
  (84,431), `dim_eleccion` (1).
- Facts: `fact_resultados_oficiales` (3,461,671, grano `idActa x nposicion`),
  `fact_qa_modelo` (29,106, predicho vs oficial con `error`, `abs_error`,
  `match_flag`, `confianza_baja`).
- Marts: `mart_kpis_globales` (digit/field/acta acc, MAE, %reconstruccion, con
  `checkpoint` + `id_corrida`), `mart_calidad_por_departamento` (27),
  `mart_calidad_por_partido` (41), `mart_resultados_por_departamento` (1,230),
  `mart_peores_actas` (20), `mart_error_hist` (10).

KPIs gold reconcilian exactamente con `scripts/evaluate.py`: digit 98.12%, field
98.87%, acta-level 90.33% (626/693), MAE del total agregado 2.40.

## Como correr

```bash
pip install -r requirements.txt   # incluye deltalake, duckdb, python-dotenv, adlfs

# Build completo local (bronze -> silver -> gold -> calidad -> export):
python scripts/build_lakehouse.py --capa todo --destino local

# Una sola capa:
python scripts/build_lakehouse.py --capa silver
```

El destino se controla con `--destino {local,adls,ambos}` o con `LAKEHOUSE_DESTINO`
en `.env` (default `local`). La copia local siempre se escribe (es la fuente de
trabajo de las capas siguientes); `adls`/`ambos` ademas espejan a ADLS.

### Provenance de las predicciones
Antes de construir bronze, regenerar las salidas del modelo con el checkpoint oficial
para que los KPIs del dashboard sean trazables:

```bash
python scripts/evaluate.py --split val   # escribe evaluate_val.csv + eval_logits_val.parquet
```

## ADLS Gen2

Cuenta `actaslake25067`, filesystem `lakehouse`, carpetas `bronze/ silver/ gold/`
(region eastus2, RG `rg-actas-lakehouse`).

```bash
# Obtener la access key y dejarla en .env (gitignored):
ACCT=actaslake25067; RG=rg-actas-lakehouse
KEY=$(az storage account keys list --account-name $ACCT -g $RG --query '[0].value' -o tsv)
printf 'AZURE_STORAGE_ACCOUNT_KEY=%s\nLAKEHOUSE_DESTINO=ambos\n' "$KEY" > .env

# Subir todo (local + ADLS):
python scripts/build_lakehouse.py --capa todo --destino ambos
```

delta-rs escribe Delta estandar a `abfss://...dfs.core.windows.net` con
`storage_options={account_name, account_key}` (ver `lakehouse/paths.py`).

## Calidad de datos

`lakehouse/quality.py` corre 8 expectativas DuckDB sobre silver/gold y escribe
`QUALITY_REPORT.md` (mismo patron `CheckResult` que `scripts/audit.py`, pero sobre
las tablas Delta en vez de PNGs):

1. Conteo silver vs fuente (reconciliacion del join).
2. Integridad de FK del star schema (anti-joins == 0).
3. Dominio de `nposicion` en {1..38, 80, 81, 82}.
4. Sin nulls en `nvotos_oficial`.
5. `avg(match_flag)` gold == field_acc de evaluate (±1pp).
6. MAE del total agregado gold == MAE de evaluate (±0.01). **Assert clave.**
7. Sin leak: actas evaluadas todas en val, ninguna en train/test.
8. Grano unico en `fact_resultados_oficiales` y `silver_predicciones` (29,106).

```bash
python scripts/build_lakehouse.py --capa calidad   # genera QUALITY_REPORT.md
```

Los 6 chequeos de `scripts/audit.py` (sobre PDFs/PNGs/crops) se conservan tal cual;
la capa de calidad del lakehouse los complementa, no los reemplaza.

## Dashboard

### Looker Studio (primario)
`lakehouse/export_sheets.py` escribe cada mart a `data/lakehouse/gold_export/*.csv`.
Conectar Looker Studio a esos CSV (subida directa) o cargarlos a Google Sheets (un
tab por mart; activar el push automatico con `GOOGLE_SHEETS_SA_JSON` +
`GOOGLE_SHEETS_SPREADSHEET_ID`). Paginas sugeridas:

| Pagina / visual | Mart |
|---|---|
| KPIs de extraccion (scorecards) | mart_kpis_globales |
| Votos por organizacion politica (barra) | mart_resultados_por_departamento (rollup nposicion) |
| Resultados por departamento (coropletico) | mart_resultados_por_departamento |
| Calidad por departamento (coropletico + tabla) | mart_calidad_por_departamento |
| Calidad por partido (barra/tabla) | mart_calidad_por_partido |
| Donde fallan las extracciones (tabla) | mart_peores_actas |
| Distribucion de error (histograma) | mart_error_hist |

### Databricks SQL (Free Edition, aditivo)
Subir el gold parquet a un volumen administrado de Free Edition, `CREATE TABLE ...
USING DELTA`, y construir un dashboard Databricks SQL nativo. Es un segundo artefacto
para el portafolio; no monta ADLS externo (restringido en el free tier).

## Migracion a Azure y costos

Solo se suben los ~120MB analiticos (parquets + predicciones + gold); NUNCA los ~2TB
de PDFs (storage + egress se comerian el credito de estudiante y no aportan al
lakehouse). El raw frio queda en HF/R2/GCS (redundancia existente, ver
`docs/02-migracion.md`). Footprint en ADLS < ~300MB con overhead Delta. Egress a
Looker via los CSV/Sheets, no por lectura de ADLS. Recomendado: alerta de presupuesto
(~$30) en el portal de Azure.
```
