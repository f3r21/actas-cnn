# 02 - Migracion de datos (PRIORIDAD #1)

Sacar las ~2TB de actas de GCS y dejarlas en Hugging Face + Internet Archive, sin
reducir, con redundancia. Detalle operativo y scripts en `archive/migracion/`
(fuera del camino critico del curso, side-project).

## Principios

- Hacerlo mientras el usuario tenga creditos de GCP (el egress se cobra al salir).
- No descargar 2TB a la laptop. Correr en una VM de GCP en la region del bucket
  (lectura GCS->VM gratis; solo VM->internet gasta egress).
- rclone es la herramienta base. Tiene backend nativo para Internet Archive; para
  HF se usa `huggingface_hub.upload_large_folder` sobre el bucket montado con gcsfuse.

## Pasos (resumen; ver archive/migracion/README_migracion.md)

1. Dimensionar: `gcloud storage du -s` y conteo de archivos. Define el sharding.
2. Configurar remotos rclone (`gcs`, `ia`) y tokens (HF, IA S3 keys).
3. Google Drive (destino prioritario, respaldo personal de 5TB): `migrar_a_drive.sh`
   (streaming, reanudable). Respeta el cap de 750 GB/dia con
   `--drive-stop-on-upload-limit`; si el bucket supera ~500k archivos, empaquetar
   en tarballs por lote antes de subir. No sustituye a HF/IA como copia publica/citable.
3b. Internet Archive: `migrar_a_ia.sh` (streaming). Dividir en varios items por
   region/tipo de eleccion (IA prefiere items moderados).
4. Hugging Face: montar con gcsfuse y `migrar_a_hf.py` (reanudable). Si 2TB excede
   limites del repo, fragmentar por region en varios repos.
5. Verificar con `verificar.sh` (conteos y tamanos GCS vs destinos).
6. Solo borrar el bucket cuando ambas copias esten confirmadas.

## Lo que falta para afinar (pedir al usuario)

- Nombre real del bucket.
- Salida de dimensionamiento (tamano total, numero de archivos).
- Como estan organizadas las carpetas (por region, por mesa, por tipo).

Con eso: definir el esquema concreto de items de IA y de repos/subcarpetas de HF,
reemplazando los placeholders.

## Criterio de aceptacion

- Conteo de archivos y bytes coinciden (dentro de tolerancia) entre GCS y cada
  destino.
- Las dos copias (HF e IA) accesibles y completas.
- Documentado el mapeo bucket -> items IA / repos HF.
