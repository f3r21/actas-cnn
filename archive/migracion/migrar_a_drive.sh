#!/usr/bin/env bash
# Migra el bucket de GCS a Google Drive con rclone (streaming, reanudable).
# Requiere remotos rclone "gcs" y "gdrive" configurados, y la variable BUCKET.
# Variable opcional DRIVE_DIR: carpeta destino en Drive (por defecto el nombre del bucket).
#
# Caps de Google Drive que este script respeta o que debes tener en cuenta:
#   - 750 GB/dia de subida por cuenta (cuota dura de la Drive API). Con
#     --drive-stop-on-upload-limit rclone se detiene limpio al tocar el tope;
#     relanza el mismo comando al dia siguiente y continua donde quedo.
#   - ~500,000 items en "Mi unidad". Si el bucket tiene mas archivos que eso,
#     NO uses este script tal cual: empaqueta primero en tarballs por region/lote
#     (ver README_migracion.md, seccion Drive) para reducir el conteo de objetos.
set -euo pipefail

: "${BUCKET:?define BUCKET, p. ej. export BUCKET=tu-bucket}"
DRIVE_DIR="${DRIVE_DIR:-$BUCKET}"

rclone copy "gcs:${BUCKET}" "gdrive:${DRIVE_DIR}" \
  --transfers=8 --checkers=16 \
  --retries=5 --low-level-retries=10 \
  --drive-stop-on-upload-limit \
  --progress --log-file=migracion_drive.log

echo "Drive: subida lanzada/reanudada en la carpeta '${DRIVE_DIR}' de tu Google Drive."
echo "Si se detuvo por el cap de 750 GB/dia, relanza este mismo comando manana."
