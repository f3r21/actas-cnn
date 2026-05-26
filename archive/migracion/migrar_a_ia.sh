#!/usr/bin/env bash
# Migra el bucket de GCS a un item de Internet Archive con rclone (streaming).
# Requiere remotos rclone "gcs" e "ia" configurados y las variables BUCKET e IA_ITEM_ID.
set -euo pipefail

: "${BUCKET:?define BUCKET, p. ej. export BUCKET=tu-bucket}"
: "${IA_ITEM_ID:?define IA_ITEM_ID, p. ej. export IA_ITEM_ID=actas-peru-2026-region-XX}"

rclone copy "gcs:${BUCKET}" "ia:${IA_ITEM_ID}" \
  --transfers=8 --checkers=16 \
  --retries=5 --low-level-retries=10 \
  --progress --log-file=migracion_ia.log

echo "IA: subida lanzada. Revisa https://archive.org/details/${IA_ITEM_ID}"
