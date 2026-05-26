#!/usr/bin/env bash
# Compara conteo de archivos y tamano entre GCS y los destinos.
set -euo pipefail

: "${BUCKET:?define BUCKET}"

echo "== GCS =="
gcloud storage du -s "gs://${BUCKET}"
echo -n "archivos en GCS: "
gcloud storage ls -r "gs://${BUCKET}/**" | wc -l

if rclone listremotes | grep -q '^ia:'; then
  echo "== Internet Archive (via rclone) =="
  echo -n "archivos en IA item ${IA_ITEM_ID:-<sin definir>}: "
  rclone size "ia:${IA_ITEM_ID:-}" 2>/dev/null || echo "define IA_ITEM_ID para medir"
fi

echo "== Hugging Face =="
echo "Revisa el conteo de archivos en la pagina del dataset en huggingface.co"
