#!/usr/bin/env bash
# Publica/sincroniza el universo del Proyecto II (dataset citable) en Hugging Face.
#
# NOTA: los 5,000 PDFs y los labels YA fueron subidos manualmente por el usuario
# a f3r21/actas-cnn-dataset (PDFs en la raiz, labels en labels/). Este script
# re-sincroniza ese layout de forma idempotente y agrega el split canonico, que
# faltaba. El archivo completo (1.58 TB) va a Drive aparte (repo del scraper,
# ops/vm-hetzner/).
#
# Un guardrail anti-exfiltracion suele bloquear que el agente suba el arbol
# fuente en bloque: correr ESTE script tu mismo desde la laptop.
#
# Requisitos:
#   pip install huggingface_hub        # trae huggingface-cli
#   huggingface-cli login              # o export HF_TOKEN=hf_xxx
set -euo pipefail
cd "$(dirname "$0")/../.."

REPO="${HF_DATASET_REPO:-f3r21/actas-cnn-dataset}"
PDFS_DIR="data/pdfs_train"
LABELS_DIR="data/labels"
SPLIT="data/splits/manuscritas_full.txt"

command -v huggingface-cli >/dev/null || {
  echo "falta huggingface-cli. pip install huggingface_hub" >&2; exit 1
}

echo "[hf] creando/verificando dataset $REPO"
huggingface-cli repo create "$REPO" --repo-type dataset -y 2>/dev/null || true

# PDFs a la RAIZ (coincide con el layout ya subido; idempotente). Excluye
# data/pdfs_train/rendered/*.png. Si rendered/ ya se borro, --include es no-op.
echo "[hf] sincronizando PDFs (solo *.pdf, a la raiz)..."
huggingface-cli upload "$REPO" "$PDFS_DIR" . --repo-type dataset --include "*.pdf"

echo "[hf] sincronizando labels (parquets ONPE)..."
huggingface-cli upload "$REPO" "$LABELS_DIR" labels --repo-type dataset

echo "[hf] agregando split canonico (lo que faltaba)..."
huggingface-cli upload "$REPO" "$SPLIT" splits/manuscritas_full.txt --repo-type dataset

echo "[hf] listo: https://huggingface.co/datasets/$REPO"
echo "[hf] (opcional) redundancia en Internet Archive con archive/migracion/migrar_a_ia.sh"
