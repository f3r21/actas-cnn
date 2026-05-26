# 01 - Decisiones tomadas

Registro de decisiones del usuario (para no reabrir lo ya cerrado).

- **Framework**: PyTorch (ambos, Keras y PyTorch, se vieron en clase; se
  eligio PyTorch).
- **Tarea**: reconocimiento de cifras manuscritas en actas (clasificacion
  de digitos 0-9). Se descarto clasificar actas completas y el pipeline
  de digitalizacion end-to-end como foco principal.
- **Formato de las actas**: PDF escaneado (manuscritas, una pagina).
- **Plantilla**: una sola — Presidencial (`idEleccion=10`, `tipo=1` ACTA
  DE ESCRUTINIO), 42 campos. Los otros 4 layouts (Parlamento Andino,
  Diputados, Senadores DE Multiple, Senadores DE Unico landscape) quedan
  fuera del alcance para este entregable.
- **Filtro STAE**: las actas de Lima y Callao usan el Sistema de
  Transmision de Actas Electorales (digitos impresos por computador, 2
  paginas) — se EXCLUYEN del entrenamiento porque el problema es leer
  manuscritos, no OCR de impreso. Identificacion: PDFs de 2 paginas.
- **Etiquetas**: resueltas via cruce con parquets curados de ONPE
  (`actas_archivos`, `actas_cabecera`, `actas_votos`). No requiere
  anotacion manual.
- **Convencion ONPE de escritura**: cifras right-justified, leading zeros
  en blanco. Implementado en `extract_crops.es_celda_escrita`.
- **Nivel**: enfoque comparativo (LeNet desde cero vs CNN profunda), con
  ablations de augmentation y pretrain MNIST para Semana 2.
- **Compute**: M2 + MPS para desarrollo, Kaggle/Colab para ablations
  pesadas. Bug MPS de AdaptiveAvgPool2d arreglado cambiando LeNet pool a
  (3,3) y DeepCNN pool a (4,4) para que el cociente sea divisible.

## Stack free-tier redundante (en standby)

Solo necesario para empaquetar y publicar; no es ruta critica:
- Codigo: GitHub.
- Dataset de recortes + originales: Hugging Face (primario).
- Espejo / storage S3: Cloudflare R2 (10 GB, egress 0).
- Modelos + tracking + respaldo grande: Weights & Biases (100 GB).
- GPU: Kaggle / Colab intercambiables (Paperspace si se quiere evitar
  Google).

## Migracion del bucket 1.58 TB: fuera de alcance del curso

Originales del bucket ONPE (`gs://onpe-eg2026-pdfs-v2/`) podrian migrarse
a Hugging Face + Internet Archive como side-project futuro, pero no es
parte del proyecto evaluado. El curso evalua la CNN + informe +
presentacion. La migracion no aporta a la nota.

## Aclaracion importante

"Originales intactos" + "free tier" solo es posible porque los datos son publicos:
Hugging Face (dataset publico) e Internet Archive aceptan volumenes grandes
gratis. Ningun free tier privado normal guarda 2TB.
