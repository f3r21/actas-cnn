# 08 - Setup Colab/Kaggle para entrenar con GPU

Guia paso a paso para mover el entrenamiento desde MPS local (M2,
~35 min/corrida) a Colab con T4 (~7-10 min/corrida). **El flujo no
requiere tokens** — el dataset esta en un HF dataset publico y los
resultados se bajan directo al navegador.

## Correr el notebook (3 pasos)

### 1. Abrir el notebook en Colab

`https://colab.research.google.com/github/f3r21/actas-cnn/blob/main/notebooks/train_portable.ipynb`

### 2. Activar GPU

`Runtime` → `Change runtime type` → Hardware accelerator: **T4 GPU**
(plan gratis) o A100/V100 (Pro).

### 3. Ejecutar

`Runtime` → `Run all`. Tiempo total esperado en T4:

| Paso | Tiempo |
|---|---|
| Clone repo + pip install | ~30 seg |
| Bajar bundle 460 MB desde HF | ~30-60 seg |
| Descomprimir | ~30 seg |
| Entrenar 40 epochs ResNet-18 (con ablations) | ~15-20 min |
| Evaluate | ~1 min |
| Bajar `.pt` + `.csv` al navegador | ~30 seg |

**Total: ~20-25 min** vs ~70-80 min en MPS para 40 epochs.

Al final, el navegador descarga automaticamente:
- `resnet18_<SUFFIX>_best.pt` (~45 MB) — checkpoint del modelo.
- `evaluate_val_<SUFFIX>.csv` (~3 MB) — metricas por field.

Movelos a tu repo local:
```bash
mv ~/Downloads/resnet18_<SUFFIX>_best.pt checkpoints/
mv ~/Downloads/evaluate_val_<SUFFIX>.csv data/
```

## Iterar ablations

Editar las constantes de la celda 5 y re-run:
- `EPOCHS`: 20 / 40 / 60.
- `MIXUP`: 0.2 / 0.4 / 0.6.
- `LABEL_SMOOTHING`: 0.0 / 0.05 / 0.1 / 0.15.
- `RANDAUGMENT`: True / False.
- `COSINE_LR`: True / False.
- `SUFFIX`: nombre unico por ablation (e.g. `e40_mu04`) para que el
  archivo descargado no se sobreescriba.

Cada combinacion produce su propio `resnet18_<SUFFIX>_best.pt` en el
VM, que se descarga al navegador al ejecutar la celda final.

## Troubleshooting

### `git clone` falla
- El repo es publico, no deberia fallar. Verificar conexion.
- Si Colab tiene problemas de red, abrir el notebook subiendolo
  manualmente desde la copia local.

### "out of memory" en GPU
- Bajar `TRAIN.batch_size` en `config.py` de 128 a 64.

### Sesion de Colab se cae antes de terminar
- Plan gratis: hasta 12h de sesion, se corta por inactividad.
- Soluciones:
  - Pasarse a Kaggle (sesiones mas estables, 30 hrs/semana).
  - Activar Colab Pro ($10/mes).
  - Reducir `EPOCHS` a 20 para que entre en menos tiempo.

### El navegador no descarga los archivos al final
- Verificar que el bloqueo de descargas multiple no este activo.
- O bajar manualmente desde el panel lateral de Colab (icono de
  archivo > seleccionar el .pt y .csv > tres puntos > Descargar).

## Opcional: usar tokens (no requerido)

Si en el futuro quisieras subir el checkpoint a un HF model repo
privado en lugar de bajarlo al navegador:

1. Generar token de escritura en https://huggingface.co/settings/tokens
2. Colab: barra lateral izquierda → icono llave (Secrets) → `+`
   → name `HF_TOKEN`, value el token, "Notebook access" ON.
3. La celda 3 del notebook lo cargara automaticamente desde Secrets.
4. Modificar la celda 7 para usar `storage.upload(...)` en lugar de
   `files.download(...)`.

Para el flujo del curso esto no es necesario.
