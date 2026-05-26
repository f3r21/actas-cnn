# 08 - Setup Colab/Kaggle para entrenar con GPU

Guia paso a paso para mover el entrenamiento desde MPS local (M2,
~35 min/corrida) a Colab con T4 (~7-10 min/corrida).

## Prerequisitos: cuentas y tokens (una sola vez)

### 1. Hugging Face

1. Crear cuenta gratis en https://huggingface.co/join (si todavia no).
2. Generar token de escritura: https://huggingface.co/settings/tokens
   - "Create new token"
   - Token type: **Write**
   - Name: `actas-cnn-colab` (o lo que quieras)
   - Copiar el token. Aparece UNA vez.

### 2. Crear los 2 repos de HF

- **Dataset**: https://huggingface.co/new-dataset
  - Owner: tu usuario
  - Repository name: `actas-cnn-dataset`
  - Visibilidad: privado o publico (privado es OK si no compartis).
  - Create dataset.

- **Model**: https://huggingface.co/new
  - Owner: tu usuario
  - Repository name: `actas-cnn-model`
  - Visibilidad: idem.
  - Create model.

### 3. Subir el bundle al HF dataset repo

El archivo `data_bundle.tar.gz` (~460 MB, ya generado en la raiz del
repo local) contiene crops + manifests + parquets + templates +
anchors.

**Opcion A: web UI** (mas simple):
1. Ir al dataset repo en HF: `https://huggingface.co/datasets/<tu-usuario>/actas-cnn-dataset`
2. Tab "Files" → "Add file" → "Upload files".
3. Arrastrar `data_bundle.tar.gz` (~460 MB, sube en 2-5 min con buena conexion).
4. Commit.

**Opcion B: CLI** (mejor para reanudar):
```bash
pip install huggingface_hub
huggingface-cli login   # pegas el token
huggingface-cli upload <tu-usuario>/actas-cnn-dataset data_bundle.tar.gz --repo-type dataset
```

### 4. Subir codigo a GitHub

El notebook hace `git clone` desde GitHub, asi que el codigo tiene
que estar publico (o usar PAT para privado).

```bash
# Si todavia no tienes repo en GitHub:
gh repo create actas-cnn --public --source=. --remote=origin --push

# O manual:
# 1) https://github.com/new -> nombre actas-cnn, publico, sin README
# 2) Local:
git init                        # si no esta inicializado
git add -A
git commit -m "Sem 2: ResNet-18 + ablations + Colab setup"
git remote add origin git@github.com:<tu-usuario>/actas-cnn.git
git push -u origin main
```

### 5. Actualizar `config.py` con tus repos

Editar `config.py` (linea 11-19) y reemplazar los TODO:

```python
@dataclass(frozen=True)
class RemoteConfig:
    hf_dataset_repo: str = "<tu-usuario>/actas-cnn-dataset"  # <-- TU repo
    hf_model_repo: str = "<tu-usuario>/actas-cnn-model"      # <-- TU repo
    wandb_project: str = "actas-cnn"
    wandb_entity: str = ""
    r2_bucket: str = "actas-cnn"
    r2_endpoint: str = ""
```

Commit + push esto a GitHub. El notebook clona desde GitHub, asi que
necesita estos valores actualizados en el repo remoto.

## Correr en Colab

### 6. Abrir el notebook en Colab

Opciones:

**A** (directo desde GitHub): abrir
`https://colab.research.google.com/github/<tu-usuario>/actas-cnn/blob/main/notebooks/train_portable.ipynb`

**B** (subir manualmente): bajar el .ipynb desde tu repo, ir a
https://colab.research.google.com → Upload → seleccionar.

### 7. Activar GPU

`Runtime` → `Change runtime type` → Hardware accelerator: **T4 GPU**
(plan gratis) o A100/V100 (Pro).

### 8. Configurar HF_TOKEN como Colab Secret

En la barra lateral izquierda, icono de llave (Secrets) → "+":
- Name: `HF_TOKEN`
- Value: el token de paso 1.2
- Toggle "Notebook access": ON.

### 9. Editar `REPO_URL` en la primera celda de codigo

Cambiar `https://github.com/TU-USUARIO/actas-cnn.git` por tu URL real.

### 10. Ejecutar

`Runtime` → `Run all`. Tiempo total esperado:
- Clone + pip install: ~30 seg
- Bajar bundle 460 MB de HF: ~30-60 seg (HF tiene buena banda)
- Descomprimir: ~30 seg
- Entrenar 40 epochs ResNet-18: ~15-20 min en T4
- Evaluate: ~1 min
- Subir checkpoint: ~30 seg

**Total: ~20-25 min** vs ~70-80 min en MPS para 40 epochs.

## Iterar ablations

Editar las constantes de la celda 5 y re-run:
- `EPOCHS`: 20 / 40 / 60.
- `MIXUP`: 0.2 / 0.4 / 0.6.
- `LABEL_SMOOTHING`: 0.0 / 0.05 / 0.1 / 0.15.
- `RANDAUGMENT`: True / False.
- `COSINE_LR`: True / False.
- `SUFFIX`: nombre unico por ablation (e.g. `e40_mu04`) para no
  sobreescribir checkpoints.

Cada combinacion produce su propio `resnet18_<SUFFIX>_best.pt` en HF
model repo, asi que podes comparar despues.

## Troubleshooting

### "HF_TOKEN no esta configurado en Secrets"
- Configura HF_TOKEN en Colab Secrets (paso 8).
- Verifica que "Notebook access" este ON.

### `git clone` falla
- Si el repo es privado, configurar PAT de GitHub. Mas simple: hacer
  el repo publico.

### "out of memory" en GPU
- Bajar `TRAIN.batch_size` en `config.py` de 128 a 64.

### Sesion de Colab se cae
- Plan gratis: hasta 12h de sesion, se corta por inactividad.
- Soluciones:
  - Pasarse a Kaggle (sesiones mas estables, 30 hrs/semana).
  - Activar Colab Pro ($10/mes).
  - Salvar checkpoint en HF cada epoch (no implementado todavia).

### El bundle no esta en el HF dataset repo
- Verificar que se subio: ir al repo en HF, tab Files.
- Si no aparece: re-upload via CLI con `huggingface-cli upload`.

## Bajar el modelo entrenado a local (post-entrenamiento)

Desde la maquina local con HF_TOKEN configurado:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(
    repo_id="<tu-usuario>/actas-cnn-model",
    filename="resnet18_colab_ls_ra_mu_cos_e40_best.pt",
)
# Copiar a checkpoints/
import shutil; shutil.copy(ckpt, "checkpoints/resnet18_best.pt")
```

Luego ya podes correr `python scripts/audit.py` y `python
scripts/evaluate.py` localmente sobre el modelo entrenado en Colab.
