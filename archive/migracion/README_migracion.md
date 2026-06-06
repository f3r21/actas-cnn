# Migracion de las actas: GCS -> Hugging Face + Internet Archive

Objetivo: sacar las ~2TB de actas (PDF) de Google Cloud Storage y dejarlas, sin
reducir, en dos hogares gratis para datos publicos, con redundancia. Hazlo
**mientras tengas creditos de GCP**, porque el egress se cobra al salir.

Idea clave: no descargues 2TB a tu laptop. Corre esto en una **VM de GCP** en la
misma region del bucket (la lectura GCS->VM es gratis; solo el tramo VM->internet
gasta egress, cubierto por tus creditos). rclone transmite sin guardar todo en
disco.

## 0. Requisitos

- VM de GCP (e2-medium basta; es solo trafico) con `gcloud`, `rclone`, `gcsfuse`
  y Python con `huggingface_hub` instalados.
- Cuenta Hugging Face + token de escritura (https://huggingface.co/settings/tokens).
- Cuenta Internet Archive + llaves S3 (https://archive.org/account/s3.php).
- ADC de GCP en la VM: `gcloud auth application-default login` (o cuenta de servicio).

## 1. Configurar rclone

Crea dos remotos (ver `rclone.conf.example`):

- `gcs`  -> tipo `google cloud storage` (usa ADC: `env_auth = true`).
- `ia`   -> tipo `internetarchive` con tus llaves S3 de archive.org.

Comprueba: `rclone lsd gcs:` y `rclone about ia:` deberian responder.

## 2. Dimensiona primero

```bash
export BUCKET=tu-bucket
gcloud storage du -s "gs://$BUCKET"
gcloud storage ls -r "gs://$BUCKET/**" | wc -l
```

Anota tamano total y numero de archivos: definen como fragmentar.

## 3. Internet Archive (streaming, sin disco grande)

```bash
export BUCKET=tu-bucket
export IA_ITEM_ID=actas-peru-2026-region-XX   # un item por region/lote
./migrar_a_ia.sh
```

Nota: Internet Archive recomienda items de tamano moderado (cientos de GB y
decenas de miles de archivos como mucho). Divide por region o tipo de eleccion en
varios items en vez de uno gigante.

## 3b. Google Drive (destino prioritario, respaldo personal)

Si tienes un plan de Drive con espacio de sobra (p. ej. 5TB), sirve como copia
de respaldo. No reemplaza a HF/IA para la copia *publica/citable* (los enlaces de
Drive se rate-limitean con acceso programatico y no dan URLs estables de dataset),
pero es seguro barato.

```bash
export BUCKET=tu-bucket
export DRIVE_DIR=actas-peru-2026     # opcional; por defecto usa el nombre del bucket
./migrar_a_drive.sh
```

Caps de Drive a tener en cuenta:

- **750 GB/dia de subida por cuenta** (cuota dura). 2TB tarda ~3 dias minimo. El
  script usa `--drive-stop-on-upload-limit`: se detiene limpio al tocar el tope;
  relanzalo al dia siguiente y continua donde quedo.
- **~500,000 items en "Mi unidad"**. Si el bucket tiene mas archivos que eso,
  empaqueta primero en tarballs por region/lote y sube esos (reduce el conteo de
  objetos y ademas va mucho mas rapido). Mide el conteo con
  `gcloud storage ls -r "gs://$BUCKET/**" | wc -l` (paso 2) antes de decidir.

El remoto `gdrive` usa OAuth, no llaves estaticas: en una VM headless genera el
token con `rclone authorize "drive"` desde una maquina con navegador y pegalo
(ver `rclone.conf.example`).

## 4. Hugging Face (carpeta grande via gcsfuse)

```bash
sudo mkdir -p /mnt/actas
gcsfuse --implicit-dirs "$BUCKET" /mnt/actas      # monta el bucket de solo lectura
export HF_TOKEN=hf_xxx
python migrar_a_hf.py --folder /mnt/actas --repo TU-USUARIO/actas-peru-2026
```

`upload_large_folder` es reanudable: si se corta, vuelve a lanzarlo y continua.
Caveat: 2TB es mucho para un repo; verifica los limites vigentes de HF y, si hace
falta, fragmenta en varios repos (p. ej. uno por region) o sube por subcarpetas.

## 5. Verificar

```bash
./verificar.sh
```

Compara conteos GCS vs destinos. Para HF: revisa el numero de archivos en la web
del dataset. Para IA: revisa https://archive.org/details/$IA_ITEM_ID

## 6. Cuando termine

Confirmada la copia en ambos destinos, ya no dependes de GCP para los originales.
Recien ahi puedes apagar/borrar el bucket. La copia de trabajo (recortes
comprimidos para entrenar) se genera aparte con el pipeline del repo principal.
