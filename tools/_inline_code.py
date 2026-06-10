"""Bloques de codigo inline para los notebooks Colab (fuente de los cells).

Los notebooks entregables son *autonomos*: llevan su propio codigo en celdas, sin
clonar el repo. Para no duplicar a mano, estos bloques son la version aplanada y
didactica del paquete `actas_cnn` (misma logica, validada contra el baseline).
`tools/build_notebooks.py` los ensambla con nbformat.

Editar la deteccion de digitos = editar PREPROCESS aca y re-generar los notebooks.
"""

# --- Instalacion de dependencias (Colab ya trae torch/torchvision) -----------
INSTALL = r"""# Dependencias (Colab ya trae torch, torchvision, numpy, pandas, matplotlib)
%pip install -q pymupdf==1.27.2.3 opencv-python-headless huggingface_hub pyarrow
print("deps instaladas")"""

# --- Preprocesamiento: DONDE ESTAN LOS DIGITOS (superficie de iteracion) ------
PREPROCESS = r'''# === PREPROCESAMIENTO: deteccion de digitos (EDITAR AQUI para cambiar el metodo) ===
# Metodo OFICIAL = zonal por plantilla: cada campo se recorta por su caja relativa
# y se parte en n_digits celdas equiespaciadas. La afin del template alinea bien
# en >98% de actas. Para cambiar "donde estan los digitos", reescribe crop_fields
# / split_digits (p.ej. projection profile, deteccion por contornos, un detector
# aprendido) manteniendo la firma: localizar -> {campo: [celda_0, celda_1, ...]}.
import fitz  # PyMuPDF
from PIL import Image

TARGET_SIZE = (2339, 3309)  # tamano fijo: imagenes uniformes, detector estable

def rasterize_acta(pdf_path):
    """PDF de acta -> PIL.Image en gris (primera pagina, tamano fijo), en memoria.
    Sin PNG intermedio: encode+write+decode de un PNG de 7.7Mpx cuesta ~3/4 del
    tiempo por acta (medido en M2: 0.75s de 0.97s) y el archivo se borraria
    igual. Pixeles identicos al PNG (verificado byte a byte)."""
    with fitz.open(pdf_path) as doc:
        page = doc[0]
        if page.rect.width > page.rect.height:
            page.set_rotation(90)  # normaliza landscape -> portrait
        mat = fitz.Matrix(TARGET_SIZE[0] / page.rect.width,
                          TARGET_SIZE[1] / page.rect.height)
        pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")

def crop_fields(image, template):
    """Recorta los 42 campos por sus cajas relativas [x0,y0,x1,y1] en [0,1].
    Acepta una PIL.Image ya rasterizada o una ruta a PNG."""
    img = image if isinstance(image, Image.Image) else Image.open(image).convert("L")
    w, h = img.size
    out = {}
    for field in template["fields"]:
        x0, y0, x1, y1 = field["box"]
        out[field["name"]] = img.crop((int(x0 * w), int(y0 * h),
                                       int(x1 * w), int(y1 * h)))
    return out

def split_digits(field_img, n_digits):
    """Parte un campo en n_digits celdas equiespaciadas (izq -> der)."""
    w, h = field_img.size
    step = w / n_digits
    return [field_img.crop((int(i * step), 0, int((i + 1) * step), h))
            for i in range(n_digits)]

def localize_digits(image, template):
    """Localizador zonal: {campo: [celda_0, ...]}. Punto unico de 'donde estan'."""
    fields = crop_fields(image, template)
    return {f["name"]: split_digits(fields[f["name"]], f["n_digits"])
            for f in template["fields"]}'''

# --- Labels (ground truth ONPE) + construccion de crops/manifest -------------
# --- Compartido por 01 (armado de labels) y 02 (evaluacion downstream) --------
FIELD_VALUE_FOR = r'''def field_value_for(name, votos_acta, total_emitidos):
    """Entero del ground truth para un campo (partido / blanco-nulos-impugnados / total)."""
    if name.startswith("partido_"):
        pos = int(name.split("_")[1])
        row = votos_acta[votos_acta["nposicion"] == pos]
        return int(row.iloc[0]["nvotos"]) if len(row) else 0
    mapping = {"votos_blanco": 80, "votos_nulos": 81, "votos_impugnados": 82}
    if name in mapping:
        row = votos_acta[votos_acta["nposicion"] == mapping[name]]
        return int(row.iloc[0]["nvotos"]) if len(row) else 0
    if name == "total_ciudadanos":
        return int(total_emitidos)
    raise ValueError(name)'''

LABELS_BUILD = r'''# === Labels desde el ground truth ONPE + armado de crops/manifest ===
import csv
import pandas as pd

def es_celda_escrita(value, n_cells, pos):
    """Convencion ONPE: cifras right-justified, leading zeros en blanco.
    value=5,n=3 -> [vacio,vacio,'5']; value=0 -> todo vacio (nadie escribe '000')."""
    if value == 0:
        return False
    return pos >= n_cells - len(str(int(value)))

def int_to_digits(value, n_cells):
    """Entero -> lista de n_cells digitos right-justified. 18,3 -> [0,1,8]."""
    return [int(c) for c in str(int(value)).zfill(n_cells)]

''' + FIELD_VALUE_FOR + r'''

def build_crops_for_acta(image, archivo_id, id_acta, template,
                         votos, cabecera, crops_root, filtrar_vacias=True):
    """Imagen (PIL o ruta) + labels -> crops/<label>/<archivoId>_<campo>_c<pos>.png.
    Devuelve (guardados, filtrados)."""
    cab = cabecera[cabecera["idActa"] == id_acta]
    if len(cab) == 0 or pd.isna(cab.iloc[0]["totalVotosEmitidos"]):
        return 0, 0
    total = int(cab.iloc[0]["totalVotosEmitidos"])
    votos_acta = votos[votos["idActa"] == id_acta]
    cells = localize_digits(image, template)
    crops_root = Path(crops_root)
    n_saved = n_filt = 0
    for field in template["fields"]:
        name, n_cells = field["name"], field["n_digits"]
        value = field_value_for(name, votos_acta, total)
        labels = int_to_digits(value, n_cells)
        for pos, (label, dimg) in enumerate(zip(labels, cells[name])):
            if filtrar_vacias and not es_celda_escrita(value, n_cells, pos):
                n_filt += 1; continue
            d = crops_root / str(label); d.mkdir(parents=True, exist_ok=True)
            dimg.save(d / f"{archivo_id}_{name}_c{pos}.png")
            n_saved += 1
    return n_saved, n_filt

def build_manifest(crops_dir, out_csv):
    """crops/<label>/*.png -> manifest CSV (path,label) relativo a crops_dir."""
    crops_dir = Path(crops_dir); rows = []
    for label_dir in sorted(crops_dir.iterdir()):
        if label_dir.is_dir():
            for img in sorted(label_dir.glob("*.png")):
                rows.append((str(img.relative_to(crops_dir)), label_dir.name))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path", "label"]); w.writerows(rows)
    return len(rows)'''

# --- Modelo: ResNet-18 estilo CIFAR (modelo del proyecto) --------------------
MODEL = r'''# === Modelo del proyecto: ResNet-18 estilo CIFAR (He et al., 2015) ===
import torch.nn as nn
from torchvision.models import resnet18 as _torchvision_resnet18

def resnet18_cifar(in_channels=1, num_classes=10):
    """ResNet-18 adaptada a entradas 1x32x32: stem 3x3 stride 1, sin MaxPool
    inicial (preserva resolucion), 1 canal. Mantiene las 4 etapas residuales,
    GAP (1,1) y Linear(512, num_classes). 11.17M params."""
    m = _torchvision_resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m'''

# --- Dataset + transforms ----------------------------------------------------
DATASET = r'''# === Dataset de crops + transforms ===
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def default_transforms(image_size=32, train=True, randaugment=False):
    ops = [transforms.Grayscale(num_output_channels=1),
           transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomAffine(degrees=8, translate=(0.1, 0.1), scale=(0.9, 1.1)))
        if randaugment:
            ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops += [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    return transforms.Compose(ops)

class CropsDataset(Dataset):
    def __init__(self, manifest_csv, root=".", transform=None):
        self.df = pd.read_csv(manifest_csv)
        self.root = Path(root)
        self.transform = transform or default_transforms()
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = self.transform(Image.open(self.root / row["path"]))
        return x, torch.tensor(int(row["label"]), dtype=torch.long)'''

# --- Entrenamiento -----------------------------------------------------------
TRAIN = r'''# === Entrenamiento (ResNet-18, recipe base, ~5-8 min en T4) ===
import os
import numpy as np
from torch.utils.data import DataLoader, random_split

def _mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def run_epoch(model, loader, device, criterion, optimizer=None, mixup_alpha=0.0):
    is_train = optimizer is not None
    model.train(is_train)
    total = correct = 0; loss_sum = 0.0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train and mixup_alpha > 0:
                xm, ya, yb, lam = _mixup(x, y, mixup_alpha)
                out = model(xm)
                loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
                correct += (out.argmax(1) == ya).sum().item()
            else:
                out = model(x); loss = criterion(out, y)
                correct += (out.argmax(1) == y).sum().item()
            if is_train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * x.size(0); total += x.size(0)
    return loss_sum / total, correct / total

def train_model(manifest, root, device, epochs=20, lr=5e-4, batch_size=128,
                label_smoothing=0.0, randaugment=False, mixup=0.0, cosine_lr=False,
                seed=42):
    # Defaults = recipe BASE (sin RandAugment): rapido y consistente con el
    # checkpoint oficial resnet18_best.pt (~90.3% acta). RandAugment corre en CPU
    # por imagen y es el cuello de botella en Colab; activalo (randaugment=True,
    # mixup=0.2, cosine_lr=True, label_smoothing=0.1) solo para la ablacion ls_ra_mu_cos.
    torch.manual_seed(seed)
    full = CropsDataset(manifest, root=root,
                        transform=default_transforms(32, train=True, randaugment=randaugment))
    n_val = max(1, int(0.2 * len(full)))
    tr, va = random_split(full, [len(full) - n_val, n_val])
    pin = device.type == "cuda"
    nw = min(4, os.cpu_count() or 2)
    trl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=nw,
                     pin_memory=pin, persistent_workers=nw > 0)
    val = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=nw,
                     pin_memory=pin, persistent_workers=nw > 0)
    model = resnet18_cifar(1, 10).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cosine_lr else None
    best = 0.0
    for ep in range(1, epochs + 1):
        _, tra = run_epoch(model, trl, device, crit, opt, mixup_alpha=mixup)
        _, vaa = run_epoch(model, val, device, crit)
        if sched: sched.step()
        best = max(best, vaa)
        print(f"epoch {ep:02d}  train_acc {tra:.4f}  val_acc {vaa:.4f}")
    print(f"mejor val_acc (holdout interno): {best:.4f}")
    return model'''

# --- Evaluacion downstream (digit/field/acta-level + reconstruccion) ---------
EVAL = r'''# === Evaluacion downstream: digit / field / acta-level + reconstruccion del total ===
import numpy as np

''' + FIELD_VALUE_FOR + r'''

def parse_crop_path(rel):
    """'<label>/<aid>_<field>_c<pos>.png' -> (aid, field, pos)."""
    parts = Path(rel).stem.split("_")
    return parts[0], "_".join(parts[1:-1]), int(parts[-1][1:])

def reconstruct_value(preds_by_pos, n_cells):
    """Digitos predichos -> entero right-justified (posiciones faltantes = 0)."""
    return int("".join(str(preds_by_pos.get(p, 0)) for p in range(n_cells)))

@torch.no_grad()
def evaluate_split(model, manifest, crops_root, template, archivos, votos, cabecera, device):
    """Devuelve (df_celdas, res_campos): digit-level en df, field/acta-level en res."""
    model.eval()
    field_specs = {f["name"]: f["n_digits"] for f in template["fields"]}
    aid_to_idacta = dict(zip(archivos["archivoId"], archivos["idActa"]))
    ds = CropsDataset(manifest, root=crops_root, transform=default_transforms(32, train=False))
    df = ds.df.reset_index(drop=True)
    preds = []
    loader = DataLoader(ds, batch_size=512, shuffle=False,
                        num_workers=min(4, os.cpu_count() or 2))
    for x, _ in loader:
        preds.append(model(x.to(device)).argmax(1).cpu().numpy())
    df["pred"] = np.concatenate(preds)
    parsed = df["path"].apply(parse_crop_path).apply(pd.Series)
    parsed.columns = ["archivoId", "field", "pos"]
    df = pd.concat([df, parsed], axis=1)
    rows = []
    for aid, da in df.groupby("archivoId"):
        if aid not in aid_to_idacta:
            continue
        ida = int(aid_to_idacta[aid])
        cab = cabecera[cabecera["idActa"] == ida]
        if len(cab) == 0 or pd.isna(cab.iloc[0]["totalVotosEmitidos"]):
            continue
        total = int(cab.iloc[0]["totalVotosEmitidos"])
        va = votos[votos["idActa"] == ida]
        for fname, n_cells in field_specs.items():
            cf = da[da["field"] == fname]
            pv = reconstruct_value(dict(zip(cf["pos"], cf["pred"])), n_cells)
            rv = field_value_for(fname, va, total)
            rows.append({"archivoId": aid, "field": fname, "pred": pv, "real": rv,
                         "correct": pv == rv, "error": pv - rv})
    res = pd.DataFrame(rows)
    n_eval, n_total = res["archivoId"].nunique(), df["archivoId"].nunique()
    if n_eval < n_total:
        print(f"aviso: {n_eval}/{n_total} actas evaluadas "
              f"({n_total - n_eval} sin ground truth en los parquets)")
    return df, res

def report_metrics(df, res):
    """Imprime y devuelve el dict de metricas oficiales."""
    digit = float(df["pred"].eq(df["label"]).mean())
    field = float(res["correct"].mean())
    acta = float(res.groupby("archivoId")["correct"].all().mean())
    no_tot = res[res["field"] != "total_ciudadanos"]
    err = (no_tot.groupby("archivoId")["pred"].sum() - no_tot.groupby("archivoId")["real"].sum())
    mae = float(err.abs().mean()); exact = float((err == 0).mean() * 100)
    print(f"digit-level : {digit:.4f}  (n={len(df)})")
    print(f"field-level : {field:.4f}")
    print(f"acta-level  : {acta:.4f}")
    print(f"reconstruccion total: MAE {mae:.2f} votos, exacta {exact:.2f}% de actas")
    return {"digit": digit, "field": field, "acta": acta, "total_mae": mae,
            "total_exact_pct": exact}'''

# --- Metricas para el informe: confusion, P/R/F1, ablations ------------------
METRICS = r'''# === Metricas para el informe: matriz de confusion, P/R/F1 por clase, ablations ===
import numpy as np
import matplotlib.pyplot as plt

def confusion_and_prf(df):
    """Matriz 10x10 + tabla precision/recall/F1 por clase (digit-level)."""
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(df["label"].values, df["pred"].values):
        cm[int(t), int(p)] += 1
    recall = np.array([cm[i, i] / max(cm[i].sum(), 1) for i in range(10)])
    prec = np.array([cm[i, i] / max(cm[:, i].sum(), 1) for i in range(10)])
    f1 = np.array([2 * p * r / max(p + r, 1e-9) for p, r in zip(prec, recall)])
    prf = pd.DataFrame({"clase": range(10), "n": cm.sum(1),
                        "precision": prec, "recall": recall, "f1": f1})
    return cm, prf

def plot_confusion(cm, acc):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusion (acc={acc:.4f})")
    vmax = cm.max()
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > vmax / 2 else "black")
    fig.colorbar(im); fig.tight_layout(); plt.show()

def ablations_table(csv_map):
    """{nombre: ruta_evaluate_val_*.csv} -> tabla field/acta-level + MAE del total."""
    rows = []
    for nombre, csv_path in csv_map.items():
        r = pd.read_csv(csv_path)
        no_tot = r[r["field"] != "total_ciudadanos"]
        err = (no_tot.groupby("archivoId")["pred"].sum()
               - no_tot.groupby("archivoId")["real"].sum()).abs()
        rows.append({"variante": nombre,
                     "field_acc": r["correct"].mean(),
                     "acta_acc": r.groupby("archivoId")["correct"].all().mean(),
                     "total_mae": err.mean()})
    return pd.DataFrame(rows).set_index("variante")'''
