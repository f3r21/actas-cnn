"""Deteccion de los 15 fiducial markers en actas ONPE.

Patron observado (constante en todas las actas Presidenciales):
  - 4 corners: TL, TR, BL, BR
  - 3 top middle: T1, T2, T3 (entre TL y TR)
  - 4 left lateral: L1, L2, L3, L4 (entre TL y BL)
  - 4 right lateral: R1, R2, R3, R4 (entre TR y BR)
  - 0 markers en el margen inferior
  Total: 15

Algoritmo: detection por zonas geometricas + sort.
  - CLAHE preprocesa la imagen para mejorar contraste local
  - Para cada zona (TOP, LEFT, RIGHT, BOT-CORNERS), encuentra blobs
    cuadrados oscuros y los ordena por posicion para asignar el rol

Probamos bootstrap iterativo con anchors (docs/auditorias/fiducial-experimento.md) pero
no supero a la deteccion zonal simple — el formato ONPE es lo suficiente
regular para que las zonas predefinidas funcionen muy bien.

Uso CLI:
  python scripts/detect_fiducials.py --image <png> [--out-overlay <png>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# Parametros calibrados sobre la referencia.
# AREA_MIN/MAX se relajaron tras audit (scripts/audit_fiducial_detector.py):
# markers no-canonicos en algunas actas son mas chicos o mas grandes. Probamos
# Otsu local pero introdujo regresiones en actas que ya funcionaban — el
# threshold estatico 150 + AREA wider gana en smoke comparativo.
THRESHOLD = 150
AREA_MIN = 200
AREA_MAX = 1200
ASPECT_MIN = 0.6
ASPECT_MAX = 1.5

DEFAULT_ANCHORS_PATH = Path(__file__).resolve().parent.parent / "fiducial_anchors.json"

# Mapeo rol -> zona geometrica. Usado por transform_template para validar
# que la nube de markers cubre >=3 zonas antes de aplicar afin (evita
# afinaciones mal condicionadas por puntos casi colineales; R-NEW de
# docs/auditorias/fiducial-auditoria.md).
ZONES = {
    "TL": "TOP", "T1": "TOP", "T2": "TOP", "T3": "TOP", "TR": "TOP",
    "L1": "LEFT", "L2": "LEFT", "L3": "LEFT", "L4": "LEFT",
    "R1": "RIGHT", "R2": "RIGHT", "R3": "RIGHT", "R4": "RIGHT",
    "BL": "BOT", "BR": "BOT",
}


def _preproc(img_gray: np.ndarray) -> np.ndarray:
    """CLAHE para mejorar contraste local antes de threshold."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)


def _scan_region(img: np.ndarray, x0: int, y0: int) -> list[tuple[int, int, int]]:
    """Devuelve [(cx, cy, area)] de blobs cuadrados oscuros."""
    binary = (img < THRESHOLD).astype(np.uint8) * 255
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    out = []
    for i in range(1, n_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        if not (AREA_MIN <= a <= AREA_MAX):
            continue
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        ar = bw / max(bh, 1)
        if not (ASPECT_MIN <= ar <= ASPECT_MAX):
            continue
        cx = int(centroids[i][0]) + x0
        cy = int(centroids[i][1]) + y0
        out.append((cx, cy, int(a)))
    return out


def _local_scan(img: np.ndarray, predicted_xy: tuple[int, int],
                 window: int = 80, area_min: int = 100,
                 area_max: int = 1500) -> tuple[int, int] | None:
    """Busqueda local de un marker alrededor de una posicion predicha.

    Usado por bootstrap iterativo cuando la deteccion zonal pierde un marker
    pero hay suficientes detectados para estimar afin coarse. Filtros
    relajados (sin aspect ratio) porque el marker puede estar parcialmente
    fragmentado por watermark/sombra.
    """
    px, py = int(predicted_xy[0]), int(predicted_xy[1])
    h, w = img.shape
    half = window // 2
    x0 = max(0, px - half)
    y0 = max(0, py - half)
    x1 = min(w, px + half)
    y1 = min(h, py + half)
    region = img[y0:y1, x0:x1]
    if region.size == 0:
        return None
    binary = (region < THRESHOLD).astype(np.uint8) * 255
    n, _, stats, cents = cv2.connectedComponentsWithStats(binary, 8)
    best_dist = None
    best_xy = None
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if not (area_min <= a <= area_max):
            continue
        cx = int(cents[i][0]) + x0
        cy = int(cents[i][1]) + y0
        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_xy = (cx, cy)
    if best_xy is not None and best_dist is not None and best_dist <= half:
        return best_xy
    return None


def _bootstrap_missing(detected: dict[str, tuple[int, int]],
                        img: np.ndarray,
                        anchors: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """Predice posiciones de markers faltantes via afin coarse + busqueda local.

    Requiere >=4 markers detectados en >=3 zonas (mismo guard de R-NEW para
    afin segura). Si no cumple, retorna `detected` sin cambios.
    """
    common = sorted(set(detected) & set(anchors))
    if len(common) < 4:
        return detected
    zones_covered = {ZONES[r] for r in common if r in ZONES}
    if len(zones_covered) < 3:
        return detected

    src = np.array([detected[r] for r in common], dtype=np.float32)
    dst = np.array([anchors[r] for r in common], dtype=np.float32)
    M, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                  ransacReprojThreshold=8.0)
    if M is None or not _affine_is_sane(M):
        return detected
    M_inv = cv2.invertAffineTransform(M)
    if not _affine_is_sane(M_inv):
        return detected

    new_markers = dict(detected)
    for role, anc_xy in anchors.items():
        if role in new_markers:
            continue
        anc_pt = np.array([[[float(anc_xy[0]), float(anc_xy[1])]]],
                          dtype=np.float32)
        pred = cv2.transform(anc_pt, M_inv).reshape(2)
        found = _local_scan(img, (int(pred[0]), int(pred[1])))
        if found is not None:
            new_markers[role] = found
    return new_markers


def detect_15(png_path: Path, anchors: dict | None = None) -> dict[str, tuple[int, int]]:
    """Detecta los 15 markers etiquetados por rol via deteccion zonal.

    Si `anchors` se pasa (o si `fiducial_anchors.json` esta disponible) y la
    deteccion zonal recupera <15 markers en >=3 zonas, intenta bootstrap
    iterativo para los faltantes.

    Nota: probamos un detector search-by-prior (Sem 2 dia 2) que bajo
    std_x de 22-27x en los roles TOP, pero downstream resulto en -0.72pp
    acta-level. El detector zonal es el oficial. Ver
    docs/05-backlog.md.
    """
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(png_path)
    h, w = img.shape

    # CLAHE preprocesamiento
    img = _preproc(img)

    # Zona TOP (5 markers: TL, T1, T2, T3, TR). y<100 (antes y<60) — la zona
    # mas critica segun audit; el AREA_MAX 1200 sigue filtrando QR/barcode.
    top = _scan_region(img[0:100, :], 0, 0)
    top_sorted_x = sorted(top, key=lambda b: b[0])

    # Zona LEFT (4 markers: L1, L2, L3, L4). Ampliada de x<70 a x<100 por la
    # misma razon que TOP: markers centrados en x~70 quedaban cortados a la
    # mitad por el slice, perdian aspect ratio y caian fuera del filtro.
    left = _scan_region(img[200:h-100, 0:100], 0, 200)
    left_sorted_y = sorted(left, key=lambda b: b[1])

    # Zona RIGHT (4 markers: R1, R2, R3, R4). Ampliada de x>w-70 a x>w-100.
    right = _scan_region(img[200:h-100, w-100:w], w-100, 200)
    right_sorted_y = sorted(right, key=lambda b: b[1])

    # Zona BOTTOM-CORNERS (BL y BR)
    bot = _scan_region(img[h-150:h, :], 0, h-150)
    bot_sorted_x = sorted(bot, key=lambda b: b[0])

    markers = {}
    if len(top_sorted_x) >= 5:
        for label, b in zip(["TL", "T1", "T2", "T3", "TR"], top_sorted_x[:5]):
            markers[label] = (b[0], b[1])
    if len(left_sorted_y) >= 4:
        for label, b in zip(["L1", "L2", "L3", "L4"], left_sorted_y[:4]):
            markers[label] = (b[0], b[1])
    if len(right_sorted_y) >= 4:
        for label, b in zip(["R1", "R2", "R3", "R4"], right_sorted_y[:4]):
            markers[label] = (b[0], b[1])
    if len(bot_sorted_x) >= 2:
        markers["BL"] = (bot_sorted_x[0][0], bot_sorted_x[0][1])
        markers["BR"] = (bot_sorted_x[-1][0], bot_sorted_x[-1][1])

    # Bootstrap: si tenemos <15 pero >=3 zonas, intentar recuperar faltantes
    # via afin coarse + busqueda local. Anchors se cargan lazy si no se pasaron.
    if len(markers) < 15:
        if anchors is None:
            try:
                anchors = load_anchors()
            except FileNotFoundError:
                anchors = None
        if anchors is not None:
            markers = _bootstrap_missing(markers, img, anchors)

    return markers


def load_anchors(path: Path | None = None) -> dict[str, tuple[int, int]]:
    """Carga los anchors canonicos desde JSON."""
    if path is None:
        path = DEFAULT_ANCHORS_PATH
    raw = json.loads(Path(path).read_text())
    return {k: tuple(v) for k, v in raw.items()}


def _affine_is_sane(M: np.ndarray) -> bool:
    """Verifica que la afin sea aproximadamente identidad-like."""
    a, b, tx = M[0, 0], M[0, 1], M[0, 2]
    c, d, ty = M[1, 0], M[1, 1], M[1, 2]
    sx = (a * a + c * c) ** 0.5
    sy = (b * b + d * d) ** 0.5
    if not (0.92 <= sx <= 1.08 and 0.92 <= sy <= 1.08):
        return False
    import math
    angle = math.degrees(math.atan2(c, a))
    if abs(angle) > 5:
        return False
    if abs(tx) > 100 or abs(ty) > 100:
        return False
    return True


def transform_template(template: dict, src_markers: dict, dst_anchors: dict,
                        img_size: tuple[int, int]) -> dict:
    """Devuelve template con boxes movidos via afin src->dst.

    Safety checks: si la afin es degenerada o las cajas terminan fuera
    de la imagen, retorna el template original.
    """
    common = sorted(set(src_markers) & set(dst_anchors))
    if len(common) < 4:
        return template
    # GUARD R-NEW: rechazar afin si los markers no cubren >=3 zonas. Puntos
    # concentrados en TOP+BOT (sin laterales) o TOP-only quedan casi
    # colineales en Y; RANSAC encuentra solucion "sana" pero estira el
    # template incorrectamente. Fallback al template original (mismo path
    # que bucket [0-3] markers) acc empirica ~0.808 vs ~0.434 con afin mal.
    zones_covered = {ZONES[r] for r in common if r in ZONES}
    if len(zones_covered) < 3:
        return template
    src = np.array([src_markers[r] for r in common], dtype=np.float32)
    dst = np.array([dst_anchors[r] for r in common], dtype=np.float32)
    M, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                  ransacReprojThreshold=8.0)
    if M is None or not _affine_is_sane(M):
        return template

    M_inv = cv2.invertAffineTransform(M)
    if not _affine_is_sane(M_inv):
        return template

    w, h = img_size
    ref_w, ref_h = template.get("image_size_reference", [2339, 3309])

    new_fields = []
    for f in template["fields"]:
        x0_f, y0_f, x1_f, y1_f = f["box"]
        x0, y0 = x0_f * ref_w, y0_f * ref_h
        x1, y1 = x1_f * ref_w, y1_f * ref_h
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                       dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.transform(pts, M_inv).reshape(-1, 2)
        nx0 = max(0.0, float(warped[:, 0].min()))
        ny0 = max(0.0, float(warped[:, 1].min()))
        nx1 = min(float(w), float(warped[:, 0].max()))
        ny1 = min(float(h), float(warped[:, 1].max()))
        if nx1 - nx0 < 10 or ny1 - ny0 < 10:
            return template  # fallback completo
        new_fields.append({
            **f,
            "box": [nx0 / w, ny0 / h, nx1 / w, ny1 / h],
        })
    return {**template, "fields": new_fields}


def draw_overlay(png_path: Path, out_path: Path, markers: dict,
                 anchors: dict | None = None):
    """Dibuja markers + opcionalmente anchors canonicos."""
    img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    for role, (cx, cy) in markers.items():
        cv2.circle(img, (cx, cy), 28, (0, 0, 255), 3)
        cv2.putText(img, role, (cx + 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    if anchors:
        for role, (cx, cy) in anchors.items():
            cv2.drawMarker(img, (cx, cy), (0, 200, 0), cv2.MARKER_CROSS, 30, 2)
    cv2.putText(img, f"detected: {len(markers)}/15", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out-overlay", default=None, type=Path)
    ap.add_argument("--save-anchors", default=None, type=Path)
    args = ap.parse_args()

    markers = detect_15(args.image)
    print(f"detected {len(markers)}/15 markers")
    for role in ["TL", "T1", "T2", "T3", "TR",
                 "L1", "L2", "L3", "L4",
                 "R1", "R2", "R3", "R4",
                 "BL", "BR"]:
        if role in markers:
            cx, cy = markers[role]
            print(f"  {role}: ({cx}, {cy})")
        else:
            print(f"  {role}: MISSING")

    if args.save_anchors:
        with open(args.save_anchors, "w") as f:
            json.dump({k: list(v) for k, v in markers.items()}, f, indent=2)
        print(f"anchors -> {args.save_anchors}")
    if args.out_overlay:
        draw_overlay(args.image, args.out_overlay, markers)
        print(f"overlay -> {args.out_overlay}")


if __name__ == "__main__":
    main()
