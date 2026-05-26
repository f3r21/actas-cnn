"""Audit de calidad de los PNGs producidos por Etapa 1 (pdf_to_images.py).

Mide por cada PNG en data/pdfs_train/rendered/:
  - dimensiones (w, h) y orientacion (landscape/portrait)
  - mean intensity (0=negro, 255=blanco)
  - std intensity (uniforme => sospechoso)
  - fraccion de pixeles oscuros (< 200) -> proxy de "hay contenido"
  - fraccion de pixeles casi negros (< 30) -> proxy de "scan sobreexpuesto"

Reporta:
  - distribucion global
  - bottom-50 por contenido (probablemente vacios/corruptos)
  - top-50 por oscuridad (probablemente sobreexpuestos / fotos malas)
  - actas con dimension fuera de 2339x3309
  - actas rotadas (landscape)
  - cross-check con worst-20 de docs/auditorias/template-generalizacion.md (si hay overlap, el problema
    de bajo accuracy puede venir de Etapa 1, no del template)

Salida:
  - data/audit_render_quality.csv (una fila por acta)
  - data/visualizaciones/audit_render_quality_*.png (grids para inspeccion)
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


THRESHOLD_OSCURO = 200       # pixeles "con tinta o linea"
THRESHOLD_NEGRO = 30         # pixeles "sobreexpuestos / casi negros"
NOMINAL_SIZE = (2339, 3309)  # dimension esperada del render


@dataclass(frozen=True)
class RenderStats:
    archivo_id: str
    width: int
    height: int
    is_landscape: bool
    is_nominal_size: bool
    mean_intensity: float
    std_intensity: float
    frac_dark: float
    frac_very_dark: float


def medir_png(path: Path) -> RenderStats:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape
    archivo_id = path.stem.removesuffix("_p0")
    return RenderStats(
        archivo_id=archivo_id,
        width=w,
        height=h,
        is_landscape=w > h,
        is_nominal_size=(w, h) == NOMINAL_SIZE,
        mean_intensity=float(arr.mean()),
        std_intensity=float(arr.std()),
        frac_dark=float((arr < THRESHOLD_OSCURO).mean()),
        frac_very_dark=float((arr < THRESHOLD_NEGRO).mean()),
    )


def parse_worst_table(audit_md: Path) -> list[str]:
    """Extrae los archivoIds de la seccion 'Los 20 peores actas' de docs/auditorias/template-generalizacion.md."""
    if not audit_md.exists():
        return []
    txt = audit_md.read_text(encoding="utf-8")
    matches = re.findall(r"`([0-9a-f]{24})`\s+acc=", txt)
    return matches


def construir_grid(paths: list[Path], titulo: str, out_path: Path, cols: int = 5):
    """Construye una grilla PNG simple sin matplotlib (pure PIL)."""
    if not paths:
        return
    from PIL import ImageDraw, ImageFont
    thumb_w, thumb_h = 280, 396
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h + 40), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 10), titulo, fill="black", font=font)
    for i, p in enumerate(paths):
        r, c = divmod(i, cols)
        thumb = Image.open(p).convert("RGB").resize((thumb_w, thumb_h))
        canvas.paste(thumb, (c * thumb_w, 40 + r * thumb_h))
        d = ImageDraw.Draw(canvas)
        d.text((c * thumb_w + 5, 40 + r * thumb_h + 5),
               p.stem.removesuffix("_p0")[:12], fill="red", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rendered-dir", default=Path("data/pdfs_train/rendered"),
                    type=Path)
    ap.add_argument("--audit-template",
                    default=Path("docs/auditorias/template-generalizacion.md"),
                    type=Path)
    ap.add_argument("--out-csv", default=Path("data/audit_render_quality.csv"),
                    type=Path)
    ap.add_argument("--viz-dir", default=Path("data/visualizaciones"),
                    type=Path)
    args = ap.parse_args()

    pngs = sorted(args.rendered_dir.glob("*_p0.png"))
    print(f"midiendo {len(pngs)} PNGs...")

    stats: list[RenderStats] = []
    for i, p in enumerate(pngs):
        stats.append(medir_png(p))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pngs)}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archivo_id", "width", "height", "is_landscape",
                    "is_nominal_size", "mean_intensity", "std_intensity",
                    "frac_dark", "frac_very_dark"])
        for s in stats:
            w.writerow([s.archivo_id, s.width, s.height, int(s.is_landscape),
                        int(s.is_nominal_size), f"{s.mean_intensity:.2f}",
                        f"{s.std_intensity:.2f}", f"{s.frac_dark:.4f}",
                        f"{s.frac_very_dark:.4f}"])
    print(f"CSV -> {args.out_csv}")

    print()
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)

    n = len(stats)
    n_landscape = sum(1 for s in stats if s.is_landscape)
    n_off_size = sum(1 for s in stats if not s.is_nominal_size)
    fracs = np.array([s.frac_dark for s in stats])
    means = np.array([s.mean_intensity for s in stats])
    stds = np.array([s.std_intensity for s in stats])

    print(f"  total renders          : {n}")
    print(f"  landscape (rotados)    : {n_landscape}")
    print(f"  fuera de {NOMINAL_SIZE}: {n_off_size}")
    print()
    print("  frac_dark (proxy 'hay contenido'):")
    print(f"    media={fracs.mean():.3f}  mediana={np.median(fracs):.3f}")
    print(f"    P05={np.percentile(fracs, 5):.3f}  P95={np.percentile(fracs, 95):.3f}")
    print(f"    min={fracs.min():.3f}  max={fracs.max():.3f}")
    print()
    print("  mean_intensity (0=negro, 255=blanco):")
    print(f"    media={means.mean():.1f}  mediana={np.median(means):.1f}")
    print(f"    P05={np.percentile(means, 5):.1f}  P95={np.percentile(means, 95):.1f}")
    print()
    print("  std_intensity (uniforme = std baja = sospechoso):")
    print(f"    media={stds.mean():.1f}  mediana={np.median(stds):.1f}")
    print(f"    P05={np.percentile(stds, 5):.1f}  P95={np.percentile(stds, 95):.1f}")

    # Bottom-50 por contenido = mas probable "vacios"
    by_content = sorted(stats, key=lambda s: s.frac_dark)
    bottom = by_content[:50]
    print()
    print("BOTTOM-10 por frac_dark (sospechosos de estar vacios):")
    for s in bottom[:10]:
        print(f"  {s.archivo_id}  frac_dark={s.frac_dark:.4f}  "
              f"mean={s.mean_intensity:.1f}  std={s.std_intensity:.1f}  "
              f"size=({s.width}x{s.height})")

    # Top-50 por oscuridad
    top_dark = sorted(stats, key=lambda s: -s.frac_very_dark)[:50]
    print()
    print("TOP-10 por frac_very_dark (sobreexpuestos / fotos malas):")
    for s in top_dark[:10]:
        print(f"  {s.archivo_id}  very_dark={s.frac_very_dark:.4f}  "
              f"mean={s.mean_intensity:.1f}  std={s.std_intensity:.1f}")

    # Cross-check con worst-20 del template audit
    worst_ids = set(parse_worst_table(args.audit_template))
    if worst_ids:
        worst_stats = [s for s in stats if s.archivo_id in worst_ids]
        n_worst_blanco = sum(1 for s in worst_stats if s.frac_dark < 0.05)
        n_worst_off_size = sum(1 for s in worst_stats if not s.is_nominal_size)
        print()
        print(f"CROSS-CHECK vs worst-20 de {args.audit_template.name}:")
        print(f"  matched {len(worst_stats)}/20 actas")
        print(f"  de esos: frac_dark<0.05 -> {n_worst_blanco}  "
              f"(template falla por scan vacio)")
        print(f"           off_size       -> {n_worst_off_size}  "
              f"(template falla por dimension distinta)")
        for s in worst_stats:
            print(f"    {s.archivo_id}  frac_dark={s.frac_dark:.4f}  "
                  f"mean={s.mean_intensity:.1f}  "
                  f"size=({s.width}x{s.height})")

    # Grids visuales
    bottom_paths = [args.rendered_dir / f"{s.archivo_id}_p0.png" for s in bottom[:20]]
    top_paths = [args.rendered_dir / f"{s.archivo_id}_p0.png" for s in top_dark[:20]]
    construir_grid(bottom_paths, "BOTTOM 20 por frac_dark (posibles vacios)",
                   args.viz_dir / "audit_render_bottom20.png")
    construir_grid(top_paths, "TOP 20 por frac_very_dark (sobreexpuestos)",
                   args.viz_dir / "audit_render_top20_dark.png")
    landscape_paths = [args.rendered_dir / f"{s.archivo_id}_p0.png"
                       for s in stats if s.is_landscape]
    if landscape_paths:
        construir_grid(landscape_paths, "Actas en landscape (rotadas)",
                       args.viz_dir / "audit_render_landscape.png")

    print()
    print(f"grids -> {args.viz_dir}/audit_render_*.png")


if __name__ == "__main__":
    main()
