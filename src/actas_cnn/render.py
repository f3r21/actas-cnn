"""Convierte PDFs de actas a imagenes de pagina usando PyMuPDF (sin poppler).

Render con tamano objetivo fijo (default 2339x3309) en lugar de DPI fijo.
Razon: los PDFs originales tienen page sizes ligeramente distintos; renderizar
a un DPI fijo produce ~74 dimensiones distintas en disco. Con tamano fijo
todos los PNGs salen iguales, lo que permite que el detector fiducial (que usa
pixeles absolutos) funcione uniformemente y elimina la rama de "imagen rara".

Tambien auto-rota a portrait si el PDF viene en landscape.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
from PIL import Image


TARGET_W, TARGET_H = 2339, 3309


def rasterize_first_page(
    pdf_path: "str | Path",
    target_size: tuple[int, int] = (TARGET_W, TARGET_H),
) -> Image.Image:
    """Primera pagina del PDF -> PIL.Image en gris, en memoria (sin PNG).

    Mismo raster que pdf_to_images (pixeles identicos, verificado byte a byte);
    evita el encode/decode del PNG intermedio, que es ~3/4 del tiempo por acta
    (medido en M2: 0.75s de 0.97s). Auto-rota landscape -> portrait igual que
    el render a disco.
    """
    tw, th = target_size
    with fitz.open(pdf_path) as doc:
        page = doc[0]
        if page.rect.width > page.rect.height:
            page.set_rotation(90)
        pix = page.get_pixmap(matrix=fitz.Matrix(tw / page.rect.width,
                                                 th / page.rect.height))
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")


def _parse_size(s: str) -> tuple[int, int]:
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"target-size invalido: {s} (esperado WxH)")
    return int(parts[0]), int(parts[1])


def pdf_to_images(
    pdf_path: str | Path,
    out_dir: str | Path,
    first_page_only: bool = False,
    target_size: tuple[int, int] = (TARGET_W, TARGET_H),
) -> list[Path]:
    """Renderiza paginas del PDF a PNG con tamano exacto `target_size`.

    Si la pagina viene en landscape, la rota 90 grados para emitir portrait.
    Cada PNG se guarda como <pdf_stem>_p<i>.png.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tw, th = target_size
    paths: list[Path] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            if page.rect.width > page.rect.height:
                page.set_rotation(90)
            zoom_x = tw / page.rect.width
            zoom_y = th / page.rect.height
            mat = fitz.Matrix(zoom_x, zoom_y)
            pix = page.get_pixmap(matrix=mat)
            out = out_dir / f"{pdf_path.stem}_p{i}.png"
            pix.save(out)
            paths.append(out)
            if first_page_only:
                break
    return paths


def _iter_pdfs(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    if input_path.is_dir():
        yield from sorted(input_path.glob("*.pdf"))
        return
    raise FileNotFoundError(f"No existe la ruta: {input_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Renderiza PDFs de actas a PNG con tamano fijo."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path,
                       help="Ruta a un PDF individual.")
    group.add_argument("--input-dir", type=Path,
                       help="Directorio con PDFs (procesa todos los *.pdf).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directorio destino para los PNG generados.")
    parser.add_argument("--target-size", type=str,
                        default=f"{TARGET_W}x{TARGET_H}",
                        help=f"Tamano exacto del PNG (default {TARGET_W}x{TARGET_H}).")
    parser.add_argument("--dpi", type=int, default=None,
                        help="DEPRECATED: se ignora. El render usa --target-size.")
    parser.add_argument("--first-page-only", action="store_true",
                        help="Renderiza solo la primera pagina de cada PDF.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.dpi is not None:
        warnings.warn("--dpi esta deprecado y se ignora; usa --target-size.",
                      stacklevel=1)
    target_size = _parse_size(args.target_size)
    source = args.input if args.input is not None else args.input_dir
    pdfs = list(_iter_pdfs(source))
    if not pdfs:
        print(f"No se encontraron PDFs en {source}")
        return
    for pdf in pdfs:
        rutas = pdf_to_images(
            pdf,
            args.output_dir,
            first_page_only=args.first_page_only,
            target_size=target_size,
        )
        for ruta in rutas:
            print(ruta)


if __name__ == "__main__":
    main()
