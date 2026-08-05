#!/usr/bin/env python3
"""Render the HTML deck to PDF and produce a page-contact-sheet preview."""

from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image, ImageDraw
from weasyprint import HTML


HERE = Path(__file__).resolve().parent
HTML_PATH = HERE / "Micro_Q_Instruct_Presentation.html"
PDF_PATH = HERE / "Micro_Q_Instruct_Presentation.pdf"
PREVIEW_PATH = HERE / "Micro_Q_Instruct_Presentation_preview.png"
EXPECTED_PAGES = 32


def render_pdf() -> None:
    HTML(filename=str(HTML_PATH), base_url=str(HERE)).write_pdf(str(PDF_PATH))


def make_preview() -> int:
    document = fitz.open(PDF_PATH)
    if document.page_count != EXPECTED_PAGES:
        raise RuntimeError(f"Expected {EXPECTED_PAGES} pages, got {document.page_count}")

    thumb_w, thumb_h = 320, 180
    label_h, columns, rows = 20, 4, 8
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "#d7dddb")
    draw = ImageDraw.Draw(sheet)
    for index, page in enumerate(document):
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        image = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + label_h)
        sheet.paste(image, (x, y))
        draw.rectangle((x, y + thumb_h, x + thumb_w, y + thumb_h + label_h), fill="#0a3344")
        draw.text((x + 7, y + thumb_h + 4), f"SLIDE {index + 1:02d}", fill="white")
    sheet.save(PREVIEW_PATH, optimize=True)
    return document.page_count


def main() -> None:
    render_pdf()
    pages = make_preview()
    size_mb = PDF_PATH.stat().st_size / 1024 / 1024
    print(f"Rendered {PDF_PATH}")
    print(f"Pages: {pages}; size: {size_mb:.1f} MB")
    print(f"Preview: {PREVIEW_PATH}")


if __name__ == "__main__":
    main()
