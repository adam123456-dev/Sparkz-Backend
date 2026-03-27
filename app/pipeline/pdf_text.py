from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_pdf_pages(pdf_path: str | Path, enable_ocr: bool = False) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())

    # When OCR is explicitly enabled, always try it (do not gate on _needs_ocr).
    # Otherwise image PDFs with accidental text fragments would skip OCR and stay empty.
    if enable_ocr:
        ocr_pages = _extract_with_ocr(pdf_path)
        if ocr_pages:
            return ocr_pages
        # Fall back to native text if OCR failed (missing deps / conversion error handled inside).
        return pages

    if _needs_ocr(pages):
        ocr_pages = _extract_with_ocr(pdf_path)
        if ocr_pages:
            return ocr_pages
    return pages


def _needs_ocr(pages: list[str]) -> bool:
    if not pages:
        return True
    non_empty = [page for page in pages if page and len(page.strip()) >= 40]
    return len(non_empty) < max(1, len(pages) // 3)


def _extract_with_ocr(pdf_path: str | Path) -> list[str]:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OCR requires packages: pip install pdf2image pytesseract "
            "and system installs: Tesseract OCR + Poppler (PATH on Windows)."
        ) from exc

    try:
        images = convert_from_path(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "pdf2image failed (often missing Poppler on Windows). "
            "Install Poppler and ensure its bin folder is on PATH."
        ) from exc

    pages: list[str] = []
    for image in images:
        pages.append((pytesseract.image_to_string(image) or "").strip())
    return pages

