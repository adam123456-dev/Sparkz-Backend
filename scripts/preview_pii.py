from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview PII redaction for a local PDF file.")
    parser.add_argument("pdf", type=Path, help="Path to input PDF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Directory for preview output files",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=0,
        help="Optional max pages to process (0 = all pages)",
    )
    parser.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Enable OCR fallback for image-based PDFs (requires OCR dependencies).",
    )
    return parser.parse_args()


def main() -> None:
    try:
        from app.pipeline.pdf_text import extract_pdf_pages
        from app.pipeline.pii import redact_pii_with_audit
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}. Install backend dependencies first (e.g. pip install -e .)."
        ) from exc

    args = parse_args()
    pdf_path = args.pdf if args.pdf.is_absolute() else (REPO_ROOT / args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    try:
        pages = extract_pdf_pages(pdf_path, enable_ocr=args.enable_ocr)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.page_limit > 0:
        pages = pages[: args.page_limit]

    all_audits: list[dict[str, str | int]] = []
    redacted_lines: list[str] = []
    non_empty_pages = 0
    total_chars = 0

    for page_number, page_text in enumerate(pages, start=1):
        source_text = page_text or ""
        if source_text.strip():
            non_empty_pages += 1
            total_chars += len(source_text)

        redacted, audits = redact_pii_with_audit(source_text)
        redacted_lines.append(f"===== Page {page_number} =====")
        redacted_lines.append(redacted)
        redacted_lines.append("")

        for audit in audits:
            all_audits.append(
                {
                    "pageNumber": page_number,
                    "entityType": audit.entity_type,
                    "originalValue": audit.original_value,
                    "replacementValue": audit.replacement_value,
                }
            )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    redacted_path = output_dir / f"{stem}_pii_redacted.txt"
    audit_path = output_dir / f"{stem}_pii_audit.json"

    redacted_path.write_text("\n".join(redacted_lines), encoding="utf-8")
    audit_path.write_text(json.dumps(all_audits, indent=2, ensure_ascii=False), encoding="utf-8")

    by_type = Counter(item["entityType"] for item in all_audits)
    avg_chars = (total_chars / non_empty_pages) if non_empty_pages else 0

    print(f"PDF: {pdf_path}")
    print(f"OCR enabled: {args.enable_ocr}")
    print(f"Pages processed: {len(pages)}")
    print(f"Pages with extracted text: {non_empty_pages}")
    print(f"Average chars per non-empty page: {avg_chars:.1f}")
    if non_empty_pages == 0:
        print("WARNING: No text extracted. This PDF is likely image-based and OCR may be required.")
    print(f"PII replacements: {len(all_audits)}")
    for entity_type, count in by_type.most_common():
        print(f"  - {entity_type}: {count}")
    print(f"Redacted text: {redacted_path}")
    print(f"PII audit JSON: {audit_path}")


if __name__ == "__main__":
    main()

