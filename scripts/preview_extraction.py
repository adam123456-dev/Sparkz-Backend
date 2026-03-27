from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.checklists import ChecklistItem, parse_workbook


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview checklist extraction output for an XLSX workbook."
    )
    parser.add_argument("workbook", type=Path, help="Path to source XLSX workbook.")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of extracted rows to print in terminal preview (default: 20).",
    )
    parser.add_argument(
        "--sheet",
        action="append",
        dest="sheets",
        help="Filter to one or more sheet names. Repeat flag for multiple sheets.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        type=Path,
        help="Optional path to save full extraction as JSON.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_output",
        type=Path,
        help="Optional path to save full extraction as CSV.",
    )
    return parser


def filter_items(items: list[ChecklistItem], sheets: list[str] | None) -> list[ChecklistItem]:
    if not sheets:
        return items
    sheet_set = {name.strip().lower() for name in sheets if name.strip()}
    return [item for item in items if item.sheet_name.lower() in sheet_set]


def print_summary(items: list[ChecklistItem]) -> None:
    print(f"Total extracted items: {len(items)}")
    frameworks = Counter(item.framework for item in items)
    sheets = Counter(item.sheet_name for item in items)
    print("Framework counts:")
    for framework, count in frameworks.most_common():
        print(f"  - {framework}: {count}")
    print("Top sheets by extracted items:")
    for sheet, count in sheets.most_common(10):
        print(f"  - {sheet}: {count}")


def print_preview(items: Iterable[ChecklistItem], limit: int) -> None:
    print(f"\nPreview (first {limit} rows):")
    for index, item in enumerate(items):
        if index >= limit:
            break
        print("-" * 80)
        print(f"ID: {item.requirement_id}")
        print(f"Sheet: {item.sheet_name}")
        print(f"Section: {item.section_path or '[none]'}")
        print(f"Reference: {item.reference_text or '[none]'}")
        print(f"Text: {item.requirement_text}")


def write_json(path: Path, items: list[ChecklistItem]) -> None:
    payload = [asdict(item) | {"embedding_text": item.embedding_text} for item in items]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(f"\nJSON output written: {path}")


def write_csv(path: Path, items: list[ChecklistItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_workbook",
                "framework",
                "sheet_name",
                "section_path",
                "requirement_id",
                "requirement_text",
                "reference_text",
                "embedding_text",
            ],
        )
        writer.writeheader()
        for item in items:
            row = asdict(item)
            row["embedding_text"] = item.embedding_text
            writer.writerow(row)
    print(f"CSV output written: {path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    workbook_path = args.workbook
    if not workbook_path.exists():
        raise SystemExit(f"Workbook not found: {workbook_path}")

    all_items = parse_workbook(workbook_path)
    items = filter_items(all_items, args.sheets)

    print_summary(items)
    print_preview(items, args.limit)

    if args.json_output:
        write_json(args.json_output, items)
    if args.csv_output:
        write_csv(args.csv_output, items)


if __name__ == "__main__":
    main()

