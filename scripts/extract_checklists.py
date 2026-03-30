from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.checklists import parse_workbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract checklist items from one XLSX workbook and save to output files."
    )
    parser.add_argument("workbook", type=Path, help="Source XLSX workbook path")
    parser.add_argument("--json", dest="json_path", type=Path, help="Write extracted items to JSON file")
    parser.add_argument("--csv", dest="csv_path", type=Path, help="Write extracted items to CSV file")
    parser.add_argument("--txt", dest="txt_path", type=Path, help="Write human-readable extracted items to TXT")
    return parser.parse_args()


def write_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_txt(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write("=" * 80 + "\n")
            file.write(f"Framework: {row['framework']}\n")
            file.write(f"Sheet: {row['sheet_name']}\n")
            file.write(f"Section: {row['section_path'] or '[none]'}\n")
            file.write(f"Requirement ID: {row['requirement_id']}\n")
            file.write(f"Reference: {row['reference_text'] or '[none]'}\n")
            file.write(f"Requirement: {row['requirement_text']}\n")
            file.write(f"Embedding text:\n{row['embedding_text']}\n\n")


def main() -> None:
    args = parse_args()
    if not args.workbook.exists():
        raise SystemExit(f"Workbook not found: {args.workbook}")
    if not any([args.json_path, args.csv_path, args.txt_path]):
        raise SystemExit("At least one output option is required: --json, --csv, or --txt")

    items = parse_workbook(args.workbook)
    rows = []
    for item in items:
        row = asdict(item) | {"embedding_text": item.embedding_text}
        row["search_keywords"] = []
        row["section_hints"] = []
        rows.append(row)

    if args.json_path:
        write_json(args.json_path, rows)
        print(f"JSON written: {args.json_path}")
    if args.csv_path:
        write_csv(args.csv_path, rows)
        print(f"CSV written: {args.csv_path}")
    if args.txt_path:
        write_txt(args.txt_path, rows)
        print(f"TXT written: {args.txt_path}")

    print(f"Extracted rows: {len(rows)}")


if __name__ == "__main__":
    main()

