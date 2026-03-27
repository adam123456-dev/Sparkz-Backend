from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from .models import ChecklistItem

XML_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XML_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

NS = {"a": XML_MAIN_NS, "r": XML_REL_NS, "p": XML_PKG_REL_NS}

REQUIREMENT_ID_RE = re.compile(r"^(?:[A-Z]{1,5}\s*)?\d+(?:\.\d+)+$|^A\d+\.\d+$")
TOP_META_PREFIXES = ("client", "year", "period", "file number", "prepared by", "reviewed by", "date")
NON_REQUIREMENT_PREFIXES = (
    "appendix",
    "main checklist",
    "further guidance",
    "audited accounts",
    "assurance review report",
    "political donations",
)


@dataclass(slots=True)
class RawRow:
    first: str
    second: str
    third: str
    all_cells: tuple[str, ...]


class ChecklistWorkbookParser:
    def parse(self, workbook_path: str | Path) -> list[ChecklistItem]:
        workbook_path = Path(workbook_path)
        framework = self._infer_framework(workbook_path)
        items: list[ChecklistItem] = []

        with ZipFile(workbook_path) as archive:
            shared_strings = self._load_shared_strings(archive)
            for sheet_name, sheet_xml_path in self._iter_sheets(archive):
                rows = self._read_sheet_rows(archive, sheet_xml_path, shared_strings)
                sheet_items = self._extract_items_from_rows(
                    workbook_name=workbook_path.name,
                    framework=framework,
                    sheet_name=sheet_name,
                    rows=rows,
                )
                items.extend(sheet_items)

        return items

    def _load_shared_strings(self, archive: ZipFile) -> list[str]:
        path = "xl/sharedStrings.xml"
        if path not in archive.namelist():
            return []
        root = ET.fromstring(archive.read(path))
        out: list[str] = []
        for item in root.findall("a:si", NS):
            text_parts = [node.text or "" for node in item.findall(".//a:t", NS)]
            out.append("".join(text_parts))
        return out

    def _iter_sheets(self, archive: ZipFile) -> Iterable[tuple[str, str]]:
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"] for rel in rel_root.findall("p:Relationship", NS)
        }

        for sheet in workbook_root.findall("a:sheets/a:sheet", NS):
            name = sheet.attrib["name"].strip()
            rel_id = sheet.attrib[f"{{{XML_REL_NS}}}id"]
            target = rel_map[rel_id].lstrip("/")
            sheet_path = f"xl/{target}" if not target.startswith("xl/") else target
            yield name, sheet_path

    def _read_sheet_rows(self, archive: ZipFile, sheet_xml_path: str, shared_strings: list[str]) -> list[RawRow]:
        root = ET.fromstring(archive.read(sheet_xml_path))
        rows: list[RawRow] = []

        for row in root.findall("a:sheetData/a:row", NS):
            values_by_col: dict[int, str] = {}
            for cell in row.findall("a:c", NS):
                ref = cell.attrib.get("r", "")
                col_index = self._column_index(ref)
                if col_index is None:
                    continue

                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", NS)
                inline_node = cell.find("a:is/a:t", NS)
                value = ""

                if inline_node is not None and inline_node.text:
                    value = inline_node.text
                elif value_node is not None and value_node.text is not None:
                    if cell_type == "s":
                        idx = int(value_node.text)
                        value = shared_strings[idx] if 0 <= idx < len(shared_strings) else value_node.text
                    else:
                        value = value_node.text

                cleaned = self._normalize_text(value)
                if cleaned:
                    values_by_col[col_index] = cleaned

            if not values_by_col:
                continue

            max_col = max(values_by_col)
            row_cells = tuple(values_by_col.get(i, "") for i in range(1, max_col + 1))
            rows.append(
                RawRow(
                    first=row_cells[0] if len(row_cells) > 0 else "",
                    second=row_cells[1] if len(row_cells) > 1 else "",
                    third=row_cells[2] if len(row_cells) > 2 else "",
                    all_cells=row_cells,
                )
            )

        return rows

    def _extract_items_from_rows(
        self,
        workbook_name: str,
        framework: str,
        sheet_name: str,
        rows: list[RawRow],
    ) -> list[ChecklistItem]:
        items: list[ChecklistItem] = []
        section_path = ""
        current_item: ChecklistItem | None = None

        for row in rows:
            first = self._normalize_requirement_id_candidate(row.first)
            second = row.second
            third = row.third

            if self._is_top_metadata_row(first):
                continue

            if self._looks_like_section_header(first, second):
                section_path = self._build_section_path(section_path, first, second)
                continue

            if self._is_requirement_id(first):
                if current_item:
                    items.append(current_item)
                current_item = ChecklistItem(
                    source_workbook=workbook_name,
                    framework=framework,
                    sheet_name=sheet_name,
                    section_path=section_path,
                    requirement_id=first,
                    requirement_text=second,
                    reference_text=third,
                )
                continue

            if current_item and self._is_continuation_row(row):
                extension_text = second or first
                if extension_text:
                    current_item.requirement_text = self._append_sentence(
                        current_item.requirement_text, extension_text
                    )
                if third and third not in current_item.reference_text:
                    current_item.reference_text = self._append_sentence(current_item.reference_text, third)
                continue

            if first and not second and not third and not self._is_non_requirement_text(first):
                section_path = self._build_section_path(section_path, first, "")

        if current_item:
            items.append(current_item)

        return [item for item in items if item.requirement_text]

    def _is_requirement_id(self, value: str) -> bool:
        return bool(REQUIREMENT_ID_RE.match(value.strip()))

    def _is_top_metadata_row(self, first_cell: str) -> bool:
        lower = first_cell.lower()
        return any(lower.startswith(prefix) for prefix in TOP_META_PREFIXES)

    def _looks_like_section_header(self, first: str, second: str) -> bool:
        if self._is_requirement_id(first):
            return False
        if first and not second and not self._is_non_requirement_text(first):
            return True
        if first.isdigit() and second and second.isupper():
            return True
        return False

    def _is_non_requirement_text(self, value: str) -> bool:
        lower = value.lower()
        return any(lower.startswith(prefix) for prefix in NON_REQUIREMENT_PREFIXES)

    def _is_continuation_row(self, row: RawRow) -> bool:
        if row.first and self._is_requirement_id(row.first):
            return False
        if row.first and row.first.isdigit():
            return False
        has_text = bool(row.second or row.first)
        return has_text

    def _build_section_path(self, current: str, first: str, second: str) -> str:
        candidate = second or first
        candidate = candidate.strip(" :-")
        if not candidate:
            return current
        if current.endswith(candidate):
            return current
        if not current:
            return candidate
        return f"{current} > {candidate}"

    def _append_sentence(self, base: str, extra: str) -> str:
        base = self._normalize_text(base)
        extra = self._normalize_text(extra)
        if not base:
            return extra
        if not extra:
            return base
        if extra in base:
            return base
        separator = " " if base.endswith((".", ":", ";", ")", "]")) else "; "
        return f"{base}{separator}{extra}"

    def _normalize_requirement_id_candidate(self, value: str) -> str:
        """
        Excel sometimes serializes numeric cells with binary float artifacts
        (e.g. "4.0199999999999996"). For requirement-id-like numbers, convert to
        a stable short representation ("4.02").
        """
        text = value.strip()
        if not re.match(r"^\d+\.\d{10,}$", text):
            return value
        try:
            normalized = format(float(text), ".15g")
        except ValueError:
            return value
        return normalized if "." in normalized else value

    def _normalize_text(self, value: str) -> str:
        text = value.replace("\r", " ").replace("\n", " ").strip()
        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u00a0": " ",
            "\ufffd": "",
        }
        for key, target in replacements.items():
            text = text.replace(key, target)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _column_index(self, cell_ref: str) -> int | None:
        letters = "".join(char for char in cell_ref if char.isalpha())
        if not letters:
            return None
        index = 0
        for char in letters:
            index = index * 26 + (ord(char.upper()) - ord("A") + 1)
        return index

    def _infer_framework(self, workbook_path: Path) -> str:
        name = workbook_path.stem.lower()
        if "charit" in name:
            return "CHARITIES_FRS102"
        if "ifrs" in name:
            return "IFRS"
        if "frs102" in name and "1a" in name:
            return "FRS102_1A"
        if "frs102" in name:
            return "FRS102"
        return workbook_path.stem.upper().replace(" ", "_")


def parse_workbook(workbook_path: str | Path) -> list[ChecklistItem]:
    return ChecklistWorkbookParser().parse(workbook_path)

