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
CLAUSE_MARKER_RE = re.compile(r"^\(([A-Za-z0-9]+)\)\s*(.*)$")
# Single letters like (c) are clause markers, not Roman "100". Use an explicit list.
_ROMAN_SUBCLAUSE_TOKENS = frozenset(
    {
        "i",
        "ii",
        "iii",
        "iv",
        "v",
        "vi",
        "vii",
        "viii",
        "ix",
        "x",
        "xi",
        "xii",
        "xiii",
        "xiv",
        "xv",
        "xvi",
        "xvii",
        "xviii",
        "xix",
        "xx",
        "xxi",
        "xxii",
    }
)
TOP_META_PREFIXES = ("client", "year", "period", "file number", "prepared by", "reviewed by", "date")
NON_REQUIREMENT_PREFIXES = (
    "appendix",
    "main checklist",
    "further guidance",
    "audited accounts",
    "assurance review report",
    "political donations",
)
NOTE_PREFIXES = ("note", "notes")
GUIDANCE_PREFIXES = (
    "[guidance]",
    "guidance",
    "per the above",
    "table of equivalence",
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

        base_id = ""
        base_text = ""
        base_ref = ""
        alpha_token = ""
        alpha_text = ""
        had_roman_under_alpha = False
        alpha_child_count = 0
        note_lines: list[str] = []
        saw_atomic_for_base = False

        def emit_alpha_standalone() -> None:
            """Emit base_id + (alpha) when there are no (i)/(ii) children under that alpha."""
            nonlocal saw_atomic_for_base, alpha_token, alpha_text, had_roman_under_alpha, alpha_child_count
            if not base_id or not alpha_token:
                return
            if had_roman_under_alpha or alpha_child_count > 0:
                return
            clause_path = f"({alpha_token})"
            full_id = f"{base_id}{clause_path}"
            composed = self._compose_requirement_text(base_text, alpha_text, "")
            if not composed.strip():
                return
            items.append(
                ChecklistItem(
                    source_workbook=workbook_name,
                    framework=framework,
                    sheet_name=sheet_name,
                    section_path=section_path,
                    requirement_id=full_id,
                    requirement_text=composed,
                    requirement_text_leaf=alpha_text.strip(),
                    requirement_base_id=base_id,
                    clause_path=clause_path,
                    notes_text=" ".join(note_lines).strip(),
                    reference_text=base_ref,
                    item_kind="rule",
                )
            )
            saw_atomic_for_base = True
            alpha_token = ""
            alpha_text = ""
            had_roman_under_alpha = False
            alpha_child_count = 0

        def emit_alpha_list_leaf(leaf_text: str) -> None:
            nonlocal saw_atomic_for_base, had_roman_under_alpha, alpha_child_count
            if not base_id or not alpha_token:
                return
            alpha_child_count += 1
            token = str(alpha_child_count)
            clause_path = f"({alpha_token})({token})"
            full_id = f"{base_id}{clause_path}"
            composed = self._compose_requirement_text(base_text, alpha_text, leaf_text)
            if not composed.strip():
                return
            items.append(
                ChecklistItem(
                    source_workbook=workbook_name,
                    framework=framework,
                    sheet_name=sheet_name,
                    section_path=section_path,
                    requirement_id=full_id,
                    requirement_text=composed,
                    requirement_text_leaf=leaf_text.strip(),
                    requirement_base_id=base_id,
                    clause_path=clause_path,
                    notes_text=" ".join(note_lines).strip(),
                    reference_text=base_ref,
                    item_kind="rule",
                )
            )
            saw_atomic_for_base = True
            had_roman_under_alpha = False

        def flush_base_if_needed() -> None:
            nonlocal saw_atomic_for_base
            if not base_id:
                return
            emit_alpha_standalone()
            if saw_atomic_for_base:
                return
            if not base_text:
                return
            items.append(
                ChecklistItem(
                    source_workbook=workbook_name,
                    framework=framework,
                    sheet_name=sheet_name,
                    section_path=section_path,
                    requirement_id=base_id,
                    requirement_text=base_text,
                    requirement_text_leaf=base_text,
                    requirement_base_id=base_id,
                    clause_path="",
                    notes_text=" ".join(note_lines).strip(),
                    reference_text=base_ref,
                    item_kind="rule",
                )
            )

        for row in rows:
            first = self._pad_id_decimal_segments(self._normalize_requirement_id_candidate(row.first))
            second = row.second
            third = row.third
            row_text = self._row_text(first, second)

            if self._is_top_metadata_row(first):
                continue

            if self._looks_like_section_header(first, second):
                flush_base_if_needed()
                base_id = ""
                base_text = ""
                base_ref = ""
                alpha_token = ""
                alpha_text = ""
                had_roman_under_alpha = False
                alpha_child_count = 0
                note_lines = []
                saw_atomic_for_base = False
                section_path = self._build_section_path(section_path, first, second)
                continue

            if self._is_requirement_id(first):
                flush_base_if_needed()
                base_id = first
                base_text = second.strip()
                base_ref = third.strip()
                alpha_token = ""
                alpha_text = ""
                had_roman_under_alpha = False
                alpha_child_count = 0
                note_lines = []
                saw_atomic_for_base = False
                continue

            if not base_id:
                if first and not second and not third and not self._is_non_requirement_text(first):
                    section_path = self._build_section_path(section_path, first, "")
                continue

            if self._is_note_row(row_text):
                note_lines.append(row_text)
                continue

            token, token_text = self._extract_clause_token(row_text)
            if token:
                if self._is_alpha_token(token):
                    new_alpha = token.lower()
                    if alpha_token and new_alpha != alpha_token:
                        emit_alpha_standalone()
                    alpha_token = new_alpha
                    alpha_text = token_text
                    had_roman_under_alpha = False
                    alpha_child_count = 0
                    continue

                if self._is_roman_token(token):
                    if not alpha_token:
                        if row_text:
                            base_text = self._append_sentence(base_text, row_text)
                        if third and third not in base_ref:
                            base_ref = self._append_sentence(base_ref, third)
                        continue
                    clause_path = f"({alpha_token})({token.lower()})"
                    full_id = f"{base_id}{clause_path}"
                    composed_full = self._compose_requirement_text(base_text, alpha_text, token_text)
                    item = ChecklistItem(
                        source_workbook=workbook_name,
                        framework=framework,
                        sheet_name=sheet_name,
                        section_path=section_path,
                        requirement_id=full_id,
                        requirement_text=composed_full,
                        requirement_text_leaf=token_text,
                        requirement_base_id=base_id,
                        clause_path=clause_path,
                        notes_text=" ".join(note_lines).strip(),
                        reference_text=self._append_sentence(base_ref, third),
                        item_kind="rule",
                    )
                    items.append(item)
                    saw_atomic_for_base = True
                    had_roman_under_alpha = True
                    alpha_child_count += 1
                    continue

            if alpha_token and not had_roman_under_alpha and self._should_split_alpha_list_item(row):
                emit_alpha_list_leaf(row_text)
                if third and third not in base_ref:
                    base_ref = self._append_sentence(base_ref, third)
                continue

            if alpha_token and not had_roman_under_alpha and row_text:
                alpha_text = self._append_sentence(alpha_text, row_text)
                if third and third not in base_ref:
                    base_ref = self._append_sentence(base_ref, third)
                continue

            if saw_atomic_for_base and items and items[-1].requirement_base_id == base_id:
                extension_text = row_text
                if extension_text:
                    items[-1].requirement_text_leaf = self._append_sentence(items[-1].requirement_text_leaf, extension_text)
                    items[-1].requirement_text = self._append_sentence(items[-1].requirement_text, extension_text)
                if third and third not in items[-1].reference_text:
                    items[-1].reference_text = self._append_sentence(items[-1].reference_text, third)
                continue

            if row_text:
                base_text = self._append_sentence(base_text, row_text)
            if third and third not in base_ref:
                base_ref = self._append_sentence(base_ref, third)

        flush_base_if_needed()
        return [item for item in items if item.requirement_text]

    def _is_requirement_id(self, value: str) -> bool:
        return bool(REQUIREMENT_ID_RE.match(value.strip()))

    def _row_text(self, first: str, second: str) -> str:
        if first and second:
            if CLAUSE_MARKER_RE.match(first):
                return self._normalize_text(f"{first} {second}")
            return self._normalize_text(second)
        return self._normalize_text(first or second)

    def _is_note_row(self, text: str) -> bool:
        lower = text.lower().strip()
        if any(lower.startswith(prefix) for prefix in NOTE_PREFIXES):
            return True
        if self._is_guidance_like_text(text):
            return True
        if any(lower.startswith(prefix) for prefix in GUIDANCE_PREFIXES):
            return True
        if text.strip().startswith('"') and text.strip().endswith('"'):
            return True
        return False

    def _is_guidance_like_text(self, text: str) -> bool:
        lower = text.lower().strip()
        if not lower:
            return False
        if any(lower.startswith(prefix) for prefix in GUIDANCE_PREFIXES):
            return True
        if text.strip().startswith('"') and text.strip().endswith('"'):
            return True
        if "guidance only" in lower:
            return True
        if lower.startswith("a micro-entity may use titles for the financial statements other than"):
            return True
        return False

    def _extract_clause_token(self, text: str) -> tuple[str, str]:
        m = CLAUSE_MARKER_RE.match(text.strip())
        if not m:
            return "", ""
        token = m.group(1).strip()
        body = self._normalize_text(m.group(2))
        return token, body

    def _is_alpha_token(self, token: str) -> bool:
        t = token.strip().lower()
        return len(t) == 1 and "a" <= t <= "z" and not self._is_roman_token(t)

    def _is_roman_token(self, token: str) -> bool:
        t = token.strip().lower()
        return t in _ROMAN_SUBCLAUSE_TOKENS

    def _compose_requirement_text(self, base_text: str, alpha_text: str, leaf_text: str) -> str:
        full = base_text.strip()
        if alpha_text:
            full = self._append_sentence(full, alpha_text)
        full = self._append_sentence(full, leaf_text)
        return full.strip()

    def _should_split_alpha_list_item(self, row: RawRow) -> bool:
        first = row.first.strip()
        second = row.second.strip()
        if not second:
            return False
        if first:
            return False
        text = self._row_text(first, second)
        if not text or self._is_note_row(text) or self._is_guidance_like_text(text):
            return False
        if self._extract_clause_token(text)[0]:
            return False
        lower = text.lower()
        if any(word in lower for word in (" must ", " shall ", " should ", " may ", " is ", " are ")):
            return False
        return text.endswith(";") or len(text.split()) <= 8

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

    def _pad_id_decimal_segments(self, value: str) -> str:
        """
        Excel stores 6.10 as float 6.1 in XML. Format with two decimal places so
        6.1 -> 6.10 (not 6.01 from naive zero-padding of the last segment).
        """
        text = value.strip()
        if not text or not re.match(r"^\d+(?:\.\d+)+$", text):
            return value
        if re.match(r"^\d+\.\d{10,}$", text):
            return value
        try:
            if re.match(r"^\d+\.\d$", text):
                return f"{float(text):.2f}"
        except ValueError:
            return value
        return value

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

