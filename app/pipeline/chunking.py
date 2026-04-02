from __future__ import annotations

import hashlib
import re
from typing import Iterable

from app.pipeline.models import RedactedChunk

_HEADING_SPLIT_RE = re.compile(r"[.!?\n:;]+")
_NOISE_RE = re.compile(r"\[[A-Z_0-9]+\]|\b\d[\d,.\[\]]*\b|£m\b", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"\s+")
_LINE_SPLIT_RE = re.compile(r"\r?\n+")
_NOTE_RE = re.compile(r"^(note|notes)\s+[a-z0-9]+", re.IGNORECASE)
_NUMBERED_NOTE_RE = re.compile(r"^(\d+)\s+[A-Za-z]", re.IGNORECASE)
_NUMBERED_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*|[A-Z])[\)\.\-:]\s+\S")
_TABLEISH_RE = re.compile(r"(?:\s{2,}|\|\s*|\t)")
_NOTE_NUMBER_RE = re.compile(r"^(?:note\s+)?(\d+(?:\.\d+)*)\b", re.IGNORECASE)
_PRIMARY_STATEMENT_RE = re.compile(
    r"\b(balance sheet|statement of|income statement|cash flow|changes in equity)\b",
    re.IGNORECASE,
)
_PAGE_MARKER_RE = re.compile(r"^--\s*\d+\s+of\s+\d+\s*--$", re.IGNORECASE)
_PAGE_NUMBER_RE = re.compile(r"^\d{1,3}$")
_YEAR_ROW_RE = re.compile(r"^(?:\d{4}\s+){1,3}\d{4}$")
_CURRENCY_ROW_RE = re.compile(r"^(?:[£$€]\s*){1,4}$")
_YEAR_ENDED_RE = re.compile(r"^y\s*ear ended\b", re.IGNORECASE)


def _heading_guess(text: str, max_len: int = 120) -> str:
    """Best-effort section hint from the start of a chunk, not a raw preview."""
    t = (text or "").strip()
    if not t:
        return ""
    first_part = _HEADING_SPLIT_RE.split(t, maxsplit=1)[0].strip()
    candidate = first_part if first_part else t[:max_len]
    candidate = _NOISE_RE.sub(" ", candidate)
    candidate = _MULTISPACE_RE.sub(" ", candidate).strip(" -,:;()")
    words = candidate.split()
    if not words:
        candidate = _MULTISPACE_RE.sub(" ", t[:max_len]).strip()
        words = candidate.split()
    if len(words) > 12:
        candidate = " ".join(words[:12])
    if len(candidate) > max_len:
        candidate = candidate[:max_len].rsplit(" ", 1)[0] or candidate[:max_len]
    return candidate.strip()


def _normalize_line(line: str) -> str:
    return _MULTISPACE_RE.sub(" ", (line or "").strip())


def _has_letters(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def _is_probable_page_number(text: str) -> bool:
    return bool(_PAGE_NUMBER_RE.fullmatch(text.strip()))


def _is_trivial_noise_line(text: str) -> bool:
    normalized = _normalize_line(text)
    if not normalized:
        return True
    if _PAGE_MARKER_RE.fullmatch(normalized):
        return True
    if _CURRENCY_ROW_RE.fullmatch(normalized):
        return True
    return False


def _is_heading_line(line: str) -> bool:
    text = _normalize_line(line)
    if not text:
        return False
    if _is_probable_page_number(text):
        return False
    if _YEAR_ROW_RE.fullmatch(text) or _CURRENCY_ROW_RE.fullmatch(text):
        return False
    words = text.split()
    if len(words) > 12:
        return False
    if not _has_letters(text):
        return False
    if _NOTE_RE.match(text):
        return True
    if _NUMBERED_NOTE_RE.match(text):
        return True
    if _NUMBERED_HEADING_RE.match(text):
        return True
    if any(ch.isdigit() for ch in text):
        alpha_words = [word for word in words if any(ch.isalpha() for ch in word)]
        if len(alpha_words) <= 1:
            return False
    if text.isupper() and 1 <= len(words) <= 10 and not any(ch.isdigit() for ch in text):
        return True
    if text.endswith(":") and len(words) <= 12:
        return True
    title_like = (
        sum(1 for w in words if w[:1].isupper()) >= max(1, len(words) - 1)
        and not any(ch.isdigit() for ch in text)
    )
    if title_like and not text.endswith((".", ";")):
        return True
    return False


def _is_tableish_line(line: str) -> bool:
    text = _normalize_line(line)
    if not text:
        return False
    if _YEAR_ROW_RE.fullmatch(text) or _CURRENCY_ROW_RE.fullmatch(text):
        return True
    if _TABLEISH_RE.search(line):
        return True
    tokens = text.split()
    numericish = sum(1 for tok in tokens if re.fullmatch(r"[\[\]A-Z_0-9,.()%/-]+", tok, re.IGNORECASE))
    return len(tokens) >= 5 and numericish >= max(3, len(tokens) // 2)


def _infer_statement_area(*, heading_hint: str, text: str, page_number: int) -> str:
    heading = _normalize_line(heading_hint).lower()
    body = _normalize_line(text).lower()
    combined = f"{heading} {body}".strip()
    if _NOTE_RE.match(heading_hint) or heading.startswith("note ") or heading.startswith("notes "):
        return "notes"
    if _NUMBERED_NOTE_RE.match(_normalize_line(heading_hint)):
        return "notes"
    if _PRIMARY_STATEMENT_RE.search(combined):
        return "primary_statement"
    if page_number <= 2 and any(term in combined for term in ("contents", "independent auditor", "directors' report")):
        return "front_matter"
    return "notes" if "note" in combined else "other"


def _infer_chunk_type(text: str, heading_hint: str) -> str:
    lines = [_normalize_line(line) for line in _LINE_SPLIT_RE.split(text or "") if _normalize_line(line)]
    if heading_hint and not lines:
        return "heading_block"
    if lines and all(_is_tableish_line(line) for line in lines[: min(4, len(lines))]):
        return "table_like"
    return "narrative"


def _note_number_from_heading(heading_hint: str) -> str:
    heading = _normalize_line(heading_hint)
    if heading.lower().startswith("notes to "):
        return ""
    match = _NOTE_NUMBER_RE.match(heading)
    return (match.group(1) if match else "").strip()


def _normalize_extracted_text(text: str) -> str:
    normalized = (text or "").replace("\ufb01", "fi").replace("\ufb02", "fl").replace("\u2019", "'")
    normalized = _YEAR_ENDED_RE.sub("Year ended", normalized)
    return normalized


def _clean_page_lines(text: str) -> list[str]:
    raw_lines = [_normalize_line(line) for line in _LINE_SPLIT_RE.split(_normalize_extracted_text(text or ""))]
    lines = [line for line in raw_lines if line and not _is_trivial_noise_line(line)]
    while lines and _is_probable_page_number(lines[-1]):
        lines.pop()
    return lines


def _detect_repeated_edge_lines(pages: list[list[str]], *, edge_window: int = 3) -> tuple[set[str], set[str]]:
    head_counts: dict[str, int] = {}
    tail_counts: dict[str, int] = {}
    for lines in pages:
        if not lines:
            continue
        for line in lines[:edge_window]:
            head_counts[line] = head_counts.get(line, 0) + 1
        for line in lines[-edge_window:]:
            tail_counts[line] = tail_counts.get(line, 0) + 1
    repeated_heads = {line for line, count in head_counts.items() if count >= 3}
    repeated_tails = {line for line, count in tail_counts.items() if count >= 3}
    return repeated_heads, repeated_tails


def _strip_repeated_edge_lines(lines: list[str], repeated_heads: set[str], repeated_tails: set[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and lines[start] in repeated_heads:
        start += 1
    while end > start and lines[end - 1] in repeated_tails:
        end -= 1
    trimmed = lines[start:end]
    while trimmed and _is_probable_page_number(trimmed[-1]):
        trimmed.pop()
    return trimmed


def _iter_note_sections(lines: list[str]) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_lines
        if not current_lines:
            return
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, current_lines[:]))
        current_lines = []

    in_notes = False
    for line in lines:
        if line.upper() == "NOTES TO THE FINANCIAL STATEMENTS":
            _flush()
            in_notes = True
            current_heading = line
            continue
        if in_notes and _NUMBERED_NOTE_RE.match(line):
            _flush()
            current_heading = line
            current_lines = [line]
            continue
        if _is_heading_line(line) and not in_notes:
            _flush()
            current_heading = line.rstrip(":")
            continue
        current_lines.append(line)
    _flush()
    return [(heading, "\n".join(body_lines)) for heading, body_lines in sections if body_lines]


def _chunk_from_text(chunk_index: int, page_number: int, text: str, heading_hint: str) -> RedactedChunk:
    body = _normalize_line(text)
    heading = heading_hint or _heading_guess(body)
    section_title = heading
    statement_area = _infer_statement_area(heading_hint=heading, text=body, page_number=page_number)
    chunk_type = _infer_chunk_type(text, heading)
    note_number = _note_number_from_heading(heading)
    return RedactedChunk(
        chunk_index=chunk_index,
        page_number=page_number,
        text_redacted=body,
        text_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        heading_guess=heading,
        section_title=section_title,
        statement_area=statement_area,
        chunk_type=chunk_type,
        note_number=note_number,
    )


def _emit_windowed_chunks(
    *,
    chunk_index: int,
    page_number: int,
    text: str,
    heading_hint: str,
    chunk_word_target: int,
    chunk_word_overlap: int,
) -> tuple[list[RedactedChunk], int]:
    words = text.split()
    if not words:
        return [], chunk_index
    out: list[RedactedChunk] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_word_target)
        chunk_text = " ".join(words[start:end]).strip()
        if chunk_text:
            out.append(_chunk_from_text(chunk_index, page_number, chunk_text, heading_hint))
            chunk_index += 1
        if end >= len(words):
            break
        start = max(0, end - chunk_word_overlap)
    return out, chunk_index


def _split_page_into_sections(text: str) -> list[tuple[str, str]]:
    lines = _clean_page_lines(text)
    if not lines:
        return []
    if any(line.upper() == "NOTES TO THE FINANCIAL STATEMENTS" for line in lines):
        note_sections = _iter_note_sections(lines)
        if note_sections:
            return note_sections
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_note_heading = ""
    current_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_lines
        if not current_lines:
            return
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, current_lines[:]))
        current_lines = []

    for line in lines:
        if _is_heading_line(line):
            _flush()
            normalized = _normalize_line(line).rstrip(":")
            if _NOTE_RE.match(normalized):
                current_note_heading = normalized
                current_heading = normalized
            elif current_note_heading and _NUMBERED_NOTE_RE.match(normalized):
                current_heading = normalized
            elif current_note_heading and current_note_heading.lower().startswith("note "):
                current_heading = f"{current_note_heading} - {normalized}"
            else:
                current_heading = normalized
            continue
        current_lines.append(line)
    _flush()
    if not sections:
        return [("", "\n".join(lines))]
    return [(heading, "\n".join(body_lines)) for heading, body_lines in sections]


def build_chunks_from_redacted_pages(
    redacted_pages: list[str],
    chunk_word_target: int = 320,
    chunk_word_overlap: int = 60,
) -> list[RedactedChunk]:
    chunks: list[RedactedChunk] = []
    chunk_index = 0
    cleaned_pages = [_clean_page_lines(text) for text in redacted_pages]
    repeated_heads, repeated_tails = _detect_repeated_edge_lines(cleaned_pages)

    for page_number, lines in enumerate(cleaned_pages, start=1):
        page_lines = _strip_repeated_edge_lines(lines, repeated_heads, repeated_tails)
        text = "\n".join(page_lines)
        if not text.strip():
            continue
        sections = _split_page_into_sections(text)
        if not sections:
            continue
        for heading_hint, section_text in sections:
            body = section_text.strip()
            if not body:
                continue
            lines = [_normalize_line(line) for line in _LINE_SPLIT_RE.split(body) if _normalize_line(line)]
            if lines and all(_is_tableish_line(line) for line in lines[: min(4, len(lines))]):
                table_text = "\n".join(lines)
                chunks.append(_chunk_from_text(chunk_index, page_number, table_text, heading_hint))
                chunk_index += 1
                continue
            section_chunks, chunk_index = _emit_windowed_chunks(
                chunk_index=chunk_index,
                page_number=page_number,
                text=body,
                heading_hint=heading_hint,
                chunk_word_target=chunk_word_target,
                chunk_word_overlap=chunk_word_overlap,
            )
            chunks.extend(section_chunks)
    return chunks


def chunk_texts(chunks: Iterable[RedactedChunk]) -> list[str]:
    return [chunk.embedding_text for chunk in chunks]

