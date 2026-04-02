from __future__ import annotations

import re
from typing import Protocol


class ChunkLike(Protocol):
    chunk_id: str
    page_number: int
    text_redacted: str
    heading_guess: str
    similarity: float
    section_title: str
    statement_area: str
    chunk_type: str
    note_number: str

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "of",
    "for",
    "to",
    "in",
    "on",
    "by",
    "or",
    "its",
    "any",
}


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "") if m.group(0).lower() not in _STOP}


def select_chunks_for_check(
    *,
    check_label: str,
    chunks: list[ChunkLike],
    max_chunks: int = 2,
) -> list[ChunkLike]:
    label_tokens = _tokens(check_label)
    ranked: list[tuple[float, ChunkLike]] = []
    for chunk in chunks:
        if not chunk.chunk_id:
            continue
        chunk_text = " ".join(
            part
            for part in (
                chunk.heading_guess,
                getattr(chunk, "section_title", ""),
                getattr(chunk, "statement_area", ""),
                getattr(chunk, "chunk_type", ""),
                getattr(chunk, "note_number", ""),
                chunk.text_redacted,
            )
            if part
        )
        chunk_tokens = _tokens(chunk_text)
        overlap = len(label_tokens & chunk_tokens)
        coverage = overlap / max(1, len(label_tokens))
        metadata_bonus = 0.0
        if getattr(chunk, "section_title", ""):
            metadata_bonus += 0.05 * (len(label_tokens & _tokens(chunk.section_title)) > 0)
        if getattr(chunk, "statement_area", ""):
            metadata_bonus += 0.03 * (len(label_tokens & _tokens(chunk.statement_area)) > 0)
        score = (0.67 * float(chunk.similarity)) + (0.28 * coverage) + metadata_bonus
        ranked.append((score, chunk))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[: max(1, max_chunks)]]


def select_evidence_for_check(
    *,
    check_label: str,
    chunks: list[ChunkLike],
    max_chunks: int = 2,
    max_chars: int = 700,
) -> str:
    selected = select_chunks_for_check(check_label=check_label, chunks=chunks, max_chunks=max_chunks)
    pieces: list[str] = []
    used = 0
    for chunk in selected:
        heading = f"Section: {chunk.section_title or chunk.heading_guess}\n" if (chunk.section_title or chunk.heading_guess) else ""
        page = f"Page {chunk.page_number}: " if chunk.page_number else ""
        block = f"{heading}{page}{chunk.text_redacted}".strip()
        gap = 2 if pieces else 0
        if used + gap + len(block) > max_chars:
            room = max_chars - used - gap
            if room > 80:
                pieces.append(block[:room].rstrip() + "…")
            break
        pieces.append(block)
        used += gap + len(block)
    return "\n\n".join(pieces).strip()
