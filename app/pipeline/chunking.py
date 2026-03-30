from __future__ import annotations

import hashlib
from typing import Iterable

from app.pipeline.models import RedactedChunk


def _heading_guess(text: str, max_len: int = 120) -> str:
    """Short preview for UI / section hints (PII-redacted chunk text only)."""
    t = (text or "").strip()
    if not t:
        return ""
    if len(t) <= max_len:
        return t
    cut = t[:max_len].rsplit(" ", 1)[0]
    return cut if cut else t[:max_len]


def build_chunks_from_redacted_pages(
    redacted_pages: list[str],
    chunk_word_target: int = 220,
    chunk_word_overlap: int = 40,
) -> list[RedactedChunk]:
    chunks: list[RedactedChunk] = []
    chunk_index = 0

    for page_number, text in enumerate(redacted_pages, start=1):
        words = text.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_word_target)
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append(
                    RedactedChunk(
                        chunk_index=chunk_index,
                        page_number=page_number,
                        text_redacted=chunk_text,
                        text_hash=hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                        heading_guess=_heading_guess(chunk_text),
                    )
                )
                chunk_index += 1
            if end >= len(words):
                break
            start = max(0, end - chunk_word_overlap)
    return chunks


def chunk_texts(chunks: Iterable[RedactedChunk]) -> list[str]:
    return [chunk.text_redacted for chunk in chunks]

