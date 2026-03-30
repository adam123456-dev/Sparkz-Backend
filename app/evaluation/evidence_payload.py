"""Structured evidence payloads for API / persistence (PII-redacted chunk text only)."""

from __future__ import annotations

from typing import Any

from app.evaluation.retrieval import TopChunk


def build_evidence_blocks(
    chunks: list[TopChunk],
    *,
    max_chars_per_text: int = 560,
) -> list[dict[str, Any]]:
    """Skip empty placeholders; trim long text for JSON responses."""
    out: list[dict[str, Any]] = []
    for ch in chunks:
        if not ch.chunk_id:
            continue
        text = (ch.text_redacted or "").strip()
        if len(text) > max_chars_per_text:
            text = text[: max_chars_per_text - 1] + "…"
        out.append(
            {
                "chunkId": ch.chunk_id,
                "pageNumber": int(ch.page_number or 0),
                "similarity": round(float(ch.similarity), 4),
                "text": text,
            }
        )
    return out
