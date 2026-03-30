"""
Lexical gating: map token -> chunk indices so each rule only vector-scores relevant chunks.

Operates on PII-redacted chunk text only (same text as stored in ``analysis_chunks``).
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)


def _iter_word_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN.finditer(text or "")]


def build_inverted_index(chunk_texts: list[str]) -> dict[str, set[int]]:
    """For each lowercase token, the set of chunk row indices where it appears."""
    inverted: dict[str, set[int]] = {}
    for idx, text in enumerate(chunk_texts):
        seen_local: set[str] = set()
        for w in _iter_word_tokens(text):
            if w in seen_local:
                continue
            seen_local.add(w)
            inverted.setdefault(w, set()).add(idx)
    return inverted


def candidate_indices_for_keywords(
    keywords: list[str],
    inverted: dict[str, set[int]],
) -> set[int]:
    """
    Union of chunks that contain **any** keyword token (OR semantics).

    Empty ``keywords`` means caller should treat as “no lexical filter” (use all chunks).
    """
    out: set[int] = set()
    for kw in keywords:
        k = (kw or "").strip().lower()
        if not k:
            continue
        if k in inverted:
            out |= inverted[k]
    return out
