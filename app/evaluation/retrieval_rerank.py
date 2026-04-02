from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)


def token_set(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}


def keyword_overlap_score(keywords: list[str], chunk_tokens: set[str]) -> float:
    if not keywords or not chunk_tokens:
        return 0.0
    hits = 0
    total = 0
    for kw in keywords:
        toks = token_set(kw)
        if not toks:
            continue
        total += 1
        if toks <= chunk_tokens:
            hits += 1
    if total == 0:
        return 0.0
    return hits / total


def heading_match_score(keywords: list[str], heading_guess: str) -> float:
    heading_tokens = token_set(heading_guess)
    if not heading_tokens or not keywords:
        return 0.0
    return keyword_overlap_score(keywords, heading_tokens)


def section_hint_score(section_hints: list[str], heading_guess: str, chunk_text: str) -> float:
    if not section_hints:
        return 0.0
    haystack_tokens = token_set(f"{heading_guess} {chunk_text}")
    if not haystack_tokens:
        return 0.0
    return keyword_overlap_score(section_hints, haystack_tokens)


def final_rank_score(
    *,
    semantic_similarity: float,
    keyword_overlap: float,
    heading_match: float,
) -> float:
    return (
        0.65 * float(semantic_similarity)
        + 0.22 * float(keyword_overlap)
        + 0.13 * float(heading_match)
    )
