"""When no chat model is used, map best chunk cosine similarity to a coarse status."""

from __future__ import annotations


def status_from_similarity(similarity: float) -> str:
    if similarity >= 0.83:
        return "fully_met"
    if similarity >= 0.72:
        return "partially_met"
    return "missing"
