"""
Stable ordering for checklist requirement_id strings (e.g. 1.01 < 6.01 < 6.01(a) < 6.01(a)(i)).

PostgREST returns rows in no guaranteed order; use this for API responses so UI matches workbook logic.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z]+\d*|\d+|\([a-z0-9]+\)", re.IGNORECASE)


def requirement_id_sort_key(requirement_id: str) -> tuple:
    """Tuple usable with sorted(..., key=...) for disclosure-style IDs."""
    parts: list[tuple[int, int | str]] = []
    for raw in _TOKEN.finditer(requirement_id.strip()):
        t = raw.group(0).lower()
        if t.startswith("(") and t.endswith(")"):
            parts.append((1, t))
        elif t.isdigit():
            parts.append((0, int(t)))
        else:
            parts.append((2, t))
    return tuple(parts)
