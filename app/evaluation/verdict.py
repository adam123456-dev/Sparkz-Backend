"""Parse the judge model JSON body: {\"status\": \"FULLY|PARTIAL|NONE\"}."""

from __future__ import annotations

import json

_STATUS_MAP = {"FULLY": "fully_met", "PARTIAL": "partially_met", "NONE": "missing"}


def parse_judge_response(raw: str, *, explanation_max_chars: int = 320) -> tuple[str | None, str | None]:
    """
    Returns ``(api_status, explanation)`` or ``(None, None)`` if the body is not
    valid JSON with a recognized ``status``.
    """
    text = (raw or "").strip()
    if not text:
        return None, None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(data, dict):
        return None, None
    key = str(data.get("status", "")).strip().upper()
    status = _STATUS_MAP.get(key)
    if not status:
        return None, None
    return status, None
