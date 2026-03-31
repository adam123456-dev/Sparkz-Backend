"""Parse the disclosure judge JSON: status, optional reason, optional confidence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_STATUS_MAP = {"FULLY": "fully_met", "PARTIAL": "partially_met", "NONE": "missing"}
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """Outcome of ``parse_judge_response``."""

    status: str | None
    reason: str | None
    confidence: float | None


def _clamp_confidence(raw: object) -> float | None:
    if raw is None:
        return None
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return max(0.0, min(1.0, x))


def _normalize_reason(text: str, max_chars: int) -> str | None:
    s = _WS_RE.sub(" ", (text or "").strip())
    if not s:
        return None
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def parse_judge_response(raw: str, *, explanation_max_chars: int = 320) -> JudgeVerdict:
    """
    Parse the model JSON body.

    Expected keys: ``status`` (FULLY|PARTIAL|NONE), optional ``reason``,
    optional ``confidence`` (0–1). Unknown ``status`` yields a verdict with
    all fields cleared (caller should treat as failure).
    """
    text = (raw or "").strip()
    if not text:
        return JudgeVerdict(None, None, None)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return JudgeVerdict(None, None, None)
    if not isinstance(data, dict):
        return JudgeVerdict(None, None, None)
    key = str(data.get("status", "")).strip().upper()
    status = _STATUS_MAP.get(key)
    if not status:
        return JudgeVerdict(None, None, None)
    reason = _normalize_reason(str(data.get("reason", "")), explanation_max_chars)
    confidence = _clamp_confidence(data.get("confidence"))
    return JudgeVerdict(status, reason, confidence)
