from __future__ import annotations

import json
from urllib.request import Request, urlopen

_GENERIC_PREFIXES = (
    "verify ",
    "check ",
    "ensure ",
    "confirm ",
)


def generate_rule_checks_with_openai(
    *,
    api_key: str,
    model: str,
    requirement_text: str,
    requirement_text_leaf: str = "",
    max_checks: int = 4,
) -> list[dict[str, str]]:
    """
    Decompose one disclosure rule into atomic checks.

    Returns list items shaped as:
    {"check_id": "c1", "label": "...", "kind": "required"}
    """
    req = (requirement_text or "").strip()
    leaf = (requirement_text_leaf or "").strip()
    if not req:
        return []

    system = (
        "You decompose accounting disclosure rules into atomic checks.\n"
        "Return JSON only with key: checks.\n"
        "checks is an array of objects with keys: check_id, label, kind.\n"
        "kind must be 'required' only.\n"
        "Rules:\n"
        "- Each check must be independently verifiable from document evidence.\n"
        "- Keep checks short, concrete, and non-overlapping.\n"
        "- Start each check label with a concrete subject, not verbs like verify/check/ensure.\n"
        "- Avoid meta checks (e.g., correct year, qualifies in general) unless explicit in rule text.\n"
        "- Do not include legal citations or references.\n"
    )
    user = (
        f"Requirement (full): {req}\n"
        f"Requirement (leaf): {leaf}\n\n"
        f"Return up to {max_checks} checks."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 320,
        "response_format": {"type": "json_object"},
    }
    request = Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(request, timeout=120) as response:  # nosec B310
        body = json.loads(response.read().decode("utf-8"))
    choices = body.get("choices") or []
    if not choices:
        return []
    content = str((choices[0].get("message") or {}).get("content") or "").strip()
    if not content:
        return []
    data = json.loads(content)
    if not isinstance(data, dict):
        return []
    checks = _normalize_rule_checks(data.get("checks"), max_checks)
    return _filter_low_signal_checks(checks)


def _normalize_rule_checks(raw: object, max_checks: int) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    seen_labels: set[str] = set()
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        low = label.lower()
        if low in seen_labels:
            continue
        seen_labels.add(low)
        check_id = str(item.get("check_id") or f"c{idx}").strip() or f"c{idx}"
        out.append({"check_id": check_id, "label": label, "kind": "required"})
        if len(out) >= max_checks:
            break
    return out


def _filter_low_signal_checks(checks: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for chk in checks:
        label = str(chk.get("label") or "").strip()
        if not label:
            continue
        low = label.lower()
        if low.startswith(_GENERIC_PREFIXES):
            continue
        if "correct financial year" in low:
            continue
        if "qualifies for" in low and "exemption" in low:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append({"check_id": chk["check_id"], "label": label, "kind": "required"})
    return out

