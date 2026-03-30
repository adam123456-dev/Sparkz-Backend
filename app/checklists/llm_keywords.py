from __future__ import annotations

import json
from urllib.request import Request, urlopen


def generate_rule_keywords_with_openai(
    *,
    api_key: str,
    model: str,
    requirement_text: str,
    requirement_text_leaf: str = "",
    reference_text: str = "",
    max_keywords: int = 18,
) -> list[str]:
    """
    Return one sharp keyword/phrase list for one disclosure rule.
    """
    req = (requirement_text or "").strip()
    leaf = (requirement_text_leaf or "").strip()
    ref = (reference_text or "").strip()
    if not req:
        return []

    system = (
        "You extract sharp compliance keywords for UK financial disclosure rules.\n"
        "Return JSON ONLY with key: keywords.\n"
        "keywords must be an array of short strings/phrases.\n"
        "Rules:\n"
        "- Use concrete high-signal terms likely to appear in accounts.\n"
        "- Avoid citations, statute codes, URLs, and generic terms.\n"
        "- Keep concise; no duplicates.\n"
    )
    user = (
        f"Requirement (full): {req}\n"
        f"Requirement (leaf): {leaf}\n"
        f"Reference: {ref}\n\n"
        f"Return up to {max_keywords} keywords."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 280,
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
    return _normalize_keywords(data.get("keywords"), max_keywords)


def _normalize_keywords(raw: object, max_count: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        text = str(item or "").strip().lower()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max_count:
            break
    return out

