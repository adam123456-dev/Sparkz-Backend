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


def generate_retrieval_hints_with_openai(
    *,
    api_key: str,
    model: str,
    requirement_text: str,
    requirement_text_leaf: str = "",
    reference_text: str = "",
    sheet_name: str = "",
    section_path: str = "",
    max_keywords: int = 10,
    max_section_hints: int = 4,
) -> dict[str, list[str]]:
    """
    Return direct retrieval hints for one disclosure rule.

    Shape:
    {"keywords": [...], "section_hints": [...]}
    """
    req = (requirement_text or "").strip()
    leaf = (requirement_text_leaf or "").strip()
    ref = (reference_text or "").strip()
    sheet = (sheet_name or "").strip()
    section = (section_path or "").strip()
    if not req:
        return {"keywords": [], "section_hints": []}

    system = (
        "You prepare retrieval hints for financial disclosure checklist rows.\n"
        "Return JSON ONLY with keys: keywords, section_hints.\n"
        "keywords: 3-10 short high-signal phrases likely to appear in accounts.\n"
        "section_hints: 1-4 likely headings or document zones.\n"
        "Rules:\n"
        "- Preserve the exact scope of the row, especially if Requirement (leaf) is a narrow atomic subclause.\n"
        "- Do not include sibling obligations or broader parent requirements unless needed as context.\n"
        "- Avoid generic terms such as date, period, financial statements, disclosure, details, notes.\n"
        "- Prefer phrases accountants would actually search for in a report.\n"
        "- section_hints should be short labels like accounting policies, balance sheet, income statement, directors, related party.\n"
        "- No duplicates.\n"
    )
    user = (
        f"Requirement (full): {req}\n"
        f"Requirement (leaf): {leaf}\n"
        f"Reference: {ref}\n"
        f"Sheet: {sheet}\n"
        f"Section path: {section}\n\n"
        f"Return up to {max_keywords} keywords and up to {max_section_hints} section_hints."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 260,
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
        return {"keywords": [], "section_hints": []}
    content = str((choices[0].get("message") or {}).get("content") or "").strip()
    if not content:
        return {"keywords": [], "section_hints": []}
    data = json.loads(content)
    if not isinstance(data, dict):
        return {"keywords": [], "section_hints": []}
    return {
        "keywords": _normalize_keywords(data.get("keywords"), max_keywords),
        "section_hints": _normalize_keywords(data.get("section_hints"), max_section_hints),
    }


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

