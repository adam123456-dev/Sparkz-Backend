from __future__ import annotations

import json
import logging
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You assess whether retrieved document excerpts satisfy a disclosure requirement. "
    "Return JSON only with keys: "
    'status (string, exactly FULLY or PARTIAL or NONE), '
    "reason (string, max 12 words), "
    "confidence (number from 0 to 1). "
    "FULLY if the evidence clearly covers the atomic check; PARTIAL if only partly; "
    "NONE if the evidence does not support it or is absent. "
    "The reason must only restate what is supported by the Evidence block below—do not "
    "use outside knowledge or invent facts. Keep the reason very short and direct. "
    "If status is NONE, say what is missing in a few words only."
)


def judge_disclosure(
    *,
    api_key: str,
    model: str,
    requirement_text: str,
    evidence_text: str,
) -> str:
    req = (requirement_text or "").strip()
    ev = (evidence_text or "").strip()
    user = (
        f"Requirement:\n{req}\n\n"
        "Evidence from document (retrieved chunks; each paragraph may start with \"Page N:\" "
        "for the PDF page that chunk came from):\n"
        f"{ev}\n\n"
        'Respond with JSON: {"status":"FULLY|PARTIAL|NONE","reason":"...","confidence":0.0}'
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": 80,
        "temperature": 0,
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
    try:
        with urlopen(request, timeout=120) as response:  # nosec B310
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("LLM judge request failed: %s", exc)
        raise

    choices = body.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "").strip()
