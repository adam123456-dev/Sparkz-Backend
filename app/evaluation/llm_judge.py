from __future__ import annotations

import json
import logging
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_SYSTEM = (
    "Return JSON only. Key: status (string, exactly FULLY or PARTIAL or NONE). "
    "FULLY if evidence clearly satisfies the requirement; PARTIAL if only some; "
    "NONE if evidence does not support it or is absent."
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
        'Respond with JSON: {"status":"FULLY|PARTIAL|NONE"}'
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": 96,
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
