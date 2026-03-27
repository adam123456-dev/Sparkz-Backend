from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import get_settings


def create_embeddings(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    vectors: list[list[float]] = []
    for index in range(0, len(texts), batch_size):
        vectors.extend(_embed_batch(texts[index : index + batch_size]))
    return vectors


def _embed_batch(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    key = (settings.openai_api_key or "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is empty. Set it in the server environment "
            "(e.g. Render Dashboard → Environment → add OPENAI_API_KEY, then redeploy)."
        )
    payload = {
        "model": settings.openai_embedding_model,
        "input": texts,
    }
    request = Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:  # nosec B310
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        snippet = ""
        if exc.fp is not None:
            try:
                snippet = exc.fp.read().decode("utf-8", errors="replace")[:800]
            except Exception:
                pass
        if exc.code == 401:
            raise RuntimeError(
                "OpenAI returned 401 Unauthorized: invalid or missing API key. "
                "On Render, set Environment variable OPENAI_API_KEY (no quotes, no `Bearer ` prefix, full sk-...). "
                "Redeploy after saving. Rotate the key if it was revoked."
            ) from exc
        raise RuntimeError(
            f"OpenAI embeddings failed with HTTP {exc.code}: {snippet or exc.reason}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAI embeddings network error: {exc.reason}") from exc

    data = body.get("data", [])
    return [entry["embedding"] for entry in data]

