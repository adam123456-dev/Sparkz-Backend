from __future__ import annotations

import json
from urllib.request import Request, urlopen

from app.core.config import get_settings


def create_embeddings(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    vectors: list[list[float]] = []
    for index in range(0, len(texts), batch_size):
        vectors.extend(_embed_batch(texts[index : index + batch_size]))
    return vectors


def _embed_batch(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    payload = {
        "model": settings.openai_embedding_model,
        "input": texts,
    }
    request = Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.openai_api_key}",
        },
        method="POST",
    )
    with urlopen(request, timeout=120) as response:  # nosec B310
        body = json.loads(response.read().decode("utf-8"))
    data = body.get("data", [])
    return [entry["embedding"] for entry in data]

