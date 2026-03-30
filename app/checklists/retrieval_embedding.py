"""
Text used for the **checklist side** of vector retrieval (OpenAI embeddings).

``embedding_text`` (framework, sheet, IDs, references) is great for audit trails
and storage, but it is a poor match for PDF chunks that contain plain statutory
wording. Embedding only the obligation text aligns cosine similarity with
“does this chunk read like the requirement?” much better than threshold games.
"""

from __future__ import annotations


def retrieval_embedding_source_text(
    *,
    requirement_text: str,
    requirement_text_leaf: str,
) -> str:
    """
    Short obligation wording only — same semantic space as typical disclosure prose.

    Prefer the leaf clause when present; otherwise full requirement text.
    """
    leaf = (requirement_text_leaf or "").strip()
    full = (requirement_text or "").strip()
    core = leaf if leaf else full
    if not core:
        return "disclosure requirement"
    return core
