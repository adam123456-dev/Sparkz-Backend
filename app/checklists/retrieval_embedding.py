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
    Retrieval-oriented obligation wording only — richer than a raw leaf, but still
    aligned to the semantic space of disclosure prose.

    Keep parent context when the leaf is too short/generic on its own.
    """
    leaf = (requirement_text_leaf or "").strip()
    full = (requirement_text or "").strip()
    if not full and not leaf:
        return "disclosure requirement"
    if not leaf or leaf == full:
        return full or leaf

    leaf_words = len(leaf.rstrip(".;:").split())
    generic_leaf_prefixes = (
        "the ",
        "any ",
        "its ",
        "their ",
        "details of ",
        "amount of ",
    )
    should_keep_parent = leaf_words <= 5 or leaf.lower().startswith(generic_leaf_prefixes)
    if should_keep_parent:
        return f"Requirement context: {full}\nFocus: {leaf}"
    return leaf
