"""
Top-k chunks per checklist item by cosine similarity.

Embeddings live in pgvector tables but are fetched as rows and scored in NumPy
(not the ``match_analysis_chunks`` RPC). No LLM tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.db.supabase import get_supabase_client
from app.evaluation.embedding_vector import embedding_to_float_vector


@dataclass(slots=True)
class TopChunk:
    chunk_id: str
    page_number: int
    text_redacted: str
    similarity: float


def _paginate_eq(
    table: str,
    select: str,
    eq_col: str,
    eq_val: Any,
    page_size: int = 500,
) -> list[dict[str, Any]]:
    client = get_supabase_client()
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        resp = (
            client.table(table)
            .select(select)
            .eq(eq_col, eq_val)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _rows_to_matrix(rows: list[dict[str, Any]], embedding_key: str = "embedding") -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = [embedding_to_float_vector(row[embedding_key]) for row in rows]
    return np.stack(vectors, axis=0)


_EMPTY = TopChunk(chunk_id="", page_number=0, text_redacted="", similarity=0.0)


def match_checklist_items_top_k(
    analysis_id: str,
    checklist_type_key: str,
    top_k: int,
) -> tuple[list[str], list[list[TopChunk]]]:
    """
    For every checklist item with an embedding, return the ``top_k`` best matching
    chunks (by cosine similarity), strongest first. Parallel lists: ``item_keys``,
    ``tops_per_item``.
    """
    k = max(1, int(top_k))

    chunk_text_rows = _paginate_eq(
        "analysis_chunks",
        "id,page_number,text_redacted",
        "analysis_id",
        analysis_id,
    )
    text_by_id: dict[str, dict[str, Any]] = {str(r["id"]): r for r in chunk_text_rows}

    checklist_rows = _paginate_eq(
        "checklist_item_embeddings",
        "item_key,embedding",
        "checklist_type_key",
        checklist_type_key,
    )
    if not checklist_rows:
        return [], []

    item_keys = [str(r["item_key"]) for r in checklist_rows]

    emb_rows = _paginate_eq(
        "analysis_chunk_embeddings",
        "chunk_id,embedding",
        "analysis_id",
        analysis_id,
    )
    if not emb_rows:
        return item_keys, [[_EMPTY] * k for _ in item_keys]

    chunk_ids_in_order: list[str] = []
    chunk_metas: list[dict[str, Any]] = []
    chunk_embs: list[Any] = []
    for row in emb_rows:
        cid = str(row["chunk_id"])
        meta = text_by_id.get(cid)
        if meta is None:
            continue
        chunk_ids_in_order.append(cid)
        chunk_metas.append(meta)
        chunk_embs.append(row["embedding"])

    if not chunk_embs:
        return item_keys, [[_EMPTY] * k for _ in item_keys]

    chunk_mat = np.stack([embedding_to_float_vector(v) for v in chunk_embs], axis=0)
    chunk_mat = _l2_normalize_rows(chunk_mat)

    checklist_mat = _rows_to_matrix(checklist_rows, "embedding")
    checklist_mat = _l2_normalize_rows(checklist_mat)

    sims = checklist_mat @ chunk_mat.T
    n_items, n_chunks = sims.shape
    k_eff = min(k, n_chunks)

    tops_per_item: list[list[TopChunk]] = []
    if k_eff == n_chunks:
        for i in range(n_items):
            order = np.argsort(-sims[i])
            row_chunks: list[TopChunk] = []
            for j in order:
                meta = chunk_metas[int(j)]
                row_chunks.append(
                    TopChunk(
                        chunk_id=chunk_ids_in_order[int(j)],
                        page_number=int(meta.get("page_number") or 0),
                        text_redacted=str(meta.get("text_redacted") or ""),
                        similarity=float(sims[i, j]),
                    )
                )
            tops_per_item.append(row_chunks)
    else:
        part = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
        for i in range(n_items):
            idx_row = part[i]
            scores = sims[i, idx_row]
            order = np.argsort(-scores)
            sorted_idx = idx_row[order]
            row_chunks = []
            for j in sorted_idx:
                ji = int(j)
                meta = chunk_metas[ji]
                row_chunks.append(
                    TopChunk(
                        chunk_id=chunk_ids_in_order[ji],
                        page_number=int(meta.get("page_number") or 0),
                        text_redacted=str(meta.get("text_redacted") or ""),
                        similarity=float(sims[i, j]),
                    )
                )
            tops_per_item.append(row_chunks)

    return item_keys, tops_per_item
