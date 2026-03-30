"""
Hybrid retrieval: lexical keyword gate (union over chunk token index), then cosine
top-k among **candidate** chunks only.

Falls back to global vector match when a rule has **no** ``search_keywords`` (legacy rows).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.db.supabase import get_supabase_client
from app.db.supabase_retry import execute_with_retry
from app.evaluation.embedding_vector import embedding_to_float_vector
from app.evaluation.lexical import build_inverted_index, candidate_indices_for_keywords

logger = logging.getLogger(__name__)


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
        off = offset

        def _fetch() -> Any:
            return (
                client.table(table)
                .select(select)
                .eq(eq_col, eq_val)
                .range(off, off + page_size - 1)
                .execute()
            )

        resp = execute_with_retry(_fetch, label=f"paginate {table}@{eq_val} off={off}")
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


def _keywords_by_item_key(checklist_type_key: str) -> dict[str, list[str]]:
    try:
        rows = _paginate_eq(
            "checklist_items",
            "item_key,search_keywords",
            "checklist_type_key",
            checklist_type_key,
        )
    except Exception as exc:
        logger.warning(
            "Could not read checklist_items.search_keywords (%s). "
            "Falling back to item_key-only (empty keywords = full-chunk vector match). "
            "Run sql/migrations if this column is missing.",
            exc,
        )
        rows = _paginate_eq("checklist_items", "item_key", "checklist_type_key", checklist_type_key)
        return {str(r["item_key"]): [] for r in rows}
    out: dict[str, list[str]] = {}
    for row in rows:
        ik = str(row["item_key"])
        raw = row.get("search_keywords")
        if raw is None:
            out[ik] = []
        elif isinstance(raw, list):
            out[ik] = [str(x).strip().lower() for x in raw if str(x).strip()]
        elif isinstance(raw, str):
            # PostgREST / drivers sometimes return jsonb as a serialized string.
            s = raw.strip()
            if s.startswith("["):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        out[ik] = [str(x).strip().lower() for x in parsed if str(x).strip()]
                    else:
                        out[ik] = []
                except json.JSONDecodeError:
                    out[ik] = []
            else:
                out[ik] = []
        else:
            out[ik] = []
    return out


_EMPTY = TopChunk(chunk_id="", page_number=0, text_redacted="", similarity=0.0)


def _normalize_keyword_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    return []


def _top_k_cosine_on_columns(
    rule_vec: np.ndarray,
    chunk_mat_norm: np.ndarray,
    column_indices: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (column_indices ordered by score desc, scores) length up to k."""
    if column_indices.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    sub = chunk_mat_norm[column_indices]
    sims = rule_vec @ sub.T
    k_eff = min(max(1, k), sims.size)
    if k_eff >= sims.size:
        order = np.argsort(-sims)
    else:
        part = np.argpartition(-sims, kth=k_eff - 1)[:k_eff]
        order = part[np.argsort(-sims[part])]
    return column_indices[order], sims[order]


def match_checklist_items_top_k(
    analysis_id: str,
    checklist_type_key: str,
    top_k: int,
    *,
    keyword_prefilter: bool = True,
) -> tuple[list[str], list[list[TopChunk]], list[bool]]:
    """
    For each checklist item with an embedding, return top similar chunks among
    **lexically filtered** candidates when ``search_keywords`` exist.

    Returns:
        item_keys, tops_per_item, lexical_miss_flags — ``lexical_miss_flags[i]`` is
        True when the rule had non-empty keywords but **no** chunk matched any token.
    """
    k = max(1, int(top_k))
    use_kw = keyword_prefilter

    chunk_text_rows = _paginate_eq(
        "analysis_chunks",
        "id,page_number,text_redacted",
        "analysis_id",
        analysis_id,
    )
    text_by_id: dict[str, dict[str, Any]] = {str(r["id"]): r for r in chunk_text_rows}

    checklist_emb_rows = _paginate_eq(
        "checklist_item_embeddings",
        "item_key,embedding",
        "checklist_type_key",
        checklist_type_key,
    )
    if not checklist_emb_rows:
        return [], [], []

    keywords_map = _keywords_by_item_key(checklist_type_key)

    item_keys = [str(r["item_key"]) for r in checklist_emb_rows]

    emb_rows = _paginate_eq(
        "analysis_chunk_embeddings",
        "chunk_id,embedding",
        "analysis_id",
        analysis_id,
    )
    if not emb_rows:
        return item_keys, [[_EMPTY] * k for _ in item_keys], [False] * len(item_keys)

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
        return item_keys, [[_EMPTY] * k for _ in item_keys], [False] * len(item_keys)

    chunk_texts = [str(m.get("text_redacted") or "") for m in chunk_metas]
    inverted = build_inverted_index(chunk_texts)

    chunk_mat = np.stack([embedding_to_float_vector(v) for v in chunk_embs], axis=0)
    chunk_mat = _l2_normalize_rows(chunk_mat)
    n_chunks = chunk_mat.shape[0]
    all_col_idx = np.arange(n_chunks, dtype=np.int64)

    checklist_mat = _rows_to_matrix(checklist_emb_rows, "embedding")
    checklist_mat = _l2_normalize_rows(checklist_mat)
    n_items = checklist_mat.shape[0]

    tops_per_item: list[list[TopChunk]] = []
    lexical_miss_flags: list[bool] = []

    for i in range(n_items):
        ik = item_keys[i]
        rule_vec = checklist_mat[i]
        keywords = _normalize_keyword_list(keywords_map.get(ik))

        lexical_miss = False
        if use_kw and keywords:
            cand_set = candidate_indices_for_keywords(keywords, inverted)
            if not cand_set:
                lexical_miss = True
                tops_per_item.append([_EMPTY] * k)
                lexical_miss_flags.append(True)
                continue
            col_idx = np.array(sorted(cand_set), dtype=np.int64)
        else:
            col_idx = all_col_idx

        picked_idx, scores = _top_k_cosine_on_columns(rule_vec, chunk_mat, col_idx, k)
        row_chunks: list[TopChunk] = []
        for j in range(picked_idx.size):
            ji = int(picked_idx[j])
            meta = chunk_metas[ji]
            row_chunks.append(
                TopChunk(
                    chunk_id=chunk_ids_in_order[ji],
                    page_number=int(meta.get("page_number") or 0),
                    text_redacted=str(meta.get("text_redacted") or ""),
                    similarity=float(scores[j]),
                )
            )
        while len(row_chunks) < k:
            row_chunks.append(_EMPTY)
        tops_per_item.append(row_chunks[:k])
        lexical_miss_flags.append(lexical_miss)

    return item_keys, tops_per_item, lexical_miss_flags
