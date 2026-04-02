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
from app.evaluation.retrieval_rerank import (
    final_rank_score,
    heading_match_score,
    keyword_overlap_score,
    section_hint_score,
    token_set,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TopChunk:
    chunk_id: str
    page_number: int
    text_redacted: str
    heading_guess: str
    similarity: float
    section_title: str = ""
    statement_area: str = ""
    chunk_type: str = ""
    note_number: str = ""


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


def _section_hints_by_item_key(checklist_type_key: str) -> dict[str, list[str]]:
    try:
        rows = _paginate_eq(
            "checklist_items",
            "item_key,section_hints",
            "checklist_type_key",
            checklist_type_key,
        )
    except Exception as exc:
        logger.warning(
            "Could not read checklist_items.section_hints (%s). Falling back to empty hints.",
            exc,
        )
        rows = _paginate_eq("checklist_items", "item_key", "checklist_type_key", checklist_type_key)
        return {str(r["item_key"]): [] for r in rows}
    out: dict[str, list[str]] = {}
    for row in rows:
        ik = str(row["item_key"])
        raw = row.get("section_hints")
        if raw is None:
            out[ik] = []
        elif isinstance(raw, list):
            out[ik] = [str(x).strip().lower() for x in raw if str(x).strip()]
        elif isinstance(raw, str):
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


_EMPTY = TopChunk(chunk_id="", page_number=0, text_redacted="", heading_guess="", similarity=0.0)


def _chunk_search_text(meta: dict[str, Any]) -> str:
    return " ".join(
        str(meta.get(key) or "")
        for key in ("heading_guess", "section_title", "statement_area", "chunk_type", "note_number", "text_redacted")
        if str(meta.get(key) or "").strip()
    ).strip()


def _metadata_bias_indices(
    *,
    section_hints: list[str],
    chunk_metas: list[dict[str, Any]],
    limit: int,
) -> np.ndarray:
    if not section_hints:
        return np.array([], dtype=np.int64)
    scored: list[tuple[float, int]] = []
    for idx, meta in enumerate(chunk_metas):
        heading_guess = str(meta.get("heading_guess") or "")
        chunk_text = " ".join(
            str(meta.get(key) or "")
            for key in ("section_title", "statement_area", "chunk_type", "note_number", "text_redacted")
        )
        score = section_hint_score(section_hints, heading_guess, chunk_text)
        if score > 0:
            scored.append((score, idx))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return np.array([idx for _, idx in scored[: max(1, limit)]], dtype=np.int64)
def _normalize_keyword_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    return []


def _rank_and_select_candidates(
    *,
    rule_vec: np.ndarray,
    candidate_indices: np.ndarray,
    chunk_mat_norm: np.ndarray,
    chunk_metas: list[dict[str, Any]],
    chunk_ids_in_order: list[str],
    keywords: list[str],
    section_hints: list[str],
    final_k: int,
) -> list[TopChunk]:
    if candidate_indices.size == 0:
        return [_EMPTY] * final_k

    selected_idx, semantic_scores = _top_k_cosine_on_columns(
        rule_vec,
        chunk_mat_norm,
        candidate_indices,
        candidate_indices.size,
    )
    ranked: list[tuple[float, TopChunk]] = []
    seen_ids: set[str] = set()
    for raw_idx, sem in zip(selected_idx.tolist(), semantic_scores.tolist()):
        meta = chunk_metas[int(raw_idx)]
        chunk_id = chunk_ids_in_order[int(raw_idx)]
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        heading_guess = str(meta.get("heading_guess") or "")
        chunk_text = str(meta.get("text_redacted") or "")
        chunk_search_text = _chunk_search_text(meta)
        chunk_tokens = token_set(chunk_search_text)
        key_score = keyword_overlap_score(keywords, chunk_tokens)
        heading_score = heading_match_score(keywords, heading_guess)
        section_score = section_hint_score(section_hints, heading_guess, chunk_search_text)
        final_score = final_rank_score(
            semantic_similarity=float(sem),
            keyword_overlap=min(1.0, key_score + 0.35 * section_score),
            heading_match=max(heading_score, section_score),
        )
        ranked.append(
            (
                final_score,
                TopChunk(
                    chunk_id=chunk_id,
                    page_number=int(meta.get("page_number") or 0),
                    text_redacted=str(meta.get("text_redacted") or ""),
                    heading_guess=heading_guess,
                    similarity=float(sem),
                    section_title=str(meta.get("section_title") or ""),
                    statement_area=str(meta.get("statement_area") or ""),
                    chunk_type=str(meta.get("chunk_type") or ""),
                    note_number=str(meta.get("note_number") or ""),
                ),
            )
        )
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    out = [chunk for _, chunk in ranked[:final_k]]
    while len(out) < final_k:
        out.append(_EMPTY)
    return out


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
    candidate_k: int = 12,
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
    candidate_count = max(k, int(candidate_k))
    use_kw = keyword_prefilter

    chunk_text_rows = _paginate_eq(
        "analysis_chunks",
        "id,page_number,text_redacted,heading_guess,section_title,statement_area,chunk_type,note_number",
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
    section_hints_map = _section_hints_by_item_key(checklist_type_key)

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

    chunk_texts = [
        _chunk_search_text(m)
        for m in chunk_metas
    ]
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
        section_hints = _normalize_keyword_list(section_hints_map.get(ik))

        lexical_miss = False
        global_idx, _ = _top_k_cosine_on_columns(rule_vec, chunk_mat, all_col_idx, candidate_count)
        metadata_idx = _metadata_bias_indices(
            section_hints=section_hints,
            chunk_metas=chunk_metas,
            limit=candidate_count,
        )
        if use_kw and keywords:
            cand_set = candidate_indices_for_keywords(keywords, inverted)
            if not cand_set:
                lexical_miss = True
                col_idx = np.unique(np.concatenate([metadata_idx, global_idx])) if metadata_idx.size else global_idx
            else:
                shortlist_idx = np.array(sorted(cand_set), dtype=np.int64)
                shortlist_top_idx, _ = _top_k_cosine_on_columns(
                    rule_vec,
                    chunk_mat,
                    shortlist_idx,
                    candidate_count,
                )
                if shortlist_top_idx.size < candidate_count:
                    extra = [shortlist_top_idx, global_idx]
                    if metadata_idx.size:
                        extra.append(metadata_idx)
                    col_idx = np.unique(np.concatenate(extra))
                else:
                    col_idx = (
                        np.unique(np.concatenate([shortlist_top_idx, metadata_idx]))
                        if metadata_idx.size
                        else shortlist_top_idx
                    )
        else:
            col_idx = np.unique(np.concatenate([metadata_idx, global_idx])) if metadata_idx.size else global_idx

        row_chunks = _rank_and_select_candidates(
            rule_vec=rule_vec,
            candidate_indices=col_idx,
            chunk_mat_norm=chunk_mat,
            chunk_metas=chunk_metas,
            chunk_ids_in_order=chunk_ids_in_order,
            keywords=keywords,
            section_hints=section_hints,
            final_k=k,
        )
        tops_per_item.append(row_chunks[:k])
        lexical_miss_flags.append(lexical_miss)

    return item_keys, tops_per_item, lexical_miss_flags
