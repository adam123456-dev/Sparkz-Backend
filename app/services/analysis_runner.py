from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.db.supabase import get_supabase_client
from app.db.supabase_retry import execute_with_retry
from app.evaluation.llm_judge import judge_disclosure
from app.evaluation.requirements import requirement_text_by_item_key
from app.evaluation.evidence_payload import build_evidence_blocks
from app.evaluation.retrieval import TopChunk, match_checklist_items_top_k
from app.evaluation.verdict import parse_judge_response
from app.pipeline.chunking import build_chunks_from_redacted_pages, chunk_texts
from app.pipeline.embeddings import create_embeddings
from app.pipeline.models import RedactedChunk
from app.pipeline.pdf_text import extract_pdf_pages
from app.pipeline.pii import redact_pii

logger = logging.getLogger(__name__)


STEP_TEMPLATE = [
    {"id": "ingestion", "label": "Document ingestion", "state": "waiting"},
    {"id": "redaction", "label": "PII redaction and chunking", "state": "waiting"},
    {"id": "embedding", "label": "Embedding generation", "state": "waiting"},
    {"id": "evaluation", "label": "Disclosure evaluation", "state": "waiting"},
]


def run_analysis_job(analysis_id: str, pdf_path: str, checklist_type_key: str) -> None:
    settings = get_settings()
    steps = _new_steps()

    try:
        _update_analysis(
            analysis_id,
            status="processing",
            progress=10,
            message="Extracting PDF content.",
            steps=_mark_step(steps, "ingestion", "in_progress"),
        )

        pages = extract_pdf_pages(pdf_path, enable_ocr=settings.enable_ocr)
        _upsert_document(analysis_id, Path(pdf_path).name, pdf_path, len(pages))

        _update_analysis(
            analysis_id,
            status="processing",
            progress=30,
            message="Redacting sensitive data and chunking.",
            steps=_mark_step(steps, "redaction", "in_progress"),
        )

        redacted_pages: list[str] = []
        for page_number, page_text in enumerate(pages, start=1):
            redacted_pages.append(redact_pii(page_text or ""))

        chunks = build_chunks_from_redacted_pages(redacted_pages)
        if not chunks:
            # Without chunks there is nothing to embed or match; evaluation would mark
            # every checklist row as "missing" and look like a retrieval bug.
            raise RuntimeError(
                "No extractable text from this PDF (zero chunks after extraction). "
                "Common causes: image-only pages with no text layer, or OCR not available. "
                "Enable OCR (ENABLE_OCR / pdf2image + Tesseract + Poppler) or use a PDF with selectable text."
            )
        _insert_chunks(analysis_id, chunks)
        _set_step_state(steps, "redaction", "completed")
        _set_step_state(steps, "embedding", "in_progress")

        _update_analysis(
            analysis_id,
            status="processing",
            progress=55,
            message="Generating document embeddings.",
            steps=steps,
        )

        chunk_vectors = create_embeddings(chunk_texts(chunks), batch_size=50) if chunks else []
        _insert_chunk_embeddings(analysis_id, chunks, chunk_vectors)

        _set_step_state(steps, "embedding", "completed")
        _set_step_state(steps, "evaluation", "in_progress")
        _update_analysis(
            analysis_id,
            status="processing",
            progress=75,
            message="Evaluating disclosure requirements.",
            steps=steps,
        )

        _evaluate_checklist(analysis_id, checklist_type_key)

        _set_step_state(steps, "evaluation", "completed")
        _update_analysis(
            analysis_id,
            status="completed",
            progress=100,
            message="Analysis completed.",
            steps=steps,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analysis job failed for %s", analysis_id)
        _update_analysis(
            analysis_id,
            status="failed",
            progress=100,
            message="Analysis failed.",
            steps=steps,
            error_message=str(exc),
        )


def _new_steps() -> list[dict[str, str]]:
    return [dict(step) for step in STEP_TEMPLATE]


def _mark_step(steps: list[dict[str, str]], step_id: str, state: str) -> list[dict[str, str]]:
    for step in steps:
        if step["id"] == step_id:
            step["state"] = state
    return steps


def _set_step_state(steps: list[dict[str, str]], step_id: str, state: str) -> None:
    for step in steps:
        if step["id"] == step_id:
            step["state"] = state
            return


def _update_analysis(
    analysis_id: str,
    status: str,
    progress: int,
    message: str,
    steps: list[dict[str, str]],
    error_message: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "progress": progress,
        "message": message,
        "steps": steps,
    }
    if error_message:
        payload["error_message"] = error_message
    get_supabase_client().table("analyses").update(payload).eq("id", analysis_id).execute()


def _upsert_document(analysis_id: str, original_filename: str, storage_path: str, page_count: int) -> None:
    get_supabase_client().table("analysis_documents").upsert(
        [
            {
                "analysis_id": analysis_id,
                "original_filename": original_filename,
                "storage_path": storage_path,
                "page_count": page_count,
            }
        ]
    ).execute()


def _insert_chunks(analysis_id: str, chunks: list[RedactedChunk]) -> None:
    payload = [
        {
            "analysis_id": analysis_id,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "text_redacted": chunk.text_redacted,
            "text_hash": chunk.text_hash,
            "heading_guess": chunk.heading_guess or "",
        }
        for chunk in chunks
    ]
    if payload:
        _upsert_in_batches(
            table_name="analysis_chunks",
            rows=payload,
            on_conflict="analysis_id,chunk_index",
            batch_size=200,
        )


def _insert_chunk_embeddings(
    analysis_id: str, chunks: list[RedactedChunk], vectors: list[list[float]]
) -> None:
    if not chunks or not vectors:
        return

    supabase = get_supabase_client()
    rows = (
        supabase.table("analysis_chunks")
        .select("id,chunk_index")
        .eq("analysis_id", analysis_id)
        .execute()
        .data
        or []
    )
    chunk_id_by_index = {int(row["chunk_index"]): row["id"] for row in rows}
    model = get_settings().openai_embedding_model

    payload = []
    for chunk, vector in zip(chunks, vectors):
        chunk_id = chunk_id_by_index.get(chunk.chunk_index)
        if not chunk_id:
            continue
        payload.append(
            {
                "chunk_id": chunk_id,
                "analysis_id": analysis_id,
                "model": model,
                "embedding": vector,
            }
        )
    if payload:
        _upsert_in_batches(
            table_name="analysis_chunk_embeddings",
            rows=payload,
            on_conflict="chunk_id",
            batch_size=50,
        )


def _merge_evidence_texts(chunks: list[TopChunk], max_chars: int) -> tuple[str, float]:
    """Join top chunk texts with optional page labels; return best cosine score."""
    if not chunks or not chunks[0].chunk_id:
        return "", 0.0
    best_sim = float(chunks[0].similarity)
    seen: set[str] = set()
    pieces: list[str] = []
    used = 0
    for ch in chunks:
        cid = ch.chunk_id
        if not cid or cid in seen:
            continue
        seen.add(cid)
        text = (ch.text_redacted or "").strip()
        if not text:
            continue
        page = int(ch.page_number or 0)
        label = f"Page {page}: " if page > 0 else ""
        block = f"{label}{text}"
        gap = 2 if pieces else 0
        if used + gap + len(block) > max_chars:
            room = max_chars - used - gap - len(label)
            if room > 80:
                pieces.append(f"{label}{text[:room]}")
            break
        pieces.append(block)
        used += gap + len(block)
    return "\n\n".join(pieces), best_sim


def _evaluate_checklist(analysis_id: str, checklist_type_key: str) -> None:
    """
    Lexical keyword gate → cosine top-k on candidate chunks → OpenAI final verdict.

    **Evidence is always extractive** (``build_evidence_blocks`` from retrieved chunks).
    The chat model is **never** the source of evidence; it only returns status/explanation
    from requirement text + retrieved evidence text.
    """
    settings = get_settings()
    item_keys, tops_per_item, lexical_miss = match_checklist_items_top_k(
        analysis_id,
        checklist_type_key,
        top_k=settings.evaluation_top_k,
        keyword_prefilter=settings.evaluation_keyword_prefilter,
    )
    if not item_keys:
        logger.warning("No checklist embeddings for type_key=%s", checklist_type_key)
        return

    req_by_key = requirement_text_by_item_key(checklist_type_key, item_keys)
    kw_by_key = _search_keywords_by_item_key(checklist_type_key, item_keys)
    if not settings.openai_api_key.strip():
        raise RuntimeError("OPENAI_API_KEY is required for final rule judgment.")

    results_payload = []
    for item_key, tops, kw_miss in zip(item_keys, tops_per_item, lexical_miss):
        merged, best_sim = _merge_evidence_texts(tops, settings.evaluation_evidence_max_chars)
        evidence_json = build_evidence_blocks(tops)
        requirement = req_by_key.get(item_key, "")[: settings.evaluation_requirement_max_chars]
        evidence_for_model = merged if merged.strip() else "(No matching document text.)"
        keywords = kw_by_key.get(item_key, [])
        keyword_score = _keyword_match_ratio(keywords, merged)
        semantic_score = max(0.0, min(1.0, float(best_sim)))
        combined_score = 0.5 * keyword_score + 0.5 * semantic_score

        if combined_score >= 0.65:
            status = "fully_met"
            explanation = None
        elif combined_score < 0.30:
            status = "missing"
            explanation = None
        else:
            try:
                raw = judge_disclosure(
                    api_key=settings.openai_api_key,
                    model=settings.openai_chat_model,
                    requirement_text=requirement,
                    evidence_text=evidence_for_model,
                )
                parsed_status, _ = parse_judge_response(
                    raw,
                    explanation_max_chars=settings.evaluation_explanation_max_chars,
                )
                status = parsed_status or "missing"
                explanation = None
            except Exception:  # noqa: BLE001
                logger.exception("LLM judge failed for item_key=%s; defaulting to missing", item_key)
                status = "missing"
                explanation = None

        evidence_snippet = merged[:900] if merged.strip() and status != "missing" else None
        evidence_value = evidence_json if status != "missing" and evidence_json else None
        results_payload.append(
            {
                "analysis_id": analysis_id,
                "item_key": item_key,
                "status": status,
                "evidence_snippet": evidence_snippet,
                "evidence": evidence_value,
                "explanation": explanation,
                "similarity": best_sim,
            }
        )

    if results_payload:
        _upsert_in_batches(
            table_name="analysis_results",
            rows=results_payload,
            on_conflict="analysis_id,item_key",
            batch_size=75,
        )


def _upsert_in_batches(
    table_name: str,
    rows: list[dict[str, Any]],
    on_conflict: str,
    batch_size: int,
    max_retries: int = 4,
) -> None:
    """
    PostgREST/Supabase has payload/connection limits; large upserts can trigger HTTP/2 stream resets.
    This helper batches writes and retries transient transport failures.
    """
    supabase = get_supabase_client()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        execute_with_retry(
            lambda b=batch: supabase.table(table_name).upsert(b, on_conflict=on_conflict).execute(),
            max_retries=max_retries,
            label=f"upsert {table_name} rows {start}-{start + len(batch) - 1}",
        )


def _search_keywords_by_item_key(checklist_type_key: str, item_keys: list[str]) -> dict[str, list[str]]:
    if not item_keys:
        return {}
    supabase = get_supabase_client()
    out: dict[str, list[str]] = {}
    batch_size = 200
    for start in range(0, len(item_keys), batch_size):
        batch = item_keys[start : start + batch_size]
        rows = (
            supabase.table("checklist_items")
            .select("item_key,search_keywords")
            .eq("checklist_type_key", checklist_type_key)
            .in_("item_key", batch)
            .execute()
            .data
            or []
        )
        for row in rows:
            raw = row.get("search_keywords")
            if isinstance(raw, list):
                out[str(row["item_key"])] = [str(x).strip().lower() for x in raw if str(x).strip()]
            else:
                out[str(row["item_key"])] = []
    return out


def _keyword_match_ratio(keywords: list[str], evidence_text: str) -> float:
    if not keywords:
        return 0.0
    hay = (evidence_text or "").lower()
    if not hay.strip():
        return 0.0
    matched = 0
    for kw in keywords:
        token = (kw or "").strip().lower()
        if token and token in hay:
            matched += 1
    return matched / max(1, len(keywords))

