from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.db.supabase import get_supabase_client
from app.evaluation.fallback import status_from_similarity
from app.evaluation.llm_judge import judge_disclosure
from app.evaluation.requirements import requirement_text_by_item_key
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
    Retrieve top-k chunks per checklist item (embeddings + NumPy), merge text for
    context, then set status via a short LLM verdict when configured, else cosine
    fallback (no chat tokens).
    """
    settings = get_settings()
    item_keys, tops_per_item = match_checklist_items_top_k(
        analysis_id,
        checklist_type_key,
        top_k=settings.evaluation_top_k,
    )
    if not item_keys:
        logger.warning("No checklist embeddings for type_key=%s", checklist_type_key)
        return

    req_by_key = requirement_text_by_item_key(checklist_type_key, item_keys)
    use_llm = bool(settings.evaluation_use_llm and settings.openai_api_key.strip())

    results_payload = []
    for item_key, tops in zip(item_keys, tops_per_item):
        merged, best_sim = _merge_evidence_texts(tops, settings.evaluation_evidence_max_chars)
        requirement = req_by_key.get(item_key, "")[: settings.evaluation_requirement_max_chars]
        evidence_for_model = merged if merged.strip() else "(No matching document text.)"

        explanation: str | None = None
        if use_llm:
            try:
                raw = judge_disclosure(
                    api_key=settings.openai_api_key,
                    model=settings.openai_chat_model,
                    requirement_text=requirement,
                    evidence_text=evidence_for_model,
                )
                parsed_status, explanation = parse_judge_response(
                    raw,
                    explanation_max_chars=settings.evaluation_explanation_max_chars,
                )
                status = parsed_status if parsed_status is not None else status_from_similarity(best_sim)
                if parsed_status is None:
                    explanation = None
            except Exception:  # noqa: BLE001
                logger.exception("LLM judge failed for item_key=%s; using similarity fallback", item_key)
                status = status_from_similarity(best_sim)
                explanation = None
        else:
            status = status_from_similarity(best_sim)

        if status == "missing":
            explanation = "No evidence found."

        evidence = merged[:900] if merged.strip() and status != "missing" else None
        results_payload.append(
            {
                "analysis_id": analysis_id,
                "item_key": item_key,
                "status": status,
                "evidence_snippet": evidence,
                "explanation": explanation,
                "similarity": best_sim,
            }
        )

    if results_payload:
        _upsert_in_batches(
            table_name="analysis_results",
            rows=results_payload,
            on_conflict="analysis_id,item_key",
            batch_size=200,
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
        attempt = 0
        while True:
            try:
                supabase.table(table_name).upsert(batch, on_conflict=on_conflict).execute()
                break
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt > max_retries:
                    raise
                # Exponential backoff for transient HTTP/2 resets/timeouts.
                sleep_s = min(8.0, 0.5 * (2 ** (attempt - 1)))
                logger.warning(
                    "Upsert retry %s/%s for table=%s batch=%s..%s due to %s",
                    attempt,
                    max_retries,
                    table_name,
                    start,
                    start + len(batch) - 1,
                    type(exc).__name__,
                )
                time.sleep(sleep_s)

