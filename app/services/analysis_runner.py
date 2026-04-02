from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.db.supabase import get_supabase_client
from app.db.supabase_retry import execute_with_retry
from app.evaluation.check_evidence import select_chunks_for_check, select_evidence_for_check
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
            "section_title": chunk.section_title or "",
            "statement_area": chunk.statement_area or "",
            "chunk_type": chunk.chunk_type or "",
            "note_number": chunk.note_number or "",
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


_NO_RETRIEVED_TEXT = "(No matching document text.)"

_NO_KEYWORD_MSG = "No document passages matched the rule keyword filter."
_NO_USABLE_TEXT_MSG = "Retrieval did not return usable document text for this requirement."


def _synthesize_row_explanation(check_results: list[dict[str, Any]], *, max_chars: int) -> str | None:
    """Compact deterministic summary from per-check results only."""
    if not check_results:
        return None
    total = len(check_results)
    full = sum(1 for r in check_results if r.get("status") == "fully_met")
    partial = sum(1 for r in check_results if r.get("status") == "partially_met")
    missing = sum(1 for r in check_results if r.get("status") == "missing")

    if total == 1:
        reason = str(check_results[0].get("reason") or "").strip()
        if not reason:
            return None
        return reason[: max_chars - 1].rstrip() + "…" if len(reason) > max_chars else reason

    if full == total:
        text = f"All {total} atomic checks matched."
    elif missing == total:
        text = f"All {total} atomic checks missing."
    else:
        parts: list[str] = []
        if full:
            parts.append(f"{full} met")
        if partial:
            parts.append(f"{partial} partial")
        if missing:
            parts.append(f"{missing} missing")
        text = f"{', '.join(parts)} out of {total} checks."
    return text[: max_chars - 1].rstrip() + "…" if len(text) > max_chars else text


def _uniform_reason(check_results: list[dict[str, Any]]) -> str | None:
    reasons = [str(r.get("reason") or "").strip() for r in check_results]
    if not reasons:
        return None
    first = reasons[0]
    if first and all(r == first for r in reasons):
        return first
    return None


def _row_explanation(
    *,
    has_lexical_candidates: bool,
    lexical_miss: bool,
    check_results: list[dict[str, Any]],
    max_chars: int,
) -> str | None:
    if not has_lexical_candidates:
        return _NO_KEYWORD_MSG
    uniform = _uniform_reason(check_results)
    if uniform:
        return uniform
    if lexical_miss:
        compact = _synthesize_row_explanation(check_results, max_chars=max_chars)
        if compact:
            return f"Keyword shortlist missed; fallback retrieval used. {compact}"
    return _synthesize_row_explanation(check_results, max_chars=max_chars)


def _truncate_text(text: str, max_chars: int) -> str:
    s = (text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


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
    Lexical keyword gate → cosine top-k on candidate chunks → OpenAI per-atomic-check verdict.

    **Evidence is always extractive** (``build_evidence_blocks`` from retrieved chunks).
    The chat model must not invent document facts; it returns status, a short reason tied
    to the evidence, and a confidence score. Row ``explanation`` is assembled only from
    those per-check reasons (or a deterministic message when retrieval found nothing).
    """
    settings = get_settings()
    item_keys, tops_per_item, lexical_miss = match_checklist_items_top_k(
        analysis_id,
        checklist_type_key,
        top_k=settings.evaluation_top_k,
        candidate_k=settings.evaluation_candidate_k,
        keyword_prefilter=settings.evaluation_keyword_prefilter,
    )
    if not item_keys:
        logger.warning("No checklist embeddings for type_key=%s", checklist_type_key)
        return

    req_by_key = requirement_text_by_item_key(checklist_type_key, item_keys)
    rule_checks_by_key = _rule_checks_by_item_key(checklist_type_key, item_keys)
    if not settings.openai_api_key.strip():
        raise RuntimeError("OPENAI_API_KEY is required for final rule judgment.")

    results_payload = []
    for item_key, tops, kw_miss in zip(item_keys, tops_per_item, lexical_miss):
        merged, best_sim = _merge_evidence_texts(tops, settings.evaluation_evidence_max_chars)
        has_retrieved_evidence = bool(merged.strip())
        evidence_json = build_evidence_blocks(tops)
        requirement = req_by_key.get(item_key, "")[: settings.evaluation_requirement_max_chars]
        evidence_for_model = merged if merged.strip() else _NO_RETRIEVED_TEXT
        checks = _normalize_rule_checks(rule_checks_by_key.get(item_key), fallback_requirement=requirement)
        check_results = _evaluate_checks_for_rule(
            checks=checks,
            requirement_text=requirement,
            evidence_text=evidence_for_model,
            evidence_chunks=tops,
            has_lexical_candidates=has_retrieved_evidence,
            openai_api_key=settings.openai_api_key,
            openai_chat_model=settings.openai_chat_model,
            explanation_max_chars=settings.evaluation_explanation_max_chars,
        )
        status, coverage = _aggregate_rule_status(check_results)
        explanation = _row_explanation(
            has_lexical_candidates=has_retrieved_evidence,
            lexical_miss=kw_miss,
            check_results=check_results,
            max_chars=settings.evaluation_row_explanation_max_chars,
        )

        evidence_snippet = _truncate_text(merged, 160) if merged.strip() and status != "missing" else None
        evidence_value = evidence_json if status != "missing" and evidence_json else None
        results_payload.append(
            {
                "analysis_id": analysis_id,
                "item_key": item_key,
                "status": status,
                "evidence_snippet": evidence_snippet,
                "evidence": evidence_value,
                "check_results": check_results,
                "coverage": coverage,
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


def _rule_checks_by_item_key(checklist_type_key: str, item_keys: list[str]) -> dict[str, list[dict[str, str]]]:
    if not item_keys:
        return {}
    supabase = get_supabase_client()
    out: dict[str, list[dict[str, str]]] = {}
    batch_size = 200
    for start in range(0, len(item_keys), batch_size):
        batch = item_keys[start : start + batch_size]
        rows = (
            supabase.table("checklist_items")
            .select("item_key,rule_checks")
            .eq("checklist_type_key", checklist_type_key)
            .in_("item_key", batch)
            .execute()
            .data
            or []
        )
        for row in rows:
            raw = row.get("rule_checks")
            if isinstance(raw, list):
                out[str(row["item_key"])] = [x for x in raw if isinstance(x, dict)]
            else:
                out[str(row["item_key"])] = []
    return out


def _normalize_rule_checks(raw: list[dict[str, str]] | None, *, fallback_requirement: str) -> list[dict[str, str]]:
    if not raw:
        label = (fallback_requirement or "").strip() or "Disclosure requirement"
        return [{"check_id": "c1", "label": label, "kind": "required"}]
    out: list[dict[str, str]] = []
    for idx, entry in enumerate(raw, start=1):
        label = str(entry.get("label") or "").strip()
        if not label:
            continue
        cid = str(entry.get("check_id") or f"c{idx}").strip() or f"c{idx}"
        kind = str(entry.get("kind") or "required").strip().lower()
        if kind not in {"required", "supporting"}:
            kind = "required"
        out.append({"check_id": cid, "label": label, "kind": kind})
    if not out:
        label = (fallback_requirement or "").strip() or "Disclosure requirement"
        return [{"check_id": "c1", "label": label, "kind": "required"}]
    return out


def _evaluate_checks_for_rule(
    *,
    checks: list[dict[str, str]],
    requirement_text: str,
    evidence_text: str,
    evidence_chunks: list[TopChunk],
    has_lexical_candidates: bool,
    openai_api_key: str,
    openai_chat_model: str,
    explanation_max_chars: int,
) -> list[dict[str, Any]]:
    trimmed = (evidence_text or "").strip()
    if not has_lexical_candidates:
        return [
            {
                "checkId": c["check_id"],
                "label": c["label"],
                "kind": c.get("kind", "required"),
                "status": "missing",
                "reason": _NO_KEYWORD_MSG,
                "confidence": None,
                "selectedChunkIds": [],
                "evidenceSnippet": None,
            }
            for c in checks
        ]
    if not trimmed or trimmed == _NO_RETRIEVED_TEXT:
        return [
            {
                "checkId": c["check_id"],
                "label": c["label"],
                "kind": c.get("kind", "required"),
                "status": "missing",
                "reason": _NO_USABLE_TEXT_MSG,
                "confidence": None,
                "selectedChunkIds": [],
                "evidenceSnippet": None,
            }
            for c in checks
        ]

    out: list[dict[str, Any]] = []
    for c in checks:
        check_label = c["label"]
        row_context = _truncate_text(requirement_text, 220) or requirement_text
        check_prompt = (
            f"Row context: {row_context}\n"
            f"Atomic check: {check_label}\n"
            "Judge only this atomic check against the evidence."
        )
        selected_chunks = select_chunks_for_check(
            check_label=check_label,
            chunks=evidence_chunks,
            max_chunks=2,
        )
        check_evidence = select_evidence_for_check(
            check_label=check_label,
            chunks=selected_chunks or evidence_chunks,
            max_chunks=2,
            max_chars=max(280, min(700, len(evidence_text) or 700)),
        )
        evidence_for_check = check_evidence or evidence_text
        selected_chunk_ids = [chunk.chunk_id for chunk in selected_chunks if chunk.chunk_id]
        evidence_snippet = _truncate_text(evidence_for_check, 180) if evidence_for_check.strip() else None
        try:
            raw = judge_disclosure(
                api_key=openai_api_key,
                model=openai_chat_model,
                requirement_text=check_prompt,
                evidence_text=evidence_for_check,
            )
            verdict = parse_judge_response(raw, explanation_max_chars=explanation_max_chars)
            if verdict.status:
                status = verdict.status
                reason = verdict.reason
                confidence = verdict.confidence
                if status == "missing" and not reason:
                    reason = "The excerpts do not support this atomic check."
            else:
                status = "missing"
                reason = "The model returned an invalid or empty judgment."
                confidence = None
        except Exception:  # noqa: BLE001
            logger.exception("LLM check evaluation failed for check_id=%s", c["check_id"])
            status = "missing"
            reason = "Automatic evaluation failed (upstream error)."
            confidence = None
        out.append(
            {
                "checkId": c["check_id"],
                "label": check_label,
                "kind": c.get("kind", "required"),
                "status": status,
                "reason": reason,
                "confidence": confidence,
                "selectedChunkIds": selected_chunk_ids,
                "evidenceSnippet": evidence_snippet,
            }
        )
    return out


def _aggregate_rule_status(check_results: list[dict[str, Any]]) -> tuple[str, float]:
    if not check_results:
        return "missing", 0.0
    weighted_total = 0.0
    weighted_score = 0.0
    required_results = [r for r in check_results if r.get("kind") != "supporting"]
    if not required_results:
        required_results = check_results
    for result in check_results:
        weight = 0.35 if result.get("kind") == "supporting" else 1.0
        weighted_total += weight
        if result.get("status") == "fully_met":
            weighted_score += weight
        elif result.get("status") == "partially_met":
            weighted_score += 0.5 * weight
    coverage = weighted_score / max(weighted_total, 1e-9)
    required_statuses = [str(r.get("status") or "") for r in required_results]
    if required_statuses and all(status == "fully_met" for status in required_statuses):
        return "fully_met", coverage
    if required_statuses and all(status == "missing" for status in required_statuses):
        return "missing", coverage
    return "partially_met", coverage

