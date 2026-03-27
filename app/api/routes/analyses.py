from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app.core.checklist_type_keys import resolve_framework_form_value
from app.core.requirement_order import requirement_id_sort_key
from app.core.config import get_settings
from app.db.supabase import get_supabase_client
from app.schemas.analysis import (
    AnalysisChecklistItem,
    AnalysisResultResponse,
    AnalysisStatusResponse,
    AnalysisStep,
    StartAnalysisResponse,
)
from app.services.analysis_runner import run_analysis_job

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post("", response_model=StartAnalysisResponse)
async def start_analysis(
    background_tasks: BackgroundTasks,
    companyName: str = Form(...),
    framework: str = Form(...),
    file: UploadFile = File(...),
) -> StartAnalysisResponse:
    try:
        type_key = resolve_framework_form_value(framework)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    analysis_id = str(uuid.uuid4())
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / f"{analysis_id}.pdf"
    content = await file.read()
    saved_path.write_bytes(content)

    steps = [
        {"id": "ingestion", "label": "Document ingestion", "state": "waiting"},
        {"id": "redaction", "label": "PII redaction and chunking", "state": "waiting"},
        {"id": "embedding", "label": "Embedding generation", "state": "waiting"},
        {"id": "evaluation", "label": "Disclosure evaluation", "state": "waiting"},
    ]
    get_supabase_client().table("analyses").insert(
        {
            "id": analysis_id,
            "company_name": companyName.strip(),
            "checklist_type_key": type_key,
            "status": "queued",
            "progress": 0,
            "message": "Queued for processing.",
            "steps": steps,
        }
    ).execute()

    background_tasks.add_task(run_analysis_job, analysis_id, str(saved_path), type_key)
    return StartAnalysisResponse(analysisId=analysis_id)


@router.get("/{analysis_id}/status", response_model=AnalysisStatusResponse)
def get_status(analysis_id: str) -> AnalysisStatusResponse:
    data = (
        get_supabase_client()
        .table("analyses")
        .select("id,status,progress,message,steps")
        .eq("id", analysis_id)
        .limit(1)
        .execute()
        .data
    )
    if not data:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    row = data[0]
    steps = [AnalysisStep(**step) for step in row.get("steps", [])]
    return AnalysisStatusResponse(
        analysisId=row["id"],
        progress=int(row["progress"]),
        status=row["status"],
        message=row["message"],
        steps=steps,
    )


@router.get("/{analysis_id}/result", response_model=AnalysisResultResponse)
def get_result(analysis_id: str) -> AnalysisResultResponse:
    supabase = get_supabase_client()
    analysis_rows = (
        supabase.table("analyses")
        .select("id,company_name,checklist_type_key,status")
        .eq("id", analysis_id)
        .limit(1)
        .execute()
        .data
    )
    if not analysis_rows:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    analysis = analysis_rows[0]
    if analysis["status"] != "completed":
        raise HTTPException(status_code=409, detail="Analysis not completed yet.")

    result_rows = (
        supabase.table("analysis_results")
        .select("item_key,status,evidence_snippet,explanation")
        .eq("analysis_id", analysis_id)
        .execute()
        .data
        or []
    )
    checklist_rows = (
        supabase.table("checklist_items")
        .select("item_key,requirement_id,requirement_text,sheet_name")
        .eq("checklist_type_key", analysis["checklist_type_key"])
        .execute()
        .data
        or []
    )
    checklist_rows.sort(
        key=lambda r: (
            (r.get("sheet_name") or "").lower(),
            requirement_id_sort_key(str(r.get("requirement_id") or "")),
        )
    )
    result_by_item = {row["item_key"]: row for row in result_rows}

    items: list[AnalysisChecklistItem] = []
    missing_count = 0
    partial_count = 0
    fully_count = 0
    for checklist in checklist_rows:
        result = result_by_item.get(checklist["item_key"])
        status = result["status"] if result else "missing"
        evidence = result.get("evidence_snippet") if result else None
        explanation = result.get("explanation") if result else None
        if status == "missing":
            missing_count += 1
        elif status == "partially_met":
            partial_count += 1
        else:
            fully_count += 1
        items.append(
            AnalysisChecklistItem(
                id=checklist["requirement_id"],
                itemKey=str(checklist["item_key"]),
                requirement=checklist["requirement_text"],
                status=status,
                evidence=evidence,
                explanation=explanation,
            )
        )

    return AnalysisResultResponse(
        analysisId=analysis_id,
        companyName=analysis["company_name"],
        framework=analysis["checklist_type_key"],
        total=len(items),
        missing=missing_count,
        partial=partial_count,
        fullyMet=fully_count,
        items=items,
    )

