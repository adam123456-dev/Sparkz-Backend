from __future__ import annotations

from pydantic import BaseModel, Field


class StartAnalysisResponse(BaseModel):
    analysisId: str


class AnalysisStep(BaseModel):
    id: str
    label: str
    state: str


class AnalysisStatusResponse(BaseModel):
    analysisId: str
    progress: int
    status: str
    message: str
    steps: list[AnalysisStep]


class EvidenceBlock(BaseModel):
    """Extractive snippet from a PII-redacted chunk (page + similarity)."""

    chunkId: str
    pageNumber: int
    similarity: float
    text: str


class AnalysisChecklistItem(BaseModel):
    id: str
    itemKey: str
    requirement: str
    status: str
    bestSimilarity: float | None = Field(
        default=None,
        description="Best retrieved chunk cosine vs rule (0–1), shown for evidence ranking context.",
    )
    evidence: str | None = None
    evidenceBlocks: list[EvidenceBlock] | None = None
    explanation: str | None = None


class AnalysisResultResponse(BaseModel):
    analysisId: str
    companyName: str
    framework: str
    total: int
    missing: int
    partial: int
    fullyMet: int
    items: list[AnalysisChecklistItem]

