from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


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


class CheckResult(BaseModel):
    checkId: str
    label: str
    kind: str | None = None
    status: str
    reason: str | None = None
    confidence: float | None = Field(
        default=None,
        description="Model self-reported certainty for this atomic check (0–1).",
    )
    selectedChunkIds: list[str] | None = None
    evidenceSnippet: str | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, value: object) -> float | None:
        if value is None:
            return None
        try:
            x = float(value)
        except (TypeError, ValueError):
            return None
        if x != x:  # NaN
            return None
        return max(0.0, min(1.0, x))


class AnalysisChecklistItem(BaseModel):
    id: str
    itemKey: str
    requirement: str
    status: str
    bestSimilarity: float | None = Field(
        default=None,
        description="Best retrieved chunk cosine vs rule (0–1), shown for evidence ranking context.",
    )
    coverage: float | None = None
    checkResults: list[CheckResult] | None = None
    evidence: str | None = None
    evidenceBlocks: list[EvidenceBlock] | None = None
    explanation: str | None = None
    needsReview: bool = Field(
        default=False,
        description="True when the item should be human-reviewed (low confidence, missing evidence, etc.).",
    )


class AnalysisResultResponse(BaseModel):
    analysisId: str
    companyName: str
    framework: str
    total: int
    missing: int
    partial: int
    fullyMet: int
    items: list[AnalysisChecklistItem]

