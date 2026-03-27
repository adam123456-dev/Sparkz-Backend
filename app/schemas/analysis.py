from __future__ import annotations

from pydantic import BaseModel


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


class AnalysisChecklistItem(BaseModel):
    id: str
    itemKey: str
    requirement: str
    status: str
    evidence: str | None
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

