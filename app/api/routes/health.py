from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.health import HealthResponse
from app.schemas.supabase import SupabaseHealthResponse
from app.services.supabase_health import check_supabase_connectivity

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", app=settings.app_name, env=settings.app_env)


@router.get("/health/supabase", response_model=SupabaseHealthResponse)
def health_supabase() -> SupabaseHealthResponse:
    result = check_supabase_connectivity()
    return SupabaseHealthResponse(
        connected=result.connected,
        status_code=result.status_code,
        message=result.message,
    )

