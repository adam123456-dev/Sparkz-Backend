from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import get_settings


@dataclass(slots=True)
class SupabaseHealthCheckResult:
    connected: bool
    status_code: int | None
    message: str


def check_supabase_connectivity(timeout_seconds: int = 8) -> SupabaseHealthCheckResult:
    settings = get_settings()
    settings.validate_external_services()

    endpoint = f"{settings.supabase_url.rstrip('/')}/rest/v1/"
    headers = {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Accept": "application/json",
    }
    request = Request(endpoint, headers=headers, method="GET")

    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
            status_code = response.getcode()
            _ = response.read()
            if status_code and 200 <= status_code < 300:
                return SupabaseHealthCheckResult(
                    connected=True,
                    status_code=status_code,
                    message="Supabase connectivity check succeeded.",
                )
            return SupabaseHealthCheckResult(
                connected=False,
                status_code=status_code,
                message=f"Supabase responded with non-success status code {status_code}.",
            )
    except HTTPError as error:
        body = ""
        try:
            payload = error.read().decode("utf-8", errors="ignore")
            body = json.loads(payload).get("message", payload)
        except Exception:
            body = "HTTP error during Supabase check."
        return SupabaseHealthCheckResult(
            connected=False,
            status_code=error.code,
            message=body.strip() or f"HTTP {error.code}",
        )
    except URLError as error:
        return SupabaseHealthCheckResult(
            connected=False,
            status_code=None,
            message=f"Network error while connecting to Supabase: {error.reason}",
        )

