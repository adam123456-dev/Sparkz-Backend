from __future__ import annotations

from functools import lru_cache

from httpx import Client as HttpxClient
from httpx import Timeout
from supabase import Client, ClientOptions, create_client

from app.core.config import get_settings


def _supabase_httpx_client() -> HttpxClient:
    # Force HTTP/1.1. HTTP/2 + connection reuse against Supabase/Cloudflare often
    # surfaces as httpx.RemoteProtocolError: Server disconnected on Windows.
    return HttpxClient(http2=False, timeout=Timeout(120.0, connect=30.0))


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    settings = get_settings()
    settings.validate_external_services()
    return create_client(
        settings.supabase_url,
        settings.supabase_service_role_key,
        options=ClientOptions(httpx_client=_supabase_httpx_client()),
    )

