"""Retry transient httpx/httpcore failures against Supabase (HTTP/2 stream resets, etc.)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def execute_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 4,
    label: str = "supabase",
) -> T:
    """Exponential backoff; re-raises the last error after ``max_retries`` attempts."""
    last: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except BaseException as exc:
            last = exc
            if attempt >= max_retries:
                raise
            sleep_s = min(8.0, 0.5 * (2 ** (attempt - 1)))
            logger.warning(
                "%s attempt %s/%s failed (%s): %s; retrying in %.1fs",
                label,
                attempt,
                max_retries,
                type(exc).__name__,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)
    raise RuntimeError("execute_with_retry exhausted without result") from last
