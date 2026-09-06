from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class RetriesExhaustedError(Exception):
    pass

import re

def _extract_retry_delay(exc: Exception, default_delay: float) -> float:
    msg = str(exc)
    match = re.search(r"(?:try again in|retry[ -]after|retry_after)\s*([0-9.]+)\s*s?", msg, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) + random.uniform(0.5, 1.5)
        except ValueError:
            pass
    return default_delay

async def with_retry[T](
    func: Callable[[], Awaitable[T]],
    max_retries: int | None = None,
    base_delay_s: float | None = None,
) -> T:
    if max_retries is None or base_delay_s is None:
        from config import get_config

        cfg = get_config().resilience
        max_retries = max_retries if max_retries is not None else cfg.max_retries
        base_delay_s = base_delay_s if base_delay_s is not None else cfg.base_delay_s

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_retries:
                break
            base_delay = base_delay_s * (2**attempt) + random.uniform(0, base_delay_s)
            delay = _extract_retry_delay(exc, base_delay)
            await asyncio.sleep(delay)
    raise RetriesExhaustedError(str(last_exc)) from last_exc