"""
Single call site every other module uses to reach an LLM (FR-19's
tiering + NFR-10's cross-provider failover). Callers pass a ModelSlugPair
from config.get_config().model_tiers.* — this module doesn't know or
care which tier it's serving, only how to try primary, then fail over to
fallback on exhausted retries.
"""
from __future__ import annotations

from config import ModelRef, ModelSlugPair
from llm_clients.groq_client import call_groq
from llm_clients.openrouter_client import call_openrouter
from logging_config import get_logger
from resilience import RetriesExhaustedError, with_retry


async def _dispatch(model_ref: ModelRef, system_prompt: str, user_prompt: str) -> str:
    if model_ref.provider == "groq":
        return await call_groq(model_ref.model, system_prompt, user_prompt)
    if model_ref.provider == "openrouter":
        return await call_openrouter(model_ref.model, system_prompt, user_prompt)
    raise ValueError(f"Unknown provider: {model_ref.provider!r}")


async def call_with_failover(
    slug_pair: ModelSlugPair,
    system_prompt: str,
    user_prompt: str,
    trace_id: str = "llm_call",
) -> str:
    logger = get_logger(trace_id=trace_id)

    try:
        return await with_retry(
            lambda: _dispatch(slug_pair.primary, system_prompt, user_prompt)
        )
    except RetriesExhaustedError as exc:
        logger.warning(
            "primary_provider_failed_over",
            stage="llm_call",
            primary_provider=slug_pair.primary.provider,
            primary_model=slug_pair.primary.model,
            fallback_provider=slug_pair.fallback.provider,
            fallback_model=slug_pair.fallback.model,
            error=str(exc),
        )
        return await with_retry(
            lambda: _dispatch(slug_pair.fallback, system_prompt, user_prompt)
        )