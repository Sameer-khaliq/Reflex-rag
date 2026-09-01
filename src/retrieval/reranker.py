"""
Local cross-encoder reranking (FR-2) — runs entirely on CPU, no API
call.

Simplified from the prior project's version: that one ran a dedicated
worker thread with a manual queue to keep rerank calls off the asyncio
event loop. asyncio.to_thread() does the same job in three lines and is
the standard-library way to do it — the manual thread+queue pool bought
nothing extra at this project's scale and was cut, not because it was
wrong, just unneeded complexity here.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import torch
from sentence_transformers import CrossEncoder

from config import get_config
from logging_config import get_logger

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is not None:
        return _reranker

    torch.set_num_threads(min(4, torch.get_num_threads()))
    model_name = get_config().retrieval.rerank.model

    try:
        _reranker = CrossEncoder(
            model_name,
            max_length=256,
            device="cpu",
            local_files_only=True,
        )
    except (OSError, RuntimeError):
        _reranker = CrossEncoder(
            model_name,
            max_length=256,
            device="cpu",
            local_files_only=False,
        )

    if hasattr(_reranker, "model"):
        _reranker.model.eval()

    return _reranker


def preload_reranker() -> None:
    """Eager load and warmup — call once at process startup, not per-query."""
    logger = get_logger(trace_id="rerank")
    model = _get_reranker()
    try:
        with torch.inference_mode():
            model.predict([("warmup query", "warmup text")], show_progress_bar=False)
    except Exception as e:  # noqa: BLE001
        logger.debug("reranker_warmup_failed", error=str(e))


def _candidate_text(candidate: dict[str, Any]) -> str:
    payload = candidate.get("payload") or {}
    text = payload.get("text") or candidate.get("text") or candidate.get("content", "")
    max_chars = get_config().retrieval.rerank.max_candidate_chars
    return str(text)[:max_chars]


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int | None = None,
    trace_id: str = "rerank",
) -> list[dict[str, Any]]:
    logger = get_logger(trace_id=trace_id)

    if not candidates:
        return []

    pool_size = max(top_k or 0, get_config().retrieval.rerank.candidate_pool)
    pool = candidates[:pool_size]

    pairs = [(query, _candidate_text(c)) for c in pool]

    model = _get_reranker()
    start = time.perf_counter()

    try:
        with torch.inference_mode():
            scores = model.predict(
                pairs,
                batch_size=len(pairs),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning("rerank_inference_failed", stage="rerank", error=str(e))
        return pool[:top_k] if top_k is not None else pool

    elapsed_ms = (time.perf_counter() - start) * 1000

    scored = [
        {**candidate, "rerank_score": float(score)}
        for candidate, score in zip(pool, scores)
    ]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)

    result = scored[:top_k] if top_k is not None else scored

    logger.info(
        "rerank_complete",
        stage="rerank",
        num_candidates=len(pool),
        top_k=top_k,
        latency_ms=round(elapsed_ms, 2),
    )
    return result


async def rerank_async(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int | None = None,
    trace_id: str = "rerank",
) -> list[dict[str, Any]]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(rerank, query, candidates, top_k, trace_id),
            timeout=get_config().retrieval.rerank.timeout_s,
        )
    except TimeoutError:
        logger = get_logger(trace_id=trace_id)
        logger.warning(
            "rerank_timeout_fallback",
            stage="rerank",
            detail="Exceeded timeout cap, skipping rerank",
        )
        return candidates[:top_k] if top_k is not None else candidates