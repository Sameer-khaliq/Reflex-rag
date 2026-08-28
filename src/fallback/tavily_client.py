"""
Tavily fallback fetch (FR-7).

Adapted from the prior project's tavily_fallback.py. Kept: the retry-
wrapped Tavily search call and source tagging. Removed: is_low_confidence()
and needs_current_info(), which triggered fallback off a rerank-score
floor and a currency-keyword regex — a different decision mechanism than
this project's spec. Here, fallback triggers strictly off the CRAG
document-grader's p_correct aggregation (§3.2 of the implementation
plan): p_correct == 0 fires fallback directly; 0 < p_correct < 0.5 after
the iteration cap is exhausted fires it as a last resort. That decision
lives in the LangGraph orchestration node (§3.3's fallback_retrieval
branch), not in this module — this module only knows how to fetch and
tag, not when to be called.

Injection gating (FR-8) is NOT handled here and must not be skipped:
results returned by fetch_tavily_results() are raw, ungated web content.
The orchestration node must pass them through the injection guard
(fallback/injection_guard.py — not yet built, flagged separately) before
they reach generation context. NFR-11 requires zero exceptions to this.
"""
from __future__ import annotations

from typing import Any

from tavily import AsyncTavilyClient

from config import get_config
from logging_config import get_logger
from resilience import RetriesExhaustedError, with_retry

_tavily_client: AsyncTavilyClient | None = None


def _get_tavily_client() -> AsyncTavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = AsyncTavilyClient(api_key=get_config().settings.tavily_api_key)
    return _tavily_client


async def _search_tavily_raw(query: str) -> list[dict[str, Any]]:
    client = _get_tavily_client()
    response = await client.search(query=query, max_results=get_config().retrieval.dense_top_n)
    return response.get("results", [])


def _tag_source(items: list[dict], source: str) -> list[dict]:
    return [{**item, "source": source} for item in items]


async def fetch_tavily_results(
    query: str,
    trace_id: str = "tavily",
) -> dict:
    """
    Fetches Tavily results with retry/backoff. Caller (the orchestration
    node) decides whether to call this at all — see module docstring.

    Returns:
        {
            "results": [...],   # tagged source="web", UNGATED — must
                                 # pass through the injection guard before
                                 # reaching generation context
            "degraded": bool,   # True if Tavily failed after retries
        }
    """
    logger = get_logger(trace_id=trace_id)

    try:
        raw_results = await with_retry(lambda: _search_tavily_raw(query))
    except RetriesExhaustedError as exc:
        logger.warning("tavily_retries_exhausted", stage="tavily_fallback", error=str(exc))
        return {"results": [], "degraded": True}

    tagged = _tag_source(raw_results, "web")
    logger.info("tavily_fetch_complete", stage="tavily_fallback", num_results=len(tagged))
    return {"results": tagged, "degraded": False}