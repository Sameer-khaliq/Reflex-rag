"""
Single entrypoint for FR-7 + FR-8 together: fetch Tavily results, then
gate them through the injection guard, in that fixed order, as one call.

This function exists specifically so NFR-11's zero-exception policy is
structurally hard to violate — fetch_tavily_results() alone returns raw,
ungated content (see its docstring), and a caller reaching for it
directly has to consciously skip a step to reach generation with
ungated content. fetch_gated_fallback() is the one call the orchestration
node (§3.3's fallback_retrieval branch) should actually use.
"""
from __future__ import annotations

from fallback.injection_guard import injection_guard
from fallback.tavily_client import fetch_tavily_results
from logging_config import get_logger


async def fetch_gated_fallback(query: str, trace_id: str = "fallback") -> dict:
    """
    Returns:
        {
            "chunks": [...],        # clean, gated web results, tagged source="web"
            "tavily_degraded": bool,   # True if the Tavily call itself failed after retries
            "injection_flagged_count": int,
            "all_flagged": bool,    # True if fetch succeeded but every
                                     # result was flagged — treat identically
                                     # to zero corpus-equivalent results
                                     # (§3 error taxonomy), don't retry
                                     # against the same flagged content
        }
    """
    logger = get_logger(trace_id=trace_id)

    fetch_result = await fetch_tavily_results(query, trace_id=trace_id)

    if fetch_result["degraded"]:
        return {
            "chunks": [],
            "tavily_degraded": True,
            "injection_flagged_count": 0,
            "all_flagged": False,
        }

    gated = injection_guard(fetch_result["results"], trace_id=trace_id)

    logger.info(
        "fetch_gated_fallback_complete",
        stage="fallback_retrieval",
        clean=len(gated["clean"]),
        flagged=len(gated["flagged"]),
        all_flagged=gated["all_flagged"],
    )

    return {
        "chunks": gated["clean"],
        "tavily_degraded": False,
        "injection_flagged_count": len(gated["flagged"]),
        "all_flagged": gated["all_flagged"],
    }