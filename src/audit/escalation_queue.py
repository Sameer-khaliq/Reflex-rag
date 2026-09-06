"""
Escalation queue interface (FR-15).

Phase 7 wires the *interface* only — every low-confidence terminal case
calls this function with the full final state so the call site never
needs to change once Phase 8 replaces the body with real persistence
(a database table, a message queue, whatever). For now this just logs
the escalation event; nothing is actually queued for human review yet.

REQUIREMENTS.md §0.2 gap #5: escalation is always asynchronous and
non-blocking — the system never holds a response waiting on human
review. This stub already returns essentially immediately (it's just a
log call); the real Phase 8 implementation must preserve that property
— a slow enqueue call would silently violate FR-14's "return the
best-effort answer" requirement.
"""
from __future__ import annotations

from logging_config import get_logger

logger = get_logger(trace_id="escalation_queue")


async def push_to_escalation_queue(final_state: dict) -> None:
    """Phase 7 stub — logs the escalation event. Phase 8 replaces this
    body with real persistence (DB row, queue message, etc.) without
    changing this function's signature or any call site that already
    calls it."""
    logger.warning(
        "escalation_queue_push",
        stage="escalation",
        original_query=final_state.get("original_query"),
        low_confidence_reason=final_state.get("low_confidence_reason"),
        iteration_count=final_state.get("iteration_count"),
        generation_attempts=final_state.get("generation_attempts"),
        fallback_used=final_state.get("fallback_used"),
    )