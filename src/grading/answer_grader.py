"""
Answer-level self-correction graders (FR-10 groundedness, FR-11 relevance).

Per IMPLEMENTATION_PLAN.md §3's error taxonomy: malformed grader output
gets one retry with a stricter "return only valid JSON matching this
schema" reprompt; if still malformed, fail closed. Phase 3's document
grader fails closed to AMBIGUOUS because that state exists in its
three-way schema. Answer grading has no AMBIGUOUS state, so failing
closed here means defaulting the score to 0.0 — treat an unparseable
grade as a failing grade, never as a passing one. A broken grader must
never silently wave an ungrounded or off-topic answer through as if it
had been verified.

Both graders run through llm_clients/router.py's call_with_failover — the
same seam Phase 3's document grader uses, so retry/backoff/failover
behavior is identical across every grading call site (NFR-10).
"""
from __future__ import annotations

import json

from pydantic import BaseModel

from config import get_config
from llm_clients.router import call_with_failover
from logging_config import get_logger
from schemas.answer_grade import GroundednessGrade, RelevanceGrade

logger = get_logger(trace_id="answer_grader")

_GROUNDEDNESS_SYSTEM_PROMPT = (
    "You are a strict groundedness grader for a RAG system.\n"
    "Given a CONTEXT and an ANSWER, determine whether every factual claim "
    "in the ANSWER is directly supported by the CONTEXT. Do not reward "
    "plausible-sounding claims that are not actually stated in the "
    "CONTEXT — an answer that \"sounds right\" but adds unsupported "
    "specifics must be scored low.\n\n"
    "Respond with ONLY a JSON object, no other text, no markdown fences, "
    "matching exactly:\n"
    '{"score": <float between 0.0 and 1.0, where 1.0 means fully grounded '
    'and 0.0 means entirely unsupported>, "reasoning": "<one sentence '
    'explaining the score>"}'
)

_GROUNDEDNESS_RETRY_SYSTEM_PROMPT = _GROUNDEDNESS_SYSTEM_PROMPT + (
    "\n\nYour previous response could not be parsed. Return ONLY the raw "
    "JSON object above and nothing else — no markdown code fences, no "
    "preamble, no trailing text."
)

_RELEVANCE_SYSTEM_PROMPT = (
    "You are a strict relevance grader for a RAG system.\n"
    "Given an ORIGINAL QUESTION and an ANSWER, determine whether the "
    "ANSWER actually addresses what was asked — not whether it's "
    "factually correct, only whether it's on-topic. An answer can be "
    "fully grounded in real content and still fail this check if it "
    "dodges, over-generalizes, or fails to address the specific question "
    "asked.\n\n"
    "Respond with ONLY a JSON object, no other text, no markdown fences, "
    "matching exactly:\n"
    '{"score": <float between 0.0 and 1.0, where 1.0 means fully '
    'addresses the question and 0.0 means entirely off-topic>, '
    '"reasoning": "<one sentence explaining the score>"}'
)

_RELEVANCE_RETRY_SYSTEM_PROMPT = _RELEVANCE_SYSTEM_PROMPT + (
    "\n\nYour previous response could not be parsed. Return ONLY the raw "
    "JSON object above and nothing else — no markdown code fences, no "
    "preamble, no trailing text."
)


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = [
            line for line in text.split("\n") if not line.strip().startswith("```")
        ]
        text = "\n".join(lines).strip()
    return text


async def _grade_with_fail_closed(
    system_prompt: str,
    retry_system_prompt: str,
    user_prompt: str,
    model_cls: type[BaseModel],
    trace_id: str,
) -> BaseModel:
    cfg = get_config()
    slug_pair = cfg.model_tiers.tier1_grading

    last_error: Exception | None = None
    for attempt, prompt in enumerate((system_prompt, retry_system_prompt)):
        try:
            raw = await call_with_failover(
                slug_pair, prompt, user_prompt, trace_id=trace_id
            )
            parsed = json.loads(_strip_fences(raw))
            return model_cls(**parsed)
        except Exception as exc:  # malformed JSON, schema mismatch, etc.
            last_error = exc
            logger.warning(
                "answer_grade_parse_failed",
                trace_id=trace_id,
                attempt=attempt,
                error=str(exc),
            )
            continue

    logger.error(
        "answer_grade_fail_closed",
        trace_id=trace_id,
        reason="malformed_output_after_retry",
        last_error=str(last_error),
    )
    return model_cls(
        score=0.0,
        reasoning="Grading failed after retry — defaulted to fail-closed (score 0.0).",
    )


async def grade_groundedness(
    answer: str, context: str, trace_id: str = "groundedness_grade"
) -> GroundednessGrade:
    """FR-10. Scores whether `answer` is supported by `context`.
    Fail-closed on malformed grader output — never defaults to a passing
    score."""
    user_prompt = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    grade = await _grade_with_fail_closed(
        _GROUNDEDNESS_SYSTEM_PROMPT,
        _GROUNDEDNESS_RETRY_SYSTEM_PROMPT,
        user_prompt,
        GroundednessGrade,
        trace_id,
    )
    return grade  # type: ignore[return-value]


async def grade_relevance(
    answer: str, original_query: str, trace_id: str = "relevance_grade"
) -> RelevanceGrade:
    """FR-11. Scores whether `answer` addresses `original_query`.
    Fail-closed on malformed grader output — never defaults to a passing
    score."""
    user_prompt = f"ORIGINAL QUESTION:\n{original_query}\n\nANSWER:\n{answer}"
    grade = await _grade_with_fail_closed(
        _RELEVANCE_SYSTEM_PROMPT,
        _RELEVANCE_RETRY_SYSTEM_PROMPT,
        user_prompt,
        RelevanceGrade,
        trace_id,
    )
    return grade  # type: ignore[return-value]


def passes_answer_gate(
    groundedness: GroundednessGrade, relevance: RelevanceGrade
) -> bool:
    """§3.4's accept condition: both scores must independently clear their
    configured thresholds (default 0.7/0.7, config-driven per FR-21).
    Callers needing the three-way branch (both pass / relevance fails /
    groundedness fails) should inspect the two scores directly rather than
    relying on this boolean alone — Phase 7's orchestration owns that
    branch logic."""
    cfg = get_config()
    return (
        groundedness.score >= cfg.thresholds.groundedness_threshold
        and relevance.score >= cfg.thresholds.relevance_threshold
    )