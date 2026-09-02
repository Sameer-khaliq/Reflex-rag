"""
Answer-level self-correction grading (FR-10 groundedness, FR-11 relevance).

Both scores come from a single LLM call against the existing `AnswerGrade`
schema (schemas/answer_grade.py) — groundedness_score and relevance_score
are graded together in one pass, then gated independently downstream. A
single call rather than two is more efficient and still satisfies FR-10/
FR-11's requirement that the two be "scored and gated separately": the
schema keeps them as two distinct fields, and `classify_outcome()` below
branches on them independently, exactly per REQUIREMENTS.md §3.4's
decision tree (relevance failure and groundedness failure route
differently, and relevance is checked first).

Per IMPLEMENTATION_PLAN.md §3's error taxonomy: malformed grader output
gets one retry with a stricter "return only valid JSON matching this
schema" reprompt; if still malformed, fail closed. There is no AMBIGUOUS
state here (unlike Phase 3's document grader), so failing closed means
defaulting BOTH scores to 0.0 — an unparseable grade must never be treated
as a passing grade on either axis.

Runs through llm_clients/router.py's call_with_failover — the same seam
Phase 3's document grader uses (NFR-10).
"""
from __future__ import annotations

import json
from typing import Literal

from config import get_config
from llm_clients.router import call_with_failover
from logging_config import get_logger
from schemas.answer_grade import AnswerGrade

logger = get_logger(trace_id="answer_grader")

_SYSTEM_PROMPT = (
    "You are a strict answer grader for a RAG system. Given a CONTEXT, the "
    "ORIGINAL QUESTION, and a generated ANSWER, produce two independent "
    "scores:\n\n"
    "groundedness_score: whether every factual claim in the ANSWER is "
    "directly supported by the CONTEXT. Do not reward plausible-sounding "
    "claims that are not actually stated in the CONTEXT — an answer that "
    "\"sounds right\" but adds unsupported specifics must score low here, "
    "even if it correctly addresses the question.\n\n"
    "relevance_score: whether the ANSWER actually addresses the ORIGINAL "
    "QUESTION — not whether it's factually correct, only whether it's "
    "on-topic. Score this independently of groundedness: a fully grounded "
    "answer can still dodge or fail to address the question, and an "
    "on-topic answer can still be ungrounded.\n\n"
    "Respond with ONLY a JSON object, no other text, no markdown fences, "
    "matching exactly:\n"
    '{"groundedness_score": <float 0.0-1.0>, "relevance_score": '
    "<float 0.0-1.0>}"
)

_RETRY_SYSTEM_PROMPT = _SYSTEM_PROMPT + (
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


async def grade_answer(
    answer: str,
    context: str,
    original_query: str,
    trace_id: str = "answer_grade",
) -> AnswerGrade:
    """FR-10 + FR-11. Grades `answer` for groundedness against `context`
    and relevance against `original_query` in a single LLM call.

    Fail-closed on malformed output: one retry with a stricter reprompt,
    then both scores default to 0.0 (never to a passing score).
    """
    cfg = get_config()
    slug_pair = cfg.model_tiers.tier1_grading
    user_prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"ORIGINAL QUESTION:\n{original_query}\n\n"
        f"ANSWER:\n{answer}"
    )

    last_error: Exception | None = None
    for attempt, prompt in enumerate((_SYSTEM_PROMPT, _RETRY_SYSTEM_PROMPT)):
        try:
            raw = await call_with_failover(
                slug_pair, prompt, user_prompt, trace_id=trace_id
            )
            parsed = json.loads(_strip_fences(raw))
            return AnswerGrade(**parsed)
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
    return AnswerGrade(groundedness_score=0.0, relevance_score=0.0)


def classify_outcome(
    grade: AnswerGrade,
) -> Literal["accept", "relevance_fail", "groundedness_fail"]:
    """REQUIREMENTS.md §3.4's exact branch order:

        if groundedness >= 0.7 and relevance >= 0.7: accept
        elif relevance < 0.7: relevance_fail   (checked first)
        elif groundedness < 0.7: groundedness_fail

    Relevance is checked before groundedness because an off-topic answer
    signals a likely retrieval-adequacy problem (route back to rewrite),
    while an on-topic-but-ungrounded answer signals a generation problem
    (route to regenerate) — two different failure modes, two different
    corrective actions. Phase 7's orchestration wires each branch to its
    corrective node; this function only classifies, it doesn't route.
    """
    cfg = get_config()
    groundedness_ok = grade.groundedness_score >= cfg.thresholds.groundedness_threshold
    relevance_ok = grade.relevance_score >= cfg.thresholds.relevance_threshold

    if groundedness_ok and relevance_ok:
        return "accept"
    if not relevance_ok:
        return "relevance_fail"
    return "groundedness_fail"