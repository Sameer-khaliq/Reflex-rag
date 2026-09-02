"""
CRAG-style per-chunk document grading (FR-4).

Each reranked chunk is graded CORRECT / AMBIGUOUS / INCORRECT relative
to the query by the tier1_grading model (config.model_tiers.tier1_grading,
Groq primary / OpenRouter fallback per llm_clients.router).

Deliberately does NOT trust the LLM to echo back the chunk_id correctly
— chunk_id is already known before the call, so only "grade" is parsed
from the response and validated against ChunkGrade's Literal constraint.
This sidesteps a whole failure class (LLM mistypes/truncates an ID) that
has nothing to do with whether the grade itself is trustworthy.

Two independent failure classes both fail closed to AMBIGUOUS, never to
CORRECT, per IMPLEMENTATION_PLAN.md §3: "A broken grader should never
silently pass bad content through as if it were verified."

  1. Malformed output: response isn't valid JSON matching ChunkGrade's
     schema. Gets one stricter reprompt; still malformed -> AMBIGUOUS.
  2. Call failure: both providers (Groq primary, OpenRouter fallback)
     exhaust their retries — llm_clients.router.call_with_failover only
     wraps the primary attempt in try/except, so a fallback exhaustion
     propagates as an uncaught RetriesExhaustedError (or any other
     transient exception) to the caller. This is treated as a per-chunk
     grading failure, not a system-wide infra failure like a Qdrant
     outage (IMPLEMENTATION_PLAN.md §3 reserves hard-failure/5xx
     propagation for that class specifically) — one chunk having a bad
     moment on both providers shouldn't crash grading for every other
     chunk in the same retrieval-grading pass, especially since chunks
     are graded concurrently via grade_chunks(). Caught here and failed
     closed the same way malformed output is.

grade_chunks() also passes return_exceptions=True to asyncio.gather as a
second line of defense — if grade_chunk() itself were ever changed to
raise again in the future, one bad chunk still can't take down the whole
batch; any escaped exception is converted to an AMBIGUOUS grade for that
chunk_id rather than propagating.
"""
from __future__ import annotations

import asyncio
import json

from pydantic import ValidationError

from config import get_config
from llm_clients.router import call_with_failover
from logging_config import get_logger
from schemas.chunk_grade import ChunkGrade

_SYSTEM_PROMPT = (
    "You are a strict relevance grader for a retrieval-augmented "
    "generation system. Given a user query and a single retrieved "
    "document chunk, classify the chunk as CORRECT (directly and "
    "specifically answers the query), AMBIGUOUS (topically related but "
    "does not directly answer the query, or answers it only partially), "
    "or INCORRECT (unrelated to the query). Respond with ONLY a JSON "
    'object matching this exact schema, no other text: {"grade": '
    '"CORRECT"|"AMBIGUOUS"|"INCORRECT"}'
)

_MALFORMED_REPROMPT_SUFFIX = (
    "\n\nYour previous response did not match the required JSON schema. "
    "Respond with ONLY valid JSON matching exactly this schema, no "
    'markdown, no code fences, no explanation: {"grade": '
    '"CORRECT"|"AMBIGUOUS"|"INCORRECT"}'
)


def _build_user_prompt(query: str, chunk_text: str) -> str:
    return f"Query: {query}\n\nChunk text:\n{chunk_text}"


def _strip_fences(raw: str) -> str:
    """Strips ```json ... ``` / ``` ... ``` wrapping. LLMs frequently add
    this even when explicitly instructed not to — without stripping it,
    a perfectly valid grade gets misclassified as malformed output and
    silently degrades grading accuracy (confirmed: a CORRECT grade wrapped
    in fences fails both the initial parse and the reprompt-retry parse,
    landing on AMBIGUOUS despite the model having graded correctly both
    times)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = [
            line for line in text.split("\n") if not line.strip().startswith("```")
        ]
        text = "\n".join(lines).strip()
    return text


async def _call_or_none(
    slug_pair,
    system_prompt: str,
    user_prompt: str,
    trace_id: str,
    logger,
    chunk_id: str,
) -> str | None:
    """Wraps call_with_failover so a total provider exhaustion (or any
    other transient exception escaping the router) is treated as a failed
    grading attempt rather than an unhandled crash — same fail-closed
    class as malformed output, per this module's docstring."""
    try:
        return await call_with_failover(
            slug_pair, system_prompt, user_prompt, trace_id=trace_id
        )
    except Exception as exc:
        logger.warning(
            "chunk_grade_call_failed",
            stage="document_grading",
            chunk_id=chunk_id,
            error=str(exc),
        )
        return None


async def grade_chunk(
    query: str,
    chunk_id: str,
    chunk_text: str,
    trace_id: str = "grade_chunk",
) -> ChunkGrade:
    logger = get_logger(trace_id=trace_id)
    slug_pair = get_config().model_tiers.tier1_grading
    user_prompt = _build_user_prompt(query, chunk_text)

    raw_response = await _call_or_none(
        slug_pair, _SYSTEM_PROMPT, user_prompt, trace_id, logger, chunk_id
    )
    if raw_response is not None:
        grade = _try_parse(raw_response, chunk_id)
        if grade is not None:
            return grade
        logger.warning(
            "chunk_grade_malformed_retrying",
            stage="document_grading",
            chunk_id=chunk_id,
            raw_response=raw_response[:200],
        )

    retry_response = await _call_or_none(
        slug_pair,
        _SYSTEM_PROMPT + _MALFORMED_REPROMPT_SUFFIX,
        user_prompt,
        trace_id,
        logger,
        chunk_id,
    )
    if retry_response is not None:
        grade = _try_parse(retry_response, chunk_id)
        if grade is not None:
            return grade

    logger.warning(
        "chunk_grade_fail_closed",
        stage="document_grading",
        chunk_id=chunk_id,
        reason="malformed_output_or_call_failure_after_retry",
        raw_response=(retry_response or "")[:200],
    )
    return ChunkGrade(chunk_id=chunk_id, grade="AMBIGUOUS")


def _try_parse(raw_response: str, chunk_id: str) -> ChunkGrade | None:
    try:
        payload = json.loads(_strip_fences(raw_response))
        return ChunkGrade(chunk_id=chunk_id, grade=payload["grade"])
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError):
        return None


async def grade_chunks(
    query: str,
    chunks: list[dict],
    trace_id: str = "grade_chunks",
) -> list[ChunkGrade]:
    """Grades every chunk concurrently. Each chunk dict must carry
    'chunk_id' and 'text'.

    return_exceptions=True is deliberate defense-in-depth: grade_chunk()
    already catches its own call failures internally and never raises,
    but this ensures that if it ever did (a future bug, an exception
    type not covered by the broad `except Exception` above), one bad
    chunk still fails closed to AMBIGUOUS for its own chunk_id instead of
    crashing the whole batch and losing every other chunk's valid grade.
    """
    logger = get_logger(trace_id=trace_id)
    chunk_ids = [str(c["chunk_id"]) for c in chunks]
    tasks = [
        grade_chunk(query, str(c["chunk_id"]), c.get("text", ""), trace_id=trace_id)
        for c in chunks
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    graded: list[ChunkGrade] = []
    for chunk_id, result in zip(chunk_ids, results):
        if isinstance(result, Exception):
            logger.warning(
                "chunk_grade_unhandled_exception_fail_closed",
                stage="document_grading",
                chunk_id=chunk_id,
                error=str(result),
            )
            graded.append(ChunkGrade(chunk_id=chunk_id, grade="AMBIGUOUS"))
        else:
            graded.append(result)
    return graded