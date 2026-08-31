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

Malformed output gets one stricter reprompt. If that's still malformed,
this fails closed to AMBIGUOUS — never to CORRECT — per
IMPLEMENTATION_PLAN.md §3: "A broken grader should never silently pass
bad content through as if it were verified."
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


async def grade_chunk(
    query: str,
    chunk_id: str,
    chunk_text: str,
    trace_id: str = "grade_chunk",
) -> ChunkGrade:
    logger = get_logger(trace_id=trace_id)
    slug_pair = get_config().model_tiers.tier1_grading
    user_prompt = _build_user_prompt(query, chunk_text)

    raw_response = await call_with_failover(
        slug_pair, _SYSTEM_PROMPT, user_prompt, trace_id=trace_id
    )

    grade = _try_parse(raw_response, chunk_id)
    if grade is not None:
        return grade

    logger.warning(
        "chunk_grade_malformed_retrying",
        stage="document_grading",
        chunk_id=chunk_id,
        raw_response=raw_response[:200],
    )
    retry_response = await call_with_failover(
        slug_pair,
        _SYSTEM_PROMPT + _MALFORMED_REPROMPT_SUFFIX,
        user_prompt,
        trace_id=trace_id,
    )
    grade = _try_parse(retry_response, chunk_id)
    if grade is not None:
        return grade

    logger.warning(
        "chunk_grade_malformed_fail_closed",
        stage="document_grading",
        chunk_id=chunk_id,
        raw_response=retry_response[:200],
    )
    return ChunkGrade(chunk_id=chunk_id, grade="AMBIGUOUS")


def _try_parse(raw_response: str, chunk_id: str) -> ChunkGrade | None:
    try:
        payload = json.loads(raw_response)
        return ChunkGrade(chunk_id=chunk_id, grade=payload["grade"])
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError):
        return None


async def grade_chunks(
    query: str,
    chunks: list[dict],
    trace_id: str = "grade_chunks",
) -> list[ChunkGrade]:
    """Grades every chunk concurrently. Each chunk dict must carry
    'chunk_id' and 'text'."""
    tasks = [
        grade_chunk(query, str(c["chunk_id"]), c.get("text", ""), trace_id=trace_id)
        for c in chunks
    ]
    return await asyncio.gather(*tasks)