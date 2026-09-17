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


def _extract_chunk_id(chunk: dict, index: int) -> str:
    """Extracts chunk_id or fallback identifiers, defaulting to deterministic index."""
    val = chunk.get("chunk_id") or chunk.get("id") or chunk.get("url")
    return str(val) if val is not None else f"fallback_chunk_{index}"


def _extract_chunk_text(chunk: dict) -> str:
    """Extracts raw text content across internal corpus or external web fallback formats."""
    return str(
        chunk.get("text")
        or chunk.get("content")
        or chunk.get("snippet")
        or ""
    )


def _build_user_prompt(query: str, chunk_text: str) -> str:
    return f"Query: {query}\n\nChunk text:\n{chunk_text}"


def _strip_fences(raw: str) -> str:
    """Strips Markdown code fences to prevent JSON parse failures."""
    text = raw.strip()
    if text.startswith("```"):
        lines = [
            line
            for line in text.split("\n")
            if not line.strip().startswith("```")
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
    """Executes LLM call, returning None on transient/provider failures."""
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


def _try_parse(raw_response: str, chunk_id: str) -> ChunkGrade | None:
    try:
        payload = json.loads(_strip_fences(raw_response))
        return ChunkGrade(chunk_id=chunk_id, grade=payload["grade"])
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError):
        return None


async def grade_chunk(
    query: str,
    chunk_id: str,
    chunk_text: str,
    trace_id: str = "grade_chunk",
) -> ChunkGrade:
    """Evaluates a single chunk against the query with reprompt retry and fail-closed logic."""
    logger = get_logger(trace_id=trace_id)
    slug_pair = get_config().model_tiers.tier1_grading
    user_prompt = _build_user_prompt(query, chunk_text)

    # 1. Primary Attempt
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

    # 2. Stricter Reprompt Retry
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

    # 3. Fail closed to AMBIGUOUS
    logger.warning(
        "chunk_grade_fail_closed",
        stage="document_grading",
        chunk_id=chunk_id,
        reason="malformed_output_or_call_failure_after_retry",
        raw_response=(retry_response or "")[:200],
    )
    return ChunkGrade(chunk_id=chunk_id, grade="AMBIGUOUS")


async def grade_chunks(
    query: str,
    chunks: list[dict],
    trace_id: str = "grade_chunks",
) -> list[ChunkGrade]:
    """Grades multiple chunks concurrently using a semaphore to avoid rate limits."""
    logger = get_logger(trace_id=trace_id)

    normalized_chunks = [
        (
            _extract_chunk_id(chunk, idx),
            _extract_chunk_text(chunk),
        )
        for idx, chunk in enumerate(chunks)
    ]

    semaphore = asyncio.Semaphore(2)

    async def _throttled_grade(cid: str, ctext: str) -> ChunkGrade:
        async with semaphore:
            return await grade_chunk(query, cid, ctext, trace_id=trace_id)

    tasks = [
        _throttled_grade(chunk_id, chunk_text)
        for chunk_id, chunk_text in normalized_chunks
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    graded: list[ChunkGrade] = []
    for (chunk_id, _), result in zip(normalized_chunks, results):
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


# Verification & Testing Main Function
async def main():
    test_query = "What are the rate limit headers returned by the API?"

    # 3 Distinct chunks: 1 Correct, 1 Ambiguous/Borderline, 1 Incorrect
    mock_chunks = [
        {
            "chunk_id": "chunk_api_headers_01",
            "text": (
                "Document: API Rate Limits > Section: Headers\n"
                "Every API response includes three headers indicating current status: "
                "X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset."
            ),
        },
        {
            "chunk_id": "chunk_billing_02",
            "text": (
                "Document: Billing Overview > Section: Plan Tiers\n"
                "The Starter tier includes up to 500 requests per month. "
                "Any additional requests incur overage charges."
            ),
        },
        {
            "chunk_id": "chunk_failed_payments_03",
            "text": (
                "Document: Payments > Section: Dunning\n"
                "Once a valid payment method successfully clears the outstanding balance, "
                "account access is restored automatically within minutes."
            ),
        },
    ]

    print(f"Testing Document Grading for query: '{test_query}'\n")
    graded_results = await grade_chunks(
        test_query, mock_chunks, trace_id="grader_test"
    )

    for item in graded_results:
        print(f"Chunk ID : {item.chunk_id}")
        print(f"Grade    : {item.grade}")
        print("-" * 35)


if __name__ == "__main__":
    asyncio.run(main())