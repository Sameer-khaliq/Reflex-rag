"""
Query rewriting (FR-6).

Given an under-specified or insufficiently-retrieved query and the full
rewrite_history so far, produces a genuinely different, more specific
reformulation via the tier2_rewriting model. Two independent retry
conditions, each with its own reprompt:

  1. Malformed JSON output -> reprompt with the schema restated.
  2. A rewrite that exactly repeats a prior entry in rewrite_history ->
     reprompt explicitly told to avoid the repeated phrasing. This is
     what FR-6's "doesn't repeat a prior failed reformulation" actually
     means in practice — a rewrite that's byte-for-byte identical to one
     that already failed to retrieve enough clearly isn't a genuine
     reformulation.

If 3 attempts are exhausted without a valid, non-duplicate rewrite, this
raises RewriteGenerationFailed rather than silently returning something.
REQUIREMENTS.md's error taxonomy defines a fail-closed default for a bad
chunk grade (AMBIGUOUS) but there's no equivalent safe default for a
rewrite — a fabricated "safe" rewrite could just as easily send the loop
in a worse direction than the original query. The orchestration node
(Phase 7) should catch this and route to terminal low_confidence rather
than ever handing generation a rewrite this module wasn't confident in.
"""
from __future__ import annotations

import json
import re

from config import get_config
from llm_clients.router import call_with_failover
from logging_config import get_logger
from schemas.rewrite import RewriteOutput

_MAX_ATTEMPTS = 3


class RewriteGenerationFailed(Exception):
    pass


_SYSTEM_PROMPT = (
    "You rewrite search queries for a retrieval system when the original "
    "query failed to retrieve sufficient relevant results. Given the "
    "original query and a list of prior rewrites that also failed, "
    "produce ONE new reformulation that is genuinely different from "
    "every prior rewrite — more specific, using different phrasing or "
    "disambiguating terms, not a near-paraphrase.\n"
    "Respond with ONLY a valid JSON object matching this schema, no markdown, no thinking tags:\n"
    '{"rewritten_query": "your actual rewritten query text here"}'
)

_MALFORMED_REPROMPT_SUFFIX = (
    "\n\nYour previous response was malformed. Respond with ONLY valid JSON containing your actual search query, "
    'matching: {"rewritten_query": "your rewritten query text here"}'
)

_DUPLICATE_REPROMPT_SUFFIX = (
    "\n\nYour previous rewrite repeated an entry in the history or used a placeholder. "
    "Produce a genuine search query with specific terms different from past attempts."
)


def _build_user_prompt(original_query: str, rewrite_history: list[str]) -> str:
    history_block = (
        "\n".join(f"- {r}" for r in rewrite_history) if rewrite_history else "(none yet)"
    )
    return (
        f"Original query: {original_query}\n\n"
        f"Prior rewrites that already failed to retrieve enough:\n{history_block}"
    )


def _is_duplicate(candidate: str, rewrite_history: list[str]) -> bool:
    normalized_candidate = candidate.strip().lower()
    placeholders = {
        "<the new query>", "the new query", "<the rewritten query>",
        "<new query>", "<query>", "the rewritten query", "your actual rewritten query text here",
        "your rewritten query text here", ""
    }
    if normalized_candidate in placeholders:
        return True
    return any(normalized_candidate == r.strip().lower() for r in rewrite_history)


def _clean_response(raw: str) -> str:
    """Removes <think>...</think> tags and markdown code blocks."""
    # Strip thinking blocks (e.g. Qwen / DeepSeek)
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = [line for line in text.split("\n") if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _try_parse(raw_response: str) -> str | None:
    try:
        cleaned = _clean_response(raw_response)
        payload = json.loads(cleaned)
        parsed = RewriteOutput(rewritten_query=payload["rewritten_query"])
        return parsed.rewritten_query
    except Exception:
        # Secondary fallback: regex to extract JSON object if surrounding text exists
        match = re.search(r'\{.*"rewritten_query"\s*:\s*".*?".*\}', raw_response, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
                return str(payload.get("rewritten_query"))
            except Exception:
                pass
        return None


async def rewrite_query(
    original_query: str,
    rewrite_history: list[str],
    trace_id: str = "rewrite_query",
) -> str:
    logger = get_logger(trace_id=trace_id)
    slug_pair = get_config().model_tiers.tier2_rewriting
    user_prompt = _build_user_prompt(original_query, rewrite_history)
    system_prompt = _SYSTEM_PROMPT

    for attempt in range(_MAX_ATTEMPTS):
        try:
            # max_tokens cap added to keep response under Groq's 1000 OTPM ceiling
            raw_response = await call_with_failover(
                slug_pair,
                system_prompt,
                user_prompt,
                trace_id=trace_id,
                max_tokens=250,
            )
        except TypeError:
            # If call_with_failover signature doesn't take kwargs yet, fall back without max_tokens
            raw_response = await call_with_failover(
                slug_pair, system_prompt, user_prompt, trace_id=trace_id
            )
        except Exception as exc:
            logger.warning(
                "rewrite_call_failed_retrying",
                stage="query_rewriting",
                attempt=attempt,
                error=str(exc),
            )
            continue

        candidate = _try_parse(raw_response)

        if candidate is None:
            logger.warning(
                "rewrite_malformed_retrying",
                stage="query_rewriting",
                attempt=attempt,
                raw_response=raw_response[:200],
            )
            system_prompt = _SYSTEM_PROMPT + _MALFORMED_REPROMPT_SUFFIX
            continue

        if _is_duplicate(candidate, rewrite_history):
            logger.warning(
                "rewrite_duplicate_retrying",
                stage="query_rewriting",
                attempt=attempt,
                rewritten_query=candidate,
            )
            system_prompt = _SYSTEM_PROMPT + _DUPLICATE_REPROMPT_SUFFIX
            continue

        return candidate

    raise RewriteGenerationFailed(
        f"Failed to produce a valid, non-duplicate rewrite for {original_query!r} "
        f"after {_MAX_ATTEMPTS} attempts."
    )