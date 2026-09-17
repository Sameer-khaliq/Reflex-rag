from __future__ import annotations

import asyncio
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
    "\n\nYour previous response was malformed or did not match JSON. "
    "Do NOT include thinking, markdown, or code fences. "
    'Respond with ONLY valid JSON: {"rewritten_query": "your rewritten query text here"}'
)

_DUPLICATE_REPROMPT_SUFFIX = (
    "\n\nYour previous rewrite repeated an entry in the history or used a placeholder. "
    "Produce a genuine search query with specific terms different from past attempts."
)


def _build_user_prompt(original_query: str, rewrite_history: list[str]) -> str:
    history_block = (
        "\n".join(f"- {r}" for r in rewrite_history)
        if rewrite_history
        else "(none yet)"
    )
    return (
        f"Original query: {original_query}\n\n"
        f"Prior rewrites that already failed to retrieve enough:\n{history_block}"
    )


def _is_duplicate(candidate: str, rewrite_history: list[str]) -> bool:
    normalized_candidate = candidate.strip().lower()
    placeholders = {
        "<the new query>",
        "the new query",
        "<the rewritten query>",
        "<new query>",
        "<query>",
        "the rewritten query",
        "your actual rewritten query text here",
        "your rewritten query text here",
        "",
    }
    if normalized_candidate in placeholders:
        return True
    return any(
        normalized_candidate == r.strip().lower() for r in rewrite_history
    )


def _clean_response(raw: str) -> str:
    """Removes thinking blocks (<think>...</think>) and markdown code fences."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = [
            line
            for line in text.split("\n")
            if not line.strip().startswith("```")
        ]
        text = "\n".join(lines).strip()
    return text


def _try_parse(raw_response: str) -> str | None:
    try:
        cleaned = _clean_response(raw_response)
        payload = json.loads(cleaned)
        parsed = RewriteOutput(rewritten_query=payload["rewritten_query"])
        return parsed.rewritten_query
    except Exception:
        match = re.search(
            r'\{.*"rewritten_query"\s*:\s*".*?".*\}', raw_response, re.DOTALL
        )
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
            raw_response = await call_with_failover(
                slug_pair,
                system_prompt,
                user_prompt,
                trace_id=trace_id,
                max_tokens=250,
            )
        except TypeError:
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


async def main():
    print("--- Running Query Rewriter Verification ---\n")

    test_query = "limits"
    # Pehle fail hone wale attempts simulate karte hain
    mock_history = [
        "what are the rate limits",
        "api request limits",
    ]

    print(f"Original Query : '{test_query}'")
    print(f"Prior Failures : {mock_history}\n")

    try:
        new_query = await rewrite_query(
            original_query=test_query,
            rewrite_history=mock_history,
            trace_id="test_rewriter",
        )
        print("Success!")
        print(f"Rewritten Query: '{new_query}'")
    except Exception as e:
        print(f"Failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())