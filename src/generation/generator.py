"""
Answer generation (FR-9 correction-path, FR-18 fast-path).

Both paths call the same function with the same strong-tier model —
IMPLEMENTATION_PLAN.md §1 explicitly defaults fast-path to the same
generation model as correction-path rather than a cheaper one, to keep
answer quality consistent and defer the fast-path-model-tiering question
to Phase 11 as a measured optimization, not a day-one assumption.

`strict=True` is the §3.4 regeneration path: when groundedness fails and
generation_attempts < max_generation_attempts, the orchestration layer
(Phase 7) calls this again with strict=True to get a stricter "stick to
context" instruction — same model, tighter prompt, not a second code path.

Runs through llm_clients/router.py's call_with_failover — the single seam
for Groq-primary / OpenRouter-fallback failover and retry/backoff (NFR-10).
This module never talks to a provider SDK directly.
"""
from __future__ import annotations

from config import get_config
from llm_clients.router import call_with_failover
from logging_config import get_logger

logger = get_logger(trace_id="generator")

_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the "
    "provided context. Answer the question directly and concisely, using "
    "only information present in the context. If the context does not "
    "contain enough information to answer, say so plainly rather than "
    "guessing."
)

_STRICT_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using ONLY the "
    "provided context. This is a regeneration after your previous answer "
    "was flagged as insufficiently grounded in the context.\n\n"
    "Stick strictly to what is explicitly stated in the context. Do not "
    "add any fact, number, date, or claim that is not directly present in "
    "the context below. If the context does not fully answer the "
    "question, say explicitly what is missing rather than filling the gap "
    "with a plausible-sounding guess."
)


class GenerationFailed(Exception):
    """Raised when call_with_failover exhausts both providers (Groq and
    OpenRouter) during answer generation. Unlike grading, there is no
    safe fabricated default answer to fall back to here — the
    orchestration layer (Phase 7's generate_node) should catch this and
    route to terminal low_confidence, the same way rewrite_node catches
    RewriteGenerationFailed, rather than ever letting a raw provider
    exception crash the graph. Confirmed necessary by a real run where
    Groq and OpenRouter failed simultaneously during query rewriting —
    the identical failure mode is possible here."""


def _format_context(context_chunks: list[str]) -> str:
    return "\n\n---\n\n".join(context_chunks)


async def generate_answer(
    query: str,
    context_chunks: list[str],
    strict: bool = False,
    trace_id: str = "generate",
) -> str:
    """FR-9 / FR-18. Generates an answer to `query` from `context_chunks`.

    `context_chunks` is a flat list of chunk text strings — the accepted
    context after document grading (and/or gated fallback content). This
    function is intentionally state-agnostic: it takes explicit args
    rather than a LangGraph state dict, so Phase 7's orchestration layer
    owns all state-shape decisions and this stays a pure, independently
    testable function.

    Raises GenerationFailed if both providers are exhausted — never
    returns a fabricated answer on failure.
    """
    if not context_chunks:
        logger.warning("generation_called_with_empty_context", trace_id=trace_id)

    cfg = get_config()
    slug_pair = cfg.model_tiers.tier3_generation
    system_prompt = _STRICT_SYSTEM_PROMPT if strict else _SYSTEM_PROMPT
    context = _format_context(context_chunks)
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

    logger.info(
        "generation_started",
        trace_id=trace_id,
        strict=strict,
        num_chunks=len(context_chunks),
    )

    try:
        answer = await call_with_failover(
            slug_pair, system_prompt, user_prompt, trace_id=trace_id
        )
    except Exception as exc:
        logger.warning(
            "generation_call_failed",
            trace_id=trace_id,
            strict=strict,
            error=str(exc),
        )
        raise GenerationFailed(
            f"Failed to generate an answer (trace_id={trace_id!r}) after "
            f"provider failover was exhausted."
        ) from exc

    logger.info(
        "generation_completed",
        trace_id=trace_id,
        strict=strict,
        answer_length=len(answer),
    )

    return answer