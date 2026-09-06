"""
LangGraph node functions and post-node routing selectors for the
correction loop (FR-3, FR-12-FR-15), wiring together Phases 2-6.

Nodes mutate state by returning a patch dict (LangGraph merges it into
the running state). Routing selectors are the paired conditional-edge
functions that decide which node runs next — they read state only,
never mutate it (LangGraph doesn't let a conditional-edge function
change state; only a node's return value does). Where a routing
decision needs a value a node just computed (document grading's
p_correct, answer grading's two scores), that value has to be in the
node's returned patch precisely so the selector that runs immediately
afterward can see it in the merged state.
"""
from __future__ import annotations

from typing import Literal

from audit.escalation_queue import push_to_escalation_queue
from config import get_config
from fallback.fetch_and_gate import fetch_gated_fallback
from generation.generator import GenerationFailed, generate_answer
from grading.aggregation import Verdict, aggregate, compute_p_correct
from grading.answer_grader import classify_outcome, grade_answer
from grading.document_grader import grade_chunks
from logging_config import get_logger
from retrieval.retriever import retrieve
from rewriting.query_rewriter import RewriteGenerationFailed, rewrite_query
from schemas.answer_grade import AnswerGrade
from schemas.state import GraphState


# ---------------------------------------------------------------------------
# Retrieval (shared by both fast_path and correction_path)
# ---------------------------------------------------------------------------
async def retrieve_node(state: GraphState) -> dict:
    logger = get_logger(trace_id="retrieve")
    result = await retrieve(state["query"], trace_id="retrieve")
    logger.info(
        "retrieve_node_complete",
        stage="orchestration",
        chunk_count=len(result["chunks"]),
        reranked=result["reranked"],
    )
    return {"retrieved_chunks": result["chunks"]}


# ---------------------------------------------------------------------------
# Fast path (FR-18) — grading skipped entirely, by design
# ---------------------------------------------------------------------------
async def fast_generate_node(state: GraphState) -> dict:
    context_chunks = [c.get("text", "") for c in state["retrieved_chunks"]]
    logger = get_logger(trace_id="generate_fast")
    try:
        answer = await generate_answer(
            query=state["original_query"],
            context_chunks=context_chunks,
            strict=False,
            trace_id="generate_fast",
        )
    except GenerationFailed as exc:
        # Fast path has no correction loop to fall into — if generation
        # itself fails outright (both providers exhausted), the only
        # honest outcome is terminal low_confidence, never a crash and
        # never a fabricated answer.
        logger.warning(
            "fast_generate_failed_terminal", stage="orchestration", error=str(exc)
        )
        return {
            "accepted_context": context_chunks,
            "low_confidence": True,
            "low_confidence_reason": "generation_call_failed",
        }
    return {"answer": answer, "accepted_context": context_chunks}


def route_after_fast_generate(state: GraphState) -> Literal["done", "terminal"]:
    return "terminal" if state.get("low_confidence") else "done"


# ---------------------------------------------------------------------------
# Document grading + aggregation (FR-4, FR-5) — §3.2
# ---------------------------------------------------------------------------
def _run_aggregation(chunks: list[dict], chunk_grades: list, state: GraphState, cfg) -> dict:
    """Single source of truth for the aggregation call — used by both
    grade_documents_node (to compute accepted_context) and
    route_after_grading (to decide where to go next) with identical
    inputs, so the two can never disagree on the verdict."""
    return aggregate(
        chunks=chunks,
        chunk_grades=chunk_grades,
        iteration_count=state["iteration_count"],
        max_iterations=state["max_iterations"],
        fallback_used=state["fallback_used"],
        p_correct_threshold=cfg.thresholds.p_correct_threshold,
    )


async def grade_documents_node(state: GraphState) -> dict:
    grades = await grade_chunks(
        state["query"], state["retrieved_chunks"], trace_id="grade_documents"
    )
    cfg = get_config()
    agg = _run_aggregation(state["retrieved_chunks"], grades, state, cfg)
    accepted_context = [c.get("text", "") for c in agg["accepted_chunks"]]

    logger = get_logger(trace_id="grade_documents")
    logger.info(
        "grade_documents_node_complete",
        stage="orchestration",
        p_correct=agg["p_correct"],
        verdict=agg["verdict"].value,
        accepted_count=len(accepted_context),
    )
    return {"chunk_grades": grades, "accepted_context": accepted_context}


def route_after_grading(
    state: GraphState,
) -> Literal["generate", "rewrite", "fallback", "terminal"]:
    cfg = get_config()
    agg = _run_aggregation(state["retrieved_chunks"], state["chunk_grades"], state, cfg)
    verdict = agg["verdict"]

    if verdict == Verdict.GENERATE:
        return "generate"

    if verdict == Verdict.REWRITE:
        return "rewrite"

    if verdict in (Verdict.FALLBACK_DIRECT, Verdict.FALLBACK_LAST_RESORT):
        if state["fallback_used"]:
            # Fallback is single-shot (REQUIREMENTS.md §0.2 gap #2).
            # decide_verdict()'s 0 < p_correct < threshold branch
            # doesn't itself check fallback_used — only the
            # p_correct == 0 branch does. Enforced here instead: a
            # partial-correct-but-capped verdict reached AFTER fallback
            # already ran must not silently re-trigger Tavily.
            return "terminal"
        return "fallback"

    return "terminal"  # Verdict.TERMINAL_LOW_CONFIDENCE


# ---------------------------------------------------------------------------
# Query rewriting (FR-6) — §3.3
# ---------------------------------------------------------------------------
async def rewrite_node(state: GraphState) -> dict:
    logger = get_logger(trace_id="rewrite")
    try:
        new_query = await rewrite_query(
            state["original_query"], state["rewrite_history"], trace_id="rewrite"
        )
    except RewriteGenerationFailed as exc:
        # No safe fabricated fallback exists for a failed rewrite
        # (query_rewriter.py's own docstring) — route straight to
        # terminal rather than ever handing generation a rewrite the
        # rewriter itself wasn't confident in.
        logger.warning("rewrite_failed_terminal", stage="orchestration", error=str(exc))
        return {
            "low_confidence": True,
            "low_confidence_reason": "rewrite_generation_failed",
        }

    return {
        "query": new_query,
        "iteration_count": state["iteration_count"] + 1,
        "rewrite_history": state["rewrite_history"] + [new_query],
    }


def route_after_rewrite(state: GraphState) -> Literal["retry_retrieval", "terminal"]:
    if state.get("low_confidence"):
        return "terminal"
    return "retry_retrieval"


# ---------------------------------------------------------------------------
# Tavily fallback + injection guard (FR-7, FR-8) — §3.3
# ---------------------------------------------------------------------------
async def fallback_node(state: GraphState) -> dict:
    result = await fetch_gated_fallback(state["query"], trace_id="fallback")
    logger = get_logger(trace_id="fallback")
    logger.info(
        "fallback_node_complete",
        stage="orchestration",
        tavily_degraded=result["tavily_degraded"],
        chunk_count=len(result["chunks"]),
        injection_flagged_count=result["injection_flagged_count"],
        all_flagged=result["all_flagged"],
    )
    # An empty chunks list here (Tavily degraded, or every result
    # flagged by the injection guard) is not special-cased — it flows
    # into grade_documents_node exactly like a zero-result corpus
    # retrieval would, per IMPLEMENTATION_PLAN.md §3's error taxonomy,
    # and aggregation.compute_p_correct() already returns 0.0 for an
    # empty grade list by design.
    return {"retrieved_chunks": result["chunks"], "fallback_used": True}


# ---------------------------------------------------------------------------
# Generation + dual answer-grading (FR-9, FR-10, FR-11) — §3.4
# ---------------------------------------------------------------------------
async def generate_node(state: GraphState) -> dict:
    # generation_attempts starts at 0. This node increments it on every
    # call, including the very first — so with max_generation_attempts
    # = 2, at most 2 total generate() calls happen per query (1 initial
    # + 1 regeneration), matching REQUIREMENTS.md §4's explicit
    # worst-case tally ("up to 2 generation calls, 2x answer-grading
    # calls"). Deliberate reconciliation, flagged rather than silently
    # picked: §3.4's pseudocode phrasing ("if attempts < max: attempts
    # += 1; regenerate") could also be read as counting only
    # regenerations, which would total 3 generate calls at max=2 — that
    # reading conflicts with §4's own numeric commitment, so this
    # implementation follows §4.
    logger = get_logger(trace_id="generate")
    is_regeneration = state["generation_attempts"] >= 1
    try:
        answer = await generate_answer(
            query=state["original_query"],
            context_chunks=state["accepted_context"],
            strict=is_regeneration,
            trace_id="generate",
        )
    except GenerationFailed as exc:
        # Total provider exhaustion during generation — no safe
        # fabricated answer exists to fall back to, so this routes to
        # terminal exactly like a rewrite failure does, rather than
        # crashing the graph.
        logger.warning(
            "generate_node_failed_terminal", stage="orchestration", error=str(exc)
        )
        return {
            "generation_attempts": state["generation_attempts"] + 1,
            "low_confidence": True,
            "low_confidence_reason": "generation_call_failed",
        }
    return {
        "answer": answer,
        "generation_attempts": state["generation_attempts"] + 1,
    }


def route_after_generate(state: GraphState) -> Literal["grade", "terminal"]:
    return "terminal" if state.get("low_confidence") else "grade"


async def grade_answer_node(state: GraphState) -> dict:
    context = "\n\n".join(state["accepted_context"])
    grade = await grade_answer(
        answer=state["answer"] or "",
        context=context,
        original_query=state["original_query"],
        trace_id="grade_answer",
    )
    logger = get_logger(trace_id="grade_answer")
    logger.info(
        "grade_answer_node_complete",
        stage="orchestration",
        groundedness_score=grade.groundedness_score,
        relevance_score=grade.relevance_score,
    )
    return {
        "groundedness_score": grade.groundedness_score,
        "relevance_score": grade.relevance_score,
    }


def route_after_answer_grading(
    state: GraphState,
) -> Literal["accept", "rewrite", "regenerate", "terminal"]:
    grade = AnswerGrade(
        groundedness_score=state["groundedness_score"],
        relevance_score=state["relevance_score"],
    )
    outcome = classify_outcome(grade)

    if outcome == "accept":
        return "accept"

    if outcome == "relevance_fail":
        # §3.4: an off-topic answer is treated as a retrieval-adequacy
        # problem, so it consumes the RETRIEVAL loop's budget
        # (iteration_count / max_iterations), not the generation budget.
        if state["iteration_count"] < state["max_iterations"]:
            return "rewrite"
        return "terminal"

    # groundedness_fail
    if state["generation_attempts"] < state["max_generation_attempts"]:
        return "regenerate"
    return "terminal"


# ---------------------------------------------------------------------------
# Terminal / escalation (FR-13, FR-14, FR-15) — §3.5
# ---------------------------------------------------------------------------
def _derive_terminal_reason(state: GraphState) -> str:
    """Best-effort human-readable reason code when a node didn't already
    set one explicitly (rewrite_node does, on RewriteGenerationFailed).
    Phase 8's full audit trail is the authoritative decision-by-decision
    record — this is just a single summary label for FR-14."""
    cfg = get_config()

    if (
        state["generation_attempts"] >= state["max_generation_attempts"]
        and state.get("groundedness_score") is not None
        and state["groundedness_score"] < cfg.thresholds.groundedness_threshold
    ):
        return "groundedness_check_failed_generation_attempts_exhausted"

    if (
        state["iteration_count"] >= state["max_iterations"]
        and state.get("relevance_score") is not None
        and state["relevance_score"] < cfg.thresholds.relevance_threshold
    ):
        return "relevance_check_failed_iteration_cap_exhausted"

    if state["fallback_used"]:
        if compute_p_correct(state["chunk_grades"]) == 0.0:
            return "fallback_exhausted_zero_correct_chunks"
        return "fallback_already_used_iteration_cap_exhausted"

    return "iteration_cap_exhausted"


async def terminal_node(state: GraphState) -> dict:
    reason = state.get("low_confidence_reason") or _derive_terminal_reason(state)
    patch = {"low_confidence": True, "low_confidence_reason": reason}

    await push_to_escalation_queue({**state, **patch})

    logger = get_logger(trace_id="terminal")
    logger.warning(
        "terminal_low_confidence",
        stage="orchestration",
        reason=reason,
        iteration_count=state["iteration_count"],
        generation_attempts=state["generation_attempts"],
        fallback_used=state["fallback_used"],
    )
    return patch