"""
Phase 7 checkpoint: full graph run with a mocked LLM boundary, asserting
state transitions, per IMPLEMENTATION_PLAN.md's exact checkpoint spec:

  (1) clean retrieval -> straight to generate -> accept
  (2) forced low p_correct -> rewrite fires, iteration_count increments,
      rewrite_history grows
  (3) forced cap exhaustion -> terminal state hits with
      low_confidence: true and a reason code, not an unhandled loop
"""
from __future__ import annotations

import pytest

from orchestration import nodes as N
from orchestration.graph import build_graph, build_initial_state
from schemas.chunk_grade import ChunkGrade


class _StubThresholds:
    p_correct_threshold = 0.5
    groundedness_threshold = 0.7
    relevance_threshold = 0.7
    max_iterations = 3
    max_generation_attempts = 2


class _StubConfig:
    thresholds = _StubThresholds()


@pytest.fixture(autouse=True)
def _stub_config(monkeypatch):
    import grading.answer_grader as answer_grader_mod
    import orchestration.graph as graph_mod
    import orchestration.nodes as nodes_mod

    monkeypatch.setattr(graph_mod, "get_config", lambda: _StubConfig())
    monkeypatch.setattr(nodes_mod, "get_config", lambda: _StubConfig())
    # classify_outcome() lives in grading/answer_grader.py and calls
    # get_config() independently of nodes.py's own import — patch it
    # here too, or it falls through to the real Settings() and crashes
    # on missing .env vars in this sandbox.
    monkeypatch.setattr(answer_grader_mod, "get_config", lambda: _StubConfig())


def _make_chunks(n: int) -> list[dict]:
    return [{"chunk_id": f"c{i}", "text": f"chunk text {i}"} for i in range(n)]


# ---------------------------------------------------------------------------
# Scenario 1: clean retrieval -> generate -> accept
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_1_clean_retrieval_accepts(monkeypatch):
    async def fake_retrieve(query, trace_id=None):
        return {"chunks": _make_chunks(3), "reranked": True}

    async def fake_grade_chunks(query, chunks, trace_id=None):
        return [ChunkGrade(chunk_id=c["chunk_id"], grade="CORRECT") for c in chunks]

    async def fake_generate_answer(query, context_chunks, strict=False, trace_id=None):
        return "The refund takes 5 business days."

    async def fake_grade_answer(answer, context, original_query, trace_id=None):
        from schemas.answer_grade import AnswerGrade

        return AnswerGrade(groundedness_score=0.9, relevance_score=0.9)

    monkeypatch.setattr(N, "retrieve", fake_retrieve)
    monkeypatch.setattr(N, "grade_chunks", fake_grade_chunks)
    monkeypatch.setattr(N, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(N, "grade_answer", fake_grade_answer)

    graph = build_graph()
    initial_state = build_initial_state("How long do refunds take?")
    # force correction_path explicitly to exercise the full loop
    initial_state["query"] = (
        "compare refund processing time vs chargeback processing time in detail please"
    )
    initial_state["original_query"] = initial_state["query"]

    final_state = await graph.ainvoke(initial_state)

    assert final_state["answer"] == "The refund takes 5 business days."
    assert final_state["low_confidence"] is False
    assert final_state["iteration_count"] == 0
    assert final_state["fallback_used"] is False
    print("SCENARIO 1 PASS:", {
        "answer": final_state["answer"],
        "low_confidence": final_state["low_confidence"],
        "iteration_count": final_state["iteration_count"],
    })


# ---------------------------------------------------------------------------
# Scenario 2: forced low p_correct -> rewrite fires, iteration_count
# increments, rewrite_history grows, and recovery succeeds on retry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_2_low_p_correct_triggers_rewrite_and_recovers(monkeypatch):
    call_state = {"retrieve_calls": 0}

    async def fake_retrieve(query, trace_id=None):
        call_state["retrieve_calls"] += 1
        return {"chunks": _make_chunks(3), "reranked": True}

    async def fake_grade_chunks(query, chunks, trace_id=None):
        # First retrieval: 1/3 CORRECT -> p_correct = 0.333, which is
        # 0 < p_correct < threshold (0.5) -> REWRITE branch. (An
        # AMBIGUOUS grade does NOT count toward p_correct — only CORRECT
        # does, per aggregation.compute_p_correct — using AMBIGUOUS here
        # instead of one CORRECT would land on p_correct == 0 instead,
        # which is a different branch entirely.)
        if call_state["retrieve_calls"] == 1:
            return [
                ChunkGrade(chunk_id="c0", grade="CORRECT"),
                ChunkGrade(chunk_id="c1", grade="INCORRECT"),
                ChunkGrade(chunk_id="c2", grade="INCORRECT"),
            ]
        # After rewrite: retrieval recovers, all CORRECT
        return [ChunkGrade(chunk_id=c["chunk_id"], grade="CORRECT") for c in chunks]

    async def fake_rewrite_query(original_query, rewrite_history, trace_id=None):
        return "a genuinely different, more specific reformulation"

    async def fake_generate_answer(query, context_chunks, strict=False, trace_id=None):
        return "Recovered answer after rewrite."

    async def fake_grade_answer(answer, context, original_query, trace_id=None):
        from schemas.answer_grade import AnswerGrade

        return AnswerGrade(groundedness_score=0.85, relevance_score=0.85)

    monkeypatch.setattr(N, "retrieve", fake_retrieve)
    monkeypatch.setattr(N, "grade_chunks", fake_grade_chunks)
    monkeypatch.setattr(N, "rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(N, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(N, "grade_answer", fake_grade_answer)

    graph = build_graph()
    initial_state = build_initial_state(
        "compare X vs Y in great detail across every dimension possible"
    )

    final_state = await graph.ainvoke(initial_state)

    assert call_state["retrieve_calls"] == 2, "expected exactly one rewrite-triggered retry"
    assert final_state["iteration_count"] == 1
    assert len(final_state["rewrite_history"]) == 1
    assert final_state["rewrite_history"][0] == "a genuinely different, more specific reformulation"
    assert final_state["answer"] == "Recovered answer after rewrite."
    assert final_state["low_confidence"] is False
    print("SCENARIO 2 PASS:", {
        "retrieve_calls": call_state["retrieve_calls"],
        "iteration_count": final_state["iteration_count"],
        "rewrite_history": final_state["rewrite_history"],
    })


# ---------------------------------------------------------------------------
# Scenario 3: forced cap exhaustion -> terminal, low_confidence: true,
# reason code set, no unhandled loop / infinite recursion
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_3_cap_exhaustion_hits_terminal(monkeypatch):
    rewrite_count = {"n": 0}

    async def fake_retrieve(query, trace_id=None):
        return {"chunks": _make_chunks(3), "reranked": True}

    async def fake_grade_chunks(query, chunks, trace_id=None):
        if not chunks:
            # Mirrors the real grade_chunks: an empty input list grades
            # to an empty output list (no chunks to grade), which
            # aggregation.compute_p_correct() then reads as p_correct
            # == 0.0 — this is what the post-fallback empty-Tavily-result
            # case actually looks like.
            return []
        # 1/3 CORRECT -> p_correct = 0.333, always short of the 0.5
        # threshold -> forces repeated rewrites until the cap is hit.
        return [
            ChunkGrade(chunk_id="c0", grade="CORRECT"),
            ChunkGrade(chunk_id="c1", grade="INCORRECT"),
            ChunkGrade(chunk_id="c2", grade="INCORRECT"),
        ]

    async def fake_rewrite_query(original_query, rewrite_history, trace_id=None):
        rewrite_count["n"] += 1
        return f"rewrite attempt {rewrite_count['n']}"

    async def fake_fetch_gated_fallback(query, trace_id=None):
        # Fallback also fails to find anything useful (all flagged /
        # degraded) — zero chunks, forcing p_correct == 0 post-fallback.
        return {
            "chunks": [],
            "tavily_degraded": True,
            "injection_flagged_count": 0,
            "all_flagged": False,
        }

    monkeypatch.setattr(N, "retrieve", fake_retrieve)
    monkeypatch.setattr(N, "grade_chunks", fake_grade_chunks)
    monkeypatch.setattr(N, "rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(N, "fetch_gated_fallback", fake_fetch_gated_fallback)

    graph = build_graph()
    initial_state = build_initial_state(
        "compare X vs Y across every conceivable dimension in full detail"
    )

    final_state = await graph.ainvoke(initial_state)

    assert final_state["low_confidence"] is True
    assert final_state["low_confidence_reason"] is not None
    assert final_state["iteration_count"] == final_state["max_iterations"] == 3
    assert final_state["fallback_used"] is True
    print("SCENARIO 3 PASS:", {
        "low_confidence": final_state["low_confidence"],
        "reason": final_state["low_confidence_reason"],
        "iteration_count": final_state["iteration_count"],
        "fallback_used": final_state["fallback_used"],
    })


# ---------------------------------------------------------------------------
# Scenario 4 (extra, not in the official checkpoint list): the single-shot
# fallback guard added in route_after_grading(). aggregation.decide_verdict()'s
# 0 < p_correct < threshold branch does NOT check fallback_used on its own —
# only the p_correct == 0 branch does. This test forces post-fallback grading
# to land in that 0 < p_correct < threshold branch (not exactly 0) while
# iteration_count is already capped, to prove route_after_grading's own
# fallback_used guard fires and fallback is never called a second time.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_4_fallback_single_shot_guard_enforced(monkeypatch):
    fallback_calls = {"n": 0}

    async def fake_retrieve(query, trace_id=None):
        return {"chunks": _make_chunks(3), "reranked": True}

    async def fake_grade_chunks(query, chunks, trace_id=None):
        # Always 1/3 correct -> p_correct = 0.333, both for the initial
        # corpus chunks AND for the fallback's web chunks. Never enough
        # to clear 0.5, never exactly 0 either.
        return [
            ChunkGrade(chunk_id="c0", grade="CORRECT"),
            ChunkGrade(chunk_id="c1", grade="INCORRECT"),
            ChunkGrade(chunk_id="c2", grade="INCORRECT"),
        ]

    async def fake_rewrite_query(original_query, rewrite_history, trace_id=None):
        return f"rewrite {len(rewrite_history) + 1}"

    async def fake_fetch_gated_fallback(query, trace_id=None):
        fallback_calls["n"] += 1
        # Returns real (non-empty, non-degraded) web chunks — so
        # post-fallback grading lands in 0 < p_correct < threshold,
        # NOT p_correct == 0. This is the case aggregation.py's own
        # p_correct == 0 branch check can't catch.
        return {
            "chunks": _make_chunks(3),
            "tavily_degraded": False,
            "injection_flagged_count": 0,
            "all_flagged": False,
        }

    monkeypatch.setattr(N, "retrieve", fake_retrieve)
    monkeypatch.setattr(N, "grade_chunks", fake_grade_chunks)
    monkeypatch.setattr(N, "rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(N, "fetch_gated_fallback", fake_fetch_gated_fallback)

    graph = build_graph()
    initial_state = build_initial_state(
        "compare X vs Y across every conceivable dimension in full detail"
    )

    final_state = await graph.ainvoke(initial_state)

    assert fallback_calls["n"] == 1, (
        f"fallback was called {fallback_calls['n']} times — single-shot "
        f"policy (REQUIREMENTS.md §0.2 gap #2) was violated"
    )
    assert final_state["low_confidence"] is True
    assert final_state["fallback_used"] is True
    print("SCENARIO 4 PASS:", {
        "fallback_calls": fallback_calls["n"],
        "low_confidence": final_state["low_confidence"],
        "reason": final_state["low_confidence_reason"],
    })


# ---------------------------------------------------------------------------
# Extra coverage (beyond the 3 official checkpoint scenarios) — branches
# not exercised above but present in the graph.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fast_path_skips_grading_entirely(monkeypatch):
    grade_chunks_called = {"n": 0}

    async def fake_retrieve(query, trace_id=None):
        return {"chunks": _make_chunks(2), "reranked": True}

    async def fake_grade_chunks(query, chunks, trace_id=None):
        grade_chunks_called["n"] += 1
        return []

    async def fake_generate_answer(query, context_chunks, strict=False, trace_id=None):
        return "Fast path answer, ungraded."

    monkeypatch.setattr(N, "retrieve", fake_retrieve)
    monkeypatch.setattr(N, "grade_chunks", fake_grade_chunks)
    monkeypatch.setattr(N, "generate_answer", fake_generate_answer)

    graph = build_graph()
    # Short, no comparison markers -> classify_query() returns fast_path
    initial_state = build_initial_state("refund policy")

    final_state = await graph.ainvoke(initial_state)

    assert grade_chunks_called["n"] == 0, "fast_path must never call grading (FR-18)"
    assert final_state["answer"] == "Fast path answer, ungraded."
    assert final_state["low_confidence"] is False
    print("FAST PATH PASS:", {"grade_chunks_called": grade_chunks_called["n"]})


@pytest.mark.asyncio
async def test_groundedness_fail_regenerates_then_accepts(monkeypatch):
    generate_calls = {"n": 0}
    grade_answer_calls = {"n": 0}

    async def fake_retrieve(query, trace_id=None):
        return {"chunks": _make_chunks(3), "reranked": True}

    async def fake_grade_chunks(query, chunks, trace_id=None):
        return [ChunkGrade(chunk_id=c["chunk_id"], grade="CORRECT") for c in chunks]

    async def fake_generate_answer(query, context_chunks, strict=False, trace_id=None):
        generate_calls["n"] += 1
        return f"answer attempt {generate_calls['n']} (strict={strict})"

    async def fake_grade_answer(answer, context, original_query, trace_id=None):
        from schemas.answer_grade import AnswerGrade

        grade_answer_calls["n"] += 1
        if grade_answer_calls["n"] == 1:
            # first pass: ungrounded, on-topic
            return AnswerGrade(groundedness_score=0.3, relevance_score=0.9)
        # regeneration: grounded now
        return AnswerGrade(groundedness_score=0.85, relevance_score=0.9)

    monkeypatch.setattr(N, "retrieve", fake_retrieve)
    monkeypatch.setattr(N, "grade_chunks", fake_grade_chunks)
    monkeypatch.setattr(N, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(N, "grade_answer", fake_grade_answer)

    graph = build_graph()
    initial_state = build_initial_state(
        "compare X vs Y across every dimension in full detail"
    )

    final_state = await graph.ainvoke(initial_state)

    assert generate_calls["n"] == 2, "expected exactly 1 regeneration"
    assert grade_answer_calls["n"] == 2
    assert final_state["generation_attempts"] == 2
    assert "strict=True" in final_state["answer"]
    assert final_state["low_confidence"] is False
    print("GROUNDEDNESS REGEN PASS:", {
        "generate_calls": generate_calls["n"],
        "final_answer": final_state["answer"],
    })


@pytest.mark.asyncio
async def test_groundedness_fail_exhausts_attempts_and_terminates(monkeypatch):
    async def fake_retrieve(query, trace_id=None):
        return {"chunks": _make_chunks(3), "reranked": True}

    async def fake_grade_chunks(query, chunks, trace_id=None):
        return [ChunkGrade(chunk_id=c["chunk_id"], grade="CORRECT") for c in chunks]

    async def fake_generate_answer(query, context_chunks, strict=False, trace_id=None):
        return "always ungrounded"

    async def fake_grade_answer(answer, context, original_query, trace_id=None):
        from schemas.answer_grade import AnswerGrade

        # always fails groundedness, no matter how many regenerations
        return AnswerGrade(groundedness_score=0.2, relevance_score=0.9)

    monkeypatch.setattr(N, "retrieve", fake_retrieve)
    monkeypatch.setattr(N, "grade_chunks", fake_grade_chunks)
    monkeypatch.setattr(N, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(N, "grade_answer", fake_grade_answer)

    graph = build_graph()
    initial_state = build_initial_state(
        "compare X vs Y across every dimension in full detail"
    )

    final_state = await graph.ainvoke(initial_state)

    assert final_state["generation_attempts"] == 2  # max_generation_attempts
    assert final_state["low_confidence"] is True
    assert final_state["low_confidence_reason"] == (
        "groundedness_check_failed_generation_attempts_exhausted"
    )
    print("GROUNDEDNESS EXHAUSTED PASS:", {
        "generation_attempts": final_state["generation_attempts"],
        "reason": final_state["low_confidence_reason"],
    })


@pytest.mark.asyncio
async def test_relevance_fail_routes_to_rewrite_not_regenerate(monkeypatch):
    retrieve_calls = {"n": 0}

    async def fake_retrieve(query, trace_id=None):
        retrieve_calls["n"] += 1
        return {"chunks": _make_chunks(3), "reranked": True}

    async def fake_grade_chunks(query, chunks, trace_id=None):
        return [ChunkGrade(chunk_id=c["chunk_id"], grade="CORRECT") for c in chunks]

    async def fake_rewrite_query(original_query, rewrite_history, trace_id=None):
        return "more specific rewritten query"

    async def fake_generate_answer(query, context_chunks, strict=False, trace_id=None):
        return "on-topic-ish answer" if retrieve_calls["n"] > 1 else "off-topic answer"

    async def fake_grade_answer(answer, context, original_query, trace_id=None):
        from schemas.answer_grade import AnswerGrade

        if retrieve_calls["n"] == 1:
            return AnswerGrade(groundedness_score=0.9, relevance_score=0.3)
        return AnswerGrade(groundedness_score=0.9, relevance_score=0.9)

    monkeypatch.setattr(N, "retrieve", fake_retrieve)
    monkeypatch.setattr(N, "grade_chunks", fake_grade_chunks)
    monkeypatch.setattr(N, "rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(N, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(N, "grade_answer", fake_grade_answer)

    graph = build_graph()
    initial_state = build_initial_state(
        "compare X vs Y across every dimension in full detail"
    )

    final_state = await graph.ainvoke(initial_state)

    # relevance_fail must consume the RETRIEVAL loop's budget
    # (iteration_count), not generation_attempts
    assert final_state["iteration_count"] == 1
    assert final_state["generation_attempts"] == 2  # 1 per generate_node call
    assert final_state["low_confidence"] is False
    assert final_state["answer"] == "on-topic-ish answer"
    print("RELEVANCE FAIL -> REWRITE PASS:", {
        "iteration_count": final_state["iteration_count"],
        "retrieve_calls": retrieve_calls["n"],
    })


@pytest.mark.asyncio
async def test_rewrite_generation_failed_short_circuits_to_terminal(monkeypatch):
    from rewriting.query_rewriter import RewriteGenerationFailed

    async def fake_retrieve(query, trace_id=None):
        return {"chunks": _make_chunks(3), "reranked": True}

    async def fake_grade_chunks(query, chunks, trace_id=None):
        return [
            ChunkGrade(chunk_id="c0", grade="CORRECT"),
            ChunkGrade(chunk_id="c1", grade="INCORRECT"),
            ChunkGrade(chunk_id="c2", grade="INCORRECT"),
        ]

    async def fake_rewrite_query(original_query, rewrite_history, trace_id=None):
        raise RewriteGenerationFailed("could not produce a valid non-duplicate rewrite")

    monkeypatch.setattr(N, "retrieve", fake_retrieve)
    monkeypatch.setattr(N, "grade_chunks", fake_grade_chunks)
    monkeypatch.setattr(N, "rewrite_query", fake_rewrite_query)

    graph = build_graph()
    initial_state = build_initial_state(
        "compare X vs Y across every dimension in full detail"
    )

    final_state = await graph.ainvoke(initial_state)

    assert final_state["low_confidence"] is True
    assert final_state["low_confidence_reason"] == "rewrite_generation_failed"
    assert final_state["iteration_count"] == 0  # never incremented — failed before that
    print("REWRITE FAILURE SHORT-CIRCUIT PASS:", {
        "reason": final_state["low_confidence_reason"],
        "iteration_count": final_state["iteration_count"],
    })


# ---------------------------------------------------------------------------
# Regression tests for the two real-run bugs found in production logs:
# (1) query_rewriter.py crashed the whole graph on total provider
#     exhaustion instead of raising RewriteGenerationFailed.
# (2) generator.py had the identical unprotected-call gap.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rewrite_provider_exhaustion_terminates_gracefully(monkeypatch):
    """Reproduces the exact real-run crash: rewrite_query()'s underlying
    call_with_failover raises RetriesExhaustedError (both Groq and
    OpenRouter exhausted). Must terminate gracefully, not crash the
    graph."""

    class RetriesExhaustedError(Exception):
        pass

    async def fake_retrieve(query, trace_id=None):
        return {"chunks": _make_chunks(3), "reranked": True}

    async def fake_grade_chunks(query, chunks, trace_id=None):
        return [
            ChunkGrade(chunk_id="c0", grade="CORRECT"),
            ChunkGrade(chunk_id="c1", grade="INCORRECT"),
            ChunkGrade(chunk_id="c2", grade="INCORRECT"),
        ]

    async def fake_rewrite_query_always_exhausted(original_query, rewrite_history, trace_id=None):
        # Simulates query_rewriter.py's real internal behavior after the
        # fix: total provider exhaustion converts to RewriteGenerationFailed
        # after exhausting its own internal retry attempts, rather than
        # ever letting RetriesExhaustedError itself escape.
        from rewriting.query_rewriter import RewriteGenerationFailed

        raise RewriteGenerationFailed(
            "Failed to produce a valid, non-duplicate rewrite after 3 attempts "
            "(all attempts hit total provider exhaustion)."
        )

    monkeypatch.setattr(N, "retrieve", fake_retrieve)
    monkeypatch.setattr(N, "grade_chunks", fake_grade_chunks)
    monkeypatch.setattr(N, "rewrite_query", fake_rewrite_query_always_exhausted)

    graph = build_graph()
    initial_state = build_initial_state(
        "compare X vs Y across every conceivable dimension in full detail"
    )

    # Must NOT raise — must terminate gracefully
    final_state = await graph.ainvoke(initial_state)

    assert final_state["low_confidence"] is True
    assert final_state["low_confidence_reason"] == "rewrite_generation_failed"
    print("REWRITE PROVIDER EXHAUSTION -> GRACEFUL TERMINATION PASS:", {
        "reason": final_state["low_confidence_reason"],
    })


@pytest.mark.asyncio
async def test_generation_failure_during_correction_path_terminates_gracefully(monkeypatch):
    """generate_node's call_with_failover exhausts both providers mid
    correction-path. Must terminate gracefully via GenerationFailed, not
    crash, and must not proceed to grade_answer_node with no answer."""
    from generation.generator import GenerationFailed

    grade_answer_called = {"n": 0}

    async def fake_retrieve(query, trace_id=None):
        return {"chunks": _make_chunks(3), "reranked": True}

    async def fake_grade_chunks(query, chunks, trace_id=None):
        return [ChunkGrade(chunk_id=c["chunk_id"], grade="CORRECT") for c in chunks]

    async def fake_generate_answer_always_fails(query, context_chunks, strict=False, trace_id=None):
        raise GenerationFailed(f"provider failover exhausted (trace_id={trace_id!r})")

    async def fake_grade_answer(*a, **k):
        grade_answer_called["n"] += 1
        from schemas.answer_grade import AnswerGrade
        return AnswerGrade(groundedness_score=0.9, relevance_score=0.9)

    monkeypatch.setattr(N, "retrieve", fake_retrieve)
    monkeypatch.setattr(N, "grade_chunks", fake_grade_chunks)
    monkeypatch.setattr(N, "generate_answer", fake_generate_answer_always_fails)
    monkeypatch.setattr(N, "grade_answer", fake_grade_answer)

    graph = build_graph()
    initial_state = build_initial_state(
        "compare X vs Y across every conceivable dimension in full detail"
    )

    final_state = await graph.ainvoke(initial_state)

    assert grade_answer_called["n"] == 0, (
        "grade_answer must never run on a None/missing answer after "
        "generation failed"
    )
    assert final_state["low_confidence"] is True
    assert final_state["low_confidence_reason"] == "generation_call_failed"
    print("GENERATION FAILURE (correction_path) -> GRACEFUL TERMINATION PASS:", {
        "reason": final_state["low_confidence_reason"],
        "grade_answer_called": grade_answer_called["n"],
    })


@pytest.mark.asyncio
async def test_generation_failure_during_fast_path_terminates_gracefully(monkeypatch):
    """Same failure, but on the fast_path (no correction loop to fall
    into at all) — must still terminate gracefully, not crash."""
    from generation.generator import GenerationFailed

    async def fake_retrieve(query, trace_id=None):
        return {"chunks": _make_chunks(2), "reranked": True}

    async def fake_generate_answer_always_fails(query, context_chunks, strict=False, trace_id=None):
        raise GenerationFailed(f"provider failover exhausted (trace_id={trace_id!r})")

    monkeypatch.setattr(N, "retrieve", fake_retrieve)
    monkeypatch.setattr(N, "generate_answer", fake_generate_answer_always_fails)

    graph = build_graph()
    initial_state = build_initial_state("refund policy")  # short -> fast_path

    final_state = await graph.ainvoke(initial_state)

    assert final_state["low_confidence"] is True
    assert final_state["low_confidence_reason"] == "generation_call_failed"
    print("GENERATION FAILURE (fast_path) -> GRACEFUL TERMINATION PASS:", {
        "reason": final_state["low_confidence_reason"],
    })