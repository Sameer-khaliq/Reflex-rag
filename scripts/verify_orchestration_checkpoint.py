"""
scripts/verify_orchestration_checkpoint.py

Phase 7 real-model checkpoint.

IMPLEMENTATION_PLAN.md §5, on why this script exists at all: "a mocked
test proves your code parses a response correctly, it doesn't prove your
*prompt* actually grades well." tests/integration/
test_correction_loop_orchestration.py already proves the graph's WIRING
is correct with a mocked LLM boundary (9/9 passing). This script proves
the WHOLE THING actually behaves per REQUIREMENTS.md §6's three DoD trap
cases against the real corpus, real Groq/OpenRouter, real Qdrant, and
real Tavily — no mocks anywhere in this run.

Fill in TRAP_CASE_QUERIES below with your actual NimbusPay corpus
trap-case queries before running. This script does NOT invent queries
for you — guessing at generic queries and quietly declaring a pass would
be worse than not running this at all, since it would validate nothing
about the specific traps you built into the corpus.

Usage:
    uv run python scripts/verify_orchestration_checkpoint.py

Each of the 3 checks runs the graph exactly ONCE (via graph.astream in
"updates" mode) and reconstructs both the step-by-step trace and the
final state from that single run — never two separate invocations for
the same query, since LLM calls aren't guaranteed reproducible and a
second run could silently diverge from the traced one.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orchestration.graph import build_graph, build_initial_state

# ---------------------------------------------------------------------------
# NimbusPay DoD trap-case queries (REQUIREMENTS.md §6).
#
# The first two queries were deliberately lengthened from their original
# short-question phrasing before being finalized here. Verified against
# routing_gate.classify_query() directly (not assumed): the original
# 6-word and 9-word phrasings both classified as fast_path, which skips
# grading entirely (FR-18) and would have made this checkpoint validate
# nothing for those two cases — the exact gotcha the WARNING further
# down in check_hallucination_trap() exists to catch. All three queries
# below are confirmed to route to correction_path:
#
#   term_collision_bad_retrieval : 18 words -> correction_path
#   hallucination_trap           : 22 words -> correction_path
#   unanswerable                 : 14 words -> correction_path
#
# The lengthening only added surrounding phrasing (comparison framing,
# "as opposed to" specificity) — it did not change what each query is
# actually testing. Still worth double-checking against your real
# corpus: term_collision_bad_retrieval needs to reference whatever
# specific ambiguous term your corpus's trap case actually collides on
# (not just "payment methods" in the abstract, if that's not the term
# your corpus disambiguates) — if it doesn't reference the real
# colliding term, this will pass trivially without ever exercising the
# rewrite loop.
# ---------------------------------------------------------------------------
TRAP_CASE_QUERIES = {
    "term_collision_bad_retrieval": (
        "How do I resolve a dispute about my last invoice, and how does the "
        "review process work vs a customer chargeback?"
    ),
    "hallucination_trap": (
        "How long does a partial refund specifically take to process, as "
        "opposed to a standard full refund, across all supported payment "
        "methods?"
    ),
    "unanswerable": (
        "What is NimbusPay's refund policy for a country not mentioned "
        "anywhere in your documentation?"
    ),
}

_PLACEHOLDER_MARKER = "<FILL IN"


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass


def _summarize_patch(patch: dict) -> dict:
    summary = {}
    for k, v in patch.items():
        if k == "retrieved_chunks" and isinstance(v, list):
            summary[k] = f"[{len(v)} chunks: {[c.get('chunk_id') or c.get('source_doc_id') for c in v[:3]]}...]"
        elif k == "accepted_context" and isinstance(v, list):
            summary[k] = f"[{len(v)} context strings]"
        else:
            summary[k] = v
    return summary


async def _run_and_trace(label: str, query: str) -> dict:
    """Runs the graph once, printing each node's patch as it fires, and
    returns the fully-merged final state — reconstructed from the same
    single run that produced the trace, not a second invocation."""
    print(f"\n{'=' * 70}\n{label}\nquery: {query!r}\n{'=' * 70}")
    graph = build_graph()
    initial_state = build_initial_state(query)

    current_state = dict(initial_state)
    async for update in graph.astream(initial_state, stream_mode="updates"):
        for node_name, patch in update.items():
            print(f"  -> node '{node_name}' returned: {_summarize_patch(patch)}")
            current_state.update(patch)

    return current_state


def _assert(condition: bool, message: str, failures: list[str]) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        failures.append(message)


async def check_term_collision(query: str) -> list[str]:
    """REQUIREMENTS.md §6, case 1: initial retrieval must be graded
    INCORRECT/AMBIGUOUS, must trigger a rewrite, and the rewrite must
    recover a CORRECT-graded chunk on retry — proving the rewrite loop
    actually improves retrieval, not just re-runs the same query."""
    failures: list[str] = []
    final_state = await _run_and_trace(
        "DoD Case 1: term-collision bad retrieval", query
    )

    print("\nAssertions (REQUIREMENTS.md §6, case 1):")
    _assert(
        len(final_state["rewrite_history"]) >= 1,
        "at least one rewrite fired",
        failures,
    )
    _assert(
        final_state["iteration_count"] >= 1,
        "iteration_count incremented",
        failures,
    )
    _assert(
        final_state["low_confidence"] is False,
        "query ultimately resolved (not low_confidence) — proves the "
        "rewrite actually recovered a usable chunk, not just retried the "
        "same query into the same failure",
        failures,
    )
    _assert(bool(final_state["answer"]), "an answer was produced", failures)

    print(f"\nFinal answer: {final_state['answer']!r}")
    print(f"Rewrite history: {final_state['rewrite_history']}")
    return failures


async def check_hallucination_trap(query: str) -> list[str]:
    """REQUIREMENTS.md §6, case 2: context is graded CORRECT (topically
    relevant) but doesn't contain the specific fact asked — a naive
    generator would hallucinate. The groundedness check must catch this:
    either force a stricter regeneration, or escalate to low_confidence.
    It must NOT be silently accepted on the first pass."""
    failures: list[str] = []
    final_state = await _run_and_trace("DoD Case 2: hallucination trap", query)

    # A hallucination-trap query MUST go through correction_path to ever
    # reach the groundedness check at all — if the routing gate classified
    # it as fast_path (FR-18 explicitly skips grading), this test can't
    # validate anything, and the failure below would misleadingly look
    # like "the groundedness check didn't catch it" when the real cause
    # is "the groundedness check never ran." Surface that distinction
    # explicitly rather than letting it hide inside a generic failure.
    grading_occurred = bool(final_state.get("chunk_grades"))
    if not grading_occurred and final_state.get("groundedness_score") is None:
        print(
            "\n  [WARNING] No chunk_grades and no groundedness_score in "
            "final state — this query was likely fast-pathed by the "
            "routing gate (FR-18 skips grading by design), NOT caught or "
            "missed by the groundedness check. If your real trap-case "
            "query is short and has no comparison markers, classify_query() "
            "in routing_gate.py may be sending it down fast_path. Rephrase "
            "the query to be unambiguously correction_path-eligible, or "
            "test classify_query() on it directly first."
        )

    print("\nAssertions (REQUIREMENTS.md §6, case 2):")
    caught = (
        final_state["generation_attempts"] >= 2
        or final_state["low_confidence"] is True
    )
    _assert(
        caught,
        "groundedness check caught the hallucination — either forced a "
        "regeneration (generation_attempts >= 2) or escalated to "
        "low_confidence, rather than being accepted on the first pass",
        failures,
    )
    _assert(
        final_state.get("groundedness_score") is not None,
        "groundedness_score was actually computed and stored",
        failures,
    )

    if final_state["low_confidence"]:
        print(f"Escalated: {final_state['low_confidence_reason']}")
    else:
        print(f"Recovered after regeneration: {final_state['answer']!r}")
    print(f"generation_attempts: {final_state['generation_attempts']}")
    print(f"groundedness_score (final pass): {final_state.get('groundedness_score')}")
    return failures


async def check_unanswerable(query: str) -> list[str]:
    """REQUIREMENTS.md §6, case 3: neither the corpus nor the Tavily
    fallback contains the answer. The system must not hallucinate a
    confident-sounding response — it must exhaust the loop, return
    low_confidence: true with a reason code, and have actually attempted
    fallback (not skip straight to giving up)."""
    failures: list[str] = []
    final_state = await _run_and_trace(
        "DoD Case 3: unanswerable from corpus + fallback", query
    )

    print("\nAssertions (REQUIREMENTS.md §6, case 3):")
    _assert(
        final_state["low_confidence"] is True,
        "terminated with low_confidence: true (never a confident "
        "hallucination)",
        failures,
    )
    _assert(
        bool(final_state["low_confidence_reason"]),
        "a machine-readable reason code was set",
        failures,
    )
    _assert(
        final_state["fallback_used"] is True,
        "Tavily fallback was actually attempted before giving up",
        failures,
    )

    print(f"Reason: {final_state['low_confidence_reason']}")
    print(
        f"Iteration count: {final_state['iteration_count']} / "
        f"{final_state['max_iterations']}"
    )
    return failures


async def main() -> int:
    unfilled = [k for k, v in TRAP_CASE_QUERIES.items() if _PLACEHOLDER_MARKER in v]
    if unfilled:
        print(
            "ERROR: TRAP_CASE_QUERIES still has placeholder values for: "
            f"{unfilled}\n\n"
            "Fill in your actual NimbusPay DoD trap-case queries at the "
            "top of this script before running it — a generic query "
            "can't validate a trap case it wasn't specifically designed "
            "to trip."
        )
        return 1

    all_failures: list[str] = []
    all_failures += await check_term_collision(
        TRAP_CASE_QUERIES["term_collision_bad_retrieval"]
    )
    all_failures += await check_hallucination_trap(
        TRAP_CASE_QUERIES["hallucination_trap"]
    )
    all_failures += await check_unanswerable(TRAP_CASE_QUERIES["unanswerable"])

    print(f"\n{'=' * 70}")
    if all_failures:
        print(f"CHECKPOINT FAILED — {len(all_failures)} assertion(s) failed:")
        for f in all_failures:
            print(f"  - {f}")
        return 1

    print("CHECKPOINT PASSED — all 3 DoD trap cases behaved as required.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))