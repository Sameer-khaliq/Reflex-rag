"""
Pre-correction routing gate (FR-3).

Rule-based, zero LLM calls, zero added latency on the hot path — a
classifier here would defeat the entire purpose of a *fast* path.
Per IMPLEMENTATION_PLAN.md §1: default to correction_path on uncertainty,
because the two failure modes are asymmetric — a false positive (an easy
query routed to correction_path) costs one unnecessary grading pass; a
false negative (a hard query fast-pathed) ships an ungraded, unverified
answer with no safety net at all.

Three intended signals (IMPLEMENTATION_PLAN.md §1):
  1. Query token count — long queries are more likely multi-part.
  2. Multi-clause / comparison markers — "vs", "compare", "and also",
     multiple question marks.
  3. Corpus vocabulary overlap — a query that shares almost no vocabulary
     with the ingested corpus is itself a signal it probably needs
     correction (rewrite or fallback), not confident fast-path
     generation.

KNOWN GAP: signal 3 needs the corpus's known entity/topic vocabulary
set, built at ingestion time (Phase 2's scripts/ingest_corpus.py). That
artifact wasn't available when this module was written, so
`known_vocabulary` defaults to None and that signal is skipped entirely
rather than guessed at — skipping it does NOT make queries more
fast-path-eligible by default; it just means the gate is currently
running on 2 of its 3 intended signals. Wire the real vocabulary set in
via `known_vocabulary` once Phase 2's ingestion pipeline exposes it, and
flag this in PROJECT_OVERVIEW.md until it's done — don't silently call
the gate "complete" without it.
"""
from __future__ import annotations

from typing import Literal

from schemas.state import GraphState

_COMPARISON_MARKERS = (
    " vs ",
    " vs. ",
    " versus ",
    "compare",
    " and also",
    " as well as ",
)

_DEFAULT_MAX_FAST_PATH_WORDS = 12
_DEFAULT_MIN_VOCABULARY_OVERLAP = 0.3


def _has_comparison_marker(query: str) -> bool:
    lowered = f" {query.lower()} "
    if any(marker in lowered for marker in _COMPARISON_MARKERS):
        return True
    if lowered.count("?") > 1:
        return True
    return False


def _vocabulary_overlap_ratio(query: str, known_vocabulary: set[str]) -> float:
    words = {w.strip(".,!?").lower() for w in query.split() if len(w) > 2}
    if not words:
        return 1.0  # nothing meaningful to compare against; don't penalize
    overlap = words & known_vocabulary
    return len(overlap) / len(words)


def classify_query(
    query: str,
    known_vocabulary: set[str] | None = None,
    max_fast_path_words: int = _DEFAULT_MAX_FAST_PATH_WORDS,
    min_vocabulary_overlap: float = _DEFAULT_MIN_VOCABULARY_OVERLAP,
) -> Literal["fast_path", "correction_path"]:
    """Pure, independently-testable classifier — no state-shape
    dependency. `routing_gate_selector` below is the thin LangGraph
    adapter around this."""
    word_count = len(query.split())

    if word_count > max_fast_path_words:
        return "correction_path"

    if _has_comparison_marker(query):
        return "correction_path"

    if known_vocabulary is not None:
        overlap = _vocabulary_overlap_ratio(query, known_vocabulary)
        if overlap < min_vocabulary_overlap:
            return "correction_path"

    return "fast_path"


def routing_gate_selector(state: GraphState) -> Literal["fast_path", "correction_path"]:
    """The LangGraph conditional-edge function wired to START in
    graph.py."""
    return classify_query(state["query"])