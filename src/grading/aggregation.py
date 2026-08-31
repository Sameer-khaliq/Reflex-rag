"""
Aggregation of per-chunk grades into a retrieval-level verdict (FR-5).

Pure functions — no I/O, no LLM calls — implementing
IMPLEMENTATION_PLAN.md §3.2 exactly:

    let p_correct = proportion of chunks graded CORRECT

    if p_correct >= threshold:
        -> drop INCORRECT chunks, keep CORRECT + AMBIGUOUS -> generate

    elif 0 < p_correct < threshold:
        if iteration_count < max_iterations:
            -> rewrite_query
        else:
            -> fallback_retrieval (last resort before terminal)

    elif p_correct == 0:
        if not fallback_used:
            -> fallback_retrieval directly
        else:
            -> terminal: low_confidence
"""
from __future__ import annotations

from enum import Enum

from schemas.chunk_grade import ChunkGrade


class Verdict(str, Enum):
    GENERATE = "generate"
    REWRITE = "rewrite"
    FALLBACK_DIRECT = "fallback_direct"
    FALLBACK_LAST_RESORT = "fallback_last_resort"
    TERMINAL_LOW_CONFIDENCE = "terminal_low_confidence"


def compute_p_correct(chunk_grades: list[ChunkGrade]) -> float:
    """
    Empty chunk_grades returns 0.0, not a ZeroDivisionError — deliberate,
    not an oversight. An empty retrieval result routes down the same
    p_correct == 0 branch as a retrieval that returned chunks but graded
    all of them INCORRECT (see IMPLEMENTATION_PLAN.md §5's explicit
    empty-results edge case).
    """
    if not chunk_grades:
        return 0.0
    correct_count = sum(1 for g in chunk_grades if g.grade == "CORRECT")
    return correct_count / len(chunk_grades)


def decide_verdict(
    p_correct: float,
    iteration_count: int,
    max_iterations: int,
    fallback_used: bool,
    p_correct_threshold: float,
) -> Verdict:
    if p_correct >= p_correct_threshold:
        return Verdict.GENERATE

    if p_correct == 0.0:
        return Verdict.FALLBACK_DIRECT if not fallback_used else Verdict.TERMINAL_LOW_CONFIDENCE

    # 0 < p_correct < threshold
    return Verdict.REWRITE if iteration_count < max_iterations else Verdict.FALLBACK_LAST_RESORT


def filter_accepted_chunks(chunks: list[dict], chunk_grades: list[ChunkGrade]) -> list[dict]:
    """
    Drops INCORRECT chunks, keeps CORRECT + AMBIGUOUS. Only meaningful
    when decide_verdict() returned GENERATE — the accepted context for
    generation is the CORRECT+AMBIGUOUS subset, not the full retrieved
    set.
    """
    incorrect_ids = {g.chunk_id for g in chunk_grades if g.grade == "INCORRECT"}
    return [c for c in chunks if str(c.get("chunk_id")) not in incorrect_ids]


def aggregate(
    chunks: list[dict],
    chunk_grades: list[ChunkGrade],
    iteration_count: int,
    max_iterations: int,
    fallback_used: bool,
    p_correct_threshold: float,
) -> dict:
    """
    Single entrypoint tying the three functions above together — this
    is what the orchestration node (Phase 7) actually calls.

    Returns:
        {
            "p_correct": float,
            "verdict": Verdict,
            "accepted_chunks": [...],  # non-empty only when verdict == GENERATE
        }
    """
    p_correct = compute_p_correct(chunk_grades)
    verdict = decide_verdict(
        p_correct, iteration_count, max_iterations, fallback_used, p_correct_threshold
    )
    accepted_chunks = (
        filter_accepted_chunks(chunks, chunk_grades) if verdict == Verdict.GENERATE else []
    )
    return {"p_correct": p_correct, "verdict": verdict, "accepted_chunks": accepted_chunks}