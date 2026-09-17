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
    has_ambiguous: bool = False,
) -> Verdict:
    # 1. Enough correct context found -> proceed to answer synthesis
    if p_correct >= p_correct_threshold:
        return Verdict.GENERATE

    # 2. Attempt rewrites if iteration budget remains
    if iteration_count < max_iterations:
        return Verdict.REWRITE

    # 3. Budget exhausted: Fallback web search if not yet used
    if not fallback_used:
        # Partial ambiguity defaults to last resort; total zero defaults to direct
        return (
            Verdict.FALLBACK_LAST_RESORT
            if (p_correct > 0.0 or has_ambiguous)
            else Verdict.FALLBACK_DIRECT
        )

    # 4. Fallback already attempted and budget dead -> terminal stop
    return Verdict.TERMINAL_LOW_CONFIDENCE


def filter_accepted_chunks(
    chunks: list[dict], chunk_grades: list[ChunkGrade]
) -> list[dict]:
    incorrect_ids = {g.chunk_id for g in chunk_grades if g.grade == "INCORRECT"}
    accepted = []

    for idx, c in enumerate(chunks):
        cid = c.get("chunk_id") or c.get("id") or c.get("url")
        cid_str = str(cid) if cid is not None else f"fallback_chunk_{idx}"
        if cid_str not in incorrect_ids:
            accepted.append(c)

    return accepted


def aggregate(
    chunks: list[dict],
    chunk_grades: list[ChunkGrade],
    iteration_count: int,
    max_iterations: int,
    fallback_used: bool,
    p_correct_threshold: float,
) -> dict:
    p_correct = compute_p_correct(chunk_grades)
    has_ambiguous = any(g.grade == "AMBIGUOUS" for g in chunk_grades)

    verdict = decide_verdict(
        p_correct=p_correct,
        iteration_count=iteration_count,
        max_iterations=max_iterations,
        fallback_used=fallback_used,
        p_correct_threshold=p_correct_threshold,
        has_ambiguous=has_ambiguous,
    )

    accepted_chunks = (
        filter_accepted_chunks(chunks, chunk_grades)
        if verdict == Verdict.GENERATE
        else []
    )

    return {
        "p_correct": p_correct,
        "verdict": verdict,
        "accepted_chunks": accepted_chunks,
    }


def main():
    print("--- Running Retrieval Aggregation Verification ---\n")

    sample_chunks = [
        {"chunk_id": "c1", "text": "API Rate Limits are 500 requests per minute."},
        {"chunk_id": "c2", "text": "Billing invoice generation occurs monthly."},
        {"chunk_id": "c3", "text": "Rate limit headers include X-RateLimit-Reset."},
    ]

    # Scenario 1: High confidence (2 CORRECT, 1 INCORRECT) -> GENERATE
    grades_high = [
        ChunkGrade(chunk_id="c1", grade="CORRECT"),
        ChunkGrade(chunk_id="c2", grade="INCORRECT"),
        ChunkGrade(chunk_id="c3", grade="CORRECT"),
    ]
    res_high = aggregate(
        chunks=sample_chunks,
        chunk_grades=grades_high,
        iteration_count=0,
        max_iterations=2,
        fallback_used=False,
        p_correct_threshold=0.5,
    )
    print(f"Scenario 1 (High Quality Retrieval):")
    print(f"  p_correct: {res_high['p_correct']:.2f}")
    print(f"  Verdict  : {res_high['verdict']}")
    print(f"  Accepted Chunks: {[c['chunk_id'] for c in res_high['accepted_chunks']]}")
    print("-" * 50)

    # Scenario 2: Low confidence, within budget -> REWRITE
    grades_low = [
        ChunkGrade(chunk_id="c1", grade="INCORRECT"),
        ChunkGrade(chunk_id="c2", grade="AMBIGUOUS"),
        ChunkGrade(chunk_id="c3", grade="INCORRECT"),
    ]
    res_rewrite = aggregate(
        chunks=sample_chunks,
        chunk_grades=grades_low,
        iteration_count=0,
        max_iterations=2,
        fallback_used=False,
        p_correct_threshold=0.5,
    )
    print(f"Scenario 2 (Ambiguous Context, Iteration 0 of 2):")
    print(f"  p_correct: {res_rewrite['p_correct']:.2f}")
    print(f"  Verdict  : {res_rewrite['verdict']}")
    print("-" * 50)

    # Scenario 3: Iterations exhausted, fallback available -> FALLBACK_LAST_RESORT
    res_fallback = aggregate(
        chunks=sample_chunks,
        chunk_grades=grades_low,
        iteration_count=2,
        max_iterations=2,
        fallback_used=False,
        p_correct_threshold=0.5,
    )
    print(f"Scenario 3 (Budget Exhausted, Fallback Unused):")
    print(f"  p_correct: {res_fallback['p_correct']:.2f}")
    print(f"  Verdict  : {res_fallback['verdict']}")
    print("-" * 50)

    # Scenario 4: Everything exhausted -> TERMINAL_LOW_CONFIDENCE
    res_terminal = aggregate(
        chunks=sample_chunks,
        chunk_grades=grades_low,
        iteration_count=2,
        max_iterations=2,
        fallback_used=True,
        p_correct_threshold=0.5,
    )
    print(f"Scenario 4 (All Attempts Dead):")
    print(f"  Verdict  : {res_terminal['verdict']}")


if __name__ == "__main__":
    main()