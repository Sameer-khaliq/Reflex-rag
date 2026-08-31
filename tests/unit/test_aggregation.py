from grading.aggregation import (
    Verdict,
    aggregate,
    compute_p_correct,
    decide_verdict,
    filter_accepted_chunks,
)
from schemas.chunk_grade import ChunkGrade

THRESHOLD = 0.5


def _grades(*grades: str) -> list[ChunkGrade]:
    return [ChunkGrade(chunk_id=str(i), grade=g) for i, g in enumerate(grades)]


def test_compute_p_correct_empty_list_is_zero_not_a_crash():
    assert compute_p_correct([]) == 0.0


def test_compute_p_correct_basic_proportion():
    grades = _grades("CORRECT", "CORRECT", "INCORRECT", "AMBIGUOUS")
    assert compute_p_correct(grades) == 0.5


def test_compute_p_correct_all_correct():
    assert compute_p_correct(_grades("CORRECT", "CORRECT")) == 1.0


def test_compute_p_correct_all_incorrect():
    assert compute_p_correct(_grades("INCORRECT", "INCORRECT")) == 0.0


def test_boundary_p_correct_exactly_threshold_routes_to_generate():
    verdict = decide_verdict(
        p_correct=0.5,
        iteration_count=0,
        max_iterations=3,
        fallback_used=False,
        p_correct_threshold=0.5,
    )
    assert verdict == Verdict.GENERATE


def test_boundary_p_correct_just_below_threshold_routes_to_rewrite():
    verdict = decide_verdict(
        p_correct=0.49,
        iteration_count=0,
        max_iterations=3,
        fallback_used=False,
        p_correct_threshold=0.5,
    )
    assert verdict == Verdict.REWRITE


def test_p_correct_between_zero_and_threshold_rewrites_when_under_cap():
    verdict = decide_verdict(
        p_correct=0.3,
        iteration_count=1,
        max_iterations=3,
        fallback_used=False,
        p_correct_threshold=0.5,
    )
    assert verdict == Verdict.REWRITE


def test_p_correct_between_zero_and_threshold_falls_back_at_cap():
    verdict = decide_verdict(
        p_correct=0.3,
        iteration_count=3,
        max_iterations=3,
        fallback_used=False,
        p_correct_threshold=0.5,
    )
    assert verdict == Verdict.FALLBACK_LAST_RESORT


def test_boundary_p_correct_exactly_zero_triggers_fallback_direct_first_time():
    verdict = decide_verdict(
        p_correct=0.0,
        iteration_count=0,
        max_iterations=3,
        fallback_used=False,
        p_correct_threshold=0.5,
    )
    assert verdict == Verdict.FALLBACK_DIRECT


def test_boundary_p_correct_exactly_zero_terminates_after_fallback_already_used():
    verdict = decide_verdict(
        p_correct=0.0,
        iteration_count=1,
        max_iterations=3,
        fallback_used=True,
        p_correct_threshold=0.5,
    )
    assert verdict == Verdict.TERMINAL_LOW_CONFIDENCE


def test_filter_accepted_chunks_drops_incorrect_keeps_correct_and_ambiguous():
    chunks = [
        {"chunk_id": "1", "text": "a"},
        {"chunk_id": "2", "text": "b"},
        {"chunk_id": "3", "text": "c"},
    ]
    grades = [
        ChunkGrade(chunk_id="1", grade="CORRECT"),
        ChunkGrade(chunk_id="2", grade="INCORRECT"),
        ChunkGrade(chunk_id="3", grade="AMBIGUOUS"),
    ]
    accepted = filter_accepted_chunks(chunks, grades)
    assert {c["chunk_id"] for c in accepted} == {"1", "3"}


def test_aggregate_empty_chunk_list_routes_to_fallback_direct():
    result = aggregate(
        chunks=[],
        chunk_grades=[],
        iteration_count=0,
        max_iterations=3,
        fallback_used=False,
        p_correct_threshold=THRESHOLD,
    )
    assert result["p_correct"] == 0.0
    assert result["verdict"] == Verdict.FALLBACK_DIRECT
    assert result["accepted_chunks"] == []


def test_aggregate_generate_case_returns_filtered_accepted_chunks():
    chunks = [{"chunk_id": "1", "text": "a"}, {"chunk_id": "2", "text": "b"}]
    grades = [
        ChunkGrade(chunk_id="1", grade="CORRECT"),
        ChunkGrade(chunk_id="2", grade="INCORRECT"),
    ]
    result = aggregate(
        chunks=chunks,
        chunk_grades=grades,
        iteration_count=0,
        max_iterations=3,
        fallback_used=False,
        p_correct_threshold=THRESHOLD,
    )
    assert result["verdict"] == Verdict.GENERATE
    assert [c["chunk_id"] for c in result["accepted_chunks"]] == ["1"]