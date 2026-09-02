import json

import pytest

from grading import answer_grader
from schemas.answer_grade import AnswerGrade


class _StubModelTiers:
    tier1_grading = "tier1_grading_stub"


class _StubThresholds:
    groundedness_threshold = 0.7
    relevance_threshold = 0.7


class _StubConfig:
    thresholds = _StubThresholds()
    model_tiers = _StubModelTiers()


@pytest.fixture(autouse=True)
def _stub_config(monkeypatch):
    monkeypatch.setattr(answer_grader, "get_config", lambda: _StubConfig())


@pytest.mark.asyncio
async def test_grade_answer_parses_valid_response(monkeypatch):
    async def fake_call(*args, **kwargs):
        return json.dumps({"groundedness_score": 0.9, "relevance_score": 0.85})

    monkeypatch.setattr(answer_grader, "call_with_failover", fake_call)

    grade = await answer_grader.grade_answer("answer", "context", "query")
    assert isinstance(grade, AnswerGrade)
    assert grade.groundedness_score == 0.9
    assert grade.relevance_score == 0.85


@pytest.mark.asyncio
async def test_grade_answer_strips_markdown_fences(monkeypatch):
    async def fake_call(*args, **kwargs):
        return '```json\n{"groundedness_score": 0.4, "relevance_score": 0.6}\n```'

    monkeypatch.setattr(answer_grader, "call_with_failover", fake_call)

    grade = await answer_grader.grade_answer("answer", "context", "query")
    assert grade.groundedness_score == 0.4
    assert grade.relevance_score == 0.6


@pytest.mark.asyncio
async def test_grade_answer_fails_closed_on_malformed_output(monkeypatch):
    """Per IMPLEMENTATION_PLAN.md §3: malformed output gets one retry, then
    BOTH scores fail closed to 0.0 — never defaults to a passing score."""
    call_count = 0

    async def fake_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return "this is not json at all"

    monkeypatch.setattr(answer_grader, "call_with_failover", fake_call)

    grade = await answer_grader.grade_answer("answer", "context", "query")
    assert grade.groundedness_score == 0.0
    assert grade.relevance_score == 0.0
    assert call_count == 2  # one initial attempt + one stricter reprompt retry


@pytest.mark.asyncio
async def test_grade_answer_recovers_on_retry(monkeypatch):
    """First response malformed, second (stricter reprompt) response valid —
    proves the retry path actually recovers, not just fails closed."""
    responses = iter(
        ["not json", json.dumps({"groundedness_score": 0.8, "relevance_score": 0.75})]
    )

    async def fake_call(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr(answer_grader, "call_with_failover", fake_call)

    grade = await answer_grader.grade_answer("answer", "context", "query")
    assert grade.groundedness_score == 0.8
    assert grade.relevance_score == 0.75


@pytest.mark.asyncio
async def test_grade_answer_fails_closed_on_out_of_range_score(monkeypatch):
    """A score outside [0.0, 1.0] fails Pydantic validation just like
    malformed JSON — same fail-closed path, not a crash."""
    call_count = 0

    async def fake_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return json.dumps({"groundedness_score": 1.5, "relevance_score": 0.9})

    monkeypatch.setattr(answer_grader, "call_with_failover", fake_call)

    grade = await answer_grader.grade_answer("answer", "context", "query")
    assert grade.groundedness_score == 0.0
    assert grade.relevance_score == 0.0
    assert call_count == 2


def test_classify_outcome_accepts_when_both_pass(monkeypatch):
    monkeypatch.setattr(answer_grader, "get_config", lambda: _StubConfig())
    grade = AnswerGrade(groundedness_score=0.8, relevance_score=0.75)
    assert answer_grader.classify_outcome(grade) == "accept"


def test_classify_outcome_relevance_fail_checked_first(monkeypatch):
    """Per REQUIREMENTS.md §3.4, relevance is checked before groundedness —
    even if groundedness also fails, a low-relevance case must classify as
    relevance_fail, not groundedness_fail."""
    monkeypatch.setattr(answer_grader, "get_config", lambda: _StubConfig())
    grade = AnswerGrade(groundedness_score=0.3, relevance_score=0.2)
    assert answer_grader.classify_outcome(grade) == "relevance_fail"


def test_classify_outcome_groundedness_fail_when_relevance_ok(monkeypatch):
    monkeypatch.setattr(answer_grader, "get_config", lambda: _StubConfig())
    grade = AnswerGrade(groundedness_score=0.4, relevance_score=0.9)
    assert answer_grader.classify_outcome(grade) == "groundedness_fail"


def test_classify_outcome_relevance_fail_even_with_high_groundedness(monkeypatch):
    """A fully-grounded but off-topic answer must still route to
    relevance_fail, not accept — the two gates are independent."""
    monkeypatch.setattr(answer_grader, "get_config", lambda: _StubConfig())
    grade = AnswerGrade(groundedness_score=0.95, relevance_score=0.5)
    assert answer_grader.classify_outcome(grade) == "relevance_fail"


def test_score_out_of_range_rejected_by_schema_directly():
    with pytest.raises(Exception):
        AnswerGrade(groundedness_score=1.5, relevance_score=0.5)