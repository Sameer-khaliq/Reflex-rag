import json

import pytest

from grading import answer_grader
from schemas.answer_grade import GroundednessGrade, RelevanceGrade


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
async def test_grade_groundedness_parses_valid_response(monkeypatch):
    async def fake_call(*args, **kwargs):
        return json.dumps({"score": 0.9, "reasoning": "Fully supported by context."})

    monkeypatch.setattr(answer_grader, "call_with_failover", fake_call)

    grade = await answer_grader.grade_groundedness(
        "The refund takes 5 days.", "Refunds take 5 business days."
    )
    assert isinstance(grade, GroundednessGrade)
    assert grade.score == 0.9


@pytest.mark.asyncio
async def test_grade_relevance_parses_valid_response(monkeypatch):
    async def fake_call(*args, **kwargs):
        return json.dumps({"score": 0.95, "reasoning": "Directly answers the question."})

    monkeypatch.setattr(answer_grader, "call_with_failover", fake_call)

    grade = await answer_grader.grade_relevance(
        "Refunds take 5 days.", "How long do refunds take?"
    )
    assert isinstance(grade, RelevanceGrade)
    assert grade.score == 0.95


@pytest.mark.asyncio
async def test_grade_groundedness_strips_markdown_fences(monkeypatch):
    async def fake_call(*args, **kwargs):
        return '```json\n{"score": 0.4, "reasoning": "Partially supported."}\n```'

    monkeypatch.setattr(answer_grader, "call_with_failover", fake_call)

    grade = await answer_grader.grade_groundedness("answer", "context")
    assert grade.score == 0.4


@pytest.mark.asyncio
async def test_grade_groundedness_fails_closed_on_malformed_output(monkeypatch):
    """Per IMPLEMENTATION_PLAN.md §3: malformed output gets one retry, then
    fails closed to score 0.0 — never defaults to a passing score."""
    call_count = 0

    async def fake_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return "this is not json at all"

    monkeypatch.setattr(answer_grader, "call_with_failover", fake_call)

    grade = await answer_grader.grade_groundedness("answer", "context")
    assert grade.score == 0.0
    assert call_count == 2  # one initial attempt + one stricter reprompt retry


@pytest.mark.asyncio
async def test_grade_relevance_fails_closed_on_malformed_output(monkeypatch):
    call_count = 0

    async def fake_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return "garbage output"

    monkeypatch.setattr(answer_grader, "call_with_failover", fake_call)

    grade = await answer_grader.grade_relevance("answer", "query")
    assert grade.score == 0.0
    assert call_count == 2


@pytest.mark.asyncio
async def test_grade_groundedness_recovers_on_retry(monkeypatch):
    """First response malformed, second (stricter reprompt) response valid —
    proves the retry path actually recovers, not just fails closed."""
    responses = iter(["not json", json.dumps({"score": 0.85, "reasoning": "ok"})])

    async def fake_call(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr(answer_grader, "call_with_failover", fake_call)

    grade = await answer_grader.grade_groundedness("answer", "context")
    assert grade.score == 0.85


@pytest.mark.asyncio
async def test_score_out_of_schema_range_triggers_fail_closed(monkeypatch):
    """A score outside [0.0, 1.0] fails Pydantic validation just like
    malformed JSON does — same fail-closed path, not a crash."""
    call_count = 0

    async def fake_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return json.dumps({"score": 1.5, "reasoning": "out of range"})

    monkeypatch.setattr(answer_grader, "call_with_failover", fake_call)

    grade = await answer_grader.grade_groundedness("answer", "context")
    assert grade.score == 0.0
    assert call_count == 2


def test_passes_answer_gate_both_above_threshold(monkeypatch):
    monkeypatch.setattr(answer_grader, "get_config", lambda: _StubConfig())
    g = GroundednessGrade(score=0.8, reasoning="ok")
    r = RelevanceGrade(score=0.75, reasoning="ok")
    assert answer_grader.passes_answer_gate(g, r) is True


def test_passes_answer_gate_fails_if_groundedness_below_threshold(monkeypatch):
    monkeypatch.setattr(answer_grader, "get_config", lambda: _StubConfig())
    g = GroundednessGrade(score=0.5, reasoning="ungrounded")
    r = RelevanceGrade(score=0.9, reasoning="on topic")
    assert answer_grader.passes_answer_gate(g, r) is False


def test_passes_answer_gate_fails_if_relevance_below_threshold(monkeypatch):
    monkeypatch.setattr(answer_grader, "get_config", lambda: _StubConfig())
    g = GroundednessGrade(score=0.9, reasoning="grounded")
    r = RelevanceGrade(score=0.5, reasoning="off topic")
    assert answer_grader.passes_answer_gate(g, r) is False


def test_score_out_of_range_rejected_by_schema_directly():
    with pytest.raises(Exception):
        GroundednessGrade(score=1.5, reasoning="invalid")