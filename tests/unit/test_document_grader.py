import pytest

import grading.document_grader as document_grader
from schemas.chunk_grade import ChunkGrade


class _CannedResponses:
    """Returns each response in order on successive calls, ignoring args."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0

    async def __call__(self, slug_pair, system_prompt, user_prompt, trace_id="x"):
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


@pytest.mark.asyncio
async def test_correct_grade_parses(monkeypatch):
    monkeypatch.setattr(
        document_grader, "call_with_failover", _CannedResponses(['{"grade": "CORRECT"}'])
    )
    grade = await document_grader.grade_chunk("q", "chunk-1", "some text")
    assert grade == ChunkGrade(chunk_id="chunk-1", grade="CORRECT")


@pytest.mark.asyncio
async def test_ambiguous_grade_parses(monkeypatch):
    monkeypatch.setattr(
        document_grader, "call_with_failover", _CannedResponses(['{"grade": "AMBIGUOUS"}'])
    )
    grade = await document_grader.grade_chunk("q", "chunk-2", "some text")
    assert grade == ChunkGrade(chunk_id="chunk-2", grade="AMBIGUOUS")


@pytest.mark.asyncio
async def test_incorrect_grade_parses(monkeypatch):
    monkeypatch.setattr(
        document_grader, "call_with_failover", _CannedResponses(['{"grade": "INCORRECT"}'])
    )
    grade = await document_grader.grade_chunk("q", "chunk-3", "some text")
    assert grade == ChunkGrade(chunk_id="chunk-3", grade="INCORRECT")


@pytest.mark.asyncio
async def test_malformed_once_then_valid_on_retry(monkeypatch):
    monkeypatch.setattr(
        document_grader,
        "call_with_failover",
        _CannedResponses(["not json at all", '{"grade": "CORRECT"}']),
    )
    grade = await document_grader.grade_chunk("q", "chunk-4", "some text")
    assert grade == ChunkGrade(chunk_id="chunk-4", grade="CORRECT")


@pytest.mark.asyncio
async def test_malformed_twice_fails_closed_to_ambiguous(monkeypatch):
    canned = _CannedResponses(["not json", "still not json"])
    monkeypatch.setattr(document_grader, "call_with_failover", canned)
    grade = await document_grader.grade_chunk("q", "chunk-5", "some text")
    assert grade == ChunkGrade(chunk_id="chunk-5", grade="AMBIGUOUS")
    assert canned.call_count == 2


@pytest.mark.asyncio
async def test_invalid_enum_value_fails_closed_to_ambiguous(monkeypatch):
    canned = _CannedResponses(['{"grade": "MAYBE"}', '{"grade": "ALSO_INVALID"}'])
    monkeypatch.setattr(document_grader, "call_with_failover", canned)
    grade = await document_grader.grade_chunk("q", "chunk-6", "some text")
    assert grade == ChunkGrade(chunk_id="chunk-6", grade="AMBIGUOUS")


@pytest.mark.asyncio
async def test_grade_chunks_runs_concurrently_and_preserves_order(monkeypatch):
    # Map by chunk text (present in user_prompt) rather than call order —
    # grade_chunks runs chunks concurrently via asyncio.gather, so a
    # shared response iterator would be a race condition, not a real test.
    grade_by_text = {
        "text a": '{"grade": "CORRECT"}',
        "text b": '{"grade": "INCORRECT"}',
        "text c": '{"grade": "AMBIGUOUS"}',
    }

    async def _fake_call(slug_pair, system_prompt, user_prompt, trace_id="x"):
        for text, response in grade_by_text.items():
            if text in user_prompt:
                return response
        raise AssertionError(f"Unexpected user_prompt: {user_prompt}")

    monkeypatch.setattr(document_grader, "call_with_failover", _fake_call)

    chunks = [
        {"chunk_id": "a", "text": "text a"},
        {"chunk_id": "b", "text": "text b"},
        {"chunk_id": "c", "text": "text c"},
    ]
    grades = await document_grader.grade_chunks("q", chunks)
    assert {g.chunk_id: g.grade for g in grades} == {"a": "CORRECT", "b": "INCORRECT", "c": "AMBIGUOUS"}
    assert [g.chunk_id for g in grades] == ["a", "b", "c"]