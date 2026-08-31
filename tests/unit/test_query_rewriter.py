import pytest

import rewriting.query_rewriter as query_rewriter


class _CannedResponses:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0

    async def __call__(self, slug_pair, system_prompt, user_prompt, trace_id="x"):
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


@pytest.mark.asyncio
async def test_rewrite_differs_from_original_with_empty_history(monkeypatch):
    monkeypatch.setattr(
        query_rewriter,
        "call_with_failover",
        _CannedResponses(['{"rewritten_query": "how long does an ACH refund take specifically"}']),
    )
    result = await query_rewriter.rewrite_query("refund timing", [])
    assert result == "how long does an ACH refund take specifically"
    assert result != "refund timing"


@pytest.mark.asyncio
async def test_second_rewrite_does_not_repeat_history(monkeypatch):
    canned = _CannedResponses(
        [
            '{"rewritten_query": "already tried this"}',  # duplicate of history entry
            '{"rewritten_query": "a genuinely new reformulation"}',
        ]
    )
    monkeypatch.setattr(query_rewriter, "call_with_failover", canned)

    result = await query_rewriter.rewrite_query(
        "original query", rewrite_history=["already tried this"]
    )
    assert result == "a genuinely new reformulation"
    assert canned.call_count == 2


@pytest.mark.asyncio
async def test_duplicate_check_is_case_and_whitespace_insensitive(monkeypatch):
    canned = _CannedResponses(
        [
            '{"rewritten_query": "  Already Tried This  "}',
            '{"rewritten_query": "different this time"}',
        ]
    )
    monkeypatch.setattr(query_rewriter, "call_with_failover", canned)

    result = await query_rewriter.rewrite_query(
        "original query", rewrite_history=["already tried this"]
    )
    assert result == "different this time"


@pytest.mark.asyncio
async def test_malformed_response_retries_then_succeeds(monkeypatch):
    canned = _CannedResponses(["not json", '{"rewritten_query": "valid rewrite"}'])
    monkeypatch.setattr(query_rewriter, "call_with_failover", canned)

    result = await query_rewriter.rewrite_query("original query", [])
    assert result == "valid rewrite"
    assert canned.call_count == 2


@pytest.mark.asyncio
async def test_exhausted_attempts_raises_rewrite_generation_failed(monkeypatch):
    canned = _CannedResponses(["not json", "still not json", "still not json either"])
    monkeypatch.setattr(query_rewriter, "call_with_failover", canned)

    with pytest.raises(query_rewriter.RewriteGenerationFailed):
        await query_rewriter.rewrite_query("original query", [])
    assert canned.call_count == 3