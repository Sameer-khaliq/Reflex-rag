import pytest

from generation import generator


class _StubModelTiers:
    tier3_generation = "tier3_generation_stub"


class _StubConfig:
    model_tiers = _StubModelTiers()


@pytest.fixture(autouse=True)
def _stub_config(monkeypatch):
    monkeypatch.setattr(generator, "get_config", lambda: _StubConfig())


@pytest.mark.asyncio
async def test_generate_answer_calls_llm_with_context_and_query(monkeypatch):
    captured = {}

    async def fake_call(slug_pair, system_prompt, user_prompt, trace_id=None):
        captured["slug_pair"] = slug_pair
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return "The refund takes 5 business days."

    monkeypatch.setattr(generator, "call_with_failover", fake_call)

    answer = await generator.generate_answer(
        query="How long do refunds take?",
        context_chunks=["Refunds are processed within 5 business days."],
    )

    assert answer == "The refund takes 5 business days."
    assert "Refunds are processed within 5 business days." in captured["user_prompt"]
    assert "How long do refunds take?" in captured["user_prompt"]
    assert captured["slug_pair"] == "tier3_generation_stub"


@pytest.mark.asyncio
async def test_generate_answer_uses_default_prompt_when_not_strict(monkeypatch):
    captured = {}

    async def fake_call(slug_pair, system_prompt, user_prompt, trace_id=None):
        captured["system_prompt"] = system_prompt
        return "answer"

    monkeypatch.setattr(generator, "call_with_failover", fake_call)

    await generator.generate_answer(query="q", context_chunks=["c"], strict=False)

    assert "regeneration" not in captured["system_prompt"].lower()


@pytest.mark.asyncio
async def test_generate_answer_uses_strict_prompt_on_regeneration(monkeypatch):
    captured = {}

    async def fake_call(slug_pair, system_prompt, user_prompt, trace_id=None):
        captured["system_prompt"] = system_prompt
        return "strict answer"

    monkeypatch.setattr(generator, "call_with_failover", fake_call)

    await generator.generate_answer(query="q", context_chunks=["c"], strict=True)

    assert "regeneration" in captured["system_prompt"].lower()
    assert "ONLY" in captured["system_prompt"]


@pytest.mark.asyncio
async def test_generate_answer_joins_multiple_chunks(monkeypatch):
    captured = {}

    async def fake_call(slug_pair, system_prompt, user_prompt, trace_id=None):
        captured["user_prompt"] = user_prompt
        return "answer"

    monkeypatch.setattr(generator, "call_with_failover", fake_call)

    await generator.generate_answer(query="q", context_chunks=["chunk one", "chunk two"])

    assert "chunk one" in captured["user_prompt"]
    assert "chunk two" in captured["user_prompt"]


@pytest.mark.asyncio
async def test_generate_answer_handles_empty_context_without_crashing(monkeypatch):
    async def fake_call(slug_pair, system_prompt, user_prompt, trace_id=None):
        return "no context to work with"

    monkeypatch.setattr(generator, "call_with_failover", fake_call)

    answer = await generator.generate_answer(query="q", context_chunks=[])
    assert answer == "no context to work with"