from __future__ import annotations

from openai import AsyncOpenAI

from config import get_config

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        cfg = get_config()
        _client = AsyncOpenAI(
            api_key=cfg.settings.openrouter_api_key,
            base_url=cfg.providers.openrouter.base_url,
        )
    return _client


async def call_openrouter(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int | None = None,
    **kwargs,
) -> str:
    client = get_client()
    create_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens
    create_kwargs.update(kwargs)

    response = await client.chat.completions.create(**create_kwargs)
    return response.choices[0].message.content or ""