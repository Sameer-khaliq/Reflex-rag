from __future__ import annotations

from groq import AsyncGroq

from config import get_config

_client: AsyncGroq | None = None


def get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=get_config().settings.groq_api_key)
    return _client


async def call_groq(
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