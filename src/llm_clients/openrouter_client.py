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


async def call_openrouter(model: str, system_prompt: str, user_prompt: str) -> str:
    client = get_client()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""