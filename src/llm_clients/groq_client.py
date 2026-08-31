from __future__ import annotations

from groq import AsyncGroq

from config import get_config

_client: AsyncGroq | None = None


def get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=get_config().settings.groq_api_key)
    return _client


async def call_groq(model: str, system_prompt: str, user_prompt: str) -> str:
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