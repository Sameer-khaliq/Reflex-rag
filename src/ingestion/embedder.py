"""
Gemini embedding wrapper — pins model/version/task_type/dimension in one place.
Ingestion uses task_type="document"; query path reuses this module with
task_type="query". Kept on Gemini per project decision — this is the one
non-Groq/OpenRouter provider dependency in the stack, isolated to embeddings
only,
"""
from __future__ import annotations

from google import genai
from google.genai import types

from config import get_config
from logging_config import get_logger


def get_genai_client() -> genai.Client:
    return genai.Client(api_key=get_config().settings.google_api_key)

def embed_texts(
    texts: list[str],
    task_type: str,
    trace_id: str = "embed",
) -> list[list[float]]:
    """
    Embed a batch of texts with the pinned model/dimension/task_type.

    task_type: "document" (ingestion) or "query" (retrieval) — never mix
    these silently, the two produce differently-optimized vectors.
    """
    logger = get_logger(trace_id=trace_id)
    client = get_genai_client()
    embedding_cfg = get_config().embedding

    gemini_task_type = "RETRIEVAL_DOCUMENT" if task_type == "document" else "RETRIEVAL_QUERY"

    result = client.models.embed_content(
        model=embedding_cfg.model,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=gemini_task_type,
            output_dimensionality=embedding_cfg.dimension,
        ),
    )

    embeddings = [e.values for e in result.embeddings]

    logger.info(
        "embedding_call",
        stage="embedding",
        task_type=task_type,
        model=embedding_cfg.model,
        dimension=embedding_cfg.dimension,
        num_texts=len(texts),
    )

    return embeddings


def get_embedding_config() -> dict:
    """Config snapshot logged alongside every stored vector."""
    embedding_cfg = get_config().embedding
    return {
        "model": embedding_cfg.model,
        "version": embedding_cfg.version,
        "dimension": embedding_cfg.dimension,
        "task_type": "document",
    }