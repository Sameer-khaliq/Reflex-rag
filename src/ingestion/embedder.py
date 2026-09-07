"""
Gemini embedding wrapper — pins model/version/task_type/dimension in one place.
Ingestion uses task_type="document"; query path reuses this module with
task_type="query". Includes batch slicing (<=90 items) and retry backoff.
"""
from __future__ import annotations

import time
from typing import Any
from google import genai
from google.genai import types

from config import get_config
from logging_config import get_logger

MAX_EMBED_BATCH_SIZE = 90  # Safe limit below Gemini's 100 items ceiling


def get_genai_client() -> genai.Client:
    return genai.Client(api_key=get_config().settings.google_api_key)


def _embed_batch_with_retry(
    client: genai.Client,
    model: str,
    contents: list[str],
    gemini_task_type: str,
    dimension: int,
    max_retries: int = 3,
    base_delay_s: float = 1.0,
) -> list[list[float]]:
    """Executes single batch call with exponential backoff on transient errors."""
    config = types.EmbedContentConfig(
        task_type=gemini_task_type,
        output_dimensionality=dimension,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            result = client.models.embed_content(
                model=model,
                contents=contents,
                config=config,
            )
            return [e.values for e in result.embeddings]
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                sleep_time = base_delay_s * (2 ** attempt)
                time.sleep(sleep_time)
            else:
                break

    raise RuntimeError(
        f"Embedding batch failed after {max_retries} retries. Cause: {last_error}"
    ) from last_error


def embed_texts(
    texts: list[str],
    task_type: str,
    trace_id: str = "embed",
) -> list[list[float]]:
    """
    Embed a batch of texts with pinned model/dimension/task_type.
    Auto-slices lists exceeding Gemini API payload limits.
    """
    if not texts:
        return []

    if task_type not in ("document", "query"):
        raise ValueError(f"task_type must be 'document' or 'query', received: '{task_type}'")

    logger = get_logger(trace_id=trace_id)
    client = get_genai_client()
    cfg = get_config()
    embedding_cfg = cfg.embedding
    resilience_cfg = cfg.resilience

    gemini_task_type = "RETRIEVAL_DOCUMENT" if task_type == "document" else "RETRIEVAL_QUERY"
    all_embeddings: list[list[float]] = []

    # Process in safe batches (handles 1 text at query time or 200+ at ingestion time)
    for i in range(0, len(texts), MAX_EMBED_BATCH_SIZE):
        batch = texts[i : i + MAX_EMBED_BATCH_SIZE]
        batch_embeddings = _embed_batch_with_retry(
            client=client,
            model=embedding_cfg.model,
            contents=batch,
            gemini_task_type=gemini_task_type,
            dimension=embedding_cfg.dimension,
            max_retries=resilience_cfg.max_retries,
            base_delay_s=resilience_cfg.base_delay_s,
        )
        all_embeddings.extend(batch_embeddings)

    # Runtime dimension guard
    if all_embeddings and len(all_embeddings[0]) != embedding_cfg.dimension:
        raise ValueError(
            f"Embedding dimension mismatch: expected {embedding_cfg.dimension}, got {len(all_embeddings[0])}"
        )

    logger.info(
        "embedding_call",
        stage="embedding",
        task_type=task_type,
        model=embedding_cfg.model,
        dimension=embedding_cfg.dimension,
        num_texts=len(texts),
        batches_processed=(len(texts) + MAX_EMBED_BATCH_SIZE - 1) // MAX_EMBED_BATCH_SIZE,
    )

    return all_embeddings


def get_embedding_config() -> dict[str, Any]:
    """Config snapshot logged alongside every stored vector."""
    embedding_cfg = get_config().embedding
    return {
        "model": embedding_cfg.model,
        "version": embedding_cfg.version,
        "dimension": embedding_cfg.dimension,
        "task_type": "document",
    }


if __name__ == "__main__":
    print("=== Testing Embedder Functionality ===")
    cfg = get_config()
    print(f"Target Model: {cfg.embedding.model} | Expected Dimension: {cfg.embedding.dimension}")

    # 1. Test Ingestion Task Type (Document)
    sample_docs = [
        "Document: Billing > Section: Overview\nSubscription renewals happen every 30 days.",
        "Document: Disputes > Section: Timeline\nMerchants have 7 days to provide evidence."
    ]
    print(f"\nEmbedding {len(sample_docs)} document texts...")
    doc_vecs = embed_texts(sample_docs, task_type="document", trace_id="test-embed-doc")
    print(f"Produced: {len(doc_vecs)} vectors | Vector Dimension: {len(doc_vecs[0])}")
    assert len(doc_vecs) == 2
    assert len(doc_vecs[0]) == cfg.embedding.dimension

    # 2. Test Query Task Type (Lightweight Query Path)
    sample_query = ["How long does dispute review take?"]
    print(f"\nEmbedding query...")
    query_vecs = embed_texts(sample_query, task_type="query", trace_id="test-embed-query")
    print(f"Produced: {len(query_vecs)} vector | Vector Dimension: {len(query_vecs[0])}")
    assert len(query_vecs) == 1
    assert len(query_vecs[0]) == cfg.embedding.dimension

    print("\n[OK] embedder.py is verified and ready for production!")