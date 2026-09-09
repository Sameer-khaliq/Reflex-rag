from __future__ import annotations

from typing import Any

from config import get_config
from logging_config import get_logger
from qdrant_client_singleton import get_client


def rrf_fuse(
    sparse_ranked: list[dict], dense_ranked: list[dict], k: int | None = None
) -> list[dict]:
    """Combines sparse (BM25) and dense search results using Reciprocal Rank Fusion (RRF).

    Returns chunks sorted descending by fused score:
    [{"chunk_id": ..., "rrf_score": ..., "payload": ...}, ...]
    """
    k = k if k is not None else get_config().retrieval.rrf_k
    scores: dict[Any, float] = {}
    payloads: dict[Any, dict[str, Any]] = {}

    for rank, item in enumerate(sparse_ranked):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        if item.get("payload"):
            payloads[cid] = item["payload"]

    for rank, item in enumerate(dense_ranked):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        if item.get("payload"):
            payloads[cid] = item["payload"]

    fused = [
        {"chunk_id": cid, "rrf_score": score, "payload": payloads.get(cid)}
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


def hydrate_missing_payloads(
    fused_items: list[dict], trace_id: str = "fusion"
) -> list[dict]:
    """Fetches payloads from Qdrant in a single round-trip for chunks that

    only appeared in sparse/BM25 results and thus lack text/metadata.
    """
    missing_ids = [
        item["chunk_id"] for item in fused_items if not item.get("payload")
    ]
    if not missing_ids:
        return fused_items

    logger = get_logger(trace_id=trace_id)
    cfg = get_config()

    try:
        client = get_client()
        points = client.retrieve(
            collection_name=cfg.settings.qdrant_collection_name,
            ids=missing_ids,
            with_payload=True,
            with_vectors=False,
        )

        fetched_map = {p.id: p.payload for p in points if p.payload}

        for item in fused_items:
            cid = item["chunk_id"]
            if not item.get("payload") and cid in fetched_map:
                item["payload"] = fetched_map[cid]

        logger.info(
            "hydrated_missing_payloads",
            stage="fusion",
            requested=len(missing_ids),
            found=len(fetched_map),
        )
    except Exception as e:
        logger.error(
            "failed_hydrating_payloads", stage="fusion", error=str(e)
        )

    return fused_items


def _payload_text(payload: dict[str, Any] | None, chunk_id: Any = None) -> str:
    """Extract chunk text from a fused item's payload for reranker input."""
    if payload and "text" in payload:
        return payload["text"]
    logger = get_logger(trace_id="fusion")
    logger.warning("missing_payload_text", stage="rerank", chunk_id=chunk_id)
    return ""


def to_rerank_candidates(
    fused_slice: list[dict], trace_id: str = "fusion"
) -> list[dict[str, Any]]:
    """Hydrates missing payloads from Qdrant, extracts text, and adapts the data

    into the shape expected by the cross-encoder reranker.
    """
    hydrated_slice = hydrate_missing_payloads(fused_slice, trace_id=trace_id)

    candidates = []
    for item in hydrated_slice:
        payload = item.get("payload")
        text = _payload_text(payload, chunk_id=item["chunk_id"])
        if payload is None:
            payload = {"text": text}

        candidates.append(
            {
                "chunk_id": item["chunk_id"],
                "text": text,
                "rrf_score": item["rrf_score"],
                "payload": payload,
            }
        )
    return candidates


def main():
    # 1. Simulate Sparse (BM25) Results (Chunk IDs only, no payloads initially)
    sparse_results = [
        {"chunk_id": 101, "score": 4.52},
        {"chunk_id": 102, "score": 2.31},
        {"chunk_id": 103, "score": 1.10},
    ]

    # 2. Simulate Dense (Vector Search) Results (with payloads from Qdrant)
    dense_results = [
        {
            "chunk_id": 104,
            "score": 0.82,
            "payload": {
                "text": "Retrieval-Augmented Generation bridges LLMs with dynamic external knowledge bases."
            },
        },
        {
            "chunk_id": 101,
            "score": 0.79,
            "payload": {
                "text": "BM25 and dense retrieval combine via reciprocal rank fusion to boost recall."
            },
        },
        {
            "chunk_id": 105,
            "score": 0.65,
            "payload": {
                "text": "API rate limits prevent cascading service disruptions."
            },
        },
    ]

    print("--- Running RRF Fusion (k=60) ---")
    fused_results = rrf_fuse(
        sparse_ranked=sparse_results, dense_ranked=dense_results, k=60
    )

    for rank, item in enumerate(fused_results, 1):
        print(
            f"Rank {rank} -> Chunk ID: {item['chunk_id']} | Fused RRF Score: {item['rrf_score']:.6f}"
        )

    print("\n--- Formatting Candidates for Reranker ---")
    rerank_candidates = to_rerank_candidates(
        fused_results, trace_id="test_fusion"
    )

    for candidate in rerank_candidates:
        print(f"ID: {candidate['chunk_id']}")
        print(f"  RRF Score: {candidate['rrf_score']:.6f}")
        print(
            f"  Text     : {candidate['text'][:60] if candidate['text'] else '[NO TEXT FOUND]'}"
        )
        print("-" * 30)


if __name__ == "__main__":
    main()