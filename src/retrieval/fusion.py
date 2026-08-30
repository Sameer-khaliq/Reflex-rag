"""
RRF fusion (FR-1).

rrf_fuse(), _payload_text(), and _backfill_missing_payloads() are kept
as-is from the prior project's proven design. apply_conditional_rerank()
is NOT reused — that function branched on a fast/deep routing decision
specific to Project 1's model-tiering router. This project's FR-2 is
unconditional: reranking happens on the fused set before any grading,
full stop. The fast_path/correction_path split in this project happens
later, at the routing gate (FR-3), and it skips grading — not reranking.
See retriever.py for the single unconditional call into reranker.py.
"""
from __future__ import annotations

from typing import Any

from config import get_config
from logging_config import get_logger
from qdrant_client_singleton import get_client


def rrf_fuse(sparse_ranked: list[dict], dense_ranked: list[dict], k: int | None = None) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) of two ranked lists of chunks.

    score = sum over lists containing chunk_id of 1 / (k + rank + 1)
    (rank+1 so the top rank contributes 1/(k+1), not 1/k — standard RRF convention)

    Returns chunks sorted descending by fused score:
    [{"chunk_id": ..., "rrf_score": ..., "payload": ...}, ...]
    """
    k = k if k is not None else get_config().retrieval.rrf_k
    scores: dict = {}
    payloads: dict = {}

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


def _payload_text(payload: dict[str, Any] | None, chunk_id: Any = None) -> str:
    """
    Extract chunk text from a fused item's payload for reranker input.

    Confirmed against ingestion/pipeline.py's embed_and_upsert(): payload
    = {**chunk, "embedding_config": ...}, and chunk always carries "text"
    straight from chunking.py. So "text" is the correct key to read here
    for dense/Qdrant-sourced payloads.
    """
    if payload and "text" in payload:
        return payload["text"]
    logger = get_logger(trace_id="fusion")
    logger.warning("missing_payload_text", stage="rerank", chunk_id=chunk_id)
    return ""


def _backfill_missing_payloads(items: list[dict], trace_id: str = "fusion") -> list[dict]:
    """
    sparse_retriever.py's query_bm25() returns only {"chunk_id", "score"} -
    no payload. A chunk that appears in sparse_ranked but NOT in
    dense_ranked (BM25 surfaced it, dense search didn't) therefore has
    payload=None after rrf_fuse(), even though the chunk's full payload
    genuinely exists in Qdrant - it just wasn't attached at retrieval
    time on the sparse leg.

    Without this backfill, sparse-only chunks would silently rerank as
    empty text, quietly defeating BM25's exact-match contribution to the
    hybrid result.

    Only backfills for the items actually passed in (call this AFTER
    slicing to top_k, not on the full fused list) - keeps the extra
    Qdrant round-trip cheap and scoped to what's actually used
    downstream.
    """
    missing_ids = [item["chunk_id"] for item in items if not item.get("payload")]
    if not missing_ids:
        return items

    logger = get_logger(trace_id=trace_id)
    client = get_client()
    points = client.retrieve(
        collection_name=get_config().settings.qdrant_collection_name,
        ids=missing_ids,
        with_payload=True,
    )
    payload_by_id = {p.id: p.payload for p in points}

    logger.info(
        "backfilled_missing_payloads",
        stage="rerank",
        num_missing=len(missing_ids),
        num_recovered=len(payload_by_id),
    )

    return [
        {**item, "payload": item.get("payload") or payload_by_id.get(item["chunk_id"])}
        for item in items
    ]


def to_rerank_candidates(fused_slice: list[dict]) -> list[dict[str, Any]]:
    """Adapt rrf_fuse()'s {"chunk_id", "rrf_score", "payload"} shape into
    what retrieval.reranker expects: dicts carrying a "text" key, with
    everything else passed through untouched."""
    return [
        {
            "chunk_id": item["chunk_id"],
            "text": _payload_text(item.get("payload"), chunk_id=item["chunk_id"]),
            "rrf_score": item["rrf_score"],
            "payload": item.get("payload"),
        }
        for item in fused_slice
    ]