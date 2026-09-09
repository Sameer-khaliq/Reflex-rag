"""Single entrypoint tying dense + sparse + RRF fusion + rerank together (FR-1,

FR-2).
"""

from __future__ import annotations

import asyncio
import json

from config import get_config
from logging_config import get_logger
from retrieval.dense_retriever import query_dense

try:
    from retrieval.fusion import _backfill_missing_payloads
except ImportError:
    from retrieval.fusion import (
        hydrate_missing_payloads as _backfill_missing_payloads,
    )

from retrieval.fusion import rrf_fuse, to_rerank_candidates
from retrieval.reranker import rerank_async
from retrieval.sparse_retriever import query_bm25


async def _dense_leg(query: str, top_n: int, trace_id: str) -> list[dict]:
    return await asyncio.to_thread(query_dense, query, top_n, trace_id)


async def _sparse_leg(
    query: str, top_n: int, trace_id: str = "sparse_query"
) -> list[dict]:
    return await asyncio.to_thread(query_bm25, query, top_n, trace_id)


async def retrieve(
    query: str,
    sparse_top_n: int | None = None,
    dense_top_n: int | None = None,
    rerank_top_k: int | None = None,
    trace_id: str = "retrieve",
) -> dict:
    cfg = get_config()
    sparse_top_n = sparse_top_n or cfg.retrieval.sparse_top_n
    dense_top_n = dense_top_n or cfg.retrieval.dense_top_n
    rerank_top_k = rerank_top_k or cfg.retrieval.rerank.top_k
    candidate_pool = cfg.retrieval.rerank.candidate_pool
    logger = get_logger(trace_id=trace_id)

    # 1. Fetch sparse and dense in parallel (non-blocking)
    sparse_task = _sparse_leg(query, sparse_top_n, trace_id=trace_id)
    dense_task = _dense_leg(query, dense_top_n, trace_id=trace_id)
    sparse_results, dense_results = await asyncio.gather(
        sparse_task, dense_task
    )

    # 2. RRF Fusion (rank blending)
    fused = rrf_fuse(sparse_results, dense_results)

    # 3. Slice to candidate_pool and backfill missing payloads from Qdrant
    pool_slice = fused[:candidate_pool]
    sliced = _backfill_missing_payloads(pool_slice, trace_id=trace_id)
    candidates = to_rerank_candidates(sliced)

    # 4. Cross-Encoder reranks the candidate pool down to rerank_top_k
    reranked = await rerank_async(
        query, candidates, top_k=rerank_top_k, trace_id=trace_id
    )
    did_rerank = bool(reranked) and "rerank_score" in reranked[0]

    logger.info(
        "retrieve_complete",
        stage="retrieval",
        sparse_count=len(sparse_results),
        dense_count=len(dense_results),
        fused_count=len(fused),
        reranked_count=len(reranked),
        reranked=did_rerank,
    )

    return {"chunks": reranked, "reranked": did_rerank}


async def main():
    test_query = "What is the policy on API rate limits?"
    print(f"Executing End-to-End Hybrid Retrieval for: '{test_query}'...\n")

    response = await retrieve(
        query=test_query,
        sparse_top_n=5,
        dense_top_n=5,
        rerank_top_k=3,
        trace_id="e2e_retrieval_test",
    )

    print(f"Reranking Applied: {response['reranked']}")
    print(f"Total Chunks Returned: {len(response['chunks'])}\n")

    for rank, chunk in enumerate(response["chunks"], 1):
        print(f"=== Rank {rank} ===")
        print(f"Chunk ID    : {chunk.get('chunk_id')}")
        print(f"Rerank Score: {chunk.get('rerank_score', 'N/A')}")
        print(f"RRF Score   : {chunk.get('rrf_score', 'N/A')}")
        print(f"Text Snippet: {chunk.get('text', '')[:120]}...")
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(main())