"""
Single entrypoint tying dense + sparse + RRF fusion + rerank together
(FR-1, FR-2). This replaces the prior project's retrieve_and_route_concurrent(),
which gathered retrieval alongside a routing decision (layer0_rules.route_query()).
That coupling doesn't apply here: this project's routing gate (FR-3) is a
separate, pre-retrieval decision that only affects whether grading runs
afterward — it never affects retrieval or reranking itself. Reranking is
unconditional here, per FR-2.

The concurrent dense+sparse pattern (asyncio.gather, both legs scheduled
in the same event-loop tick) is kept from the prior project's
retrieve_concurrent() — that part was correct and worth reusing as-is.
"""
from __future__ import annotations

import asyncio

from config import get_config
from logging_config import get_logger
from retrieval.dense_retriever import query_dense
from retrieval.fusion import _backfill_missing_payloads, rrf_fuse, to_rerank_candidates
from retrieval.reranker import rerank_async
from retrieval.sparse_retriever import query_bm25


async def _dense_leg(query: str, top_n: int, trace_id: str) -> list[dict]:
    return await asyncio.to_thread(query_dense, query, top_n, trace_id)


async def _sparse_leg(query: str, top_n: int) -> list[dict]:
    return await asyncio.to_thread(query_bm25, query, top_n)


async def retrieve(
    query: str,
    sparse_top_n: int | None = None,
    dense_top_n: int | None = None,
    rerank_top_k: int | None = None,
    trace_id: str = "retrieve",
) -> dict:
    """
    Runs sparse and dense retrieval concurrently, fuses via RRF,
    backfills payloads missing on sparse-only hits, then reranks the
    fused set unconditionally (FR-2 — no fast/deep branch here).

    Returns:
        {
            "chunks": [...],   # final ordered candidate list:
                                # [{"chunk_id", "text", "rerank_score"?,
                                #   "rrf_score", "payload"}, ...]
            "reranked": bool,  # False if the reranker timed out and this
                                # fell back to RRF-fused order (see
                                # reranker.rerank_async) — downstream
                                # grading/audit needs to know the
                                # difference, not just receive a flat
                                # list either way
        }
    """
    cfg = get_config()
    sparse_top_n = sparse_top_n or cfg.retrieval.sparse_top_n
    dense_top_n = dense_top_n or cfg.retrieval.dense_top_n
    rerank_top_k = rerank_top_k or cfg.retrieval.rerank.top_k
    logger = get_logger(trace_id=trace_id)

    sparse_task = _sparse_leg(query, sparse_top_n)
    dense_task = _dense_leg(query, dense_top_n, trace_id)
    sparse_results, dense_results = await asyncio.gather(sparse_task, dense_task)

    fused = rrf_fuse(sparse_results, dense_results)
    sliced = _backfill_missing_payloads(fused[:rerank_top_k], trace_id=trace_id)
    candidates = to_rerank_candidates(sliced)

    reranked = await rerank_async(query, candidates, top_k=rerank_top_k, trace_id=trace_id)
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