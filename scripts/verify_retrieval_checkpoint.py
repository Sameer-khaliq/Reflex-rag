"""
Phase 2 checkpoint (IMPLEMENTATION_PLAN.md): for a known query against the
ingested demo corpus, retriever.retrieve(query) must return the expected
document in the top reranked results.

Usage:
    uv run python scripts/verify_retrieval_checkpoint.py \
        --query "how long does an ACH refund take" \
        --expect-source-doc "billing_refunds_ach"

Run this once per hand-picked query you want to sanity-check — including
at least one query that should land on each of the three DoD trap-case
documents, so you catch a retrieval problem before Phase 3's grading
calibration builds on top of it.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from retrieval.reranker import preload_reranker
from retrieval.retriever import retrieve


async def main(query: str, expect_source_doc: str | None) -> None:
    print("Warming up reranker model (one-time cost, not counted against query latency)...")
    preload_reranker()

    result = await retrieve(query)
    results = result["chunks"]

    if not results:
        print(f"FAIL: retrieve('{query}') returned zero results.")
        sys.exit(1)

    print(f"Query: {query!r}")
    print(f"Reranked: {result['reranked']}")
    print(f"Top {len(results)} results:\n")
    for i, r in enumerate(results):
        payload = r.get("payload") or {}
        source_doc = payload.get("source_doc_id", "?")
        score = r.get("rerank_score", r.get("rrf_score"))
        text_preview = (r.get("text") or "")[:100].replace("\n", " ")
        print(f"  [{i}] source_doc_id={source_doc!r} score={score:.4f}")
        print(f"      {text_preview}...")

    if expect_source_doc is None:
        print("\nNo --expect-source-doc given — inspect the ranking above manually.")
        return

    top_docs = [
        (r.get("payload") or {}).get("source_doc_id") for r in results
    ]
    if expect_source_doc in top_docs:
        rank = top_docs.index(expect_source_doc)
        print(f"\nPASS: {expect_source_doc!r} found at rank {rank}.")
    else:
        print(f"\nFAIL: {expect_source_doc!r} not found in top {len(results)} results.")
        print(f"      Got source_doc_ids: {top_docs}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--expect-source-doc",
        default=None,
        help="source_doc_id (filename stem) the top results should contain",
    )
    args = parser.parse_args()

    asyncio.run(main(args.query, args.expect_source_doc))