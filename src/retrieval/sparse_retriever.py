"""
BM25 sparse index — build once at ingestion, load at query time (FR-1).

Simplified from the prior project's version: that one supported
hot-reload/atomic-swap for a corpus that changes under a live-serving
process. This demo corpus (14 static NimbusPay docs) doesn't need that
complexity — build once during ingestion, persist to disk, load at
startup. If the corpus becomes dynamic later, the atomic-swap pattern
from the prior project is the right thing to bring back; not needed now.
"""
from __future__ import annotations

import json
from pathlib import Path

import bm25s

from config import get_config
from logging_config import get_logger

_active_index: bm25s.BM25 | None = None
_active_chunk_ids: list[int] | None = None


def _index_dir() -> Path:
    return Path(get_config().retrieval.bm25_index_dir)


def build_index(chunks: list[dict], trace_id: str = "bm25_build") -> tuple[bm25s.BM25, list[int]]:
    logger = get_logger(trace_id=trace_id)

    corpus_texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    tokenized_corpus = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)

    retriever = bm25s.BM25()
    retriever.index(tokenized_corpus)

    logger.info("bm25_index_built", stage="bm25_build", num_docs=len(chunks))
    return retriever, chunk_ids


def save_index(retriever: bm25s.BM25, chunk_ids: list[int], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    retriever.save(str(path))
    (path / "chunk_ids.json").write_text(json.dumps(chunk_ids))


def load_index(path: Path) -> tuple[bm25s.BM25, list[int]]:
    retriever = bm25s.BM25.load(str(path), load_corpus=False)
    chunk_ids = json.loads((path / "chunk_ids.json").read_text())
    return retriever, chunk_ids


def get_or_build_index(
    chunks: list[dict] | None = None, rebuild: bool = False, trace_id: str = "bm25_startup"
) -> tuple[bm25s.BM25, list[int]]:
    """Startup path: load from disk unless missing or --rebuild passed."""
    global _active_index, _active_chunk_ids
    logger = get_logger(trace_id=trace_id)
    index_dir = _index_dir()

    if index_dir.exists() and not rebuild:
        retriever, chunk_ids = load_index(index_dir)
        logger.info("bm25_loaded_from_disk", stage="bm25_startup", num_docs=len(chunk_ids))
    else:
        if chunks is None:
            raise ValueError("No persisted index found and no chunks provided to build one.")
        retriever, chunk_ids = build_index(chunks, trace_id=trace_id)
        save_index(retriever, chunk_ids, index_dir)
        logger.info("bm25_built_fresh", stage="bm25_startup", num_docs=len(chunk_ids))

    _active_index = retriever
    _active_chunk_ids = chunk_ids
    return retriever, chunk_ids


def get_active_index() -> tuple[bm25s.BM25, list[int]]:
    if _active_index is None:
        raise RuntimeError("BM25 index not initialized — call get_or_build_index() first.")
    return _active_index, _active_chunk_ids


def query_bm25(query: str, top_n: int | None = None) -> list[dict]:
    """Returns [{chunk_id, score}, ...] sorted descending by score."""
    top_n = top_n or get_config().retrieval.sparse_top_n
    retriever, chunk_ids = get_active_index()

    query_tokens = bm25s.tokenize([query], stopwords="en", return_ids=False, show_progress=False)
    results, scores = retriever.retrieve(query_tokens, k=min(top_n, len(chunk_ids)))
    return [
        {"chunk_id": chunk_ids[idx], "score": float(score)}
        for idx, score in zip(results[0], scores[0])
    ]


if __name__ == "__main__":
    from qdrant_client_singleton import get_client

    client = get_client()
    all_points = client.scroll(
        collection_name=get_config().settings.qdrant_collection_name, limit=100000
    )[0]
    chunks = [{"chunk_id": p.id, "text": p.payload["text"]} for p in all_points]
    get_or_build_index(chunks=chunks, rebuild=True)
    print(f"BM25 index built with {len(chunks)} chunks.")