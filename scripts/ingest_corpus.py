"""
Ingest the NimbusPay corpus into Qdrant and build the BM25 index.
Usage:
    uv run python scripts/ingest_corpus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add src to sys.path if not present
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from config import get_config
from ingestion.pipeline import embed_and_upsert, process_document
from retrieval.sparse_retriever import get_or_build_index

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


def main():
    if not CORPUS_DIR.exists():
        print(f"Error: Corpus directory not found at {CORPUS_DIR}")
        sys.exit(1)

    doc_files = sorted(list(CORPUS_DIR.glob("*.md")) + list(CORPUS_DIR.glob("*.txt")) + list(CORPUS_DIR.glob("*.pdf")))
    if not doc_files:
        print(f"Error: No documents found in {CORPUS_DIR}")
        sys.exit(1)

    print(f"Found {len(doc_files)} documents in {CORPUS_DIR}. Processing...", flush=True)

    all_chunks = []
    total_new = 0

    for doc_path in doc_files:
        new_chunks = process_document(doc_path)
        print(f"  -> {doc_path.name}: {len(new_chunks)} new chunks", flush=True)
        all_chunks.extend(new_chunks)
        total_new += len(new_chunks)

    if all_chunks:
        print(f"\nEmbedding and upserting {len(all_chunks)} chunks to Qdrant...", flush=True)
        count = embed_and_upsert(all_chunks)
        print(f"Upserted {count} points into Qdrant collection '{get_config().settings.qdrant_collection_name}'.", flush=True)
    else:
        print("\nAll chunks were already ingested according to the hash store.", flush=True)

    # Now build/rebuild BM25 index from Qdrant scroll or from all points
    print("\nBuilding BM25 sparse index...", flush=True)
    from qdrant_client_singleton import get_client

    client = get_client()
    try:
        all_points = client.scroll(
            collection_name=get_config().settings.qdrant_collection_name,
            limit=100000,
            with_payload=True,
        )[0]
        bm25_chunks = [{"chunk_id": p.id, "text": p.payload["text"]} for p in all_points if p.payload and "text" in p.payload]
        get_or_build_index(chunks=bm25_chunks, rebuild=True)
        print(f"BM25 index successfully built and saved with {len(bm25_chunks)} chunks.", flush=True)
    except Exception as exc:
        print(f"Warning: Could not build BM25 index from Qdrant scroll ({exc}). Building from processed chunks...", flush=True)
        if all_chunks:
            bm25_chunks = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in all_chunks]
            get_or_build_index(chunks=bm25_chunks, rebuild=True)
            print(f"BM25 index built with {len(bm25_chunks)} chunks.", flush=True)


if __name__ == "__main__":
    main()
