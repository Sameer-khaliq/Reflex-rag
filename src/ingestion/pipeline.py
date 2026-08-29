"""
Metadata tagging & content-hash dedup.

process_document() is the single entrypoint that goes from "a file path"
to "a list of chunk dicts ready to store" - each one already carrying
its full metadata and having already passed the dedup check.

Dedup mechanism: a SHA-256 hash of each chunk's normalized text is
checked against a persisted hash set BEFORE the chunk is accepted.
Re-running this on an unchanged file therefore returns zero new chunks
the second time - nothing downstream ever receives a duplicate.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client.models import Distance, PointStruct, VectorParams

from config import get_config
from ingestion.chunking import chunk_text
from ingestion.embedder import embed_texts, get_embedding_config
from ingestion.loaders import load_document
from qdrant_client_singleton import get_client

HASH_STORE_PATH = Path("data/.ingestion_state/chunk_hashes.json")


def normalize_for_hash(text: str) -> str:
    return " ".join(text.split()).lower()


def compute_content_hash(text: str) -> str:
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_hash_store(path: Path | None = None) -> set[str]:
    path = path or HASH_STORE_PATH
    if path.exists():
        return set(json.loads(path.read_text(encoding="utf-8")))
    return set()


def save_hash_store(hashes: set[str], path: Path | None = None) -> None:
    path = path or HASH_STORE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(hashes)), encoding="utf-8")


def process_document(
    path: str | Path,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    overlap_min_pct: float | None = None,
    overlap_max_pct: float | None = None,
    hash_store_path: Path | None = None,
) -> list[dict]:
    """
    Loads a document, chunks it, attaches metadata, and filters out any
    chunk whose content hash has already been seen. Returns only the
    NEWLY accepted chunks - a second call on an unchanged file returns
    an empty list.

    Each returned chunk dict has:
        text, start_char, end_char, token_count   (from chunking.py)
        source_doc_id       - filename stem
        chunk_index         - position of this chunk within this document
        ingestion_timestamp - UTC ISO-8601, same for every chunk in this run
        content_hash        - SHA-256 over normalized chunk text
    """
    hash_store_path = hash_store_path or HASH_STORE_PATH

    text = load_document(path)
    raw_chunks = chunk_text(
        text,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_min_pct=overlap_min_pct,
        overlap_max_pct=overlap_max_pct,
    )

    source_doc_id = Path(path).stem
    ingestion_timestamp = datetime.now(UTC).isoformat()

    hash_store = load_hash_store(hash_store_path)
    accepted: list[dict] = []

    for chunk_index, c in enumerate(raw_chunks):
        content_hash = compute_content_hash(c["text"])
        if content_hash in hash_store:
            continue

        hash_store.add(content_hash)
        chunk_id = int(content_hash[:16], 16) % (2**63 - 1)
        accepted.append({
            **c,
            "chunk_id": chunk_id,
            "source_doc_id": source_doc_id,
            "chunk_index": chunk_index,
            "ingestion_timestamp": ingestion_timestamp,
            "content_hash": content_hash,
        })

    save_hash_store(hash_store, hash_store_path)
    return accepted


def ensure_collection(client, collection_name: str, dimension: int):
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )


def embed_and_upsert(deduped_chunks: list[dict], trace_id: str = "ingest") -> int:
    cfg = get_config()
    client = get_client()
    ensure_collection(client, cfg.settings.qdrant_collection_name, cfg.embedding.dimension)

    texts = [c["text"] for c in deduped_chunks]
    vectors = embed_texts(texts, task_type="document", trace_id=trace_id)
    embed_config = get_embedding_config()

    points = [
        PointStruct(
            id=chunk["chunk_id"],
            vector=vector,
            payload={**chunk, "embedding_config": embed_config},
        )
        for chunk, vector in zip(deduped_chunks, vectors)
    ]

    client.upsert(collection_name=cfg.settings.qdrant_collection_name, points=points)
    return len(points)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ingestion pipeline: load, chunk, dedup, embed, upsert."
    )
    parser.add_argument("--path", required=True, help="Path to source document (PDF/TXT/MD)")
    args = parser.parse_args()

    print(f"[pipeline] Processing {args.path} ...")
    new_chunks = process_document(args.path)
    print(f"[pipeline] {len(new_chunks)} new chunks after dedup (0 means already ingested).")

    if new_chunks:
        upserted_count = embed_and_upsert(new_chunks)
        print(f"[pipeline] Upserted {upserted_count} points into Qdrant collection '{get_config().settings.qdrant_collection_name}'.")
    else:
        print("[pipeline] Nothing new to embed/upsert — skipping Qdrant write.")