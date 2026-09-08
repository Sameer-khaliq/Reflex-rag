"""Dense retrieval query path"""
from __future__ import annotations
import json
from config import get_config
from ingestion.embedder import embed_texts
from logging_config import get_logger
from qdrant_client_singleton import get_client


def query_dense(query: str, top_n: int | None = None, trace_id: str = "dense_query") -> list[dict]:
    """Returns [{chunk_id, score, payload}, ...] sorted descending by cosine similarity."""
    cfg = get_config()
    top_n = top_n or cfg.retrieval.dense_top_n
    logger = get_logger(trace_id=trace_id)

    query_vector = embed_texts([query], task_type="query", trace_id=trace_id)[0]

    client = get_client()
    results = client.query_points(
        collection_name=cfg.settings.qdrant_collection_name,
        query=query_vector,
        limit=top_n,
        with_payload=True,
    ).points

    output = [{"chunk_id": r.id, "score": r.score, "payload": r.payload} for r in results]
    logger.info("dense_retrieval", stage="dense_retrieval", num_results=len(output))
    return output

def main():
    test_query = "What is billing overview?"
    print(f"Running dense query for: '{test_query}'...\n")

    try:
        results = query_dense(query=test_query, top_n=3, trace_id="test_run_1")

        if not results:
            print("Koi match nahi mila (Collection empty ho sakti hai).")
            return

        for idx, item in enumerate(results, 1):
            print(f"--- Result {idx} ---")
            print(f"Chunk ID: {item['chunk_id']}")
            print(f"Score   : {item['score']:.4f}")
            print("Payload :")
            print(json.dumps(item["payload"], indent=2))
            print("-" * 30)

    except Exception as e:
        print(f"Error during retrieval: {e}")


if __name__ == "__main__":
    main()