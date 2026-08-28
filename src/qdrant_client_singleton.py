from qdrant_client import QdrantClient

from config import get_config

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        cfg = get_config()
        _client = QdrantClient(url=cfg.settings.qdrant_url)
    return _client


def collection_name() -> str:
    return get_config().settings.qdrant_collection_name