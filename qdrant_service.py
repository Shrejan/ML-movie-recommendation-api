from __future__ import annotations  # <-- added for Python 3.9 compatibility

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import settings

_client: QdrantClient | None = None
REQUIRED_PAYLOAD_FIELDS = {"tmdb_id"}


def get_client() -> QdrantClient:
    """Reuse a single Qdrant client instance across all requests."""
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    return _client


def search(vector: list[float], top_k: int, language_filter: str | None = None):
    client = get_client()
    query_filter = None
    if language_filter:
        query_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="language", match=qmodels.MatchValue(value=language_filter))]
        )
    hits = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    return hits, language_filter


def validate_results(hits, language_filter: str | None) -> list:
    valid = []
    seen_ids = set()
    prev_score = None
    for hit in hits:
        payload = hit.payload or {}
        if not REQUIRED_PAYLOAD_FIELDS.issubset(payload.keys()):
            continue
        if language_filter and payload.get("language") != language_filter:
            continue
        if prev_score is not None and hit.score > prev_score:
            continue
        if hit.id in seen_ids:
            continue
        seen_ids.add(hit.id)
        prev_score = hit.score
        valid.append(hit)
    
    return valid