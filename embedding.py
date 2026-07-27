from __future__ import annotations  # <-- added for Python 3.9 compatibility

import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings
from models import MovieInput

_model: SentenceTransformer | None = None
_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_EMBEDDINGS)


def load_model() -> None:
    """Load the embedding model once at startup. Call from FastAPI lifespan."""
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")


def get_embedding(text: str) -> list[float]:
    """Reusable, thread-safe embedding function. Returns a normalized float32 vector."""
    if _model is None:
        raise RuntimeError("Embedding model not loaded. Call load_model() at startup.")
    vec = _model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32).tolist()


def _build_field_groups(movie: MovieInput) -> tuple[str, str, str]:
    # Three semantically distinct text groups built from the JSON payload.
    group_plot = f"title: {movie.title}. overview: {movie.overview or ''}"
    group_taxonomy = (
        f"genres: {', '.join(movie.genres or [])}. "
        f"keywords: {', '.join(movie.keywords or [])}"
    )
    group_people = (
        f"director: {movie.director or ''}. cast: {', '.join(movie.cast or [])}. "
        f"release_year: {movie.release_year or ''}. language: {movie.language or ''}"
    )
    return group_plot, group_taxonomy, group_people


async def build_movie_vector(movie: MovieInput) -> list[float]:
    """Generate 3 embeddings from distinct field groups and combine into one user vector."""
    groups = _build_field_groups(movie)
    loop = asyncio.get_running_loop()
    async with _semaphore:
        vectors = await loop.run_in_executor(
            None,
            lambda: [np.array(get_embedding(g), dtype=np.float32) for g in groups],
        )
    combined = np.mean(vectors, axis=0)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    del vectors
    return combined.astype(np.float32).tolist()


async def build_movie_vectors_batch(movies: list[MovieInput]) -> list[list[float]]:
    """Batch-generate vectors for all movies in a request, bounded by the semaphore."""
    return await asyncio.gather(*(build_movie_vector(m) for m in movies))