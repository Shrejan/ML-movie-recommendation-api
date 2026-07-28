from __future__ import annotations  # <-- added for Python 3.9 compatibility

import asyncio
import os
import time
import numpy as np
import requests

from config import settings
from models import MovieInput

_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_EMBEDDINGS)

# --- HF Inference API config ---
# Token is read directly from the environment (set via .env / Render dashboard),
# not from config.py/settings, per your setup.
_HF_TOKEN = os.environ.get("HF_TOKEN")
if not _HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN environment variable is not set. "
        "Add it to your .env file locally, or to your Render service's Environment tab."
    )

# NOTE: api-inference.huggingface.co is deprecated (returns 410 / DNS failure).
# Must use router.huggingface.co instead.
_HF_API_URL = (
    f"https://router.huggingface.co/hf-inference/models/"
    f"{settings.EMBEDDING_MODEL}/pipeline/feature-extraction"
)
_HF_HEADERS = {"Authorization": f"Bearer {_HF_TOKEN}"}
_HF_SESSION = requests.Session()  # reuse TCP connection across calls


def load_model() -> None:
    """
    No local model to load anymore — embeddings are generated via the
    HuggingFace Inference API. Kept as a no-op so existing FastAPI
    lifespan code doesn't need to change.
    """
    return None


def get_embedding(text: str, retries: int = 3) -> list[float]:
    """
    Reusable, thread-safe embedding function. Returns a normalized float32 vector.
    Same signature/return type as the local sentence-transformers version.
    """
    payload = {"inputs": [text]}

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = _HF_SESSION.post(_HF_API_URL, headers=_HF_HEADERS, json=payload, timeout=30)
        except requests.RequestException as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
            continue

        if response.status_code == 200:
            data = response.json()
            # Expected shape: [[float, float, ...]] -> one pooled vector per input sentence
            vec = np.array(data[0], dtype=np.float32)

            # The HF endpoint for this model already returns a mean-pooled vector,
            # but it does NOT guarantee L2 normalization like sentence-transformers'
            # normalize_embeddings=True does. Normalize here to match local behavior.
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.astype(np.float32).tolist()

        elif response.status_code == 503:
            # Model is loading (cold start) — wait and retry
            wait = response.json().get("estimated_time", 5)
            time.sleep(wait)
            continue

        else:
            last_error = RuntimeError(
                f"HF API error {response.status_code}: {response.text[:300]}"
            )
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Embedding request failed after {retries} retries: {last_error}")


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
