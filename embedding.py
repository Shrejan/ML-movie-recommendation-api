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

# Separate budgets: real failures (network/5xx-non-503) vs. cold-start (503) waits.
_MAX_RETRIES = 3
_MAX_COLD_START_WAITS = 5
_DEFAULT_COLD_START_WAIT = 5  # seconds, used if HF doesn't give us an estimate


def load_model() -> None:
    """
    No local model to load anymore — embeddings are generated via the
    HuggingFace Inference API. Kept as a no-op so existing FastAPI
    lifespan code doesn't need to change.
    """
    return None


def get_embedding(text: str, retries: int = _MAX_RETRIES) -> list[float]:
    """
    Reusable, thread-safe embedding function. Returns a normalized float32 vector.
    Same signature/return type as the local sentence-transformers version.

    Retry budgets are tracked separately:
      - `retries`: genuine failures (network errors, non-200/503 HTTP errors)
      - cold-start waits (HTTP 503): tracked independently via _MAX_COLD_START_WAITS,
        so a model that's simply warming up doesn't eat into the "real error" budget.
    """
    payload = {"inputs": [text]}

    last_error: Exception | None = None
    cold_start_waits = 0
    attempt = 0

    while attempt < retries:
        try:
            response = _HF_SESSION.post(
                _HF_API_URL, headers=_HF_HEADERS, json=payload, timeout=30
            )
        except requests.RequestException as e:
            last_error = e
            attempt += 1
            time.sleep(2 * attempt)
            continue

        if response.status_code == 200:
            try:
                data = response.json()
                vec = np.array(data[0], dtype=np.float32)
            except (ValueError, IndexError, KeyError) as e:
                last_error = RuntimeError(f"Malformed HF response body: {e}")
                attempt += 1
                time.sleep(2 * attempt)
                continue

            # The HF endpoint for this model already returns a mean-pooled vector,
            # but it does NOT guarantee L2 normalization like sentence-transformers'
            # normalize_embeddings=True does. Normalize here to match local behavior.
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.astype(np.float32).tolist()

        elif response.status_code == 503:
            # Model is loading (cold start) — wait and retry, on its own budget
            # so repeated cold-start waits don't get mistaken for real failures.
            if cold_start_waits >= _MAX_COLD_START_WAITS:
                last_error = RuntimeError(
                    "HF model did not finish loading after "
                    f"{_MAX_COLD_START_WAITS} cold-start waits"
                )
                break
            try:
                wait = response.json().get("estimated_time", _DEFAULT_COLD_START_WAIT)
            except ValueError:
                # Response body wasn't valid JSON (e.g. gateway timeout page)
                wait = _DEFAULT_COLD_START_WAIT
            cold_start_waits += 1
            time.sleep(wait)
            continue

        else:
            last_error = RuntimeError(
                f"HF API error {response.status_code}: {response.text[:300]}"
            )
            attempt += 1
            time.sleep(2 * attempt)

    raise RuntimeError(f"Embedding request failed after {attempt} retries: {last_error}")


def _build_movie_text(movie: MovieInput) -> str:
    return (
        f"Title: {movie.title}. "
        f"Overview: {movie.overview or ''}. "
        f"Genres: {', '.join(movie.genres or [])}. "
        f"Keywords: {', '.join(movie.keywords or [])}. "
        f"Director: {movie.director or ''}. "
        f"Cast: {', '.join(movie.cast or [])}. "
        f"Release Year: {movie.release_year or ''}. "
        f"Language: {movie.language or ''}."
    )


async def build_movie_vector(movie: MovieInput) -> list[float]:
    """Generate a single embedding from all movie information."""
    text = _build_movie_text(movie)

    loop = asyncio.get_running_loop()

    async with _semaphore:
        vector = await loop.run_in_executor(
            None,
            lambda: get_embedding(text)
        )

    return vector


async def build_movie_vectors_batch(movies: list[MovieInput]) -> list[list[float]]:
    """Batch-generate vectors for all movies in a request, bounded by the semaphore."""
    return await asyncio.gather(*(build_movie_vector(m) for m in movies))
