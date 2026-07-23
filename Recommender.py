"""
Movie recommendation engine.

Key optimizations vs the original:

1. PRECOMPUTED NORMALIZATION (biggest win):
   sklearn's cosine_similarity() re-normalizes BOTH input matrices on every
   call. Your embeddings matrix (48170 x 896) never changes, so normalizing
   it on every single request is pure waste. We normalize it once at
   startup; per request we only normalize the tiny user vector (896 floats)
   and do a single matrix-vector dot product. This turns an O(n) norm
   computation over 48k rows into a one-time cost.

2. PARTIAL SORT (argpartition) INSTEAD OF FULL SORT:
   np.argsort() over all 48170 similarities does a full O(n log n) sort when
   you only need the top ~10-20. np.argpartition() finds the top-k in O(n)
   and we only sort that small slice.

3. O(1) DICT LOOKUP INSTEAD OF PANDAS isin() + iloc IN A LOOP:
   `metadata["imdbId"].isin(...)` scans the whole dataframe, and calling
   `.iloc[idx]` inside a Python loop is slow (pandas row access has real
   overhead per call). We build a plain dict + numpy arrays once at startup
   for O(1) lookups and vectorized access.

4. TIMING LOGS AT EVERY STAGE:
   Each stage (matching, weighted average, similarity, top-k, response
   building) is timed independently and logged, so if this is ever slow
   again in production, the logs will tell you exactly which stage to
   blame instead of guessing.

5. Dropped the sklearn dependency entirely for this hot path (one less
   import, one less abstraction layer) in favor of a plain numpy dot
   product, which is what cosine_similarity does internally anyway once
   inputs are normalized.
"""

import logging
import time
from typing import List

from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("movie_rec")

# ---------------------------------------------------------------------------
# Startup: load data once, precompute everything that doesn't change per request
# ---------------------------------------------------------------------------
logger.info("=" * 40)
logger.info("Loading metadata and embeddings...")
_startup_start = time.perf_counter()

metadata_path = hf_hub_download(
    repo_id="Shrejankotyan2005/movie_vector",
    filename="movie_metadata.csv",
    repo_type="dataset",
)
logger.info(f"Metadata downloaded: {metadata_path}")

embeddings_path = hf_hub_download(
    repo_id="Shrejankotyan2005/movie_vector",
    filename="movie_embeddings_f32.npy",
    repo_type="dataset",
)
logger.info(f"Embeddings downloaded: {embeddings_path}")

metadata = pd.read_csv(metadata_path)
embeddings = np.load(embeddings_path).astype(np.float32)

assert len(metadata) == len(embeddings), (
    f"Metadata rows ({len(metadata)}) and embeddings rows ({len(embeddings)}) "
    "must match"
)

logger.info(f"Metadata shape: {metadata.shape}")
logger.info(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

# Precompute normalized embeddings ONCE. This is the single biggest fix:
# sklearn's cosine_similarity was redoing this on every request before.
_t = time.perf_counter()
_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
_norms[_norms == 0] = 1e-10  # guard against zero vectors -> division by zero
normalized_embeddings = (embeddings / _norms).astype(np.float32)
logger.info(f"Precomputed normalized embeddings in {time.perf_counter() - _t:.3f}s")

# Precompute plain numpy arrays + a dict for O(1) id -> row-index lookup.
# Avoids pandas .isin()/.iloc() overhead on every request.
imdb_ids_arr = metadata["imdbId"].to_numpy()
movie_titles_arr = metadata["movie_title"].to_numpy()
id_to_index = {int(imdb_id): i for i, imdb_id in enumerate(imdb_ids_arr)}

logger.info(f"Startup completed in {time.perf_counter() - _startup_start:.2f}s total")
logger.info("=" * 40)


def recommend_movies(user_movies: List[dict], top_n: int = 10):
    t_start = time.perf_counter()
    logger.info(f"recommend_movies() started | input={user_movies}")

    ratings_map = {movie["imdbId"]: movie["rating"] for movie in user_movies}

    # --- Match input imdbIds against our dataset (O(1) per lookup) ---
    t0 = time.perf_counter()
    watched_indices = []
    weights = []
    unmatched = []
    for imdb_id, rating in ratings_map.items():
        idx = id_to_index.get(imdb_id)
        if idx is not None:
            watched_indices.append(idx)
            weights.append(rating)
        else:
            unmatched.append(imdb_id)

    if unmatched:
        logger.warning(f"imdbIds not found in dataset: {unmatched}")

    if not watched_indices:
        logger.error(f"No matching imdbIds found. Requested: {list(ratings_map.keys())}")
        raise ValueError("None of the provided imdbIds were found.")

    watched_indices = np.array(watched_indices)
    weights = np.array(weights, dtype=np.float32)
    logger.info(
        f"Matched {len(watched_indices)}/{len(ratings_map)} imdbIds "
        f"in {time.perf_counter() - t0:.4f}s"
    )

    # --- Build weighted user preference vector from RAW (non-normalized)
    #     embeddings, since the weighted average should reflect true
    #     magnitudes before we normalize the *result* ---
    t0 = time.perf_counter()
    watched_matrix = embeddings[watched_indices]
    user_vector = np.average(watched_matrix, axis=0, weights=weights).astype(np.float32)
    logger.info(
        f"Built user_vector shape={user_vector.shape} in {time.perf_counter() - t0:.4f}s"
    )

    # Normalize just this one small vector (cheap) instead of the whole
    # embeddings matrix (expensive, and now unnecessary since it's
    # precomputed above).
    user_norm = np.linalg.norm(user_vector)
    if user_norm == 0:
        user_norm = 1e-10
    user_vector_normalized = user_vector / user_norm

    # --- Cosine similarity is now a single dot product against
    #     precomputed normalized embeddings ---
    t0 = time.perf_counter()
    similarities = normalized_embeddings @ user_vector_normalized
    logger.info(
        f"Similarity computed | shape={similarities.shape} "
        f"max={np.max(similarities):.4f} min={np.min(similarities):.4f} "
        f"in {time.perf_counter() - t0:.4f}s"
    )

    # --- Partial top-k selection instead of a full sort over 48k+ items ---
    t0 = time.perf_counter()
    watched_set = set(watched_indices.tolist())
    # Grab a few extra candidates beyond top_n to account for filtering out
    # already-watched movies, capped at the array length.
    k = min(top_n + len(watched_set) + 5, len(similarities))
    top_k_unsorted = np.argpartition(similarities, -k)[-k:]
    top_k_sorted = top_k_unsorted[np.argsort(similarities[top_k_unsorted])[::-1]]
    logger.info(f"Top-{k} candidates selected in {time.perf_counter() - t0:.4f}s")

    # --- Build the response, filtering out already-watched movies ---
    t0 = time.perf_counter()
    recommendations = []
    for idx in top_k_sorted:
        if idx in watched_set:
            continue
        recommendations.append(
            {
                "movie_title": movie_titles_arr[idx],
                "imdbId": int(imdb_ids_arr[idx]),
                "similarity": float(similarities[idx]),
            }
        )
        if len(recommendations) >= top_n:
            break
    logger.info(f"Built {len(recommendations)} recommendations in {time.perf_counter() - t0:.4f}s")

    if not recommendations:
        # Extremely unlikely given the +5 buffer above, but guard against it
        # rather than silently returning an empty list.
        logger.warning(
            "Top-k buffer exhausted without finding enough unwatched movies; "
            "widening search."
        )
        top_k_full = np.argsort(similarities)[::-1]
        for idx in top_k_full:
            if idx in watched_set:
                continue
            recommendations.append(
                {
                    "movie_title": movie_titles_arr[idx],
                    "imdbId": int(imdb_ids_arr[idx]),
                    "similarity": float(similarities[idx]),
                }
            )
            if len(recommendations) >= top_n:
                break

    total_time = time.perf_counter() - t_start
    logger.info(
        f"recommend_movies() completed in {total_time:.4f}s total | "
        f"top result: {recommendations[0] if recommendations else None}"
    )
    logger.info("=" * 40)

    return recommendations
