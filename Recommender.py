import logging
from pathlib import Path
from typing import List

from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.info("========================================")
logging.info("Loading metadata and embeddings...")

metadata_path = hf_hub_download(
    repo_id="Shrejankotyan2005/movie_vector",
    filename="movie_metadata.csv",
    repo_type="dataset"
)

logging.info(f"Metadata downloaded: {metadata_path}")

embeddings_path = hf_hub_download(
    repo_id="Shrejankotyan2005/movie_vector",
    filename="movie_embeddings_f32.npy",
    repo_type="dataset"
)

logging.info(f"Embeddings downloaded: {embeddings_path}")

metadata = pd.read_csv(metadata_path)
embeddings = np.load(embeddings_path)

logging.info(f"Metadata shape: {metadata.shape}")
logging.info(f"Embeddings shape: {embeddings.shape}")
logging.info(f"Embeddings dtype: {embeddings.dtype}")

assert len(metadata) == len(embeddings)

logging.info("Startup completed successfully.")
logging.info("========================================")


def recommend_movies(user_movies: List[dict], top_n: int = 10):

    logging.info("========================================")
    logging.info("recommend_movies() started")
    logging.info(f"Input movies: {user_movies}")

    ratings_map = {
        movie["imdbId"]: movie["rating"]
        for movie in user_movies
    }

    logging.info(f"ratings_map created")
    logging.info(f"ratings_map: {ratings_map}")

    rows = metadata[
        metadata["imdbId"].isin(ratings_map.keys())
    ].copy()

    logging.info(f"Matched rows: {len(rows)}")

    if rows.empty:
        logging.error("No matching imdbIds found")
        raise ValueError(
            "None of the provided imdbIds were found."
        )

    logging.info("Extracting watched indices")

    watched_indices = rows.index.to_numpy()

    logging.info(
        f"watched_indices shape: {watched_indices.shape}"
    )

    watched_movie_matrix = embeddings[watched_indices]

    logging.info(
        f"watched_movie_matrix shape: {watched_movie_matrix.shape}"
    )

    
    weights = rows["imdbId"].map(
    ratings_map
    ).values.astype(np.float32)


    logging.info(f"weights: {weights}")
    logging.info(f"weights shape: {weights.shape}")

    logging.info("Building user vector")

    user_vector = np.average(
    watched_movie_matrix,
    axis=0,
    weights=weights
    ).astype(np.float32).reshape(1, -1)

    logging.info(
        f"user_vector shape: {user_vector.shape}"
    )

    logging.info(
        f"user_vector dtype: {user_vector.dtype}"
    )
    logging.info(
        f"user embeddings shape: {embeddings.shape}"
    )
    logging.info(
        f"user embeddings shape: {embeddings.dtype}"
    )

    logging.info("Starting cosine similarity")

    #similarities = cosine_similarity(
     #   user_vector,
     #   embeddings
   # )[0]
    user_vector = user_vector.flatten()

    user_norm = np.linalg.norm(user_vector)

    similarities = (
    embeddings @ user_vector
    ) / (
    embedding_norms * user_norm + 1e-8
    )




    logging.info("Cosine similarity completed")

    logging.info(
        f"similarities shape: {similarities.shape}"
    )

    logging.info(
        f"Max similarity: {np.max(similarities)}"
    )

    logging.info(
        f"Min similarity: {np.min(similarities)}"
    )

    logging.info("Sorting similarities")

    top_idx = np.argsort(similarities)[::-1]

    logging.info(
        f"top_idx shape: {top_idx.shape}"
    )

    watched_ids = set(ratings_map.keys())

    recommendations = []

    logging.info("Building recommendations")

    for idx in top_idx:

        movie = metadata.iloc[idx]

        if movie["imdbId"] in watched_ids:
            continue

        recommendations.append(
            {
                "movie_title": movie["movie_title"],
                "imdbId": int(movie["imdbId"]),
                "similarity": float(similarities[idx]),
            }
        )

        if len(recommendations) % 5 == 0:
            logging.info(
                f"Recommendations collected: {len(recommendations)}"
            )

        if len(recommendations) >= top_n:
            break

    logging.info(
        f"Final recommendation count: {len(recommendations)}"
    )

    logging.info(
        f"Top recommendation: {recommendations[0] if recommendations else None}"
    )

    logging.info("recommend_movies() completed")
    logging.info("========================================")

    return recommendations
