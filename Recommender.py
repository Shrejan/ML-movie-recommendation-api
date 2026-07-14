from pathlib import Path
from typing import List
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity



BASE_DIR = Path(__file__).resolve().parent

# Load once when server starts


metadata_path = hf_hub_download(
    repo_id="Shrejankotyan2005/movie_vector",
    filename="movie_metadata.csv",
    repo_type="dataset"
)

embeddings_path = hf_hub_download(
    repo_id="Shrejankotyan2005/movie_vector",
    filename="movie_embeddings_f32.npy",
    repo_type="dataset"
)

metadata = pd.read_csv(metadata_path)
embeddings = np.load(embeddings_path)


print("Embeddings shape:", embeddings.shape)
print("Embeddings dtype:", embeddings.dtype)


# Ensure row alignment
assert len(metadata) == len(embeddings)

def recommend_movies(user_movies: List[dict], top_n: int = 10):
    print("the function has started")
    ratings_map = {
        movie["imdbId"]: movie["rating"]
        for movie in user_movies
    }

    rows = metadata[
        metadata["imdbId"].isin(ratings_map.keys())
    ].copy()

    if rows.empty:
        raise ValueError(
            "None of the provided imdbIds were found."
        )

    # Get row indices of watched movies
    watched_indices = rows.index.to_numpy()

    # Get corresponding embeddings
    watched_movie_matrix = embeddings[watched_indices]

    weights = rows["imdbId"].map(ratings_map).values

    # Build user vector
    user_vector = np.average(
        watched_movie_matrix,
        axis=0,
        weights=weights
    ).reshape(1, -1)
    print("cosine_similarity started")
    # Similarity against all movies
    similarities = cosine_similarity(
        user_vector,
        embeddings
    )[0]
    print("cosine_similarity ended")
    top_idx = np.argsort(similarities)[::-1]

    watched_ids = set(ratings_map.keys())

    recommendations = []

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
        
        if len(recommendations) >= top_n:
            break
    print(recommendations)
    return recommendations
