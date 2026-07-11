from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


COLUMNS = [
    "movieId",
    "rating_emb",
    "movie_title",
    "movie_genres",
    "genre_emb",
    "tag_emb",
    "imdbId",
    "movie_vector",
]

BASE_DIR = Path(__file__).resolve().parent

# Load once when server starts


movie_vectors = np.load(BASE_DIR / "movie_vectors.npy", allow_pickle=True)
movie_vectors_df = pd.DataFrame(movie_vectors, columns=COLUMNS)


def recommend_movies(user_movies: List[dict], top_n: int = 10):

    ratings_map = {
        movie["imdbId"]: movie["rating"]
        for movie in user_movies
    }

    rows = movie_vectors_df[
        movie_vectors_df["imdbId"].isin(ratings_map.keys())
    ].copy()

    if rows.empty:
        raise ValueError(
            "None of the provided imdbIds were found."
        )

    weights = rows["imdbId"].map(ratings_map).values

    watched_movie_matrix = np.vstack(
        rows["movie_vector"].values
    )

    user_vector = np.average(
        watched_movie_matrix,
        axis=0,
        weights=weights
    )

    user_vector = user_vector.reshape(1, -1)

    all_movie_matrix = np.vstack(
        movie_vectors_df["movie_vector"].values
    )

    similarities = cosine_similarity(
        user_vector,
        all_movie_matrix
    )[0]

    top_idx = np.argsort(similarities)[::-1]

    watched_ids = set(ratings_map.keys())

    recommendations = []

    for idx in top_idx:

        movie = movie_vectors_df.iloc[idx]

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

    return recommendations