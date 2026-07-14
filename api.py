import os
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from Recommender import recommend_movies

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)
logger.info("Starting Movie Recommendation API...")
# -----------------------------------------

app = FastAPI(
    title="Movie Recommendation API"
)

logger.info("FastAPI application initialized.")


class UserMovie(BaseModel):
    imdbId: int
    rating: float


class Recommendation(BaseModel):
    movie_title: str
    imdbId: int
    similarity: float


@app.get("/")
def home():
    logger.info("Home endpoint called.")
    return {
        "message": "Movie Recommendation API Running"
    }


@app.post(
    "/rec",
    response_model=List[Recommendation]
)
def recommend(user_movies: List[UserMovie]):
    logger.info("========================================")

    logger.info("New recommendation request received.")
    logger.info(f"Number of movies received: {len(user_movies)}")

    try:
        logger.info("Raw request data:")
        logger.info(user_movies)

        payload = []

        for i, movie in enumerate(user_movies):
            

            if hasattr(movie, "model_dump"):
                movie_dict = movie.model_dump()
            else:
                movie_dict = movie.dict()

            
            payload.append(movie_dict)

        logger.info("Payload successfully created.")
        logger.info("Calling recommend_movies()...")

        recommendations = recommend_movies(payload)

        logger.info("recommend_movies() execution completed.")

        logger.info(
            f"Number of recommendations returned: {len(recommendations)}"
        )

        for i, rec in enumerate(recommendations):
            logger.info(f"Recommendation {i+1}: {rec}")

        logger.info("Returning recommendations to client.")
        logger.info("========================================")

        return recommendations

    except ValueError as e:
        logger.error("ValueError occurred!", exc_info=True)

        raise HTTPException(
            status_code=400,
            detail=str(e)
        ) from e

    except Exception as e:
        logger.exception("Unexpected exception occurred!")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) from e


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))

    logger.info(f"Starting Uvicorn server on port {port}")
    logger.info("Server is ready to accept requests.")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
