import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from config import settings
from models import RecommendRequest, RecommendResponse, MovieRecommendationResult, RecommendedMovie
from embedding import load_model, build_movie_vectors_batch
from qdrant_service import search, validate_results

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("recommend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # load once at startup, reused for all requests
    yield


app = FastAPI(title="Movie Recommendation Service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    if not request.movies:
        raise HTTPException(status_code=400, detail="movies list must not be empty")

    top_k = request.top_k or settings.TOP_K
    vectors = await build_movie_vectors_batch(request.movies)
    
    results: list[MovieRecommendationResult] = []
    for movie, vector in zip(request.movies, vectors):
        hits, applied_filter = search(vector, top_k, request.language_filter)
        valid_hits = validate_results(hits, applied_filter)

        if request.movies and not valid_hits and hits:
            logger.warning("All hits for '%s' failed validation", movie.title)

        recommendations = [
            RecommendedMovie(
                tmdb_id=hit.payload.get("tmdb_id"),
                language=hit.payload.get("language"),
                genres=hit.payload.get("genres"),
                release_year=hit.payload.get("release_year"),
                score=hit.score,
            )
            for hit in valid_hits
        ]
       
        results.append(MovieRecommendationResult(input_title=movie.title, recommendations=recommendations))
        del vector
    
    return RecommendResponse(results=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, workers=1)