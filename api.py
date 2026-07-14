import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from Recommender import recommend_movies

app = FastAPI(
    title="Movie Recommendation API"
)


class UserMovie(BaseModel):
    imdbId: int
    rating: float


class Recommendation(BaseModel):
    movie_title: str
    imdbId: int
    similarity: float


@app.get("/")

def home():
    
    return {
        "message": "Movie Recommendation API Running"
    }


@app.post(
    "/rec",
    response_model=List[Recommendation]
)
print(response_model)

def recommend(user_movies: List[UserMovie]):

    try:
        print(user_movies)
        payload = []
        for movie in user_movies:
            if hasattr(movie, "model_dump"):
                payload.append(movie.model_dump())
            else:
                payload.append(movie.dict())

        recommendations = recommend_movies(payload)
        print(recommendations)
        print(payload)

        return recommendations

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) from e


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
