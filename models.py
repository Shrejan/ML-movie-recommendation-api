from typing import Optional
from pydantic import BaseModel, Field


class MovieInput(BaseModel):
    title: str
    overview: Optional[str] = ""
    genres: Optional[list[str]] = Field(default_factory=list)
    keywords: Optional[list[str]] = Field(default_factory=list)
    director: Optional[str] = ""
    cast: Optional[list[str]] = Field(default_factory=list)
    release_year: Optional[int] = None
    language: Optional[str] = None


class RecommendRequest(BaseModel):
    movies: list[MovieInput]
    top_k: Optional[int] = None
    language_filter: Optional[str] = None


class RecommendedMovie(BaseModel):
    tmdb_id: Optional[int] = None
    language: Optional[str] = None
    genres: Optional[list[str]] = None
    release_year: Optional[int] = None
    score: float


class MovieRecommendationResult(BaseModel):
    input_title: str
    recommendations: list[RecommendedMovie]


class RecommendResponse(BaseModel):
    results: list[MovieRecommendationResult]