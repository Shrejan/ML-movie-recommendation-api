# Movie Recommendation Service

FastAPI + SentenceTransformers + Qdrant. Designed for Render's 512 MB RAM tier.

## Run locally
```
pip install -r requirements.txt
export QDRANT_URL=http://localhost:6333
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

## API
`POST /recommend`
```json
{
  "movies": [
    {"title": "Inception", "overview": "...", "genres": ["Sci-Fi"], "keywords": ["dream"],
     "director": "Christopher Nolan", "cast": ["Leonardo DiCaprio"], "release_year": 2010, "language": "en"}
  ],
  "top_k": 5,
  "language_filter": "en"
}
```

## Why this fits in 512 MB
- **Single model instance**: `all-MiniLM-L6-v2` (~90 MB, 384-dim) is loaded once in the FastAPI `lifespan` hook and reused for every request — no per-request or per-worker duplication.
- **One Uvicorn worker**: multiple workers would each load a full copy of the model; `workers=1` plus internal `asyncio` concurrency (bounded by a semaphore) handles concurrent requests without multiplying memory.
- **CPU-only inference**: no GPU/CUDA libraries loaded, avoiding large driver/runtime overhead.
- **float32 everywhere**: embeddings are cast to `float32` (not float64), halving vector memory.
- **No dataset in memory**: the service never loads the full movie catalog — only Qdrant holds vectors + minimal payload; each request only touches the movies it was given.
- **Bounded, short-lived objects**: intermediate embeddings and vectors are computed, combined, and explicitly `del`eted; nothing is cached beyond the model itself.
- **Batching inside a request**: all movies in a request are embedded concurrently (bounded by `MAX_CONCURRENT_EMBEDDINGS`) rather than spawning unbounded parallel work.
- **Trust Qdrant's scores**: no manual re-computation of cosine similarity server-side — only lightweight integrity checks (filter respected, sorted, unique, required fields present), keeping CPU/memory work minimal per request.
