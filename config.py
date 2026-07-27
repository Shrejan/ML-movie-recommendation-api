import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    QDRANT_URL: str = os.getenv(
        "QDRANT_URL",
        "https://fdb53587-48f3-43ac-b409-c524c218ed06.eu-west-2-0.aws.cloud.qdrant.io",
    )

    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")

    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "movies_E3")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    TOP_K: int = int(os.getenv("TOP_K", "10"))
    MAX_CONCURRENT_EMBEDDINGS: int = int(os.getenv("MAX_CONCURRENT_EMBEDDINGS", "4"))
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
