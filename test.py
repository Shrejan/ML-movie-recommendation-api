from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

points, _ = client.scroll(
    collection_name="movies_E3",
    limit=1,
    with_payload=True,
    with_vectors=False,
)

print(points)