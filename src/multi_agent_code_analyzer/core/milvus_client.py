from typing import Dict, Any, List
from pymilvus import Collection, connections
from ..config import settings


class MilvusClient:
    def __init__(self):
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )
        self.collection = Collection(settings.COLLECTION_NAME)

    async def store_embeddings(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]] = None):
        """Store embeddings in Milvus."""
        if metadata is None:
            metadata = [{}] * len(embeddings)

        entities = [
            {"embedding": emb, "metadata": meta}
            for emb, meta in zip(embeddings, metadata)
        ]

        self.collection.insert(entities)

    async def find_similar(self, embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar vectors in Milvus."""
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["metadata"]
        )

        return [
            {
                "id": hit.id,
                "distance": hit.distance,
                "metadata": hit.entity.get("metadata", {})
            }
            for hit in results[0]
        ]

    def __del__(self):
        try:
            connections.disconnect("default")
        except:
            pass
