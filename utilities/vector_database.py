import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable
from utilities_2.openai_utils.embedding import EmbeddingModel
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import VectorParams
import uuid

def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

class QdrantDatabase:
    def __init__(self, embedding_model=None):
        self.qdrant_client = QdrantClient(location=":memory:")
        self.collection_name = "my_collection"
        self.embedding_model = embedding_model or EmbeddingModel(embeddings_model_name= "text-embedding-3-small", dimensions=1000)
        vector_params = VectorParams(
            size=self.embedding_model.dimensions,  # vector size 
            distance="Cosine"
          )  # distance metric
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config={"text": vector_params},
        )
        self.vectors = defaultdict(np.array)  # Still keeps a local copy if needed

    def string_to_int_id(self, s: str) -> int:
        return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (10**8)
    def get_test_vector(self):
        retrieved_vector = self.qdrant_client.retrieve(
                collection_name="my_collection",
                ids=[self.string_to_int_id("test_key")]
            )
        return retrieved_vector
    def insert(self, key: str, vector: np.array) -> None:
        point_id = str(uuid.uuid4())
        payload = {"text": key}  

        point = PointStruct(
            id=point_id,
            vector={"default": vector.tolist()},
            payload=payload
        )
        print(f"Inserting vector for key: {key}, ID: {point_id}")
        # Insert the vector into Qdrant with the associated document
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[point]  # Qdrant expects a list of PointStruct
        )


    def search(
        self,
        query_vector: np.array,
        k: int=5,
        distance_measure: Callable = cosine_similarity,
    ) -> List[Tuple[str, float]]:
        # Perform search in Qdrant
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        print(type(query_vector)) 
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,  # Pass the vector as a list
            limit=k
        )
        return [(result.payload['text'], result.score) for result in search_results]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
    ) -> List[Tuple[str, float]]:
        
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure)
        return [result[0] for result in results] if return_as_text else results

    async def abuild_from_list(self, list_of_text: List[str]) -> "QdrantDatabase":
        from qdrant_client.http import models
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        points = [
            models.PointStruct(
            id=str(uuid.uuid4()),
            vector={"text": embedding}, # Should be a named vector as per vector_config
            payload={
                "text": text
                }
            )
            for text, embedding in zip(list_of_text, embeddings)
        ]
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return self
    