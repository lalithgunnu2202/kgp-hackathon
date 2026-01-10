import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunk_embeddings = None
        self.chunks = None

    def index_chunks(self, chunks):
        """
        Embed and store novel chunks
        """
        self.chunks = chunks
        self.chunk_embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=True
        )

    def retrieve(self, query, top_k=3):
        """
        Retrieve top-k relevant chunks for a query
        """
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )

        # cosine similarity
        scores = np.dot(self.chunk_embeddings, query_embedding)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [(self.chunks[i], scores[i]) for i in top_indices]
