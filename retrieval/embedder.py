import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embed_count = 0
        self.model = SentenceTransformer(model_name)
        self.model.encode(["warmup"], convert_to_numpy=True)

    def embed(self, text: str) -> np.ndarray:
        vector = self.model.encode([text], convert_to_numpy=True)[0]
        vector = vector / np.linalg.norm(vector)
        self.embed_count += 1
        return vector
