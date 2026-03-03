import os
import pickle
from dataclasses import dataclass

import faiss
import numpy as np

# Single-threaded HNSW construction — prevents OMP segfaults on macOS/Python 3.13.
faiss.omp_set_num_threads(1)

from retrieval.embedder import Embedder


@dataclass
class Document:
    id: str
    text: str
    score: float = 0.0


class VectorStore:
    def __init__(
        self,
        embedding_dim: int = 384,
        index_path: str | None = None,
        passages_path: str | None = None,
    ):
        self.embedding_dim = embedding_dim
        self._texts: list[str] = []

        if (
            index_path is not None
            and passages_path is not None
            and os.path.exists(index_path)
            and os.path.exists(passages_path)
        ):
            self._index = faiss.read_index(index_path)
            self._index.hnsw.efSearch = 50
            with open(passages_path, "rb") as f:
                self._texts = pickle.load(f)
        else:
            self._index = faiss.IndexHNSWFlat(
                embedding_dim, 32, faiss.METRIC_INNER_PRODUCT
            )
            self._index.hnsw.efConstruction = 64
            self._index.hnsw.efSearch = 50

    def add_documents(
        self, texts: list[str], embedder: Embedder, batch_size: int = 32
    ) -> None:
        vectors = embedder.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self._index.add(vectors.astype(np.float32))
        self._texts.extend(texts)

    def search(self, query_vector: np.ndarray, k: int = 50) -> list[Document]:
        k = min(k, self.size)
        if k == 0:
            return []
        query = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(Document(
                id=str(idx),
                text=self._texts[idx],
                score=float(np.clip(score, 0.0, 1.0)),
            ))
        return results

    def save(self, index_path: str, passages_path: str) -> None:
        for path in (index_path, passages_path):
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
        faiss.write_index(self._index, index_path)
        with open(passages_path, "wb") as f:
            pickle.dump(self._texts, f)

    @property
    def size(self) -> int:
        return self._index.ntotal
