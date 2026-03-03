from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import faiss
import numpy as np

from retrieval.vector_store import Document


@dataclass
class CacheEntry:
    id: str
    query_text: str
    embedding: np.ndarray
    results: list[Document]
    quality_score: float = 1.0
    hit_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)


@dataclass
class CacheHit:
    entry: CacheEntry
    similarity: float


class SemanticCache:
    def __init__(
        self,
        embedding_dim: int = 384,
        max_size: int = 1000,
        threshold: float = 0.85,
    ):
        self.embedding_dim = embedding_dim
        self.max_size = max_size
        self.threshold = threshold

        self._entries: dict[str, CacheEntry] = {}
        self._id_map: list[str] = []
        self._index = faiss.IndexFlatIP(embedding_dim)

        self.total_hits: int = 0
        self.total_misses: int = 0

    def lookup(self, query_embedding: np.ndarray) -> Optional[CacheHit]:
        if self._index.ntotal == 0:
            self.total_misses += 1
            return None

        query = query_embedding / np.linalg.norm(query_embedding)
        query_2d = query.reshape(1, -1).astype(np.float32)

        scores, indices = self._index.search(query_2d, 1)
        similarity = float(scores[0][0])
        idx = int(indices[0][0])

        if similarity < self.threshold:
            self.total_misses += 1
            return None

        entry_id = self._id_map[idx]
        entry = self._entries[entry_id]
        entry.hit_count += 1
        entry.last_used = datetime.now()

        self.total_hits += 1
        return CacheHit(entry=entry, similarity=similarity)

    def store(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        results: list[Document],
    ) -> str:
        if len(self._entries) >= self.max_size:
            self._evict_lowest_quality()

        entry_id = str(uuid.uuid4())
        normalized = query_embedding / np.linalg.norm(query_embedding)

        entry = CacheEntry(
            id=entry_id,
            query_text=query_text,
            embedding=normalized,
            results=results,
        )

        self._entries[entry_id] = entry
        self._id_map.append(entry_id)
        self._index.add(normalized.reshape(1, -1).astype(np.float32))

        return entry_id

    def update_quality(
        self, entry_id: str, reward: float, alpha: float = 0.3
    ) -> None:
        if entry_id not in self._entries:
            return
        entry = self._entries[entry_id]
        entry.quality_score = (1 - alpha) * entry.quality_score + alpha * reward

    def _evict_lowest_quality(self) -> None:
        worst_id = min(self._entries, key=lambda eid: self._entries[eid].quality_score)
        del self._entries[worst_id]
        self._id_map.remove(worst_id)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        if not self._id_map:
            return
        vectors = np.stack(
            [self._entries[eid].embedding for eid in self._id_map]
        ).astype(np.float32)
        self._index.add(vectors)

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        if total == 0:
            return 0.0
        return self.total_hits / total
