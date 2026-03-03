from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueryTrace:
    query_id: str
    query_text: str
    path: str                           # "CACHE_HIT" or "CACHE_MISS"
    embed_ms: float = 0.0
    cache_check_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    cost_usd: float = 0.0
    cache_similarity: Optional[float] = None
    cache_quality_score: Optional[float] = None
    result_count: int = 0
    reward: Optional[float] = None

    def cost_saved(self) -> float:
        # Cache hits skip vector search + rerank — those calls cost money in production.
        # $0.00015 is a representative per-query cost for one search + rerank API call.
        # Cache misses save nothing — we had to do the full retrieval.
        if self.path == "CACHE_HIT":
            return 0.00015
        return 0.0


class LatencyTracer:

    @contextmanager
    def measure(self, trace: QueryTrace, field_name: str):
        # perf_counter_ns: monotonic, nanosecond resolution.
        # time.time() has ~1ms granularity on some platforms — unreliable for 3ms FAISS calls.
        # perf_counter_ns gives sub-microsecond precision, important when measuring
        # operations as short as a 0.1ms cache lookup or 3ms HNSW search.
        t0 = time.perf_counter_ns()
        yield
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000  # ns → ms
        # setattr lets us target any field by name from the caller without a giant if-chain.
        setattr(trace, field_name, elapsed_ms)
