from __future__ import annotations

import time
from typing import Optional

from cache.drift_detector import DriftDetector, DriftStatus
from cache.semantic_cache import SemanticCache
from feedback.simulator import FeedbackSimulator
from instrumentation.reporter import StatsReporter
from instrumentation.tracer import LatencyTracer, QueryTrace
from retrieval.embedder import Embedder
from retrieval.reranker import ReRanker
from retrieval.vector_store import Document, VectorStore

# Default configuration lives here so demos and tests can override individual
# knobs without constructing a full config dict from scratch.
_DEFAULTS = {
    "embedding_model":   "all-MiniLM-L6-v2",
    "reranker_model":    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cache_threshold":   0.85,
    "cache_max_size":    1000,
    "top_k_retrieve":    50,
    "top_k_rerank":      5,
    "drift_window_size": 50,
    "drift_threshold":   0.15,
    "cost_per_search":   0.0001,
    "cost_per_rerank":   0.00005,
}


class RetrievalPipeline:

    def __init__(self, config: Optional[dict] = None) -> None:
        # Merge caller overrides on top of defaults — callers only have to supply
        # the keys they want to change, everything else stays at sensible values.
        cfg = {**_DEFAULTS, **(config or {})}
        self._cfg = cfg

        # ── Layer 1: Embedder ──────────────────────────────────────────────────
        # Loads model + runs warmup inside __init__ so the first real query is fast.
        self.embedder = Embedder()

        # ── Layer 3: VectorStore ───────────────────────────────────────────────
        # Passing index_path + passages_path loads the pre-built HNSW index from
        # disk rather than rebuilding it on every run (would take ~30s for 10K docs).
        self.vector_store = VectorStore(
            embedding_dim=384,
            index_path="data/ms_marco_10k.index",
            passages_path="data/passages.pkl",
        )

        # ── Layer 3: ReRanker ──────────────────────────────────────────────────
        # Loads CrossEncoder + runs warmup predict() in __init__.
        self.reranker = ReRanker()

        # ── Layer 2: SemanticCache ─────────────────────────────────────────────
        self.cache = SemanticCache(
            embedding_dim=384,
            max_size=cfg["cache_max_size"],
            threshold=cfg["cache_threshold"],
        )

        # ── Layer 5: DriftDetector ─────────────────────────────────────────────
        self.drift_detector = DriftDetector(
            window_size=cfg["drift_window_size"],
            threshold=cfg["drift_threshold"],
        )

        # ── Layer 4: FeedbackSimulator ─────────────────────────────────────────
        self.feedback_sim = FeedbackSimulator()

        # ── Layer 6: LatencyTracer ─────────────────────────────────────────────
        self.tracer = LatencyTracer()

        # Ordered list of every QueryTrace produced this session.
        # Used by get_reporter() / get_stats() to aggregate results.
        self.traces: list[QueryTrace] = []

    # ── query ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        query_id: Optional[str] = None,
    ) -> tuple[list[Document], QueryTrace]:

        # ── step 1: stable, zero-padded ID so log lines sort correctly ─────────
        if query_id is None:
            query_id = f"q_{len(self.traces):04d}"

        # ── step 2: create trace — default path is CACHE_MISS ─────────────────
        trace = QueryTrace(
            query_id=query_id,
            query_text=query_text,
            path="CACHE_MISS",
        )

        # ── step 3: start wall-clock for total latency ─────────────────────────
        # We capture total_ms manually (not via tracer.measure) because it needs
        # to wrap ALL steps including cache store at the end of a miss path.
        t_total = time.perf_counter_ns()

        # ── step 4: embed ──────────────────────────────────────────────────────
        with self.tracer.measure(trace, "embed_ms"):
            embedding = self.embedder.embed(query_text)

        # ── step 5: cache lookup ───────────────────────────────────────────────
        with self.tracer.measure(trace, "cache_check_ms"):
            cache_hit = self.cache.lookup(embedding)

        # ── step 6: cache HIT path ─────────────────────────────────────────────
        if cache_hit is not None:
            trace.path               = "CACHE_HIT"
            trace.cache_similarity   = cache_hit.similarity
            trace.cache_quality_score = cache_hit.entry.quality_score
            results                  = cache_hit.entry.results
            trace.cost_usd           = 0.0   # no compute cost — served from memory
            source                   = "cache"
            cache_entry_id           = cache_hit.entry.id

        # ── step 7: cache MISS path ────────────────────────────────────────────
        else:
            with self.tracer.measure(trace, "search_ms"):
                candidates = self.vector_store.search(
                    embedding, k=self._cfg["top_k_retrieve"]
                )

            with self.tracer.measure(trace, "rerank_ms"):
                results = self.reranker.rerank(
                    query_text, candidates, top_k=self._cfg["top_k_rerank"]
                )

            # Cost = search + rerank compute expressed as dollar figure.
            trace.cost_usd = self._cfg["cost_per_search"] + self._cfg["cost_per_rerank"]
            source         = "fresh"

            # Store result in cache so the next semantically-similar query is a hit.
            cache_entry_id = self.cache.store(query_text, embedding, results)

        # ── step 8: record total latency and result count ──────────────────────
        trace.total_ms    = (time.perf_counter_ns() - t_total) / 1_000_000
        trace.result_count = len(results)

        # ── step 9: simulate user feedback, close the RL loop ──────────────────
        # relevance_scores: the FAISS/reranker scores already attached to each
        # Document are a proxy for how relevant the result is to the query.
        relevance_scores = [r.score for r in results]
        feedback         = self.feedback_sim.simulate(query_id, results, relevance_scores)
        reward           = feedback.to_reward()
        trace.reward     = reward

        # Feed reward back into cache quality (EMA update) and drift detector.
        # For a cache hit, this keeps the quality score of the served entry current.
        # For a miss, the newly stored entry starts at quality=1.0 but gets its
        # first real update immediately.
        self.cache.update_quality(cache_entry_id, reward)
        self.drift_detector.record(source, reward)

        # ── step 10: archive and return ───────────────────────────────────────
        self.traces.append(trace)
        return results, trace

    # ── convenience accessors ──────────────────────────────────────────────────

    def get_drift_status(self) -> DriftStatus:
        return self.drift_detector.check_drift()

    def get_reporter(self) -> StatsReporter:
        return StatsReporter(self.traces)

    def get_stats(self) -> dict:
        return self.get_reporter().summary()
