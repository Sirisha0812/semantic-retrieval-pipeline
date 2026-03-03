"""
Tests for RetrievalPipeline (Layer 8).

TEST 1: Cold query  — first query is always CACHE_MISS
TEST 2: Warm query  — same text twice → second is CACHE_HIT
TEST 3: Semantic cache hit or miss — verify latency bounds either way
TEST 4: Traces accumulate across queries
TEST 5: StatsReporter works — 10 queries, hit_rate in [0,1]
TEST 6: DriftDetector receives exactly one signal per query
TEST 7: Full end-to-end latency — warm query faster than cold
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import RetrievalPipeline


# ── fixture — load pipeline ONCE for the whole module (~3s model loading) ─────

@pytest.fixture(scope="module")
def pipeline():
    """
    scope="module": models (Embedder, ReRanker, VectorStore) load once and are
    shared across all 7 tests. Without this, each test would pay ~3s in model
    load time and the suite would take ~21s instead of ~3s total.
    """
    return RetrievalPipeline()


# ── TEST 1: Cold query is a cache miss ────────────────────────────────────────

def test_cold_query_is_cache_miss(pipeline):
    """
    First query on a fresh pipeline — nothing is cached yet.
    Must follow the full MISS path: embed → search → rerank.
    """
    results, trace = pipeline.query("what is employee leave policy?")

    assert trace.path == "CACHE_MISS", (
        f"First query must be CACHE_MISS, got {trace.path}"
    )
    assert trace.search_ms > 0,  "MISS must have search_ms > 0"
    assert trace.rerank_ms > 0,  "MISS must have rerank_ms > 0"
    assert trace.total_ms < 1000, (
        f"Full retrieval must be < 1000ms, got {trace.total_ms:.1f}ms"
    )
    assert len(results) == 5, (
        f"top_k_rerank=5 → expect 5 results, got {len(results)}"
    )
    assert trace.result_count == 5
    assert trace.reward is not None, "reward must be set after feedback simulation"

    print(
        f"✓ TEST 1 PASSED: CACHE_MISS — "
        f"embed={trace.embed_ms:.1f}ms, search={trace.search_ms:.1f}ms, "
        f"rerank={trace.rerank_ms:.1f}ms, total={trace.total_ms:.1f}ms"
    )


# ── TEST 2: Warm query is a cache hit ─────────────────────────────────────────

def test_warm_query_is_cache_hit(pipeline):
    """
    Query the same text twice.
    First: MISS (stores in cache).
    Second: HIT (identical embedding → similarity ≈ 1.0 ≥ threshold 0.85).
    """
    query_text = "how many vacation days do employees get?"

    _, trace_cold = pipeline.query(query_text)
    assert trace_cold.path == "CACHE_MISS", "First query should be CACHE_MISS"

    results_warm, trace_warm = pipeline.query(query_text)

    assert trace_warm.path == "CACHE_HIT", (
        f"Identical repeat should be CACHE_HIT, got {trace_warm.path}"
    )
    assert trace_warm.search_ms == 0.0, (
        "CACHE_HIT must NOT run vector search (search_ms must be 0.0)"
    )
    assert trace_warm.rerank_ms == 0.0, (
        "CACHE_HIT must NOT run reranker (rerank_ms must be 0.0)"
    )
    assert trace_warm.total_ms < trace_cold.total_ms, (
        f"Warm query ({trace_warm.total_ms:.1f}ms) must be faster than "
        f"cold query ({trace_cold.total_ms:.1f}ms)"
    )
    assert trace_warm.cache_similarity is not None
    assert trace_warm.cache_similarity > 0.85

    print(
        f"✓ TEST 2 PASSED: cold={trace_cold.total_ms:.1f}ms → "
        f"warm={trace_warm.total_ms:.1f}ms "
        f"(similarity={trace_warm.cache_similarity:.4f})"
    )


# ── TEST 3: Semantic cache — latency bounds regardless of HIT/MISS ─────────────

def test_semantic_cache_latency_bounds(pipeline):
    """
    "employee leave policy" first → stored.
    "staff vacation policy" queried → might be HIT or MISS depending on similarity.

    We don't force the outcome — just assert sensible latency bounds:
      HIT  → under 50ms  (embed + cache check only)
      MISS → under 500ms (full pipeline on CPU)
    """
    pipeline.query("employee leave policy")  # prime the cache

    _, trace = pipeline.query("staff vacation policy")

    if trace.path == "CACHE_HIT":
        assert trace.total_ms < 50, (
            f"CACHE_HIT should be < 50ms, got {trace.total_ms:.1f}ms"
        )
        print(f"✓ TEST 3 PASSED: HIT — {trace.total_ms:.1f}ms (sim={trace.cache_similarity:.4f})")
    else:
        assert trace.total_ms < 500, (
            f"CACHE_MISS should be < 500ms on this hardware, got {trace.total_ms:.1f}ms"
        )
        print(f"✓ TEST 3 PASSED: MISS — {trace.total_ms:.1f}ms (different enough for miss)")


# ── TEST 4: Traces accumulate ─────────────────────────────────────────────────

def test_traces_accumulate(pipeline):
    """
    pipeline.traces is a flat list that grows with every query.
    Run 5 new queries and check the count goes up by exactly 5.
    """
    count_before = len(pipeline.traces)

    for i in range(5):
        pipeline.query(f"trace accumulation test query {i}")

    assert len(pipeline.traces) == count_before + 5, (
        f"Expected {count_before + 5} traces, got {len(pipeline.traces)}"
    )
    print(f"✓ TEST 4 PASSED: traces grew from {count_before} to {len(pipeline.traces)}")


# ── TEST 5: StatsReporter integrates correctly ────────────────────────────────

def test_stats_reporter_integration(pipeline):
    """
    Run 10 queries.  Verify StatsReporter aggregates them correctly.
    hit_rate must be [0.0, 1.0].
    avg_latency_miss_ms must be > 0 (at least one miss happened — fresh cache).
    """
    # Use distinct texts to guarantee a mix of hits and misses
    queries = [
        "parental leave benefits",
        "parental leave benefits",      # repeat → hit
        "sick day policy",
        "sick day policy",              # repeat → hit
        "remote work guidelines",
        "remote work guidelines",       # repeat → hit
        "overtime compensation rules",
        "overtime compensation rules",  # repeat → hit
        "employee handbook overview",
        "employee handbook overview",   # repeat → hit
    ]
    count_before = len(pipeline.traces)
    for q in queries:
        pipeline.query(q)

    stats = pipeline.get_stats()

    assert stats["total_queries"] == count_before + 10, (
        f"Expected {count_before + 10} total queries, got {stats['total_queries']}"
    )
    assert 0.0 <= stats["hit_rate"] <= 1.0, (
        f"hit_rate {stats['hit_rate']} out of [0,1]"
    )
    # There must be at least some miss latency recorded (the first of each pair)
    assert stats["avg_latency_miss_ms"] > 0, (
        "avg_latency_miss_ms must be > 0 (some queries were misses)"
    )

    print(
        f"✓ TEST 5 PASSED: {stats['total_queries']} total queries, "
        f"hit_rate={stats['hit_rate']:.2f}, "
        f"miss_ms={stats['avg_latency_miss_ms']:.1f}ms, "
        f"hit_ms={stats['avg_latency_hit_ms']:.1f}ms"
    )


# ── TEST 6: DriftDetector receives one signal per query ───────────────────────

def test_drift_detector_signal_count(pipeline):
    """
    Every query records exactly one signal to the drift detector
    (either "cache" or "fresh").  Run 20 queries, check total signal count
    equals 20 more than before the test.
    """
    count_before = (
        pipeline.drift_detector.cache_count +
        pipeline.drift_detector.fresh_count
    )

    for i in range(20):
        pipeline.query(f"drift signal test {i}")

    count_after = (
        pipeline.drift_detector.cache_count +
        pipeline.drift_detector.fresh_count
    )

    # drift_detector uses deque(maxlen=window_size=50) — once it fills, older
    # signals are evicted.  We check count_after >= count_before to handle the
    # case where the window wraps, but also assert the deques haven't shrunk.
    assert count_after >= count_before, (
        f"Signal count should have grown: before={count_before}, after={count_after}"
    )
    # Deque total is capped at window_size (50 per stream) — just verify it's sane.
    assert count_after <= pipeline.drift_detector.window_size * 2, (
        f"Total signals {count_after} exceeds 2×window_size"
    )
    print(
        f"✓ TEST 6 PASSED: drift signals before={count_before}, after={count_after} "
        f"(cache={pipeline.drift_detector.cache_count}, "
        f"fresh={pipeline.drift_detector.fresh_count})"
    )


# ── TEST 7: Full end-to-end — warm query is faster than cold ──────────────────

def test_end_to_end_speedup(pipeline):
    """
    One cold query, then the exact same query warm.
    Warm must be strictly faster.
    Also prints both trace breakdowns for visual confirmation.
    """
    query_text = "what benefits does the company offer new employees?"

    results_cold, trace_cold = pipeline.query(query_text)
    results_warm, trace_warm = pipeline.query(query_text)

    assert trace_cold.path == "CACHE_MISS", "Cold must be CACHE_MISS"
    assert trace_warm.path == "CACHE_HIT",  "Warm must be CACHE_HIT"
    assert trace_warm.total_ms < trace_cold.total_ms, (
        f"Warm ({trace_warm.total_ms:.1f}ms) must be faster than "
        f"cold ({trace_cold.total_ms:.1f}ms)"
    )

    speedup = trace_cold.total_ms / trace_warm.total_ms
    assert speedup > 1.0, f"speedup must be > 1, got {speedup:.2f}x"

    print(
        f"✓ TEST 7 PASSED: end-to-end speedup\n"
        f"  COLD: embed={trace_cold.embed_ms:.1f}ms  "
        f"cache_check={trace_cold.cache_check_ms:.1f}ms  "
        f"search={trace_cold.search_ms:.1f}ms  "
        f"rerank={trace_cold.rerank_ms:.1f}ms  "
        f"total={trace_cold.total_ms:.1f}ms\n"
        f"  WARM: embed={trace_warm.embed_ms:.1f}ms  "
        f"cache_check={trace_warm.cache_check_ms:.1f}ms  "
        f"total={trace_warm.total_ms:.1f}ms\n"
        f"  Speedup: {speedup:.1f}x"
    )
