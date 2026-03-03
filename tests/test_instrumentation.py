"""
Tests for LatencyTracer (Layer 6) and StatsReporter (Layer 6).

TEST 1: QueryTrace cost_saved
TEST 2: LatencyTracer measures time — sleep(0.01) → 9ms–20ms
TEST 3: LatencyTracer sets correct field, others stay 0.0
TEST 4: StatsReporter summary — 5 hits @ 20ms, 5 misses @ 110ms
TEST 5: P95 latency matches numpy percentile
TEST 6: print_summary and print_latency_breakdown run without error
"""
import sys
import os
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instrumentation.tracer import QueryTrace, LatencyTracer
from instrumentation.reporter import StatsReporter


# ── helpers ───────────────────────────────────────────────────────────────────

def make_trace(path: str, total_ms: float = 0.0) -> QueryTrace:
    return QueryTrace(
        query_id="q",
        query_text="test query",
        path=path,
        total_ms=total_ms,
    )


# ── TEST 1: cost_saved ────────────────────────────────────────────────────────

def test_cost_saved():
    """
    CACHE_HIT → cost_saved = 0.00015
    CACHE_MISS → cost_saved = 0.0
    """
    hit  = make_trace("CACHE_HIT")
    miss = make_trace("CACHE_MISS")

    assert hit.cost_saved()  == 0.00015, f"HIT cost_saved should be 0.00015, got {hit.cost_saved()}"
    assert miss.cost_saved() == 0.0,     f"MISS cost_saved should be 0.0, got {miss.cost_saved()}"
    print("✓ TEST 1 PASSED: cost_saved CACHE_HIT=0.00015, CACHE_MISS=0.0")


# ── TEST 2: LatencyTracer measures time ───────────────────────────────────────

def test_latency_tracer_measures_time():
    """
    sleep(0.01) ≈ 10ms.
    Measured value must be in [9ms, 20ms]:
      Lower bound 9ms: allow -1ms for OS scheduler jitter
      Upper bound 20ms: allow 10ms for slow CI/test machines
    """
    tracer = LatencyTracer()
    trace  = make_trace("CACHE_MISS")

    with tracer.measure(trace, "search_ms"):
        time.sleep(0.01)

    assert 9 <= trace.search_ms <= 20, (
        f"sleep(0.01) should measure 9-20ms, got {trace.search_ms:.2f}ms"
    )
    print(f"✓ TEST 2 PASSED: measured {trace.search_ms:.2f}ms for sleep(0.01)")


# ── TEST 3: LatencyTracer sets correct field ──────────────────────────────────

def test_latency_tracer_sets_correct_field():
    """
    Measure into "embed_ms" only.
    trace.embed_ms > 0, all other timing fields still 0.0.
    """
    tracer = LatencyTracer()
    trace  = make_trace("CACHE_MISS")

    with tracer.measure(trace, "embed_ms"):
        time.sleep(0.001)  # 1ms — just needs to be nonzero

    assert trace.embed_ms > 0, "embed_ms should be > 0 after measurement"
    assert trace.cache_check_ms == 0.0, "cache_check_ms should still be 0.0"
    assert trace.search_ms      == 0.0, "search_ms should still be 0.0"
    assert trace.rerank_ms      == 0.0, "rerank_ms should still be 0.0"
    assert trace.total_ms       == 0.0, "total_ms should still be 0.0"
    print(f"✓ TEST 3 PASSED: embed_ms={trace.embed_ms:.2f}ms, others=0.0")


# ── TEST 4: StatsReporter summary ─────────────────────────────────────────────

def test_stats_reporter_summary():
    """
    5 CACHE_HIT  @ total_ms=20
    5 CACHE_MISS @ total_ms=110

    Expected:
      hit_rate            = 0.5
      avg_latency_hit_ms  = 20.0
      avg_latency_miss_ms = 110.0
      speedup_factor      = 110.0 / 20.0 = 5.5
    """
    traces = (
        [make_trace("CACHE_HIT",  total_ms=20.0)] * 5 +
        [make_trace("CACHE_MISS", total_ms=110.0)] * 5
    )
    reporter = StatsReporter(traces)
    s = reporter.summary()

    assert s["total_queries"]       == 10
    assert s["cache_hits"]          == 5
    assert s["cache_misses"]        == 5
    assert abs(s["hit_rate"] - 0.5) < 1e-9, f"hit_rate should be 0.5, got {s['hit_rate']}"
    assert abs(s["avg_latency_hit_ms"]  - 20.0)  < 1e-9
    assert abs(s["avg_latency_miss_ms"] - 110.0) < 1e-9
    assert abs(s["speedup_factor"] - 5.5) < 1e-9, (
        f"speedup should be 5.5, got {s['speedup_factor']}"
    )
    print(
        f"✓ TEST 4 PASSED: hit_rate={s['hit_rate']:.1f}, "
        f"speedup={s['speedup_factor']:.1f}x, "
        f"hit={s['avg_latency_hit_ms']:.0f}ms, miss={s['avg_latency_miss_ms']:.0f}ms"
    )


# ── TEST 5: P95 latency ───────────────────────────────────────────────────────

def test_p95_latency():
    """
    20 traces with latencies 1, 2, ..., 20 ms.
    numpy percentile(95, method="midpoint") is the ground truth.
    reporter.summary()["p95_latency_ms"] must match exactly.
    """
    latencies = list(range(1, 21))  # [1, 2, ..., 20]
    traces = [make_trace("CACHE_MISS", total_ms=float(ms)) for ms in latencies]

    reporter = StatsReporter(traces)
    s = reporter.summary()

    expected_p50 = float(np.percentile(latencies, 50, method="midpoint"))
    expected_p95 = float(np.percentile(latencies, 95, method="midpoint"))

    assert abs(s["p50_latency_ms"] - expected_p50) < 1e-9, (
        f"P50: expected {expected_p50}, got {s['p50_latency_ms']}"
    )
    assert abs(s["p95_latency_ms"] - expected_p95) < 1e-9, (
        f"P95: expected {expected_p95}, got {s['p95_latency_ms']}"
    )
    print(
        f"✓ TEST 5 PASSED: p50={s['p50_latency_ms']:.1f}ms "
        f"(expected {expected_p50:.1f}ms), "
        f"p95={s['p95_latency_ms']:.1f}ms (expected {expected_p95:.1f}ms)"
    )


# ── TEST 6: print_summary and print_latency_breakdown don't crash ─────────────

def test_print_methods_run_without_error():
    """
    Smoke test: calling print_summary() and print_latency_breakdown() must not raise.
    Uses capsys to suppress terminal output during the test.
    """
    hit_trace = QueryTrace(
        query_id="q_hit",
        query_text="cached query",
        path="CACHE_HIT",
        embed_ms=8.1,
        cache_check_ms=1.3,
        total_ms=9.4,
        cost_usd=0.0,
        cache_similarity=0.92,
        cache_quality_score=0.87,
        result_count=5,
    )
    miss_trace = QueryTrace(
        query_id="q_miss",
        query_text="fresh query",
        path="CACHE_MISS",
        embed_ms=8.3,
        cache_check_ms=1.2,
        search_ms=4.7,
        rerank_ms=98.4,
        total_ms=112.6,
        cost_usd=0.00015,
        result_count=5,
    )

    reporter = StatsReporter([hit_trace, miss_trace])

    reporter.print_summary()           # must not raise
    reporter.print_latency_breakdown() # must not raise

    print("✓ TEST 6 PASSED: print_summary and print_latency_breakdown ran without error")
