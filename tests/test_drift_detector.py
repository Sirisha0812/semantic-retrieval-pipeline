"""
Tests for DriftDetector (Layer 5).

TEST 1: Insufficient data → recommendation="INSUFFICIENT_DATA"
TEST 2: Similar distributions → no drift, JS ≈ 0
TEST 3: Clear drift: cache=0.2, fresh=0.8 → is_drifting=True
TEST 4: Cache BETTER than fresh → high JS but is_drifting=False
TEST 5: Reset clears both deques
TEST 6: Boundary — js_div == threshold → is_drifting=False (strict >)
TEST 7: Realistic demo scenario — phase 1 healthy, phase 2 drifts
"""
import sys
import os
import random

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache.drift_detector import DriftDetector, DriftStatus


# ── fixtures + helpers ────────────────────────────────────────────────────────

@pytest.fixture
def det():
    return DriftDetector(window_size=50, threshold=0.15)


def fill(detector, cache_rewards, fresh_rewards):
    for r in cache_rewards:
        detector.record("cache", r)
    for r in fresh_rewards:
        detector.record("fresh", r)


# ── TEST 1: Insufficient data ─────────────────────────────────────────────────

def test_insufficient_data(det):
    """
    5 samples per stream < _MIN_SAMPLES=10.
    Must return INSUFFICIENT_DATA and is_drifting=False regardless of values.
    """
    fill(det, [0.8] * 5, [0.8] * 5)
    status = det.check_drift()

    assert status.is_drifting is False
    assert status.recommendation == "INSUFFICIENT_DATA"
    assert status.js_divergence == 0.0
    assert status.cache_samples == 5
    assert status.fresh_samples == 5
    print("✓ TEST 1 PASSED: INSUFFICIENT_DATA when < 10 samples per stream")


def test_insufficient_data_one_stream_empty(det):
    """One full stream, one empty → still INSUFFICIENT_DATA."""
    for r in [0.8] * 30:
        det.record("cache", r)
    status = det.check_drift()
    assert status.recommendation == "INSUFFICIENT_DATA"
    assert status.fresh_samples == 0


def test_insufficient_data_asymmetric(det):
    """20 cache, 5 fresh → fresh is below MIN threshold."""
    fill(det, [0.8] * 20, [0.8] * 5)
    assert det.check_drift().recommendation == "INSUFFICIENT_DATA"


# ── TEST 2: No drift — similar distributions ──────────────────────────────────

def test_no_drift_similar_distributions(det):
    """
    Both streams at exactly 0.8 → lo == hi in _js_divergence → returns 0.0.
    is_drifting=False, recommendation="OK".
    """
    fill(det, [0.8] * 30, [0.8] * 30)
    status = det.check_drift()

    assert status.is_drifting is False
    assert status.recommendation == "OK"
    assert status.js_divergence < 0.05, (
        f"Identical distributions → JS should be ~0, got {status.js_divergence:.4f}"
    )
    assert abs(status.cache_mean_reward - 0.8) < 1e-9
    assert abs(status.fresh_mean_reward - 0.8) < 1e-9
    print(f"✓ TEST 2 PASSED: no drift, JS={status.js_divergence:.6f}")


# ── TEST 3: Clear drift detected ──────────────────────────────────────────────

def test_clear_drift_detected(det):
    """
    cache=0.2 (all mass at 0.2), fresh=0.8 (all mass at 0.8).
    Two point masses at opposite ends of the histogram → JS ≈ log(2) ≈ 0.693.
    cache_mean (0.2) < fresh_mean (0.8) → both drift conditions met.
    """
    fill(det, [0.2] * 30, [0.8] * 30)
    status = det.check_drift()

    assert status.is_drifting is True
    assert status.js_divergence > 0.15
    assert status.cache_mean_reward < status.fresh_mean_reward
    assert status.recommendation == "CACHE_DEGRADING_EVICT_LOW_QUALITY"
    assert status.cache_samples == 30
    assert status.fresh_samples == 30
    print(
        f"✓ TEST 3 PASSED: drift detected — "
        f"JS={status.js_divergence:.4f}, "
        f"cache_mean={status.cache_mean_reward:.2f}, "
        f"fresh_mean={status.fresh_mean_reward:.2f}"
    )


# ── TEST 4: High JS but cache BETTER than fresh ───────────────────────────────

def test_no_drift_when_cache_better(det):
    """
    cache=0.9, fresh=0.2. JS is high (distributions far apart),
    but cache_mean (0.9) > fresh_mean (0.2).
    Drift only fires when cache is WORSE. Cache better → no drift.
    """
    fill(det, [0.9] * 30, [0.2] * 30)
    status = det.check_drift()

    assert status.is_drifting is False
    assert status.js_divergence > 0.15, (
        "JS should still be high — distributions ARE far apart"
    )
    assert status.cache_mean_reward > status.fresh_mean_reward
    assert status.recommendation == "OK"
    print(
        f"✓ TEST 4 PASSED: no drift despite JS={status.js_divergence:.4f} — "
        f"cache ({status.cache_mean_reward:.2f}) > fresh ({status.fresh_mean_reward:.2f})"
    )


# ── TEST 5: Reset ─────────────────────────────────────────────────────────────

def test_reset(det):
    """
    After reset(), cache_count=0, fresh_count=0, check_drift()=INSUFFICIENT_DATA.
    Verifies reset() clears both deques completely.
    """
    fill(det, [0.5] * 20, [0.5] * 20)
    assert det.cache_count == 20
    assert det.fresh_count == 20

    det.reset()

    assert det.cache_count == 0
    assert det.fresh_count == 0
    assert det.check_drift().recommendation == "INSUFFICIENT_DATA"
    print("✓ TEST 5 PASSED: reset clears both deques, check_drift → INSUFFICIENT_DATA")


# ── TEST 6: Boundary — exactly at threshold ───────────────────────────────────

def test_boundary_at_threshold():
    """
    Subclass DriftDetector to pin _js_divergence return to exactly 0.15.
    The condition is `js_div > threshold` (strict greater-than).
    0.15 > 0.15 is False → is_drifting must be False.

    Also set cache_mean < fresh_mean to confirm the ONLY reason for no-drift
    is the strict inequality on JS, not the mean comparison.
    """
    class FixedJSDet(DriftDetector):
        def _js_divergence(self, p, q):
            return 0.15  # pinned exactly at threshold

    det = FixedJSDet(threshold=0.15)
    fill(det, [0.2] * 30, [0.8] * 30)  # cache worse → cache_mean < fresh_mean

    status = det.check_drift()

    assert status.js_divergence == 0.15
    assert status.cache_mean_reward < status.fresh_mean_reward  # mean condition IS met
    assert status.is_drifting is False, (
        "js_div == threshold must NOT trigger drift (requires strict js_div > threshold)"
    )
    assert status.recommendation == "OK"
    print(
        f"✓ TEST 6 PASSED: JS=threshold=0.15 → is_drifting=False "
        f"(0.15 > 0.15 is False, strict > required)"
    )


# ── TEST 7: Realistic demo scenario ──────────────────────────────────────────

def test_realistic_demo_scenario():
    """
    window_size=20 so the deque rotation is clean:
      Phase 1 fills the window with healthy data.
      Phase 2 completely replaces it with stale data.

    Phase 1 (20 queries, cache working well):
      cache rewards: uniform(0.7, 0.9) — good results
      fresh rewards: uniform(0.7, 0.9) — also good
      → similar distributions → check_drift() → no drift

    Phase 2 (20 more queries, cache now stale):
      cache rewards: uniform(0.1, 0.3) — stale/wrong cached results
      fresh rewards: uniform(0.7, 0.9) — fresh retrieval still works
      With maxlen=20, all phase-1 data is pushed out.
      cache deque: 20 × [0.1-0.3]   fresh deque: 20 × [0.7-0.9]
      → distributions diverge → drift detected
    """
    random.seed(99)
    det = DriftDetector(window_size=20, threshold=0.15)

    # Phase 1: both healthy
    for _ in range(20):
        det.record("cache", random.uniform(0.7, 0.9))
        det.record("fresh", random.uniform(0.7, 0.9))

    status_1 = det.check_drift()
    assert status_1.is_drifting is False, (
        f"Phase 1 (healthy): should NOT be drifting. JS={status_1.js_divergence:.4f}"
    )

    # Phase 2: cache degrades — deque rotates out all phase-1 data
    for _ in range(20):
        det.record("cache", random.uniform(0.1, 0.3))
        det.record("fresh", random.uniform(0.7, 0.9))

    status_2 = det.check_drift()
    assert status_2.is_drifting is True, (
        f"Phase 2 (stale cache): should BE drifting. "
        f"JS={status_2.js_divergence:.4f}, "
        f"cache_mean={status_2.cache_mean_reward:.2f}, "
        f"fresh_mean={status_2.fresh_mean_reward:.2f}"
    )
    assert status_2.recommendation == "CACHE_DEGRADING_EVICT_LOW_QUALITY"
    assert status_2.cache_mean_reward < status_2.fresh_mean_reward

    print(
        f"✓ TEST 7 PASSED: demo scenario works\n"
        f"  Phase 1: JS={status_1.js_divergence:.4f} → {status_1.recommendation}\n"
        f"  Phase 2: JS={status_2.js_divergence:.4f} → {status_2.recommendation}\n"
        f"  cache_mean: {status_1.cache_mean_reward:.2f} → {status_2.cache_mean_reward:.2f}  "
        f"  fresh_mean: {status_1.fresh_mean_reward:.2f} → {status_2.fresh_mean_reward:.2f}"
    )
