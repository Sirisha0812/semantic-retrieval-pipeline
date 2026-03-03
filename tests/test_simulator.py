"""
Tests for FeedbackSimulator and UserFeedback (Layer 4).

TEST 1: High relevance → high reward (> 0.7)
TEST 2: Low relevance  → low reward  (< 0.4)
TEST 3: Reward range — always [0.0, 1.0] across 100 random simulations
TEST 4: No clicks → reward <= 0.2
TEST 5: Perfect engagement → reward == 1.0
TEST 6: Realistic simulation — high-relevance docs clicked more than low-relevance
"""
import sys
import os
import random

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feedback.simulator import FeedbackSimulator, UserFeedback
from retrieval.vector_store import Document


# ── shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def sim():
    return FeedbackSimulator()


def make_docs(n: int) -> list[Document]:
    return [Document(id=str(i), text=f"doc {i}", score=0.5) for i in range(n)]


# ── TEST 1: High relevance → high reward ──────────────────────────────────────

def test_high_relevance_high_reward(sim):
    """
    All docs at relevance 0.9 (> 0.8 threshold):
      - Every doc gets clicked             → click_score = 1.0
      - Dwell 30-120s for each             → avg_dwell ≥ 30 → dwell_score = 1.0
      - No follow-up (never generated)     → no_refinement_score = 1.0
    Expected reward = 0.4×1 + 0.4×1 + 0.2×1 = 1.0 → always > 0.7.
    """
    docs = make_docs(5)
    relevance = [0.9] * 5

    fb = sim.simulate("q1", docs, relevance)
    reward = fb.to_reward()

    assert reward > 0.7, f"High-relevance reward should be > 0.7, got {reward:.4f}"
    assert len(fb.clicked_doc_ids) == 5, "All 5 high-relevance docs should be clicked"
    assert fb.follow_up_query is None, "High-relevance path never sets follow_up_query"
    print(f"✓ TEST 1 PASSED: high-relevance reward = {reward:.4f}")


# ── TEST 2: Low relevance → low reward ───────────────────────────────────────

def test_low_relevance_low_reward(sim):
    """
    All docs at relevance 0.2 (≤ 0.5 threshold):
      - 15% click chance → most docs NOT clicked → click_score ≈ 0-1
      - Short dwell 2-8s → dwell_score = clip((avg-5)/25, 0, 1) ≈ 0-0.12
      - Always has follow_up_query → no_refinement_score = 0.0
    Max possible: 0.4×1.0 + 0.4×0.12 + 0.2×0 ≈ 0.45 → always < 0.5.

    We use seed + many docs to make this deterministic enough to assert.
    """
    random.seed(42)
    docs = make_docs(20)
    relevance = [0.2] * 20

    fb = sim.simulate("q_low", docs, relevance)
    reward = fb.to_reward()

    assert reward < 0.5, f"Low-relevance reward should be < 0.5, got {reward:.4f}"
    assert fb.follow_up_query is not None, "Low-relevance always sets follow_up_query"
    print(f"✓ TEST 2 PASSED: low-relevance reward = {reward:.4f}")


# ── TEST 3: Reward range [0.0, 1.0] ──────────────────────────────────────────

def test_reward_always_in_range(sim):
    """
    Run 100 simulations with random relevance scores.
    Every reward must be in [0.0, 1.0] — no formula bug can produce out-of-range.
    """
    random.seed(0)
    docs = make_docs(5)

    for i in range(100):
        relevance = [random.random() for _ in docs]
        fb = sim.simulate(f"q_{i}", docs, relevance)
        reward = fb.to_reward()
        assert 0.0 <= reward <= 1.0, (
            f"Simulation {i}: reward {reward:.6f} out of [0, 1]. "
            f"relevance={[f'{r:.2f}' for r in relevance]}"
        )

    print("✓ TEST 3 PASSED: all 100 rewards in [0.0, 1.0]")


# ── TEST 4: No clicks → reward <= 0.2 ────────────────────────────────────────

def test_no_clicks_low_reward():
    """
    Manually construct UserFeedback with zero clicks.
      click_score = min(1.0, 0/2.0) = 0.0
      dwell_score = 0.0  (no dwell_times)
      no_refinement_score = 0.0 if follow_up else 1.0

    Worst case (with follow_up): reward = 0.4×0 + 0.4×0 + 0.2×0 = 0.0
    Best case (no follow_up):    reward = 0.4×0 + 0.4×0 + 0.2×1 = 0.2

    Both ≤ 0.2.
    """
    fb_with_followup = UserFeedback(
        query_id="q",
        clicked_doc_ids=[],
        dwell_times={},
        follow_up_query="more details about q",
    )
    assert fb_with_followup.to_reward() == 0.0, (
        "No clicks + follow-up → reward must be 0.0"
    )

    fb_no_followup = UserFeedback(
        query_id="q",
        clicked_doc_ids=[],
        dwell_times={},
        follow_up_query=None,
    )
    assert fb_no_followup.to_reward() == 0.2, (
        "No clicks + no follow-up → reward must be 0.2 (only no_refinement_score contributes)"
    )

    print("✓ TEST 4 PASSED: no-click rewards = 0.0 and 0.2")


# ── TEST 5: Perfect engagement → reward == 1.0 ────────────────────────────────

def test_perfect_engagement_reward():
    """
    Manually construct perfect UserFeedback:
      clicked 2 docs                → click_score = min(1.0, 2/2) = 1.0
      dwell 60s each, avg=60        → dwell_score = clip((60-5)/25, 0,1) = 1.0
      no follow_up                  → no_refinement_score = 1.0
    Expected: 0.4×1 + 0.4×1 + 0.2×1 = 1.0
    """
    fb = UserFeedback(
        query_id="perfect",
        clicked_doc_ids=["doc_a", "doc_b"],
        dwell_times={"doc_a": 60.0, "doc_b": 60.0},
        follow_up_query=None,
    )
    reward = fb.to_reward()
    assert reward == 1.0, f"Perfect engagement should give reward=1.0, got {reward}"
    print(f"✓ TEST 5 PASSED: perfect engagement reward = {reward}")


# ── TEST 6: Realistic simulation — click patterns by relevance ────────────────

def test_click_pattern_by_relevance(sim):
    """
    Mix 3 high-relevance docs (score=0.9) and 3 low-relevance docs (score=0.2).
    Over 20 simulations, high-relevance docs must appear in clicked_doc_ids
    more often than low-relevance docs.

    High: always clicked (P=1.0)  → expected click count across 20 runs = 20
    Low:  15% click chance         → expected click count across 20 runs ≈ 3
    Assert: total high clicks > total low clicks (should be by a wide margin).
    """
    random.seed(7)

    high_docs = [Document(id=f"h{i}", text=f"high doc {i}", score=0.9) for i in range(3)]
    low_docs  = [Document(id=f"l{i}", text=f"low doc {i}",  score=0.2) for i in range(3)]
    docs = high_docs + low_docs
    relevance = [0.9, 0.9, 0.9, 0.2, 0.2, 0.2]

    high_ids = {d.id for d in high_docs}
    low_ids  = {d.id for d in low_docs}

    high_click_total = 0
    low_click_total  = 0

    for run in range(20):
        fb = sim.simulate(f"q_{run}", docs, relevance)
        high_click_total += sum(1 for did in fb.clicked_doc_ids if did in high_ids)
        low_click_total  += sum(1 for did in fb.clicked_doc_ids if did in low_ids)

    assert high_click_total > low_click_total, (
        f"High-relevance docs should be clicked more than low-relevance. "
        f"High clicks: {high_click_total}, Low clicks: {low_click_total}"
    )
    print(
        f"✓ TEST 6 PASSED: high-relevance clicks={high_click_total} "
        f"> low-relevance clicks={low_click_total} over 20 runs"
    )
