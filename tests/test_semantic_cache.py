"""
Tests for SemanticCache (Layer 2).

TEST 1: Cache miss on empty cache
TEST 2: Store and retrieve exact query
TEST 3: Semantic similarity hit vs miss
TEST 4: Quality score EMA update
TEST 5: Eviction of lowest quality entry
TEST 6: Hit rate tracking
TEST 7: Full pipeline integration with real embedder
"""
import sys
import os

import numpy as np
import pytest 

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache.semantic_cache import CacheEntry, CacheHit, SemanticCache
from retrieval.vector_store import Document


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedder():
    """Load the Embedder once for the whole module — ~2s, not per-test."""
    from retrieval.embedder import Embedder
    return Embedder()


@pytest.fixture
def cache():
    """Fresh SemanticCache for each test (default params)."""
    return SemanticCache(embedding_dim=384, max_size=1000, threshold=0.85)


@pytest.fixture
def sample_results():
    """A small list of Documents to store as cache results."""
    return [
        Document(id="0", text="Employee leave entitlements overview", score=0.91),
        Document(id="1", text="How to apply for parental leave", score=0.87),
        Document(id="2", text="Leave approval workflow HR policy", score=0.83),
    ]


# ── TEST 1: Cache miss on empty cache ─────────────────────────────────────────

def test_miss_on_empty_cache(cache):
    """lookup() on a brand-new empty cache must always return None."""
    dummy_vec = np.random.randn(384).astype(np.float32)
    result = cache.lookup(dummy_vec)

    assert result is None
    assert cache.total_misses == 1
    assert cache.total_hits == 0
    print("✓ TEST 1 PASSED: Cache miss on empty cache")


# ── TEST 2: Store and retrieve exact query ────────────────────────────────────

def test_store_and_retrieve_exact(cache, sample_results):
    """
    After store(), lookup() with the identical embedding must return
    a CacheHit with similarity ~1.0.
    """
    vec = np.random.randn(384).astype(np.float32)
    vec /= np.linalg.norm(vec)

    entry_id = cache.store("test query", vec, sample_results)

    hit = cache.lookup(vec)

    assert hit is not None, "Expected a cache hit for identical embedding"
    assert isinstance(hit, CacheHit)
    assert hit.entry.id == entry_id
    assert abs(hit.similarity - 1.0) < 1e-5, (
        f"Identical vector similarity should be ~1.0, got {hit.similarity:.6f}"
    )
    assert hit.entry.hit_count == 1
    assert hit.entry.results == sample_results
    assert cache.total_hits == 1
    assert cache.total_misses == 0
    print(f"✓ TEST 2 PASSED: Store and retrieve exact (similarity={hit.similarity:.6f})")


# ── TEST 3: Semantic similarity hit vs miss ───────────────────────────────────

def test_semantic_similarity_hit_and_miss(cache, embedder, sample_results):
    """
    MISS:  "employee leave policy" vs "staff vacation rules"
           These are related but NOT near-paraphrases → similarity < 0.85 → MISS

    HIT:   "parental leave policy details" stored, "parental leave policy" queried
           Near-paraphrases → similarity > 0.85 → HIT
    """
    # ── MISS case ──
    vec_miss_stored = embedder.embed("employee leave policy")
    cache.store("employee leave policy", vec_miss_stored, sample_results)

    vec_miss_query = embedder.embed("staff vacation rules")
    miss_result = cache.lookup(vec_miss_query)

    assert miss_result is None, (
        f"Expected MISS for 'staff vacation rules' vs 'employee leave policy', "
        f"but got similarity={cache._index.search(vec_miss_query.reshape(1,-1).astype(np.float32), 1)[0][0][0]:.4f}"
    )

    # ── HIT case ──
    vec_hit_stored = embedder.embed("parental leave policy details")
    cache.store("parental leave policy details", vec_hit_stored, sample_results)

    vec_hit_query = embedder.embed("parental leave policy")
    hit_result = cache.lookup(vec_hit_query)

    assert hit_result is not None, (
        "Expected HIT for 'parental leave policy' vs 'parental leave policy details'"
    )
    assert hit_result.similarity >= 0.85, (
        f"Expected similarity >= 0.85, got {hit_result.similarity:.4f}"
    )
    print(
        f"✓ TEST 3 PASSED: Miss sim<0.85 (unrelated phrases); "
        f"Hit sim={hit_result.similarity:.4f} (near-paraphrase)"
    )


# ── TEST 4: Quality score EMA update ─────────────────────────────────────────

def test_quality_score_ema_update(cache, sample_results):
    """
    After 5 updates with reward=0.0 and alpha=0.3:
      score after k updates = (0.7)^k * 1.0
      After 5: (0.7)^5 = 0.168 — well below 1.0
    """
    vec = np.random.randn(384).astype(np.float32)
    entry_id = cache.store("quality test query", vec, sample_results)

    assert cache._entries[entry_id].quality_score == 1.0

    for _ in range(5):
        cache.update_quality(entry_id, reward=0.0, alpha=0.3)

    expected = (0.7 ** 5) * 1.0  # ~0.168
    actual = cache._entries[entry_id].quality_score

    assert actual < 0.5, (
        f"After 5 zero-reward updates, quality should be < 0.5, got {actual:.4f}"
    )
    assert abs(actual - expected) < 1e-6, (
        f"EMA formula wrong: expected {expected:.6f}, got {actual:.6f}"
    )
    print(f"✓ TEST 4 PASSED: Quality score after 5×reward=0.0 → {actual:.4f} (expected {expected:.4f})")


def test_quality_update_on_missing_id(cache):
    """update_quality() with unknown id should silently do nothing."""
    cache.update_quality("does-not-exist", reward=0.5)  # must not raise


# ── TEST 5: Eviction of lowest quality ───────────────────────────────────────

def test_eviction_lowest_quality(sample_results):
    """
    With max_size=3:
      - Store 3 entries (A, B, C), all quality=1.0
      - Lower C's quality to 0.1 via update_quality
      - Store a 4th entry → eviction fires → C is removed
      - A and B must still be in cache
      - C must be gone
    """
    small_cache = SemanticCache(embedding_dim=384, max_size=3, threshold=0.85)

    vec_a = np.random.randn(384).astype(np.float32)
    vec_b = np.random.randn(384).astype(np.float32)
    vec_c = np.random.randn(384).astype(np.float32)
    vec_d = np.random.randn(384).astype(np.float32)

    id_a = small_cache.store("query A", vec_a, sample_results)
    id_b = small_cache.store("query B", vec_b, sample_results)
    id_c = small_cache.store("query C", vec_c, sample_results)

    assert small_cache.size == 3

    # Drive C's quality down to 0.1
    for _ in range(20):
        small_cache.update_quality(id_c, reward=0.0, alpha=0.3)

    # Verify C has lowest quality before eviction
    q_a = small_cache._entries[id_a].quality_score
    q_b = small_cache._entries[id_b].quality_score
    q_c = small_cache._entries[id_c].quality_score
    assert q_c < q_a and q_c < q_b, (
        f"C should have lowest quality: A={q_a:.4f}, B={q_b:.4f}, C={q_c:.4f}"
    )

    # Storing a 4th entry triggers eviction
    id_d = small_cache.store("query D", vec_d, sample_results)

    assert small_cache.size == 3, f"Cache should still be max_size=3, got {small_cache.size}"
    assert id_c not in small_cache._entries, "Entry C (lowest quality) should have been evicted"
    assert id_a in small_cache._entries, "Entry A should survive eviction"
    assert id_b in small_cache._entries, "Entry B should survive eviction"
    assert id_d in small_cache._entries, "Entry D (new) should be in cache"

    # FAISS index and id_map must also be consistent
    assert small_cache._index.ntotal == 3
    assert len(small_cache._id_map) == 3
    assert id_c not in small_cache._id_map

    print(
        f"✓ TEST 5 PASSED: Entry C (quality={q_c:.4f}) evicted; "
        f"A ({q_a:.4f}), B ({q_b:.4f}), D remain"
    )


# ── TEST 6: Hit rate tracking ─────────────────────────────────────────────────

def test_hit_rate_tracking(cache, sample_results):
    """
    Perform 3 hits and 7 misses → hit_rate should be 0.3 exactly.

    Strategy:
      - Store one entry with a known vector (for hits)
      - Query with the exact vector 3 times → 3 hits
      - Query with 7 random orthogonal-ish vectors → 7 misses
    """
    vec_stored = np.random.randn(384).astype(np.float32)
    vec_stored /= np.linalg.norm(vec_stored)
    cache.store("hit-rate test", vec_stored, sample_results)

    # 3 hits
    for _ in range(3):
        result = cache.lookup(vec_stored)
        assert result is not None, "Expected hit for stored vector"

    # 7 misses — use random vectors unlikely to be close to vec_stored
    for _ in range(7):
        random_vec = np.random.randn(384).astype(np.float32)
        # Make it orthogonal to vec_stored to guarantee low similarity
        random_vec -= random_vec.dot(vec_stored) * vec_stored
        random_vec /= np.linalg.norm(random_vec)
        cache.lookup(random_vec)

    assert cache.total_hits == 3
    assert cache.total_misses == 7
    assert abs(cache.hit_rate - 0.3) < 1e-9, (
        f"Expected hit_rate=0.3, got {cache.hit_rate}"
    )
    print(f"✓ TEST 6 PASSED: hit_rate={cache.hit_rate:.1f} (3 hits / 10 total)")


def test_hit_rate_empty_cache():
    """hit_rate on fresh cache should be 0.0, not division error."""
    c = SemanticCache()
    assert c.hit_rate == 0.0


# ── TEST 7: Full pipeline integration ────────────────────────────────────────

def test_full_pipeline_integration(embedder, sample_results):
    """
    Real Embedder + SemanticCache:

    1. Embed "what is employee leave policy?" → store → expect hit on exact repeat
    2. Embed "employee vacation policy" → lookup → expect MISS (different topic)
    3. Embed "what is employee leave policy?" exactly → lookup → expect HIT ~1.0
    """
    cache = SemanticCache(embedding_dim=384, max_size=1000, threshold=0.85)

    query1 = "what is employee leave policy?"
    vec1 = embedder.embed(query1)
    entry_id = cache.store(query1, vec1, sample_results)

    # ── step 2: different query → MISS ──
    query2 = "employee vacation policy"
    vec2 = embedder.embed(query2)
    miss = cache.lookup(vec2)
    assert miss is None, (
        f"Expected MISS for '{query2}' vs '{query1}'"
    )

    # ── step 3: exact same query → HIT ──
    vec3 = embedder.embed(query1)
    hit = cache.lookup(vec3)
    assert hit is not None, (
        f"Expected HIT for exact repeat of '{query1}'"
    )
    assert hit.entry.id == entry_id
    assert hit.similarity >= 0.85
    assert abs(hit.similarity - 1.0) < 1e-4, (
        f"Same query re-embedded should have similarity ~1.0, got {hit.similarity:.6f}"
    )

    print(
        f"✓ TEST 7 PASSED: Pipeline integration — "
        f"miss for different query; hit={hit.similarity:.6f} for exact repeat"
    )
