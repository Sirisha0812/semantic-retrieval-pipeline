"""
Tests for ReRanker class.

TEST 1: Returns exactly top_k documents
TEST 2: Output is sorted by score descending
TEST 3: Reranker actually changes order (Paris example)
TEST 4: Real data integration test with VectorStore
"""
import os

import pytest

from retrieval.embedder import Embedder
from retrieval.reranker import ReRanker
from retrieval.vector_store import Document, VectorStore


# Load the model ONCE for the entire test module (7s instead of 35s)
@pytest.fixture(scope="module")
def reranker():
    return ReRanker()


def test_returns_exactly_top_k(reranker):
    """TEST 1: Verify rerank returns exactly top_k documents."""
    docs = [
        Document(id=str(i), text=f"Document {i}", score=0.5)
        for i in range(10)
    ]

    result = reranker.rerank("test query", docs, top_k=3)

    assert len(result) == 3, f"Expected 3 documents, got {len(result)}"
    print("✓ TEST 1 PASSED: Returns exactly top_k documents")


def test_sorted_by_score_descending(reranker):
    """TEST 2: Verify output is sorted by score descending."""
    docs = [
        Document(id="1", text="Random text A", score=0.5),
        Document(id="2", text="Random text B", score=0.5),
        Document(id="3", text="Random text C", score=0.5),
        Document(id="4", text="Random text D", score=0.5),
        Document(id="5", text="Random text E", score=0.5),
    ]

    result = reranker.rerank("query", docs, top_k=5)

    for i in range(len(result) - 1):
        assert result[i].score >= result[i + 1].score, (
            f"Not sorted: result[{i}].score={result[i].score} < "
            f"result[{i+1}].score={result[i+1].score}"
        )

    print("✓ TEST 2 PASSED: Output sorted by score descending")


def test_reranker_changes_order(reranker):
    """
    TEST 3: Verify reranker changes order to put correct answer first.

    FAISS might put the correct answer last due to poor vector similarity.
    CrossEncoder should recognize semantic relevance and move it to first.
    """
    query = "what is the capital of France?"

    docs = [
        Document("1", "Paris is in Europe", 0.9),
        Document("2", "France has good wine", 0.8),
        Document("3", "The weather in Paris", 0.7),
        Document("4", "French cuisine is famous", 0.6),
        Document("5", "Paris is the capital of France", 0.1),  # CORRECT but last
    ]

    assert docs[-1].id == "5", "Setup error: doc 5 should be last"

    result = reranker.rerank(query, docs, top_k=5)

    assert result[0].id == "5", (
        f"Expected doc 5 to be first after reranking, got doc {result[0].id}\n"
        f"Top 3 results:\n"
        f"  1. [{result[0].id}] {result[0].text[:50]} (score={result[0].score:.4f})\n"
        f"  2. [{result[1].id}] {result[1].text[:50]} (score={result[1].score:.4f})\n"
        f"  3. [{result[2].id}] {result[2].text[:50]} (score={result[2].score:.4f})"
    )

    assert result[0].score == max(doc.score for doc in result), (
        "First document doesn't have highest score"
    )

    print("✓ TEST 3 PASSED: Reranker changed order (correct answer now first)")
    print(f"  Before: doc 5 was last with score {docs[-1].score}")
    print(f"  After:  doc 5 is first with score {result[0].score:.4f}")


def test_real_data_integration(reranker):
    """
    TEST 4: Real data integration test with VectorStore.

    Flow: Query → VectorStore (50 docs) → ReRanker (top 5)
    Verify that reranking changes at least 2 positions.
    """
    index_path = "data/ms_marco_10k.index"
    passages_path = "data/passages.pkl"

    if not os.path.exists(index_path) or not os.path.exists(passages_path):
        pytest.skip("data/ms_marco_10k files not found — run prepare_index.py first")

    embedder = Embedder()
    vector_store = VectorStore(
        embedding_dim=384,
        index_path=index_path,
        passages_path=passages_path,
    )

    query = "what are employee health benefits?"

    # Step 1: embed
    query_vector = embedder.embed(query)

    # Step 2: vector search → top 50
    vector_results = vector_store.search(query_vector, k=50)
    assert len(vector_results) > 0, "No results from vector search"

    # Step 3: rerank → top 5
    reranked_results = reranker.rerank(query, vector_results, top_k=5)
    assert len(reranked_results) == 5, f"Expected 5 results, got {len(reranked_results)}"

    print("\n" + "=" * 80)
    print("BEFORE RERANKING (Top 5 from FAISS vector search):")
    print("=" * 80)
    for i, doc in enumerate(vector_results[:5]):
        print(f"{i+1}. [ID:{doc.id}] (FAISS score={doc.score:.4f})")
        print(f"   {doc.text[:100]}...")
        print()

    print("=" * 80)
    print("AFTER RERANKING (Top 5 from CrossEncoder):")
    print("=" * 80)
    for i, doc in enumerate(reranked_results):
        print(f"{i+1}. [ID:{doc.id}] (CrossEncoder score={doc.score:.4f})")
        print(f"   {doc.text[:100]}...")
        print()

    vector_top5_ids = [doc.id for doc in vector_results[:5]]
    reranked_top5_ids = [doc.id for doc in reranked_results]

    positions_changed = sum(
        1 for i in range(5)
        if vector_top5_ids[i] != reranked_top5_ids[i]
    )

    print("=" * 80)
    print(f"POSITIONS CHANGED: {positions_changed} out of 5")
    print(f"Vector order:   {vector_top5_ids}")
    print(f"Reranked order: {reranked_top5_ids}")
    print("=" * 80)

    assert positions_changed >= 2, (
        f"Expected at least 2 position changes, got {positions_changed}"
    )

    print(f"✓ TEST 4 PASSED: Real data integration successful")
    print(f"  Reranker changed {positions_changed} positions")


def test_counters(reranker):
    """Verify rerank_count and total_pairs_scored tracking.

    Uses deltas because the fixture is shared — counters accumulate
    across all tests in this module.
    """
    count_before = reranker.rerank_count
    pairs_before = reranker.total_pairs_scored

    # Call with 10 docs
    docs1 = [Document(id=str(i), text=f"doc {i}", score=0.5) for i in range(10)]
    reranker.rerank("query1", docs1, top_k=3)

    assert reranker.rerank_count == count_before + 1
    assert reranker.total_pairs_scored == pairs_before + 10

    # Call with 20 docs
    docs2 = [Document(id=str(i), text=f"doc {i}", score=0.5) for i in range(20)]
    reranker.rerank("query2", docs2, top_k=5)

    assert reranker.rerank_count == count_before + 2
    assert reranker.total_pairs_scored == pairs_before + 30  # 10 + 20

    # Call with 15 docs
    docs3 = [Document(id=str(i), text=f"doc {i}", score=0.5) for i in range(15)]
    reranker.rerank("query3", docs3, top_k=10)

    assert reranker.rerank_count == count_before + 3
    assert reranker.total_pairs_scored == pairs_before + 45  # 10 + 20 + 15

    print("✓ TEST COUNTERS PASSED: rerank_count and total_pairs_scored work correctly")
