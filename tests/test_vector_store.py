import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.embedder import Embedder
from retrieval.vector_store import Document, VectorStore

# ---------------------------------------------------------------------------
# Test data: 20 HR passages + 20 unrelated passages = 40 total
# HR passages are the "signal". Unrelated are the "noise".
# This lets us verify semantic search actually works.
# ---------------------------------------------------------------------------

HR_PASSAGES = [
    "Employees receive 15 days of paid annual leave per calendar year.",
    "Full-time employees are entitled to 20 vacation days each year.",
    "Parental leave: 12 weeks of paid time off for primary caregivers.",
    "Sick leave policy: up to 10 days per year for illness or injury.",
    "PTO accrues at a rate of 1.25 days per month for eligible employees.",
    "Vacation carry-over: up to 5 unused days may roll over to the next year.",
    "Bereavement leave: 3 paid days off for immediate family members.",
    "Personal days: 3 personal days available per year for any reason.",
    "Leave of absence: employees may request unpaid leave for up to 6 months.",
    "Holiday schedule: 10 federal holidays are observed as paid days off.",
    "Jury duty: employees are entitled to paid leave during jury service.",
    "Military leave: up to 2 weeks of paid leave for reserve duty.",
    "Maternity leave: 16 weeks paid for birth mothers under company policy.",
    "Paternity leave: 6 weeks of paid leave for non-birthing parents.",
    "FMLA: eligible employees may take up to 12 weeks of unpaid family leave.",
    "Sabbatical: employees with 7+ years may apply for a 3-month paid break.",
    "Annual leave requests must be submitted at least 2 weeks in advance.",
    "Time off in lieu: overtime can be banked as additional vacation days.",
    "Leave balances are visible in the HR portal under employee benefits.",
    "Unlimited PTO: some roles qualify for flexible time off arrangements.",
]

UNRELATED_PASSAGES = [
    "Python is a high-level programming language known for readability.",
    "Quarterly earnings exceeded analyst expectations by twelve percent.",
    "Machine learning models require large datasets for effective training.",
    "Server maintenance is scheduled for Saturday 2am to 6am UTC.",
    "The new product launch is planned for Q3 of the current fiscal year.",
    "Customer support tickets should be submitted through the help portal.",
    "Database query performance improves significantly with proper indexing.",
    "Cloud infrastructure costs increased by fifteen percent last quarter.",
    "The marketing team will present the new campaign strategy on Monday.",
    "Version 2.0 includes several bug fixes and performance improvements.",
    "Network connectivity issues have been resolved by the IT department.",
    "The annual budget review meeting is scheduled for next quarter.",
    "All employees must complete mandatory security awareness training.",
    "The cafeteria renovation project will be completed by end of month.",
    "The board approved the strategic partnership agreement yesterday.",
    "Employee ID badges must be worn at all times within the building.",
    "The company retreat is scheduled at the mountain resort in July.",
    "Kubernetes orchestrates containerized applications across clusters.",
    "The merger with the acquisition target has been approved by regulators.",
    "The new office will have open-plan seating for cross-team collaboration.",
]

ALL_PASSAGES = HR_PASSAGES + UNRELATED_PASSAGES  # 40 total


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture(scope="module")
def store(embedder):
    vs = VectorStore(embedding_dim=384)
    vs.add_documents(ALL_PASSAGES, embedder)
    return vs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_search_returns_exactly_k(store, embedder):
    query_vec = embedder.embed("employee benefits")
    results = store.search(query_vec, k=10)
    assert len(results) == 10, f"Expected 10 results, got {len(results)}"


def test_search_k_larger_than_index_returns_all(store, embedder):
    # Asking for more results than the index has should return everything
    query_vec = embedder.embed("test query")
    results = store.search(query_vec, k=9999)
    assert len(results) == store.size


def test_scores_between_0_and_1(store, embedder):
    query_vec = embedder.embed("employee leave policy")
    results = store.search(query_vec, k=20)
    for doc in results:
        assert 0.0 <= doc.score <= 1.0, (
            f"Score {doc.score:.4f} out of [0, 1] for: {doc.text[:60]}"
        )


def test_save_load_roundtrip(store, embedder, tmp_path):
    index_path = str(tmp_path / "test.index")
    passages_path = str(tmp_path / "test_passages.pkl")

    store.save(index_path, passages_path)

    loaded = VectorStore(
        embedding_dim=384,
        index_path=index_path,
        passages_path=passages_path,
    )

    # Same number of documents
    assert loaded.size == store.size

    # Same search results for the same query
    query_vec = embedder.embed("parental leave policy")
    original = store.search(query_vec, k=5)
    reloaded = loaded.search(query_vec, k=5)

    assert len(original) == len(reloaded)
    for orig, load in zip(original, reloaded):
        assert orig.id == load.id
        assert orig.text == load.text
        assert abs(orig.score - load.score) < 1e-5, (
            f"Score mismatch after reload: {orig.score:.6f} vs {load.score:.6f}"
        )


def test_relevant_documents_returned(store, embedder):
    query_vec = embedder.embed("employee leave policy")
    results = store.search(query_vec, k=10)

    leave_keywords = {"leave", "vacation", "pto", "time off", "holiday", "days off",
                      "parental", "sick", "bereavement", "fmla", "sabbatical"}
    relevant_count = sum(
        1 for doc in results
        if any(kw in doc.text.lower() for kw in leave_keywords)
    )

    assert relevant_count >= 3, (
        f"Expected >= 3 HR-relevant results in top 10, got {relevant_count}.\n"
        "Top 10 results:\n" +
        "\n".join(f"  [{doc.score:.3f}] {doc.text[:80]}" for doc in results)
    )


def test_document_fields_populated(store, embedder):
    query_vec = embedder.embed("company policy")
    results = store.search(query_vec, k=5)
    for doc in results:
        assert isinstance(doc, Document)
        assert isinstance(doc.id, str) and len(doc.id) > 0
        assert isinstance(doc.text, str) and len(doc.text) > 0
        assert isinstance(doc.score, float)


def test_size_property(store):
    assert store.size == len(ALL_PASSAGES)
