import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Both vectors are already L2-normalized, so dot product == cosine similarity
    return float(np.dot(a, b))


def test_similar_sentences_high_similarity(embedder):
    # Near-paraphrases: same concept, minor rewording.
    # This is exactly what triggers a cache HIT at threshold=0.85.
    # (Compare: "employee leave" vs "staff vacation" only scores ~0.62 —
    #  related concepts, but different enough vocabulary that the model
    #  correctly treats them as distinct. That's a cache MISS, as intended.)
    v1 = embedder.embed("what is the parental leave policy")
    v2 = embedder.embed("parental leave policy details")
    sim = cosine_sim(v1, v2)
    assert sim > 0.80, f"Expected near-paraphrases to score > 0.80, got {sim:.4f}"


def test_unrelated_sentences_low_similarity(embedder):
    v1 = embedder.embed("quarterly revenue growth exceeded expectations")
    v2 = embedder.embed("how to bake chocolate cake at home")
    sim = cosine_sim(v1, v2)
    assert sim < 0.3, f"Expected sim < 0.3, got {sim:.4f}"


def test_output_dimension(embedder):
    v = embedder.embed("any sentence")
    assert v.shape == (384,), f"Expected shape (384,), got {v.shape}"


def test_vector_is_normalized(embedder):
    v = embedder.embed("normalization check")
    length = float(np.linalg.norm(v))
    assert abs(length - 1.0) < 1e-6, f"Expected L2 norm = 1.0, got {length:.8f}"


def test_embed_count_increments(embedder):
    before = embedder.embed_count
    embedder.embed("count test one")
    embedder.embed("count test two")
    assert embedder.embed_count == before + 2


def test_model_name_stored(embedder):
    assert embedder.model_name == "all-MiniLM-L6-v2"
