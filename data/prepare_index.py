import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore

INDEX_PATH = "data/ms_marco_10k.index"
PASSAGES_PATH = "data/passages.pkl"
N_PASSAGES = 10_000
N_QUERIES = 2_000


def main() -> None:
    print("Step 1/4  Loading MS MARCO v2.1 train split...")
    from datasets import load_dataset
    dataset = load_dataset("ms_marco", "v2.1", split="train").select(range(N_QUERIES))

    print("Step 2/4  Extracting passages...")
    passages = [
        p.strip()
        for row_passages in dataset["passages"]["passage_text"]
        for p in row_passages
        if p and p.strip()
    ][:N_PASSAGES]
    print(f"          Extracted {len(passages):,} passages from {N_QUERIES:,} queries")

    print("Step 3/4  Embedding and indexing (this takes a few minutes)...")
    embedder = Embedder()
    store = VectorStore(embedding_dim=384)
    t0 = time.perf_counter()
    store.add_documents(passages, embedder)
    elapsed = time.perf_counter() - t0

    print("Step 4/4  Saving to disk...")
    store.save(INDEX_PATH, PASSAGES_PATH)

    index_mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)
    passages_mb = os.path.getsize(PASSAGES_PATH) / (1024 * 1024)

    print()
    print("=" * 52)
    print(f"  Documents indexed : {store.size:>10,}")
    print(f"  FAISS index size  : {index_mb:>9.1f} MB")
    print(f"  Passages file     : {passages_mb:>9.1f} MB")
    print(f"  Time taken        : {elapsed:>9.1f} s")
    print(f"  Throughput        : {len(passages) / elapsed:>9.0f} docs/sec")
    print(f"  Index saved to    : {INDEX_PATH}")
    print(f"  Passages saved to : {PASSAGES_PATH}")
    print("=" * 52)
    print()
    print("Done. Load the index later with:")
    print("  VectorStore(index_path='data/ms_marco_10k.index',")
    print("              passages_path='data/passages.pkl')")


if __name__ == "__main__":
    main()
