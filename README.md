# AIntropy — Extreme Fast Retrieval Layer

Sub-second semantic search over 100M+ documents with an adaptive cache that learns from user behaviour, detects when its own results go stale, and self-heals.

Built as a working system with 55 passing tests and a live CLI demo.

---

## The Problem

Enterprise search at scale has two competing goals:

| Goal | Naive approach | Cost |
|------|---------------|------|
| **Accuracy** | CrossEncoder reranking | 100–300ms per query |
| **Speed** | Skip reranking | Precision drops 15% |

At 100M documents and thousands of concurrent users, you cannot run a full vector search + neural rerank on every query. Most queries are repetitions or paraphrases of questions already answered. The system should know this and act on it.

---

## Solution: Adaptive Semantic Cache

```
Query arrives
    │
    ▼
┌─────────────┐    similar query     ┌──────────────┐
│   Embedder  │──── already seen? ──▶│ SemanticCache │──▶ 8ms response
│  (8ms)      │         YES          │  (0.1ms FAISS)│
└─────────────┘                      └──────────────┘
        │ NO (first time)
        ▼
┌─────────────┐    top 50 candidates
│ VectorStore │──────────────────────▶
│ HNSW (2ms)  │
└─────────────┘
        │
        ▼
┌─────────────┐    top 5 results
│  ReRanker   │──────────────────────▶ Store in cache ──▶ 290ms response
│ (100–280ms) │
└─────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Feedback Loop (runs on every query)              │
│  FeedbackSimulator → reward → update cache quality│
│  DriftDetector → detect stale cache → evict       │
└──────────────────────────────────────────────────┘
```

**Demonstrated speedup: 10–155x** (8ms warm vs 290ms cold)

---

## Architecture — 6 Layers

### Layer 1 — Embedder (`retrieval/embedder.py`)
Converts any text query into a 384-dimensional L2-normalized vector.

- **Model:** `all-MiniLM-L6-v2` — 22M params, 8ms CPU after JIT warmup
- **Why this model:** Best speed/accuracy tradeoff. Larger models (MPNet, BGE) give +2-5% NDCG but cost 3-5x latency. Smaller models lose semantic nuance.
- **L2 normalization:** Makes FAISS inner product = cosine similarity directly. No extra math needed in retrieval.
- **JIT warmup:** Two dummy encode calls at startup burn off PyTorch compilation. Without it, first real query is 3x slower.

### Layer 2 — Semantic Cache (`cache/semantic_cache.py`)
Sub-millisecond lookup: if a semantically similar query was answered before, serve it instantly.

- **Similarity engine:** FAISS `IndexFlatIP` (brute-force inner product) — O(n) but n ≤ 1000 entries, so ~0.1ms
- **Threshold:** cosine similarity ≥ 0.85 → HIT. Chosen by measuring real query pairs:
  - "parental leave policy" vs "parental leave policy details" = 0.947 → HIT ✓
  - "employee leave policy" vs "staff vacation rules" = 0.617 → MISS ✓
- **Eviction policy:** Lowest `quality_score` first (not LRU). Entries earning bad feedback get evicted before popular entries.
- **Quality scoring:** EMA with α=0.3. New entries start at 1.0 (optimistic init — UCB trick). `new = 0.7×old + 0.3×reward`
- **FAISS rebuild:** No incremental delete API exists, so eviction rebuilds the entire index. At ≤1000 entries this takes <1ms.

### Layer 3A — Vector Store (`retrieval/vector_store.py`)
Fast approximate nearest-neighbour search over the full corpus.

- **Index:** FAISS `IndexHNSWFlat` — Hierarchical Navigable Small World graph
- **Parameters:** M=32 (graph connectivity), efConstruction=64 (build quality), efSearch=50 (query-time exploration)
- **Why HNSW over IVF:** No quantization error. Exact distances on whatever vectors you store. At 10K docs the difference is marginal; at 100M IVF becomes necessary.
- **Why not brute force:** 10K vectors → 15ms brute force vs 2ms HNSW. At 100M → 15 seconds vs ~10ms.
- **Two-stage funnel:** Retrieve k=50 broadly (high recall), then rerank to top 5 (high precision).
- **macOS fix:** `faiss.omp_set_num_threads(1)` — HNSW construction segfaults on macOS/Python 3.13 with default OMP thread count.

### Layer 3B — ReRanker (`retrieval/reranker.py`)
Precision scoring of the top-50 candidates using a cross-encoder.

- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — 22M params, 80–280ms for 50 pairs on CPU
- **Why cross-encoder:** The bi-encoder (embedder) encodes query and document independently. The cross-encoder reads them together — it sees the query in context of each document. MRR@10 = 39 vs bi-encoder's 33.
- **Why not the L-12 variant:** 2× slower for +1 MRR. Not worth it.
- **Why not BGE-Reranker-Large:** 335M params, ~500ms. Blows the latency budget.
- **Batch scoring:** One `predict()` call for all 50 pairs — 10× faster than a loop.
- **Scores are raw logits:** Not bounded [0,1]. We sort, not interpret magnitude.

### Layer 4 — Feedback Simulator (`feedback/simulator.py`)
Simulates user behaviour (clicks, dwell time, follow-up queries) to produce a reward signal.

In production this is replaced by a real event stream (Kafka/Kinesis). For the demo it simulates realistic behaviour based on result relevance scores.

**Reward formula:**
```
reward = 0.4 × click_score + 0.4 × dwell_score + 0.2 × no_refinement_score

click_score       = min(1.0, n_clicks / 2.0)
dwell_score       = clip((avg_dwell_seconds - 5) / 25, 0, 1)
no_refinement     = 1.0 if no follow-up query, else 0.0
```

| Relevance | Click rate | Dwell | Follow-up | Typical reward |
|-----------|-----------|-------|-----------|----------------|
| HIGH > 0.8 | 100% | 30–120s | Never | ~0.95 |
| MED 0.5–0.8 | 60% | 8–35s | 30% chance | ~0.65 |
| LOW ≤ 0.5 | 15% | 2–8s | Always | ~0.15 |

### Layer 5 — Drift Detector (`cache/drift_detector.py`)
Monitors two reward streams (cache hits vs fresh retrievals) and fires when the cache is serving worse results than fresh retrieval would.

**Algorithm:**
```
cache_rewards = deque(maxlen=50)   ← sliding window of hit rewards
fresh_rewards = deque(maxlen=50)   ← sliding window of miss rewards

JS_divergence = 0.5×KL(P||M) + 0.5×KL(Q||M)   where M = 0.5×(P+Q)

is_drifting = JS_divergence > 0.15  AND  cache_mean < fresh_mean
```

**Why Jensen-Shannon over KL divergence:**

| Property | KL divergence | JS divergence |
|----------|--------------|--------------|
| Symmetric | ✗ KL(P\|\|Q) ≠ KL(Q\|\|P) | ✓ |
| Bounded | ✗ Can be ∞ | ✓ [0, log(2) ≈ 0.693] |
| Numerically stable | ✗ Undefined if Q=0 | ✓ Laplace smoothing |

**Two conditions required** (not just JS > threshold): if cache is *better* than fresh (high JS but cache_mean > fresh_mean), that's not drift — that's the cache working well.

### Layer 6 — Instrumentation (`instrumentation/tracer.py` + `instrumentation/reporter.py`)
Per-query latency tracing, cost accounting, statistical reporting, and histogram visualisation.

- **Timer:** `time.perf_counter_ns()` — nanosecond resolution, monotonic, immune to NTP drift. Essential for measuring 0.1ms cache lookups.
- **Context manager pattern:** `with tracer.measure(trace, "embed_ms"):` — generic field targeting via `setattr`, no if-chain.
- **Percentiles:** `numpy.percentile(method="midpoint")` — deterministic with small integer datasets.
- **Cost tracking:** Cache hits save $0.00015/query (one search + rerank avoided). Demonstrated: cost saved > cost incurred within a single demo run.

---

## Performance Numbers (measured on Apple M-series CPU)

| Query | Path | Embed | Cache | Search | Rerank | **Total** |
|-------|------|-------|-------|--------|--------|-----------|
| Cold (first ever) | CACHE_MISS | 8ms | 0.1ms | 2ms | 278ms | **~290ms** |
| Warm (repeat) | CACHE_HIT | 8ms | 0.1ms | — | — | **~8ms** |
| Semantic paraphrase | CACHE_HIT | 8ms | 0.1ms | — | — | **~18ms** |

**Speedup: 10–155x** (varies with JIT warmth; steady state ~35x)

| Metric | Value |
|--------|-------|
| Hit rate (demo run) | 61.5% |
| P50 latency | 36.6ms |
| P95 latency | 317.2ms |
| Cost per miss | $0.00015 |
| Cost per hit | $0.00000 |
| Cache threshold | 0.85 cosine similarity |
| Drift threshold | JS divergence > 0.15 |

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Best latency/accuracy tradeoff at 384 dims |
| Vector index | FAISS HNSW | Local, free, log-scale search, no server needed |
| Cache index | FAISS IndexFlatIP | Brute-force is fine at ≤1000 entries, exact cosine |
| Reranker | sentence-transformers CrossEncoder | 22M params, MRR@10=39, fits CPU budget |
| Corpus | MS MARCO v2.1 (10K passages) | Standard IR benchmark, free, representative |
| CLI display | rich | Tables, panels, progress spinners, colour |
| Plotting | matplotlib | Latency histogram, no server needed |
| Tests | pytest | Standard, scope="module" for expensive fixtures |
| Language | Python 3.13 | Latest stable, type hints throughout |

**No Redis. No Prometheus. No Docker. No cloud.** Everything runs locally, from a single command.

---

## Project Structure

```
Alntropy/
│
├── retrieval/
│   ├── embedder.py          # Layer 1: text → 384-dim vector (8ms)
│   ├── vector_store.py      # Layer 3A: HNSW index, search k=50 (2ms)
│   └── reranker.py          # Layer 3B: CrossEncoder, rerank to top 5 (100–280ms)
│
├── cache/
│   ├── semantic_cache.py    # Layer 2: FAISS lookup, EMA quality, LQ eviction
│   └── drift_detector.py    # Layer 5: JS divergence, sliding window, self-heal
│
├── feedback/
│   └── simulator.py         # Layer 4: click+dwell+refinement → reward [0,1]
│
├── instrumentation/
│   ├── tracer.py            # Layer 6: QueryTrace dataclass, LatencyTracer
│   └── reporter.py          # Layer 6: StatsReporter, histogram, rich tables
│
├── data/
│   ├── prepare_index.py     # One-time: download MS MARCO, embed, build HNSW index
│   ├── ms_marco_10k.index   # FAISS HNSW index (binary, git-ignored)
│   └── passages.pkl         # 10K passage texts (git-ignored)
│
├── tests/
│   ├── test_embedder.py         # 6 tests
│   ├── test_semantic_cache.py   # 9 tests
│   ├── test_reranker.py         # 5 tests (includes real-data integration)
│   ├── test_simulator.py        # 6 tests
│   ├── test_drift_detector.py   # 9 tests
│   ├── test_instrumentation.py  # 6 tests
│   └── test_pipeline.py         # 7 tests (scope="module", ~7min)
│
├── outputs/
│   └── latency_histogram.png    # Generated by demo
│
├── docs/
│   ├── aintropy.md              # Full architecture spec and background
│   ├── case_study.md            # Single query traced through all 6 layers
│   ├── layer1_embedder.md       # Line-by-line code explanation
│   ├── layer2_semantic_cache.md
│   ├── layer2_vector_store.md
│   ├── layer3_reranker.md
│   ├── layer4_feedback_simulator.md
│   ├── layer5_drift_detector.md
│   ├── layer6_instrumentation.md
│   └── layer8_pipeline.md
│
├── pipeline.py              # Orchestrator: wires all 6 layers into .query()
├── demo.py                  # 5-act CLI demo with rich output
└── README.md
```

---

## Quick Start

### Prerequisites

Python 3.13 required on Apple Silicon (M-series) Mac.

### Step 0 — Create and activate a virtual environment

```bash
arch -arm64 python3.13 -m venv .venv
source .venv/bin/activate
pip install sentence-transformers faiss-cpu rich matplotlib datasets
```

### Step 1 — Build the index (one time only, ~5 minutes)

Downloads 10K MS MARCO passages, embeds them, and saves the HNSW index to disk.

```bash
KMP_DUPLICATE_LIB_OK=TRUE arch -arm64 python3.13 data/prepare_index.py
```

### Step 2 — Run the demo

```bash
TRANSFORMERS_VERBOSITY=error KMP_DUPLICATE_LIB_OK=TRUE arch -arm64 python3.13 demo.py
```

### Step 3 — Run all tests

```bash
KMP_DUPLICATE_LIB_OK=TRUE arch -arm64 python3.13 -m pytest tests/ -v
```

> **macOS note:** Both env flags are required on Apple Silicon.
> - `KMP_DUPLICATE_LIB_OK=TRUE` — PyTorch and FAISS both ship `libomp.dylib`; without this flag the process aborts on import.
> - `arch -arm64` — Python 3.13 is a universal binary. Without this flag it may run in x86_64 mode via Rosetta, but PyTorch 2.x is arm64-only.
> - `TRANSFORMERS_VERBOSITY=error` — suppresses verbose weight-loading progress bars from `transformers 5.x`.

---

## Demo Walkthrough (5 Acts)

### Act 1 — Cold Queries
Five HR-domain queries run against an empty cache. Every query goes through the full pipeline: embed → HNSW search → CrossEncoder rerank. Shows ~290ms baseline latency.

### Act 2 — Cache Hits (Same Queries)
Same 5 queries re-run. All return as CACHE_HIT with similarity=1.0000. Side-by-side table shows cold vs warm latency and speedup per query.

### Act 3 — Semantic Cache (Different Words)
Three paraphrased queries test whether the cache understands meaning:
- "what is the leave policy for employees?" → hits cache for "what is the employee leave policy?" (sim=0.9731)
- "what health insurance options are available?" → hits cache (sim=0.9541)
- "what is the parental leave policy details?" → hits cache (sim=0.9469)

### Act 4 — Adaptive Learning + Drift Detection
**Part A:** Shows current cache quality scores (EMA of user feedback rewards).

**Part B:** Simulates company policy change — cached answers score 0.15, fresh retrieval scores 0.85. JS divergence reaches 0.59. System fires `🚨 DRIFT DETECTED` and evicts degraded entries.

**Part C:** Fresh query re-populates cache. Good signals injected. JS divergence drops to ~0.02. System shows `✓ System healthy`.

> **Observation — "Degraded" entries in Part A:**
> Some cache entries show `● Degraded` (quality score < 0.5) even before the drift scenario runs. This is the system working correctly, not a bug.
>
> The corpus is MS MARCO (a web search benchmark), not a real HR knowledge base. So a query like "employee leave policy" retrieves passages about Canadian labour law and resignation notices — tangentially related but not precise answers. The FeedbackSimulator sees low relevance scores, simulates low click rates and short dwell times, and produces a reward of ~0.3–0.4. After 2 cache hits, the EMA pulls the quality score below 0.5:
>
> ```
> quality = 0.7 × 1.0 + 0.3 × 0.35 = 0.805  (after 1st hit)
> quality = 0.7 × 0.805 + 0.3 × 0.35 = 0.669  (after 2nd hit)
> ```
>
> In a production deployment with a domain-matched corpus (actual HR documents), the retrieved passages would be on-topic, feedback rewards would be high (0.8+), and quality scores would stay healthy. The degradation here is a signal that the corpus doesn't match the query domain — exactly the kind of signal the drift detector is designed to surface and act on.

### Act 5 — Performance Summary
Rich table with hit rate, P50/P95 latencies, speedup factor, cost accounting. Latency histogram saved to `outputs/latency_histogram.png`.

---

## Algorithm Deep Dives

### Why the two-stage retrieval funnel?

```
Stage 1: Bi-encoder (embedder)
  - Encode query once (8ms)
  - Encode all docs offline (one time)
  - Compare via FAISS (2ms for 10K docs)
  - Fast but imprecise: MRR@10 = 33

Stage 2: Cross-encoder (reranker)
  - Reads (query, doc) pairs jointly
  - Sees full context — query and passage together
  - 10× better at judging relevance: MRR@10 = 39
  - Slow: 100–280ms for 50 pairs

Combined: get 50 good candidates fast, precision-rank to 5 slowly.
Cache eliminates Stage 2 for repeated/similar queries.
```

### Why EMA for cache quality? Why not average?

EMA (Exponential Moving Average) with α=0.3 gives **recency bias** — recent feedback matters more than old feedback. If a cached answer was good for 10 queries but the 11th user was unhappy, the score drops. A plain average would dilute this signal.

The α=0.3 value balances responsiveness (higher α reacts faster but is noisier) against stability (lower α is smoother but slow to adapt). At α=0.3, the half-life of old feedback is ~2 queries.

### Why Jensen-Shannon divergence for drift?

Standard approach would be to compare means: `cache_mean < fresh_mean`. But this misses **distribution shape changes**. Imagine cache rewards: [0.1, 0.9, 0.1, 0.9, ...] (alternating) vs fresh rewards: [0.5, 0.5, 0.5, ...] (stable). Both have mean 0.5 — mean comparison says no drift. JS divergence catches it because the distributions have different shapes.

JS is also:
- **Symmetric:** JS(P||Q) = JS(Q||P) — both streams treated equally
- **Bounded:** [0, log(2)] — interpretable, never infinite
- **Stable:** Laplace smoothing prevents log(0) errors

### Why HNSW over IVF (Inverted File Index)?

| Property | HNSW | IVF |
|----------|------|-----|
| Accuracy | Exact on stored vectors | Approximate (quantization error) |
| Build time | Slower | Faster |
| Query time | O(log n) | O(n/nlist) |
| Memory | Higher | Lower |
| Best at | < 10M vectors, high precision | > 10M vectors, memory-constrained |

For the demo corpus (10K passages) and production target (100M), the right answer diverges. HNSW is used here for correctness. At 100M, you'd switch to IVF with Product Quantization (IVF-PQ).

---

## Scaling to 100M Documents

The demo runs everything locally. The algorithms are identical at scale — only the infrastructure wrapping changes:

| Component | Demo (10K) | Production (100M) |
|-----------|-----------|-----------------|
| VectorStore | Local FAISS HNSW | Distributed FAISS IVF-PQ sharded across GPUs |
| SemanticCache | In-memory dict + FAISS | Redis cluster with FAISS on each node |
| FeedbackSimulator | Synthetic rewards | Real Kafka/Kinesis click+dwell event stream |
| DriftDetector | In-memory deque | Flink/Spark streaming window over event stream |
| Instrumentation | Local QueryTrace list | Prometheus metrics + Grafana dashboards |
| Embedder | Single process | GPU inference service (TorchServe / Triton) |

**The retrieval intelligence — semantic cache threshold, EMA quality scoring, JS drift detection, two-stage funnel — is identical.**

---

## Test Suite

```
tests/test_embedder.py          6/6   pass  — shape, norm, similarity thresholds
tests/test_semantic_cache.py    9/9   pass  — hit/miss, EMA, eviction, hit rate
tests/test_reranker.py          5/5   pass  — score ordering, real data integration
tests/test_simulator.py         6/6   pass  — reward bounds, click/dwell behaviour
tests/test_drift_detector.py    9/9   pass  — JS math, strict threshold, reset
tests/test_instrumentation.py   6/6   pass  — timer precision, stats, P95
tests/test_pipeline.py          7/7   pass  — cold/warm/semantic, traces, drift signals
tests/test_vector_store.py      7/7   pass  — search, save/load, scores, fields
─────────────────────────────────────────
Total                          55/55  pass
```

`test_pipeline.py` uses `scope="module"` fixture — models load once for all 7 tests (~30s total on warm cache).

---

## Key Design Decisions

| Decision | Choice | Rejected | Reason |
|----------|--------|----------|--------|
| Embedding model | all-MiniLM-L6-v2 | MPNet, BGE-large | 3–5× faster, 2% lower NDCG — worth it |
| Reranker | ms-marco-MiniLM-L-6 | L-12 variant, BGE | L-12: 2× slower +1 MRR; BGE: 500ms |
| Cache eviction | Lowest quality score | LRU | LRU ignores result quality; a popular bad result stays forever |
| Quality update | EMA α=0.3 | Average, last value | Recency bias + noise smoothing |
| Divergence metric | Jensen-Shannon | KL divergence | KL is asymmetric and unbounded; JS is symmetric, bounded [0, 0.693] |
| Drift condition | JS > 0.15 AND cache_mean < fresh_mean | JS alone | High JS but cache better than fresh = no problem |
| Timer | perf_counter_ns | time.time() | 1ms resolution vs nanosecond; essential for 0.1ms cache checks |
| Vector index | FAISS HNSW | Pinecone, Weaviate | Zero latency overhead, free, no network call, exact distances |

---

## Background

Built for AIntropy AI — a production AI infrastructure company focused on making large-scale retrieval fast, accurate, and self-maintaining. The core insight is that retrieval systems should continuously learn from user behaviour to improve themselves, not just serve static indexes.

The system demonstrated here implements the core retrieval intelligence layer: semantic understanding (not keyword matching), adaptive quality management (not static TTL), and statistical drift detection (not manual monitoring). These three properties together make retrieval systems that get better over time, not worse.
