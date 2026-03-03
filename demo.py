"""
AIntropy — Extreme Fast Retrieval Demo
Run: KMP_DUPLICATE_LIB_OK=TRUE arch -arm64 python3.13 demo.py
"""
from __future__ import annotations

import logging
import os
import sys
import time

# Suppress verbose weight-loading bars from transformers/safetensors
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ── pre-flight: fail fast with a clean message, not a traceback ───────────────
if not os.path.exists("data/ms_marco_10k.index"):
    print()
    print("  ERROR: data/ms_marco_10k.index not found.")
    print("  Run this first:  python3.13 data/prepare_index.py")
    print()
    sys.exit(1)

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

from pipeline import RetrievalPipeline

console = Console()

# ── helpers ───────────────────────────────────────────────────────────────────

def section_break(seconds: float = 1.0) -> None:
    """Pause between acts so the presenter can speak."""
    time.sleep(seconds)


def query_label(query: str, max_len: int = 28) -> str:
    """Truncate long query text for table cells."""
    return query[:max_len] + "…" if len(query) > max_len else query


def print_trace_table(trace) -> None:
    """One-row timing breakdown for a single query."""
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Path",   style="green" if trace.path == "CACHE_HIT" else "yellow")
    t.add_column("Embed",  justify="right")
    t.add_column("Cache",  justify="right")
    t.add_column("Search", justify="right")
    t.add_column("Rerank", justify="right")
    t.add_column("Total",  justify="right", style="bold")
    t.add_column("Cost",   justify="right")

    t.add_row(
        trace.path,
        f"{trace.embed_ms:.1f}ms",
        f"{trace.cache_check_ms:.1f}ms",
        f"{trace.search_ms:.1f}ms" if trace.search_ms > 0 else "—",
        f"{trace.rerank_ms:.1f}ms" if trace.rerank_ms > 0 else "—",
        f"[bold]{trace.total_ms:.1f}ms[/bold]",
        f"${trace.cost_usd:.5f}",
    )
    console.print(t)


def print_results_preview(results, n: int = 2) -> None:
    """Print text snippet for the top-N returned documents."""
    for i, doc in enumerate(results[:n]):
        snippet = doc.text[:120] + ("…" if len(doc.text) > 120 else "")
        console.print(f"  [cyan]#{i+1}[/cyan] [dim]{snippet}[/dim]")


# ── SECTION 0: Startup ────────────────────────────────────────────────────────

def section_startup() -> RetrievalPipeline:
    console.print()
    console.print(Panel(
        "[bold cyan]  AINTROPY — EXTREME FAST RETRIEVAL  [/bold cyan]\n"
        "[dim]  Semantic Cache + Adaptive Learning  [/dim]",
        box=box.DOUBLE,
        expand=False,
        padding=(1, 4),
    ))
    console.print()

    pipeline: RetrievalPipeline | None = None

    # Progress spinner — each task maps to one expensive init step.
    # We don't call pipeline components directly inside the progress context
    # because Progress redraws the terminal; mixing print() calls would corrupt
    # the display. Instead we batch-complete each task and let the spinner run
    # while we do the real work below.
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,   # keep completed tasks visible after the block
    ) as progress:

        t0 = progress.add_task("Loading embedding model…",        total=1)
        t1 = progress.add_task("Loading vector store (10K docs)…", total=1, start=False)
        t2 = progress.add_task("Loading reranker…",               total=1, start=False)
        t3 = progress.add_task("Initializing semantic cache…",    total=1, start=False)

        # All four layers initialize inside RetrievalPipeline.__init__.
        # We advance each spinner task in order so the display tracks progress
        # even though the actual work happens in a single blocking call.
        progress.start_task(t1)
        progress.start_task(t2)
        progress.start_task(t3)

        pipeline = RetrievalPipeline()

        progress.advance(t0, 1)
        progress.advance(t1, 1)
        progress.advance(t2, 1)
        progress.advance(t3, 1)

    # Extra warmup embed: Embedder.__init__ runs one warmup, but PyTorch JIT can
    # still recompile on the first encode call after a quiet period (GC, model load
    # of the reranker, etc.).  A second silent embed here burns off that recompile
    # so the first warm cache hit in Section 2 shows ~8ms, not ~45ms.
    pipeline.embedder.embed("warmup query one")
    pipeline.embedder.embed("warmup query two")

    console.print("[bold green]  ✓ Ready. 10,000 documents indexed.[/bold green]")
    console.print()
    return pipeline


# ── SECTION 1: Cold Queries ───────────────────────────────────────────────────

def section_cold_queries(pipeline: RetrievalPipeline) -> list[tuple]:
    """
    Run 5 queries against an empty cache.
    Returns list of (query, trace) for Section 2 comparison.
    """
    console.print(Panel(
        "[bold]ACT 1: COLD QUERIES[/bold] [dim](No Cache)[/dim]",
        style="yellow",
        box=box.HEAVY,
    ))
    console.print(
        "  Every query goes through the full pipeline: "
        "[cyan]embed → vector search → rerank[/cyan]"
    )
    console.print()

    queries = [
        "what is the employee leave policy?",
        "how do I submit an expense report?",
        "what are the health insurance options?",
        "how to set up VPN access?",
        "what is the parental leave policy?",
    ]

    cold_results: list[tuple] = []

    for query in queries:
        console.rule(f"[dim]  Querying: [italic]\"{query}\"[/italic][/dim]")
        results, trace = pipeline.query(query)
        print_trace_table(trace)
        print_results_preview(results, n=2)
        cold_results.append((query, trace))
        console.print()
        time.sleep(0.5)

    return cold_results


# ── SECTION 2: Cache Hits ─────────────────────────────────────────────────────

def section_cache_hits(pipeline: RetrievalPipeline, cold_results: list[tuple]) -> None:
    """
    Re-run the same 5 queries. Build a side-by-side comparison table.
    """
    section_break()
    console.print(Panel(
        "[bold]ACT 2: CACHE HITS[/bold] [dim](Same Queries)[/dim]",
        style="green",
        box=box.HEAVY,
    ))
    console.print(
        "  Same 5 queries — this time the cache responds "
        "[bold green]instantly[/bold green].\n"
        "  No vector search. No rerank. Pure memory lookup."
    )
    console.print()

    comparison = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    comparison.add_column("Query",    min_width=28)
    comparison.add_column("Cold(ms)", justify="right", style="yellow")
    comparison.add_column("Warm(ms)", justify="right", style="green")
    comparison.add_column("Speedup",  justify="right", style="bold magenta")
    comparison.add_column("Similarity", justify="right", style="cyan")

    for query, cold_trace in cold_results:
        console.print(f"  [dim]→ {query}[/dim]")
        _, warm_trace = pipeline.query(query)

        speedup = (
            cold_trace.total_ms / warm_trace.total_ms
            if warm_trace.total_ms > 0 else float("inf")
        )
        sim_str = (
            f"{warm_trace.cache_similarity:.4f}"
            if warm_trace.cache_similarity is not None else "—"
        )

        comparison.add_row(
            query_label(query),
            f"{cold_trace.total_ms:.0f}ms",
            f"{warm_trace.total_ms:.0f}ms",
            f"{speedup:.1f}x",
            sim_str,
        )
        time.sleep(0.5)

    console.print()
    console.print(comparison)


# ── SECTION 3: Semantic Similarity ────────────────────────────────────────────

def section_semantic_similarity(pipeline: RetrievalPipeline) -> None:
    section_break()
    console.print(Panel(
        "[bold]ACT 3: SEMANTIC CACHE[/bold] [dim](Different Words, Same Meaning)[/dim]",
        style="cyan",
        box=box.HEAVY,
    ))
    console.print(
        "  The cache understands [bold]MEANING[/bold], not just exact words.\n"
        "  Watch what happens with paraphrased queries:"
    )
    console.print()

    # Near-paraphrases chosen to sit above the 0.85 cosine threshold.
    # Small wording changes (reordering, "details" suffix) reliably hit;
    # semantically-related-but-different phrases (vacation days, medical benefits)
    # reliably miss — which makes both outcomes educationally useful.
    semantic_pairs = [
        (
            "what is the employee leave policy?",
            "what is the leave policy for employees?",
        ),
        (
            "what are the health insurance options?",
            "what health insurance options are available?",
        ),
        (
            "what is the parental leave policy?",
            "what is the parental leave policy details?",
        ),
    ]

    for original, paraphrase in semantic_pairs:
        # Show the pair side by side
        pair_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        pair_table.add_column(width=12, style="dim")
        pair_table.add_column()
        pair_table.add_row("[bold]Cached:[/bold]",    f"[yellow]{original}[/yellow]")
        pair_table.add_row("[bold]New query:[/bold]", f"[cyan]{paraphrase}[/cyan]")
        console.print(pair_table)

        _, trace = pipeline.query(paraphrase)

        if trace.path == "CACHE_HIT":
            console.print(
                f"  [bold green]✓ SEMANTIC HIT[/bold green]  "
                f"similarity=[bold]{trace.cache_similarity:.4f}[/bold]  "
                f"total=[bold]{trace.total_ms:.1f}ms[/bold]"
            )
            console.print(
                f"  [dim]The cache recognised these as semantically equivalent "
                f"(sim {trace.cache_similarity:.4f} ≥ threshold 0.85)[/dim]"
            )
        else:
            # Retrieve the raw similarity for the miss explanation.
            # The lookup returned None, meaning the best match was below 0.85.
            # We surface the cache_check_ms to show the system still looked.
            console.print(
                f"  [yellow]◌ SEMANTIC MISS[/yellow]  "
                f"checked in {trace.cache_check_ms:.1f}ms — "
                f"phrases too different for 0.85 threshold"
            )
            console.print(
                f"  [dim]Result retrieved fresh and stored — "
                f"future repetitions will hit.[/dim]"
            )

        console.print()
        time.sleep(0.5)


# ── SECTION 4: Quality and Drift ─────────────────────────────────────────────

def section_drift_demo(pipeline: RetrievalPipeline) -> None:
    section_break()
    console.print(Panel(
        "[bold]ACT 4: ADAPTIVE LEARNING + DRIFT DETECTION[/bold]",
        style="magenta",
        box=box.HEAVY,
    ))

    # ── Part A: current cache quality ─────────────────────────────────────────
    console.print("[bold]Part A — Current cache quality[/bold]")
    console.print("  Cache entries rated by simulated user feedback (EMA of rewards).")
    console.print()

    quality_table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    quality_table.add_column("Query",         min_width=32)
    quality_table.add_column("Quality Score", justify="right")
    quality_table.add_column("Hit Count",     justify="right")
    quality_table.add_column("Status",        justify="center")

    # Sort entries by quality descending, show top 5
    entries = sorted(
        pipeline.cache._entries.values(),
        key=lambda e: e.quality_score,
        reverse=True,
    )[:5]

    for entry in entries:
        q = entry.quality_score
        status = "[green]● Healthy[/green]" if q >= 0.5 else "[red]● Degraded[/red]"
        quality_table.add_row(
            query_label(entry.query_text, max_len=32),
            f"{q:.4f}",
            str(entry.hit_count),
            status,
        )

    console.print(quality_table)
    console.print()
    time.sleep(1)

    # ── Part B: inject stale-data drift scenario ───────────────────────────────
    console.print("[bold]Part B — Simulating policy change / stale cache[/bold]")
    console.print(Panel(
        "  Scenario: company updated leave policy.\n"
        "  Cached answers reflect old policy  → low user satisfaction (reward 0.15).\n"
        "  Fresh retrieval finds new docs     → high satisfaction (reward 0.85).\n"
        "  System must detect this divergence and trigger eviction.",
        style="dim",
        box=box.ROUNDED,
    ))
    console.print()

    console.print("  [dim]Injecting 30 divergent reward signals…[/dim]")
    for _ in range(30):
        pipeline.drift_detector.record("cache", 0.15)
        pipeline.drift_detector.record("fresh", 0.85)

    drift = pipeline.get_drift_status()

    # Drift status panel
    status_color = "red" if drift.is_drifting else "green"
    status_icon  = "🚨 DRIFT DETECTED" if drift.is_drifting else "✓ No drift"
    drift_panel  = Panel(
        f"  JS Divergence:     [bold]{drift.js_divergence:.4f}[/bold]\n"
        f"  Cache mean reward: [red]{drift.cache_mean_reward:.4f}[/red]\n"
        f"  Fresh mean reward: [green]{drift.fresh_mean_reward:.4f}[/green]\n"
        f"  Cache samples:     {drift.cache_samples}\n"
        f"  Fresh samples:     {drift.fresh_samples}\n"
        f"  Status:            [{status_color}][bold]{status_icon}[/bold][/{status_color}]\n"
        f"  Action:            [bold]{drift.recommendation}[/bold]",
        title="[bold]Drift Detector Output[/bold]",
        style=status_color,
        box=box.ROUNDED,
    )
    console.print(drift_panel)
    console.print()
    time.sleep(1)

    # ── System response: evict degraded entries ────────────────────────────────
    if drift.is_drifting:
        console.print("  [bold yellow]Evicting low-quality cache entries…[/bold yellow]")

        # Mark all entries with quality < 0.5 for removal by driving their score
        # to 0 then calling update_quality — simpler to just rebuild from scratch
        # by finding the IDs and calling _evict directly if we could, but the
        # public API only exposes update_quality. Instead we collect the IDs and
        # rebuild via the internal eviction path directly.
        ids_to_evict = [
            eid for eid, e in pipeline.cache._entries.items()
            if e.quality_score < 0.5
        ]
        evicted = 0
        for eid in ids_to_evict:
            # Drive quality to 0 so _evict_lowest_quality picks it first,
            # then trigger eviction by forcing a store that tips over max_size.
            # Simpler: just remove from _entries and _id_map then rebuild index.
            pipeline.cache._entries.pop(eid, None)
            if eid in pipeline.cache._id_map:
                pipeline.cache._id_map.remove(eid)
            evicted += 1

        if evicted:
            pipeline.cache._rebuild_index()

        console.print(
            f"  [green]✓ Evicted {evicted} degraded entr{'y' if evicted == 1 else 'ies'}.[/green]"
        )
        console.print(
            "  [dim]Cache refreshed. Next queries will retrieve fresh results\n"
            "  and rebuild cache entries from current data.[/dim]"
        )

        # Reset drift detector so the demo doesn't carry injected signals forward
        pipeline.drift_detector.reset()
        console.print("  [dim]Drift detector window reset.[/dim]")
    else:
        console.print("  [green]No action needed — system healthy.[/green]")

    console.print()
    time.sleep(1)

    # ── Part C: system recovers — drift goes green ─────────────────────────────
    console.print("[bold]Part C — System recovery[/bold]")
    console.print(
        "  Fresh queries rebuild the cache with current data.\n"
        "  Injecting 20 high-reward signals to show recovery:"
    )
    console.print()

    # Run one fresh query (evicted entries are gone, so this is a MISS that
    # re-populates the cache with up-to-date results)
    _, recovery_trace = pipeline.query("what is the employee leave policy?")
    console.print(
        f"  Fresh retrieval: [cyan]{recovery_trace.path}[/cyan]  "
        f"total={recovery_trace.total_ms:.1f}ms  "
        f"reward={recovery_trace.reward:.3f}"
    )

    # Inject 20 high-satisfaction signals to simulate users being happy
    # with the freshly retrieved results
    for _ in range(20):
        pipeline.drift_detector.record("cache", 0.85)
        pipeline.drift_detector.record("fresh", 0.80)

    recovery = pipeline.get_drift_status()

    r_color = "red" if recovery.is_drifting else "green"
    r_icon  = "🚨 DRIFT DETECTED" if recovery.is_drifting else "✓ System healthy"
    recovery_panel = Panel(
        f"  JS Divergence:     [bold]{recovery.js_divergence:.4f}[/bold]\n"
        f"  Cache mean reward: [green]{recovery.cache_mean_reward:.4f}[/green]\n"
        f"  Fresh mean reward: [green]{recovery.fresh_mean_reward:.4f}[/green]\n"
        f"  Status:            [{r_color}][bold]{r_icon}[/bold][/{r_color}]\n"
        f"  Action:            [bold]{recovery.recommendation}[/bold]",
        title="[bold]Drift Detector Output (After Recovery)[/bold]",
        style=r_color,
        box=box.ROUNDED,
    )
    console.print(recovery_panel)
    console.print(
        "  [dim]Cache is healthy again. The feedback loop closed automatically.[/dim]"
    )
    console.print()


# ── SECTION 5: Final Metrics ──────────────────────────────────────────────────

def section_final_metrics(pipeline: RetrievalPipeline) -> None:
    section_break()
    console.print(Panel(
        "[bold]ACT 5: PERFORMANCE SUMMARY[/bold]",
        style="blue",
        box=box.HEAVY,
    ))

    reporter = pipeline.get_reporter()
    reporter.print_summary()
    console.print()
    reporter.print_latency_breakdown()
    console.print()

    reporter.plot_latency_histogram()
    console.print("  [dim]Histogram saved → [bold]outputs/latency_histogram.png[/bold][/dim]")
    console.print()

    console.print(Panel(
        "[bold cyan]Architecture scales to 100M+ documents.[/bold cyan]\n\n"
        "  Same middleware, distributed FAISS index,\n"
        "  Redis semantic cache, real user event stream.\n"
        "  The retrieval intelligence is identical — only the\n"
        "  infrastructure changes, not the algorithms.",
        title="[bold]Scaling Story[/bold]",
        box=box.DOUBLE,
        padding=(1, 2),
    ))
    console.print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        pipeline = section_startup()

        cold_results = section_cold_queries(pipeline)
        section_cache_hits(pipeline, cold_results)
        section_semantic_similarity(pipeline)
        section_drift_demo(pipeline)
        section_final_metrics(pipeline)

    except KeyboardInterrupt:
        console.print()
        console.print("[dim]  Demo interrupted.[/dim]")
        sys.exit(0)

    except Exception as exc:
        # Never show a raw Python traceback to the interviewer.
        # Log the type and message only — enough to debug, clean enough to ignore.
        console.print()
        console.print(Panel(
            f"[bold red]  Demo error:[/bold red] {type(exc).__name__}: {exc}\n\n"
            f"  [dim]Check that all data files are present and run:\n"
            f"  python3.13 data/prepare_index.py[/dim]",
            title="[red]Error[/red]",
            box=box.ROUNDED,
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
