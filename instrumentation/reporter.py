from __future__ import annotations

import os
from typing import Optional

import numpy as np

from instrumentation.tracer import QueryTrace


class StatsReporter:

    def __init__(self, traces: list[QueryTrace]) -> None:
        self.traces = traces

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        hits  = [t for t in self.traces if t.path == "CACHE_HIT"]
        misses = [t for t in self.traces if t.path == "CACHE_MISS"]
        total = len(self.traces)

        hit_rate = len(hits) / total if total > 0 else 0.0

        avg_hit_ms  = float(np.mean([t.total_ms for t in hits]))  if hits   else 0.0
        avg_miss_ms = float(np.mean([t.total_ms for t in misses])) if misses else 0.0

        # speedup = how many times faster a cache hit is vs a fresh retrieval.
        # Guard: if no hits exist yet, speedup is undefined — report 1.0 (no improvement).
        speedup = avg_miss_ms / avg_hit_ms if hits and avg_hit_ms > 0 else 1.0

        total_cost   = sum(t.cost_usd         for t in self.traces)
        total_saved  = sum(t.cost_saved()     for t in self.traces)

        all_latencies = [t.total_ms for t in self.traces]
        # np.percentile with interpolation="midpoint" matches the "exact value" expectation
        # in the test where we set known latencies. Linear interpolation would shift the result
        # by fractional amounts that make deterministic assertions fragile.
        p50 = float(np.percentile(all_latencies, 50, method="midpoint")) if all_latencies else 0.0
        p95 = float(np.percentile(all_latencies, 95, method="midpoint")) if all_latencies else 0.0

        return {
            "total_queries":        total,
            "cache_hits":           len(hits),
            "cache_misses":         len(misses),
            "hit_rate":             hit_rate,
            "avg_latency_hit_ms":   avg_hit_ms,
            "avg_latency_miss_ms":  avg_miss_ms,
            "speedup_factor":       speedup,
            "total_cost_usd":       total_cost,
            "total_cost_saved_usd": total_saved,
            "p50_latency_ms":       p50,
            "p95_latency_ms":       p95,
        }

    # ── print_summary ─────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        # Import rich here so tests that don't have rich installed can still import
        # StatsReporter without an immediate crash. In practice rich is always installed.
        from rich.console import Console
        from rich.table import Table
        from rich import box

        s = self.summary()
        console = Console()

        table = Table(
            title="RETRIEVAL PERFORMANCE",
            box=box.DOUBLE,
            show_header=False,
            min_width=44,
        )
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value",  justify="right")

        table.add_row("Total queries",     str(s["total_queries"]))
        table.add_row("Cache hit rate",    f"{s['hit_rate']*100:.1f}%")
        table.add_row("Avg latency MISS",  f"{s['avg_latency_miss_ms']:.1f}ms")
        table.add_row("Avg latency HIT",   f"{s['avg_latency_hit_ms']:.1f}ms")
        table.add_row("Speedup",           f"{s['speedup_factor']:.1f}x")
        table.add_row("P50 latency",       f"{s['p50_latency_ms']:.1f}ms")
        table.add_row("P95 latency",       f"{s['p95_latency_ms']:.1f}ms")
        table.add_row("Total cost",        f"${s['total_cost_usd']:.5f}")
        table.add_row("Cost saved",        f"${s['total_cost_saved_usd']:.5f}")

        console.print(table)

    # ── print_latency_breakdown ───────────────────────────────────────────────

    def print_latency_breakdown(self) -> None:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        miss = next((t for t in self.traces if t.path == "CACHE_MISS"), None)
        hit  = next((t for t in self.traces if t.path == "CACHE_HIT"),  None)

        if miss:
            # saved_ms = what the cache would have spared: search + rerank time
            saved_ms = miss.search_ms + miss.rerank_ms
            lines = [
                f"  Embed:          {miss.embed_ms:.1f}ms",
                f"  Cache check:    {miss.cache_check_ms:.1f}ms",
                f"  Vector search:  {miss.search_ms:.1f}ms",
                f"  Rerank:         {miss.rerank_ms:.1f}ms",
                f"  Total:          {miss.total_ms:.1f}ms",
            ]
            console.print(Panel("\n".join(lines), title="COLD QUERY (cache miss)"))

        if hit:
            saved_ms = hit.total_ms  # rough proxy — how much less it took than a miss
            if miss:
                saved_ms = miss.total_ms - hit.total_ms
            lines = [
                f"  Embed:          {hit.embed_ms:.1f}ms",
                f"  Cache check:    {hit.cache_check_ms:.1f}ms",
                f"  Total:          {hit.total_ms:.1f}ms",
                f"  Saved:          {saved_ms:.1f}ms",
            ]
            console.print(Panel("\n".join(lines), title="CACHED QUERY (cache hit)"))

    # ── plot_latency_histogram ────────────────────────────────────────────────

    def plot_latency_histogram(self) -> None:
        import matplotlib.pyplot as plt

        hits   = [t.total_ms for t in self.traces if t.path == "CACHE_HIT"]
        misses = [t.total_ms for t in self.traces if t.path == "CACHE_MISS"]

        fig, ax = plt.subplots(figsize=(8, 4))

        if hits:
            ax.hist(hits,   bins=20, color="green", alpha=0.7, label="Cache Hit")
        if misses:
            ax.hist(misses, bins=20, color="red",   alpha=0.7, label="Cache Miss")

        ax.set_title("Cache Hit vs Miss Latency Distribution")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Query Count")
        ax.legend()
        fig.tight_layout()

        # Create outputs/ directory if it doesn't exist.
        os.makedirs("outputs", exist_ok=True)
        fig.savefig("outputs/latency_histogram.png", dpi=150)
        plt.close(fig)  # free memory — don't leave figures open in long-running pipelines
