"""Parallel analyst execution using ThreadPoolExecutor.

Wraps the analyst loop -- the analyze() interface stays identical.
Each analyst is I/O-bound (LLM API calls), not CPU-bound, so
ThreadPoolExecutor is the correct concurrency model.

Thread safety: each analyst creates its own data structures and returns
an independent dict. No shared mutable state.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.agents.base import BaseAnalyst
from src.models import AnalystSignal


# Cap workers to avoid overwhelming API rate limits.
MAX_WORKERS = 8


def _run_single_analyst(
    analyst: BaseAnalyst,
    tickers: list[str],
    market_data: dict[str, Any],
) -> tuple[str, dict[str, AnalystSignal]]:
    """Execute a single analyst and return (name, results).

    If the analyst raises, return neutral signals for all tickers
    so one failure does not kill the pipeline. This matches the
    existing graceful degradation behavior.
    """
    try:
        results = analyst.analyze(tickers, market_data)
        return analyst.name, results
    except Exception as e:
        # Graceful degradation: return neutral/0 for all tickers
        fallback: dict[str, AnalystSignal] = {}
        for ticker in tickers:
            fallback[ticker] = AnalystSignal(
                signal="neutral",
                confidence=0,
                reasoning=f"Analyst {analyst.name} failed: {e}",
            )
        return analyst.name, fallback


def run_analysts_parallel(
    analysts: list[BaseAnalyst],
    tickers: list[str],
    market_data: dict[str, Any],
    *,
    max_workers: int | None = None,
    verbose: bool = True,
) -> tuple[dict[str, dict[str, AnalystSignal]], float]:
    """Run analysts concurrently and collect signals.

    Args:
        analysts: List of analyst instances to execute.
        tickers: Ticker symbols to analyze.
        market_data: Pre-fetched market data (NOT parallelized).
        max_workers: Thread pool size. Defaults to min(len(analysts), MAX_WORKERS).
        verbose: Print per-analyst completion messages.

    Returns:
        Tuple of (all_signals dict, elapsed_seconds).
        all_signals maps ticker -> analyst_name -> AnalystSignal.
    """
    workers = max_workers or min(len(analysts), MAX_WORKERS)
    all_signals: dict[str, dict[str, AnalystSignal]] = {}

    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_single_analyst, analyst, tickers, market_data): analyst
            for analyst in analysts
        }

        for future in as_completed(futures):
            analyst = futures[future]
            name, results = future.result()

            for ticker, signal in results.items():
                if ticker not in all_signals:
                    all_signals[ticker] = {}
                all_signals[ticker][name] = signal

            if verbose:
                summary = ", ".join(
                    f"{t}:{results[t].signal}"
                    for t in tickers
                    if t in results
                )
                print(f"  {name} done ({summary})")

    elapsed = time.perf_counter() - t0
    return all_signals, elapsed


def run_analysts_sequential(
    analysts: list[BaseAnalyst],
    tickers: list[str],
    market_data: dict[str, Any],
    *,
    verbose: bool = True,
) -> tuple[dict[str, dict[str, AnalystSignal]], float]:
    """Run analysts sequentially (original behavior, used as fallback).

    Same interface as run_analysts_parallel for easy swap.
    """
    all_signals: dict[str, dict[str, AnalystSignal]] = {}

    t0 = time.perf_counter()

    for analyst in analysts:
        if verbose:
            print(f"  Running {analyst.name}...", end=" ", flush=True)

        name, results = _run_single_analyst(analyst, tickers, market_data)

        for ticker, signal in results.items():
            if ticker not in all_signals:
                all_signals[ticker] = {}
            all_signals[ticker][name] = signal

        if verbose:
            summary = ", ".join(
                f"{t}:{results[t].signal}"
                for t in tickers
                if t in results
            )
            print(f"done ({summary})")

    elapsed = time.perf_counter() - t0
    return all_signals, elapsed
