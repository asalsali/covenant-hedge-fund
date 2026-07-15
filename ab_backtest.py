"""Covenant Hedge Fund -- A/B Backtest: Quant-Only vs LLM-Lite.

Runs two independent backtests on the same tickers, date range, and
initial capital, then prints a side-by-side comparison table.

  Run A: Quant-only (5 quant analysts, no LLM calls, no API keys needed)
  Run B: LLM-lite  (5 quant + 4 LLM personas on 5 rebalance dates)

Usage:
    python ab_backtest.py

No arguments required. Defaults: AAPL/MSFT/NVDA, 15-month lookback,
$100k initial capital.

Run B requires Ollama running locally (free).
If Ollama is not available, Run B will produce neutral/0 LLM signals
(quant-only mode) and results will match Run A.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date

# Ensure src is importable when running from project root
sys.path.insert(0, ".")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "NVDA"]
INITIAL_CASH = 100_000.0
LOOKBACK_MONTHS = 15  # Match the validated 15-month backtest window


def _subtract_months(d: date, months: int) -> date:
    """Subtract months from a date, handling edge cases."""
    import calendar
    month = d.month - months
    year = d.year
    while month <= 0:
        month += 12
        year -= 1
    max_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, max_day))


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------

def _compute_win_rate(portfolio) -> tuple[int, int]:
    """Compute win rate from realized gains per ticker.

    A ticker is a 'win' if its total realized P&L is positive.

    Returns:
        (wins, total_closed_tickers)
    """
    realized = portfolio.state.realized_gains
    if not realized:
        return 0, 0
    wins = sum(1 for pnl in realized.values() if pnl > 0)
    return wins, len(realized)


def _extract_metrics(engine) -> dict:
    """Extract all comparison metrics from a completed BacktestEngine."""
    metrics = engine.portfolio.compute_performance()
    wins, total_closed = _compute_win_rate(engine.portfolio)
    n_trades = len(engine.portfolio.trades)

    return {
        "total_return": (metrics.total_return or 0.0) * 100,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "max_drawdown": (metrics.max_drawdown or 0.0) * 100,
        "alpha_vs_spy": (engine.alpha or 0.0) * 100,
        "spy_return": (engine.spy_return or 0.0) * 100,
        "win_rate": (wins / total_closed * 100) if total_closed > 0 else 0.0,
        "wins": wins,
        "total_closed": total_closed,
        "n_trades": n_trades,
        "final_value": (
            engine.portfolio.daily_values[-1][1]
            if engine.portfolio.daily_values
            else INITIAL_CASH
        ),
        "annualized_return": (metrics.annualized_return or 0.0) * 100,
    }


# ---------------------------------------------------------------------------
# Comparison output
# ---------------------------------------------------------------------------

def _fmt(val, fmt_str: str, suffix: str = "") -> str:
    """Format a value, handling None gracefully."""
    if val is None:
        return "N/A"
    return f"{val:{fmt_str}}{suffix}"


def _print_comparison(metrics_a: dict, metrics_b: dict, elapsed_a: float, elapsed_b: float) -> None:
    """Print side-by-side comparison table and analysis."""
    print()
    print("=" * 74)
    print("A/B BACKTEST COMPARISON")
    print("=" * 74)
    print()
    print(f"  {'Metric':<28} {'Run A (Quant-Only)':>20} {'Run B (LLM-Lite)':>20}")
    print("  " + "-" * 70)

    rows = [
        ("Total Return",        _fmt(metrics_a["total_return"], "+.2f", "%"),
                                _fmt(metrics_b["total_return"], "+.2f", "%")),
        ("Annualized Return",   _fmt(metrics_a["annualized_return"], "+.2f", "%"),
                                _fmt(metrics_b["annualized_return"], "+.2f", "%")),
        ("Sharpe Ratio",        _fmt(metrics_a["sharpe_ratio"], ".4f"),
                                _fmt(metrics_b["sharpe_ratio"], ".4f")),
        ("Sortino Ratio",       _fmt(metrics_a["sortino_ratio"], ".4f"),
                                _fmt(metrics_b["sortino_ratio"], ".4f")),
        ("Max Drawdown",        _fmt(metrics_a["max_drawdown"], ".2f", "%"),
                                _fmt(metrics_b["max_drawdown"], ".2f", "%")),
        ("Alpha vs SPY",        _fmt(metrics_a["alpha_vs_spy"], "+.2f", "%"),
                                _fmt(metrics_b["alpha_vs_spy"], "+.2f", "%")),
        ("Win Rate",            f"{metrics_a['wins']}/{metrics_a['total_closed']} "
                                f"({metrics_a['win_rate']:.0f}%)",
                                f"{metrics_b['wins']}/{metrics_b['total_closed']} "
                                f"({metrics_b['win_rate']:.0f}%)"),
        ("Number of Trades",    str(metrics_a["n_trades"]),
                                str(metrics_b["n_trades"])),
        ("Final Value",         f"${metrics_a['final_value']:,.2f}",
                                f"${metrics_b['final_value']:,.2f}"),
        ("Execution Time",      f"{elapsed_a:.1f}s",
                                f"{elapsed_b:.1f}s"),
    ]

    for label, val_a, val_b in rows:
        print(f"  {label:<28} {val_a:>20} {val_b:>20}")

    print()
    print("  " + "-" * 70)
    print(f"  {'SPY Return (benchmark)':<28} {_fmt(metrics_a['spy_return'], '+.2f', '%'):>20}"
          f" {'(same)':>20}")
    print()

    # --- Analysis paragraph ---
    print("  ANALYSIS")
    print("  " + "-" * 70)

    ret_a = metrics_a["total_return"]
    ret_b = metrics_b["total_return"]
    sharpe_a = metrics_a["sharpe_ratio"] or 0.0
    sharpe_b = metrics_b["sharpe_ratio"] or 0.0
    dd_a = abs(metrics_a["max_drawdown"])
    dd_b = abs(metrics_b["max_drawdown"])

    # Determine which run performed better
    if ret_b > ret_a + 1.0:  # >1% difference
        ret_winner = "Run B (LLM-Lite)"
        ret_delta = ret_b - ret_a
    elif ret_a > ret_b + 1.0:
        ret_winner = "Run A (Quant-Only)"
        ret_delta = ret_a - ret_b
    else:
        ret_winner = "Neither (within 1%)"
        ret_delta = abs(ret_b - ret_a)

    parts = []
    parts.append(
        f"  Return comparison: {ret_winner} leads by {ret_delta:.1f}pp."
    )

    if sharpe_b > sharpe_a:
        parts.append(
            f"  Risk-adjusted: LLM-Lite has higher Sharpe ({sharpe_b:.2f} vs {sharpe_a:.2f})."
        )
    elif sharpe_a > sharpe_b:
        parts.append(
            f"  Risk-adjusted: Quant-Only has higher Sharpe ({sharpe_a:.2f} vs {sharpe_b:.2f})."
        )
    else:
        parts.append("  Risk-adjusted: Sharpe ratios are identical.")

    if dd_b < dd_a:
        parts.append(
            f"  Drawdown: LLM-Lite had lower max drawdown ({dd_b:.1f}% vs {dd_a:.1f}%)."
        )
    elif dd_a < dd_b:
        parts.append(
            f"  Drawdown: Quant-Only had lower max drawdown ({dd_a:.1f}% vs {dd_b:.1f}%)."
        )

    if metrics_b["n_trades"] != metrics_a["n_trades"]:
        parts.append(
            f"  Activity: LLM influence changed trade count "
            f"({metrics_a['n_trades']} -> {metrics_b['n_trades']})."
        )

    parts.append("")
    parts.append(
        "  NOTE: Run B uses LLM calls which are non-deterministic. "
        "Results may vary between executions. For formal comparison, "
        "run multiple times and report medians."
    )

    for p in parts:
        print(p)

    print()
    print("=" * 74)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the A/B backtest comparison."""
    end_date = date.today()
    start_date = _subtract_months(end_date, LOOKBACK_MONTHS)

    print()
    print("Covenant Hedge Fund -- A/B Backtest Comparison")
    print("=" * 50)
    print()
    print(f"  Tickers:       {', '.join(TICKERS)}")
    print(f"  Date range:    {start_date} to {end_date}")
    print(f"  Initial cash:  ${INITIAL_CASH:,.2f}")
    print()

    # Check LLM availability for Run B
    has_ollama = False
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            has_ollama = resp.status == 200
    except Exception:
        pass

    if has_ollama:
        print("  LLM provider: Ollama (local)")
    else:
        print("  WARNING: Ollama not available -- quant-only mode.")
        print("  Run B will use neutral LLM signals. Results will match Run A.")
        print("  To enable LLM: install and start Ollama (https://ollama.com).")
    print()

    from src.backtest import BacktestEngine

    # -----------------------------------------------------------------------
    # Run A: Quant-Only
    # -----------------------------------------------------------------------
    print("=" * 50)
    print("RUN A: QUANT-ONLY (5 analysts, no LLM)")
    print("=" * 50)
    print()

    engine_a = BacktestEngine(
        tickers=TICKERS,
        start_date=start_date,
        end_date=end_date,
        initial_cash=INITIAL_CASH,
        show_reasoning=False,
        llm_lite=False,
    )

    t0 = time.time()
    engine_a.run()
    elapsed_a = time.time() - t0

    print()
    print(f"  Run A completed in {elapsed_a:.1f}s")
    print()

    # -----------------------------------------------------------------------
    # Run B: LLM-Lite
    # -----------------------------------------------------------------------
    print("=" * 50)
    print("RUN B: LLM-LITE (5 quant + 4 LLM personas)")
    print("=" * 50)
    print()

    engine_b = BacktestEngine(
        tickers=TICKERS,
        start_date=start_date,
        end_date=end_date,
        initial_cash=INITIAL_CASH,
        show_reasoning=False,
        llm_lite=True,
    )

    t0 = time.time()
    engine_b.run()
    elapsed_b = time.time() - t0

    print()
    print(f"  Run B completed in {elapsed_b:.1f}s")
    print()

    # -----------------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------------
    metrics_a = _extract_metrics(engine_a)
    metrics_b = _extract_metrics(engine_b)

    _print_comparison(metrics_a, metrics_b, elapsed_a, elapsed_b)


if __name__ == "__main__":
    main()
