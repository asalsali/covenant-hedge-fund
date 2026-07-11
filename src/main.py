"""Covenant Hedge Fund -- Entry Point.

Accepts tickers and date range, spawns analyst agents across three
epoch domains, collects signals, and produces portfolio decisions.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Covenant Hedge Fund -- AI-governed portfolio analysis",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Ticker symbols to analyze (e.g., AAPL MSFT GOOGL)",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=None,
        help="Analysis start date (YYYY-MM-DD). Defaults to 3 months ago.",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=None,
        help="Analysis end date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run in backtest mode over the date range.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100_000.0,
        help="Initial portfolio cash (default: 100000).",
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Display analyst reasoning in output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the Covenant Hedge Fund."""
    args = parse_args(argv)

    tickers = [t.upper() for t in args.tickers]
    end = args.end_date or date.today()
    start = args.start_date or date(end.year, end.month - 3, end.day)

    print("Covenant Hedge Fund initialized.")
    print(f"  Tickers:    {', '.join(tickers)}")
    print(f"  Date range: {start} to {end}")
    print(f"  Cash:       ${args.initial_cash:,.2f}")
    print(f"  Mode:       {'backtest' if args.backtest else 'analysis'}")
    print()

    # TODO: Initialize portfolio state
    # TODO: Spawn epoch containers (value, quant, macro)
    # TODO: Spawn analyst agents within each epoch container
    # TODO: Collect analyst memos
    # TODO: Apply COMPLIANCE.md risk rules
    # TODO: Make portfolio decisions
    # TODO: Write exit report with decision graph

    print("Spawning analysts...")
    print("  Value domain:  Buffett, Graham, Munger, Pabrai, Fisher, Damodaran")
    print("  Quant domain:  Technicals, Fundamentals, Valuation, Growth, Sentiment")
    print("  Macro domain:  Druckenmiller, Burry, Wood, Lynch, Ackman, Taleb, News Sentiment")
    print()
    print("[placeholder -- analyst execution not yet implemented]")


if __name__ == "__main__":
    main()
