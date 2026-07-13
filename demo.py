"""Covenant Hedge Fund -- Zero-Config Demo.

Run with: python demo.py
No arguments, no API keys, no configuration needed.
"""

from __future__ import annotations

import sys
import time
from datetime import date

# Ensure src is importable when running from project root
sys.path.insert(0, ".")


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


def main() -> None:
    print()
    print("Covenant Hedge Fund -- Quick Demo")
    print("=" * 50)
    print()
    print("Running 6-month backtest on AAPL, MSFT, NVDA")
    print("(quant-only, no API keys needed)")
    print()
    print("Tip: crypto tickers are also supported. Try:")
    print("  python -m src.main --tickers BTC ETH SOL --backtest --start-date 2025-01-01")
    print()

    tickers = ["AAPL", "MSFT", "NVDA"]
    end_date = date.today()
    start_date = _subtract_months(end_date, 6)

    from src.backtest import BacktestEngine

    engine = BacktestEngine(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_cash=100_000.0,
        show_reasoning=False,
    )

    t0 = time.time()
    engine.run()
    elapsed = time.time() - t0

    print()
    print(f"  Completed in {elapsed:.1f}s")

    # --- Attempt HTML report generation (graceful if report.py not ready) ---
    try:
        from src.report import generate_report  # type: ignore[import-not-found]

        import os
        os.makedirs("reports", exist_ok=True)

        metrics = engine.portfolio.compute_performance()
        n_days = len(engine.portfolio.daily_values)
        report_data = engine.to_report_json(metrics, n_days)
        report_path = generate_report(report_data)
        print()
        print(f"  Report saved to {report_path}")
        print("  Open it in your browser to see the full analysis.")
    except (ImportError, AttributeError):
        # report.py or to_report_json() not available yet -- that's fine
        pass

    print()


if __name__ == "__main__":
    main()
