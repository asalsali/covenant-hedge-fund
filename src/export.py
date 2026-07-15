"""Export backtest results to JSON for the web app dashboard.

Transforms BacktestEngine.to_report_json() output into the shape
expected by docs/app.html, then writes it to docs/report-data.json.

Usage:
    from src.export import export_report_json
    export_report_json(engine, metrics, n_trading_days)
"""

from __future__ import annotations

import json
import os
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest import BacktestEngine
    from src.models import PerformanceMetrics


def _to_display_pct(val: float | None) -> float | None:
    """Convert decimal ratio (0.7329) to display percentage (73.29)."""
    if val is None:
        return None
    return round(val * 100, 2)


def _build_positions(engine: "BacktestEngine") -> dict[str, Any]:
    """Build the positions dict the web app expects from portfolio state."""
    current_prices = getattr(engine, '_last_current_prices', {})
    positions: dict[str, Any] = {}
    for ticker, pos in engine.portfolio.state.positions.items():
        price = current_prices.get(ticker, 0.0)
        if pos.long_shares > 0:
            positions[ticker] = {
                "shares": pos.long_shares,
                "avg_price": round(pos.avg_long_cost, 2),
                "current_price": round(price, 2),
                "value": round(pos.long_shares * price, 2),
            }
        elif pos.short_shares > 0:
            positions[ticker] = {
                "shares": -pos.short_shares,
                "avg_price": round(pos.avg_short_cost, 2),
                "current_price": round(price, 2),
                "value": round(pos.short_shares * price, 2),
            }
    return positions


def _transform_for_webapp(report: dict[str, Any], engine: "BacktestEngine") -> dict[str, Any]:
    """Transform to_report_json() output into the web app DATA shape."""
    perf = report["performance"]

    webapp_data: dict[str, Any] = {
        "metadata": report["metadata"],
        "performance": {
            "total_return": _to_display_pct(perf.get("total_return")),
            "sharpe_ratio": round(perf["sharpe_ratio"], 2) if perf.get("sharpe_ratio") is not None else None,
            "sortino_ratio": round(perf["sortino_ratio"], 2) if perf.get("sortino_ratio") is not None else None,
            "max_drawdown": _to_display_pct(perf.get("max_drawdown")),
            "max_drawdown_date": perf.get("max_drawdown_date"),
            "annualized_return": _to_display_pct(perf.get("annualized_return")),
            "alpha_vs_spy": _to_display_pct(perf.get("alpha_vs_spy")),
            "spy_return": _to_display_pct(perf.get("spy_return")),
            "final_value": perf.get("final_value"),
            "total_transaction_costs": perf.get("total_transaction_costs"),
        },
        "positions": _build_positions(engine),
        "equity_curve": report["equity_curve"],
        "trades": report["trades"],
        "signals": report["signals"],
    }

    # Include CAR event study results if available
    if report.get("car"):
        webapp_data["car"] = report["car"]

    return webapp_data


def export_report_json(
    engine: "BacktestEngine",
    metrics: "PerformanceMetrics",
    n_trading_days: int,
) -> str:
    """Export backtest results to docs/report-data.json.

    Args:
        engine: The completed BacktestEngine instance.
        metrics: The PerformanceMetrics returned by engine.run().
        n_trading_days: Number of trading days in the backtest.

    Returns:
        Path to the written JSON file.
    """
    report = engine.to_report_json(metrics, n_trading_days)
    webapp_data = _transform_for_webapp(report, engine)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(project_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    output_path = os.path.join(docs_dir, "report-data.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(webapp_data, f, indent=2, default=str)

    return os.path.relpath(output_path, project_root)
