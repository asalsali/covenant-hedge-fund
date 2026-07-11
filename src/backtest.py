"""Backtest engine for the Covenant Hedge Fund.

Iterates over historical business days, running the same quant analysis
pipeline used in single-shot mode on each rebalancing day. Records daily
portfolio values and computes performance metrics at the end.

Usage:
    python -m src.main --tickers AAPL MSFT --backtest --start-date 2025-01-01
"""

from __future__ import annotations

from datetime import date
from typing import Any

from src.agents.quant import QUANT_ANALYSTS
from src.data.api import (
    clear_cache,
    get_financial_metrics,
    get_insider_trades,
    get_prices,
    search_line_items,
)
from src.models import AnalystSignal, PerformanceMetrics
from src.portfolio import Portfolio
from src.risk import (
    compute_allowed_actions,
    compute_correlation,
    compute_position_limit,
    compute_volatility,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOOKBACK_WINDOW = 60        # Trading days needed before first trade
REBALANCE_INTERVAL = 5      # Run analysis every N trading days
QUORUM_THRESHOLD = 3        # Minimum non-neutral signals for action
SCORE_THRESHOLD = 0.3       # Normalized score threshold for action
BENCHMARK_TICKER = "SPY"

LINE_ITEMS_TO_FETCH = [
    "revenue", "net_income", "free_cash_flow",
    "operating_cash_flow", "outstanding_shares",
]


class BacktestEngine:
    """Historical backtest engine using the quant analysis pipeline.

    Fetches all price data upfront, then walks forward day-by-day,
    running the full analyst -> risk -> decision -> execution pipeline
    on each rebalance day.

    Attributes:
        tickers: List of ticker symbols to trade.
        start_date: Backtest start date (inclusive, after lookback).
        end_date: Backtest end date (inclusive).
        initial_cash: Starting portfolio cash.
        show_reasoning: Whether to print analyst reasoning.
    """

    def __init__(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
        initial_cash: float = 100_000.0,
        show_reasoning: bool = False,
    ) -> None:
        self.tickers = [t.upper() for t in tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.show_reasoning = show_reasoning

        self.portfolio = Portfolio(initial_cash=initial_cash)
        self.analysts = [AnalystClass() for AnalystClass in QUANT_ANALYSTS]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _fetch_all_data(
        self,
    ) -> tuple[
        dict[str, list[dict[str, Any]]],       # all_prices
        dict[str, list[dict[str, Any]]],        # financial_metrics
        dict[str, list[dict[str, Any]]],        # insider_trades
        dict[str, list[dict[str, Any]]],        # line_items
        list[dict[str, Any]],                   # spy_prices
    ]:
        """Fetch all market data upfront for the full date range.

        Returns price data for all tickers plus SPY benchmark, and
        fundamental data (fetched once since it changes infrequently).
        """
        # CF-COMP-030: clear cache at start of each run
        clear_cache()

        all_prices: dict[str, list[dict[str, Any]]] = {}
        financial_metrics: dict[str, list[dict[str, Any]]] = {}
        insider_trades: dict[str, list[dict[str, Any]]] = {}
        line_items: dict[str, list[dict[str, Any]]] = {}

        for ticker in self.tickers:
            try:
                prices = get_prices(ticker, self.start_date, self.end_date)
                if not prices:
                    print(f"  WARNING: No price data for {ticker}, skipping.")
                    continue
                all_prices[ticker] = prices

                # Fundamentals fetched once (quarterly/annual, not daily)
                financial_metrics[ticker] = get_financial_metrics(
                    ticker, self.end_date, period="annual", limit=5,
                )
                insider_trades[ticker] = get_insider_trades(
                    ticker, self.end_date,
                )
                line_items[ticker] = search_line_items(
                    ticker, LINE_ITEMS_TO_FETCH, self.end_date,
                    period="annual", limit=5,
                )

                print(f"  {ticker}: {len(prices)} price bars loaded")

            except Exception as e:
                print(f"  WARNING: Failed to fetch data for {ticker}: {e}")

        # Fetch SPY benchmark
        spy_prices: list[dict[str, Any]] = []
        try:
            spy_prices = get_prices(BENCHMARK_TICKER, self.start_date, self.end_date)
            print(f"  {BENCHMARK_TICKER}: {len(spy_prices)} price bars (benchmark)")
        except Exception as e:
            print(f"  WARNING: Could not fetch SPY benchmark: {e}")

        return all_prices, financial_metrics, insider_trades, line_items, spy_prices

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_business_days(
        all_prices: dict[str, list[dict[str, Any]]],
    ) -> list[str]:
        """Extract sorted list of unique business days across all tickers."""
        dates_set: set[str] = set()
        for prices in all_prices.values():
            for bar in prices:
                dates_set.add(bar["date"])
        return sorted(dates_set)

    @staticmethod
    def _prices_up_to(
        prices: list[dict[str, Any]],
        as_of_date: str,
    ) -> list[dict[str, Any]]:
        """Return price bars up to and including as_of_date."""
        return [p for p in prices if p["date"] <= as_of_date]

    @staticmethod
    def _close_prices_list(
        prices: list[dict[str, Any]],
    ) -> list[float]:
        """Extract close prices from price bar list."""
        return [
            p["close"] for p in prices
            if p.get("close") is not None
        ]

    # ------------------------------------------------------------------
    # Analysis pipeline (mirrors main.py single-shot)
    # ------------------------------------------------------------------

    def _run_analysis_day(
        self,
        active_tickers: list[str],
        market_data: dict[str, dict[str, Any]],
        prices_dict: dict[str, list[float]],
        current_prices: dict[str, float],
    ) -> None:
        """Run the full analysis-to-execution pipeline for one day."""
        # --- Quant analysts ---
        all_signals: dict[str, dict[str, AnalystSignal]] = {}
        for analyst in self.analysts:
            results = analyst.analyze(active_tickers, market_data)
            for ticker, signal in results.items():
                if ticker not in all_signals:
                    all_signals[ticker] = {}
                all_signals[ticker][analyst.name] = signal

        # --- Risk calculations ---
        vol_metrics = compute_volatility(prices_dict)
        corr_metrics = compute_correlation(prices_dict)

        portfolio_value = self.portfolio.compute_portfolio_value(current_prices)

        position_limits: dict[str, Any] = {}
        allowed_actions: dict[str, list[str]] = {}

        for ticker in active_tickers:
            if ticker not in vol_metrics:
                continue
            limit = compute_position_limit(
                ticker, portfolio_value, vol_metrics[ticker], corr_metrics,
            )
            position_limits[ticker] = limit
            allowed = compute_allowed_actions(
                ticker, self.portfolio.state, limit,
                current_prices.get(ticker, 0),
            )
            allowed_actions[ticker] = allowed

        # --- Quorum + confidence-weighted synthesis ---
        decisions: dict[str, dict[str, Any]] = {}

        for ticker in active_tickers:
            signals = all_signals.get(ticker, {})
            non_neutral = [
                (name, sig) for name, sig in signals.items()
                if sig.signal != "neutral"
            ]

            if len(non_neutral) < QUORUM_THRESHOLD:
                decisions[ticker] = {
                    "action": "hold", "quantity": 0,
                    "reasoning": f"Quorum not met: {len(non_neutral)}/{QUORUM_THRESHOLD}",
                    "weighted_score": 0.0,
                }
                continue

            weighted_sum = 0.0
            confidence_sum = 0.0
            for _name, sig in signals.items():
                direction = {"bullish": 1.0, "bearish": -1.0}.get(sig.signal, 0.0)
                conf = sig.confidence / 100.0
                weighted_sum += conf * direction
                confidence_sum += conf

            normalized_score = (
                weighted_sum / confidence_sum if confidence_sum > 0 else 0.0
            )

            allowed = allowed_actions.get(ticker, ["hold"])
            price = current_prices.get(ticker, 0)

            if normalized_score > SCORE_THRESHOLD and "buy" in allowed:
                pl = position_limits.get(ticker)
                if pl and price > 0:
                    target_notional = pl.max_notional * 0.5
                    quantity = max(1, int(target_notional / price))
                else:
                    quantity = 0
                decisions[ticker] = {
                    "action": "buy", "quantity": quantity,
                    "reasoning": f"Bullish consensus (score={normalized_score:+.2f})",
                    "weighted_score": normalized_score,
                }
            elif normalized_score < -SCORE_THRESHOLD:
                if "sell" in allowed:
                    pos = self.portfolio.state.positions.get(ticker)
                    quantity = pos.long_shares if pos else 0
                    decisions[ticker] = {
                        "action": "sell", "quantity": quantity,
                        "reasoning": f"Bearish consensus (score={normalized_score:+.2f})",
                        "weighted_score": normalized_score,
                    }
                elif "short" in allowed:
                    pl = position_limits.get(ticker)
                    if pl and price > 0:
                        target_notional = pl.max_notional * 0.5
                        quantity = max(1, int(target_notional / price))
                    else:
                        quantity = 0
                    decisions[ticker] = {
                        "action": "short", "quantity": quantity,
                        "reasoning": f"Bearish consensus (score={normalized_score:+.2f})",
                        "weighted_score": normalized_score,
                    }
                else:
                    decisions[ticker] = {
                        "action": "hold", "quantity": 0,
                        "reasoning": f"Bearish (score={normalized_score:+.2f}) but no sell/short allowed",
                        "weighted_score": normalized_score,
                    }
            else:
                decisions[ticker] = {
                    "action": "hold", "quantity": 0,
                    "reasoning": f"Score within threshold (score={normalized_score:+.2f})",
                    "weighted_score": normalized_score,
                }

        # --- Execute trades ---
        for ticker in active_tickers:
            dec = decisions.get(ticker)
            if not dec or dec["action"] == "hold" or dec["quantity"] == 0:
                continue

            price = current_prices.get(ticker, 0)
            if price <= 0:
                continue

            try:
                if dec["action"] == "buy":
                    self.portfolio.execute_buy(
                        ticker, dec["quantity"], price,
                        reasoning=dec["reasoning"],
                    )
                elif dec["action"] == "sell":
                    self.portfolio.execute_sell(
                        ticker, dec["quantity"], price,
                        reasoning=dec["reasoning"],
                    )
                elif dec["action"] == "short":
                    self.portfolio.execute_short(
                        ticker, dec["quantity"], price,
                        reasoning=dec["reasoning"],
                    )
                elif dec["action"] == "cover":
                    self.portfolio.execute_cover(
                        ticker, dec["quantity"], price,
                        reasoning=dec["reasoning"],
                    )
            except ValueError:
                pass

        # --- Show reasoning if requested ---
        if self.show_reasoning:
            for ticker in active_tickers:
                dec = decisions.get(ticker, {})
                if dec.get("action") != "hold":
                    print(f"    {ticker}: {dec.get('action', 'hold').upper()} "
                          f"{dec.get('quantity', 0)} "
                          f"({dec.get('reasoning', '')})")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> PerformanceMetrics:
        """Execute the full backtest."""
        print("=" * 70)
        print("COVENANT HEDGE FUND -- Backtest Mode")
        print("=" * 70)
        print(f"  Tickers:    {', '.join(self.tickers)}")
        print(f"  Date range: {self.start_date} to {self.end_date}")
        print(f"  Cash:       ${self.initial_cash:,.2f}")
        print(f"  Rebalance:  every {REBALANCE_INTERVAL} trading days")
        print(f"  Lookback:   {LOOKBACK_WINDOW} trading days")
        print()

        # -- Fetch all data upfront --
        print("[1/3] Fetching market data...")
        (
            all_prices,
            financial_metrics,
            insider_trades,
            line_items,
            spy_prices,
        ) = self._fetch_all_data()

        # Remove tickers with no price data
        active_tickers = [t for t in self.tickers if t in all_prices]
        if not active_tickers:
            print("\nERROR: No valid price data for any ticker. Aborting.")
            return PerformanceMetrics()

        print()

        # -- Build trading calendar --
        business_days = self._extract_business_days(all_prices)
        total_days = len(business_days)

        if total_days <= LOOKBACK_WINDOW:
            print(
                f"\nERROR: Only {total_days} trading days available, "
                f"need at least {LOOKBACK_WINDOW + 1} "
                f"({LOOKBACK_WINDOW} lookback + 1 trading day). "
                f"Try an earlier --start-date."
            )
            return PerformanceMetrics()

        trading_days = business_days[LOOKBACK_WINDOW:]
        n_trading_days = len(trading_days)

        print(f"[2/3] Running backtest over {n_trading_days} trading days "
              f"(skipping first {LOOKBACK_WINDOW} for lookback)...")
        print()

        # -- Day-by-day iteration --
        days_since_rebalance = REBALANCE_INTERVAL  # Force rebalance on day 1

        for day_idx, current_date in enumerate(trading_days):
            # Progress reporting every 10 days
            if day_idx % 10 == 0:
                print(f"  Backtesting day {day_idx + 1}/{n_trading_days}... "
                      f"({current_date})")

            days_since_rebalance += 1

            # Build current prices for this day
            current_prices: dict[str, float] = {}
            day_active_tickers: list[str] = []

            for ticker in active_tickers:
                as_of = self._prices_up_to(all_prices[ticker], current_date)
                if as_of:
                    current_prices[ticker] = as_of[-1]["close"]
                    day_active_tickers.append(ticker)

            if not day_active_tickers:
                continue

            # Rebalance check: run analysis every REBALANCE_INTERVAL days
            if days_since_rebalance >= REBALANCE_INTERVAL:
                days_since_rebalance = 0

                # Build as-of market_data and prices_dict
                market_data: dict[str, dict[str, Any]] = {}
                prices_dict: dict[str, list[float]] = {}

                for ticker in day_active_tickers:
                    as_of_prices = self._prices_up_to(
                        all_prices[ticker], current_date,
                    )
                    closes = self._close_prices_list(as_of_prices)

                    if len(closes) < LOOKBACK_WINDOW + 1:
                        continue

                    prices_dict[ticker] = closes
                    market_data[ticker] = {
                        "prices": as_of_prices,
                        "financial_metrics": financial_metrics.get(ticker, []),
                        "insider_trades": insider_trades.get(ticker, []),
                        "line_items": line_items.get(ticker, []),
                    }

                analyzable_tickers = [
                    t for t in day_active_tickers if t in market_data
                ]

                if analyzable_tickers:
                    self._run_analysis_day(
                        analyzable_tickers,
                        market_data,
                        prices_dict,
                        current_prices,
                    )

            # Record daily portfolio value (every day, not just rebalance)
            self.portfolio.record_daily_value(current_date, current_prices)

        print()

        # -- Compute performance --
        print("[3/3] Computing performance metrics...")
        metrics = self.portfolio.compute_performance()

        # -- SPY benchmark return --
        spy_return: float | None = None
        if spy_prices and len(spy_prices) >= 2:
            first_trade_date = trading_days[0]
            last_trade_date = trading_days[-1]

            spy_in_range = [
                p for p in spy_prices
                if first_trade_date <= p["date"] <= last_trade_date
                and p.get("close") is not None
            ]
            if len(spy_in_range) >= 2:
                spy_start = spy_in_range[0]["close"]
                spy_end = spy_in_range[-1]["close"]
                spy_return = (spy_end - spy_start) / spy_start

        # -- Final portfolio value --
        final_value = self.initial_cash
        if self.portfolio.daily_values:
            final_value = self.portfolio.daily_values[-1][1]

        # -- Realized P&L --
        realized_pnl = self.portfolio.state.total_realized_gains

        # -- Print report --
        self._print_report(
            metrics=metrics,
            n_trading_days=n_trading_days,
            final_value=final_value,
            spy_return=spy_return,
            realized_pnl=realized_pnl,
        )

        return metrics

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _print_report(
        self,
        metrics: PerformanceMetrics,
        n_trading_days: int,
        final_value: float,
        spy_return: float | None,
        realized_pnl: float,
    ) -> None:
        """Print the formatted backtest performance report."""
        total_return = metrics.total_return or 0.0
        ann_return = metrics.annualized_return or 0.0
        sharpe = metrics.sharpe_ratio
        sortino = metrics.sortino_ratio
        max_dd = metrics.max_drawdown or 0.0
        max_dd_date = metrics.max_drawdown_date or "N/A"
        n_trades = len(self.portfolio.trades)

        print()
        print("=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        print(f"  Period:           {self.start_date} to {self.end_date} "
              f"({n_trading_days} trading days)")
        print(f"  Initial capital:  ${self.initial_cash:,.2f}")
        print(f"  Final value:      ${final_value:,.2f}")
        print()
        print("  PERFORMANCE")
        print("  " + "-" * 66)
        print(f"  Total return:      {total_return:+.2%}")
        print(f"  Annualized return: {ann_return:+.2%}")
        print(f"  Sharpe ratio:      {sharpe:.4f}" if sharpe is not None
              else "  Sharpe ratio:      N/A")
        print(f"  Sortino ratio:     {sortino:.4f}" if sortino is not None
              else "  Sortino ratio:     N/A")
        print(f"  Max drawdown:      {max_dd:.2%} (on {max_dd_date})")
        print()
        print("  VS BENCHMARK (SPY)")
        print("  " + "-" * 66)
        if spy_return is not None:
            alpha = total_return - spy_return
            print(f"  SPY return:        {spy_return:+.2%}")
            print(f"  Alpha:             {alpha:+.2%}")
        else:
            print("  SPY return:        N/A (benchmark data unavailable)")
            print("  Alpha:             N/A")
        print()
        print("  TRADING SUMMARY")
        print("  " + "-" * 66)
        print(f"  Total trades:      {n_trades}")
        print(f"  Realized P&L:      ${realized_pnl:+,.2f}")
        print()
        print("=" * 70)
