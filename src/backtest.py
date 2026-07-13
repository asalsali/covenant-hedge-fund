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

from src.agents.parallel import run_analysts_parallel
from src.agents.quant import QUANT_ANALYSTS
from src.agents.crypto import CRYPTO_ANALYSTS, CRYPTO_LLM_ANALYSTS
from src.data.api import (
    clear_cache,
    get_financial_metrics,
    get_insider_trades,
    get_prices,
    search_line_items,
)
from src.data.crypto import cg_get_crypto_metrics, is_crypto, resolve_coin_id
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

# LLM Lite defaults
LLM_LITE_REBALANCE_COUNT = 5  # Number of rebalance dates to run LLM on
LLM_LITE_PERSONAS = [
    "BuffettAnalyst",
    "GrahamAnalyst",
    "DruckenmillerAnalyst",
    "TalebAnalyst",
]

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
        llm_lite: bool = False,
    ) -> None:
        self.tickers = [t.upper() for t in tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.show_reasoning = show_reasoning
        self.llm_lite = llm_lite

        self.portfolio = Portfolio(initial_cash=initial_cash)
        self.analysts = [AnalystClass() for AnalystClass in QUANT_ANALYSTS]

        # Add crypto-specialized quant analysts when any ticker is crypto
        has_crypto = any(is_crypto(t) for t in self.tickers)
        if has_crypto:
            self.analysts.extend(
                AnalystClass() for AnalystClass in CRYPTO_ANALYSTS
            )

        # LLM lite: instantiate the 4 chosen LLM personas
        self.llm_analysts: list = []
        self.llm_rebalance_indices: set[int] = set()
        if llm_lite:
            from src.agents.value import BuffettAnalyst, GrahamAnalyst
            from src.agents.macro import DruckenmillerAnalyst, TalebAnalyst
            self.llm_analysts = [
                BuffettAnalyst(),
                GrahamAnalyst(),
                DruckenmillerAnalyst(),
                TalebAnalyst(),
            ]
            # Add crypto-specialized LLM analysts when any ticker is crypto
            if has_crypto:
                self.llm_analysts.extend(
                    AnalystClass() for AnalystClass in CRYPTO_LLM_ANALYSTS
                )
        # Track LLM signal diffs for the comparison report
        self.llm_signal_log: list[dict[str, Any]] = []

        # Per-rebalance signal and decision history for report persistence
        self.signal_history: dict[str, dict[str, dict[str, Any]]] = {}
        self.decision_history: dict[str, dict[str, dict[str, Any]]] = {}

        # Performance attributes set after run() completes
        self.alpha: float | None = None
        self.spy_return: float | None = None
        self.spy_daily_values: list[tuple[str, float]] = []

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

                if is_crypto(ticker):
                    # Crypto: no traditional fundamentals, fetch crypto metrics instead
                    financial_metrics[ticker] = []
                    insider_trades[ticker] = []
                    line_items[ticker] = []
                    coin_id = resolve_coin_id(ticker)
                    if coin_id:
                        crypto_m = cg_get_crypto_metrics(coin_id)
                        if not hasattr(self, '_crypto_metrics'):
                            self._crypto_metrics: dict[str, dict] = {}
                        self._crypto_metrics[ticker] = crypto_m
                    print(f"  {ticker}: {len(prices)} price bars loaded (crypto)")
                else:
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
        current_date: str = "",
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

        # --- Persist signals ---
        if current_date:
            self.signal_history[current_date] = {
                ticker: {name: {"signal": sig.signal, "confidence": sig.confidence, "reasoning": sig.reasoning}
                         for name, sig in sigs.items()}
                for ticker, sigs in all_signals.items()
            }

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

        # --- Persist decisions ---
        if current_date:
            self.decision_history[current_date] = {
                ticker: {
                    "action": dec["action"],
                    "quantity": dec["quantity"],
                    "weighted_score": dec["weighted_score"],
                    "reasoning": dec["reasoning"],
                }
                for ticker, dec in decisions.items()
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
    # LLM-enhanced analysis day
    # ------------------------------------------------------------------

    def _run_analysis_day_with_llm(
        self,
        active_tickers: list[str],
        market_data: dict[str, dict[str, Any]],
        prices_dict: dict[str, list[float]],
        current_prices: dict[str, float],
        current_date: str,
    ) -> None:
        """Run quant + LLM analysts together using the parallel runner.

        On LLM rebalance days, we run all quant analysts plus the selected
        LLM personas in parallel. The combined signals feed into the same
        quorum/scoring pipeline as quant-only days.

        Also logs the quant-only vs combined signal comparison for the
        final diversity report.
        """
        # Run quant analysts first (fast, no API calls)
        quant_signals: dict[str, dict[str, AnalystSignal]] = {}
        for analyst in self.analysts:
            results = analyst.analyze(active_tickers, market_data)
            for ticker, signal in results.items():
                if ticker not in quant_signals:
                    quant_signals[ticker] = {}
                quant_signals[ticker][analyst.name] = signal

        # Run LLM analysts in parallel
        llm_signals, llm_elapsed = run_analysts_parallel(
            self.llm_analysts,
            active_tickers,
            market_data,
            verbose=True,
        )

        print(f"    LLM analysts completed in {llm_elapsed:.1f}s")

        # Merge: quant + LLM
        all_signals: dict[str, dict[str, AnalystSignal]] = {}
        for ticker in active_tickers:
            all_signals[ticker] = {}
            # Add quant signals
            for name, sig in quant_signals.get(ticker, {}).items():
                all_signals[ticker][name] = sig
            # Add LLM signals
            for name, sig in llm_signals.get(ticker, {}).items():
                all_signals[ticker][name] = sig

        # --- Persist signals ---
        if current_date:
            self.signal_history[current_date] = {
                ticker: {name: {"signal": sig.signal, "confidence": sig.confidence, "reasoning": sig.reasoning}
                         for name, sig in sigs.items()}
                for ticker, sigs in all_signals.items()
            }

        # --- Log signal comparison for diversity report ---
        for ticker in active_tickers:
            quant_only = quant_signals.get(ticker, {})
            llm_only = llm_signals.get(ticker, {})

            # Compute quant-only score
            q_weighted_sum = 0.0
            q_conf_sum = 0.0
            for sig in quant_only.values():
                direction = {"bullish": 1.0, "bearish": -1.0}.get(sig.signal, 0.0)
                conf = sig.confidence / 100.0
                q_weighted_sum += conf * direction
                q_conf_sum += conf
            q_score = q_weighted_sum / q_conf_sum if q_conf_sum > 0 else 0.0

            # Compute combined score
            c_weighted_sum = q_weighted_sum
            c_conf_sum = q_conf_sum
            for sig in llm_only.values():
                direction = {"bullish": 1.0, "bearish": -1.0}.get(sig.signal, 0.0)
                conf = sig.confidence / 100.0
                c_weighted_sum += conf * direction
                c_conf_sum += conf
            c_score = c_weighted_sum / c_conf_sum if c_conf_sum > 0 else 0.0

            self.llm_signal_log.append({
                "date": current_date,
                "ticker": ticker,
                "quant_score": round(q_score, 4),
                "combined_score": round(c_score, 4),
                "score_delta": round(c_score - q_score, 4),
                "quant_signals": {
                    n: {"signal": s.signal, "confidence": s.confidence}
                    for n, s in quant_only.items()
                },
                "llm_signals": {
                    n: {
                        "signal": s.signal,
                        "confidence": s.confidence,
                        "reasoning": s.reasoning[:200] if s.reasoning else "",
                    }
                    for n, s in llm_only.items()
                },
            })

        # --- Risk calculations (identical to quant-only path) ---
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

        # --- Quorum + confidence-weighted synthesis (same logic) ---
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

        # --- Persist decisions ---
        if current_date:
            self.decision_history[current_date] = {
                ticker: {
                    "action": dec["action"],
                    "quantity": dec["quantity"],
                    "weighted_score": dec["weighted_score"],
                    "reasoning": dec["reasoning"],
                }
                for ticker, dec in decisions.items()
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
        mode_label = "LLM-Lite" if self.llm_lite else "Quant-Only"
        print("=" * 70)
        print(f"COVENANT HEDGE FUND -- Backtest Mode ({mode_label})")
        print("=" * 70)
        print(f"  Tickers:    {', '.join(self.tickers)}")
        print(f"  Date range: {self.start_date} to {self.end_date}")
        print(f"  Cash:       ${self.initial_cash:,.2f}")
        print(f"  Rebalance:  every {REBALANCE_INTERVAL} trading days")
        print(f"  Lookback:   {LOOKBACK_WINDOW} trading days")
        if self.llm_lite:
            print(f"  LLM personas: {', '.join(a.name for a in self.llm_analysts)}")
            print(f"  LLM dates:  {LLM_LITE_REBALANCE_COUNT} evenly spaced")
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

        # -- Compute LLM rebalance schedule --
        # First, figure out which day indices are rebalance days
        rebalance_day_indices: list[int] = []
        _counter = REBALANCE_INTERVAL  # Force rebalance on day 1
        for i in range(n_trading_days):
            _counter += 1
            if _counter >= REBALANCE_INTERVAL:
                _counter = 0
                rebalance_day_indices.append(i)

        if self.llm_lite and rebalance_day_indices:
            n_rb = len(rebalance_day_indices)
            n_llm = min(LLM_LITE_REBALANCE_COUNT, n_rb)
            # Evenly spaced indices into the rebalance list
            if n_llm >= n_rb:
                llm_rb_positions = list(range(n_rb))
            else:
                step = (n_rb - 1) / (n_llm - 1) if n_llm > 1 else 0
                llm_rb_positions = [round(i * step) for i in range(n_llm)]
            self.llm_rebalance_indices = {
                rebalance_day_indices[p] for p in llm_rb_positions
            }
            llm_dates = [trading_days[rebalance_day_indices[p]] for p in llm_rb_positions]
            print(f"  LLM rebalance dates: {', '.join(llm_dates)}")
            print(f"  Total rebalance days: {n_rb}, LLM days: {len(self.llm_rebalance_indices)}")
            est_calls = len(self.llm_rebalance_indices) * len(self.llm_analysts) * len(active_tickers)
            print(f"  Estimated LLM calls: ~{est_calls}")

        print()

        # -- Day-by-day iteration --
        days_since_rebalance = REBALANCE_INTERVAL  # Force rebalance on day 1
        rebalance_count = -1  # Will increment to 0 on first rebalance

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

                # Check if this is an LLM rebalance day
                is_llm_day = (
                    self.llm_lite
                    and day_idx in self.llm_rebalance_indices
                )

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
                    md_entry: dict[str, Any] = {
                        "prices": as_of_prices,
                        "financial_metrics": financial_metrics.get(ticker, []),
                        "insider_trades": insider_trades.get(ticker, []),
                        "line_items": line_items.get(ticker, []),
                    }
                    # Attach crypto metrics if available
                    if hasattr(self, '_crypto_metrics') and ticker in self._crypto_metrics:
                        md_entry["crypto_metrics"] = self._crypto_metrics[ticker]
                    market_data[ticker] = md_entry

                analyzable_tickers = [
                    t for t in day_active_tickers if t in market_data
                ]

                if analyzable_tickers:
                    if is_llm_day:
                        print(f"    ** LLM day: {current_date} -- "
                              f"running quant + {len(self.llm_analysts)} LLM personas")
                        self._run_analysis_day_with_llm(
                            analyzable_tickers,
                            market_data,
                            prices_dict,
                            current_prices,
                            current_date,
                        )
                    else:
                        self._run_analysis_day(
                            analyzable_tickers,
                            market_data,
                            prices_dict,
                            current_prices,
                            current_date,
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

                # Build SPY daily values normalized to initial_cash for chart overlay
                for sp in spy_in_range:
                    normalized = (sp["close"] / spy_start) * self.initial_cash
                    self.spy_daily_values.append((sp["date"], normalized))

        # Store alpha and spy_return as engine attributes
        self.spy_return = spy_return
        total_return = metrics.total_return or 0.0
        self.alpha = (total_return - spy_return) if spy_return is not None else None

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

        # -- Print LLM diversity report if applicable --
        if self.llm_lite and self.llm_signal_log:
            self._print_llm_diversity_report()

        # -- Generate HTML report --
        try:
            from src.report import generate_report
            report_data = self.to_report_json(metrics, n_trading_days)
            report_path = generate_report(report_data)
            print(f"\nReport saved to {report_path}")
        except Exception as e:
            print(f"\nWARNING: Could not generate HTML report: {e}")

        return metrics

    # ------------------------------------------------------------------
    # Report data serialization
    # ------------------------------------------------------------------

    def to_report_json(
        self,
        metrics: PerformanceMetrics,
        n_trading_days: int,
    ) -> dict[str, Any]:
        """Serialize the full backtest data into a JSON-serializable dict.

        This is the integration point for the HTML report generator.
        Captures metadata, performance, equity curve, trades, signals,
        decisions, and risk data.
        """
        from datetime import datetime as _dt

        # Collect all analysts (quant + any LLM)
        all_analysts = list(self.analysts) + list(self.llm_analysts)
        analyst_info = [
            {"name": a.name, "domain": a.domain, "uses_llm": a.uses_llm}
            for a in all_analysts
        ]

        total_return = metrics.total_return or 0.0
        final_value = self.initial_cash
        if self.portfolio.daily_values:
            final_value = self.portfolio.daily_values[-1][1]

        # Build SPY lookup for equity curve overlay
        spy_lookup: dict[str, float] = {d: v for d, v in self.spy_daily_values}

        # Equity curve
        equity_curve = [
            {
                "date": d,
                "value": round(v, 2),
                "spy_value": round(spy_lookup.get(d, 0.0), 2) if spy_lookup else None,
            }
            for d, v in self.portfolio.daily_values
        ]

        # Trades
        trades = [
            {
                "date": t.timestamp[:10] if len(t.timestamp) >= 10 else t.timestamp,
                "ticker": t.ticker,
                "action": t.action,
                "quantity": t.quantity,
                "price": round(t.price, 2),
                "notional": round(t.notional, 2),
                "reasoning": t.reasoning,
            }
            for t in self.portfolio.trades
        ]

        # Risk stub (volatility and correlation are computed per-day, store last)
        risk: dict[str, Any] = {"volatility": {}, "correlation": {}}

        return {
            "metadata": {
                "run_date": _dt.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "tickers": self.tickers,
                "mode": "llm-lite" if self.llm_lite else "quant-only",
                "start_date": str(self.start_date),
                "end_date": str(self.end_date),
                "initial_cash": self.initial_cash,
                "trading_days": n_trading_days,
                "analyst_count": len(all_analysts),
                "analysts": analyst_info,
            },
            "performance": {
                "total_return": total_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "max_drawdown_date": metrics.max_drawdown_date,
                "annualized_return": metrics.annualized_return,
                "alpha_vs_spy": self.alpha,
                "spy_return": self.spy_return,
                "final_value": round(final_value, 2),
            },
            "equity_curve": equity_curve,
            "trades": trades,
            "signals": self.signal_history,
            "decisions": self.decision_history,
            "risk": risk,
        }

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

    # ------------------------------------------------------------------
    # LLM Diversity Report
    # ------------------------------------------------------------------

    def _print_llm_diversity_report(self) -> None:
        """Print a detailed report comparing quant-only vs LLM-enhanced signals."""
        print()
        print("=" * 70)
        print("LLM SIGNAL DIVERSITY REPORT")
        print("=" * 70)
        print()

        # Group by date
        dates_seen: list[str] = []
        for entry in self.llm_signal_log:
            if entry["date"] not in dates_seen:
                dates_seen.append(entry["date"])

        total_changes = 0
        total_observations = 0
        direction_changes = 0  # Cases where LLM flipped the decision direction

        for dt in dates_seen:
            print(f"  Date: {dt}")
            print("  " + "-" * 66)
            entries = [e for e in self.llm_signal_log if e["date"] == dt]

            for entry in entries:
                ticker = entry["ticker"]
                q_score = entry["quant_score"]
                c_score = entry["combined_score"]
                delta = entry["score_delta"]
                total_observations += 1

                # Did the LLM change anything?
                changed = abs(delta) > 0.01
                if changed:
                    total_changes += 1

                # Did the direction actually flip?
                q_action = "buy" if q_score > SCORE_THRESHOLD else (
                    "sell" if q_score < -SCORE_THRESHOLD else "hold"
                )
                c_action = "buy" if c_score > SCORE_THRESHOLD else (
                    "sell" if c_score < -SCORE_THRESHOLD else "hold"
                )
                flipped = q_action != c_action
                if flipped:
                    direction_changes += 1

                flip_marker = " << DECISION CHANGED" if flipped else ""
                delta_marker = f" (delta {delta:+.4f})" if changed else " (no change)"

                print(f"    {ticker}: quant={q_score:+.4f} -> combined={c_score:+.4f}"
                      f"{delta_marker}{flip_marker}")

                # Show individual LLM signals
                for name, sig_data in entry["llm_signals"].items():
                    arrow = {"bullish": "+", "bearish": "-", "neutral": "="}[sig_data["signal"]]
                    reasoning_preview = sig_data.get("reasoning", "")[:80]
                    print(f"      {name}: {arrow}{sig_data['confidence']} "
                          f"-- {reasoning_preview}")
            print()

        # Summary
        print("  " + "=" * 66)
        print("  DIVERSITY SUMMARY")
        print("  " + "=" * 66)
        print(f"  Total observations (ticker x date):  {total_observations}")
        print(f"  Score changed by LLM:                {total_changes}/{total_observations}")
        pct_changed = (total_changes / total_observations * 100) if total_observations > 0 else 0
        print(f"  Change rate:                         {pct_changed:.0f}%")
        print(f"  Decision direction flipped:          {direction_changes}/{total_observations}")
        print()

        # Consistency analysis: are LLMs consistently bearish/bullish moderators?
        bearish_shifts = sum(1 for e in self.llm_signal_log if e["score_delta"] < -0.01)
        bullish_shifts = sum(1 for e in self.llm_signal_log if e["score_delta"] > 0.01)
        neutral_shifts = total_observations - bearish_shifts - bullish_shifts

        print(f"  LLM directional bias:")
        print(f"    Bearish shifts (LLM moderated bullishness): {bearish_shifts}")
        print(f"    Bullish shifts (LLM boosted bullishness):   {bullish_shifts}")
        print(f"    Neutral (no material change):               {neutral_shifts}")

        if bearish_shifts > bullish_shifts * 2:
            print("  --> LLMs are consistently bearish moderators")
        elif bullish_shifts > bearish_shifts * 2:
            print("  --> LLMs are consistently bullish amplifiers")
        elif bearish_shifts > 0 or bullish_shifts > 0:
            print("  --> LLMs show mixed directional influence (context-dependent)")
        else:
            print("  --> LLMs had no material effect on signals")

        print()
        print("=" * 70)
