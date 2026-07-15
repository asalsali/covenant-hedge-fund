"""Covenant Hedge Fund -- Entry Point.

Accepts tickers and date range, spawns analyst agents across three
epoch domains, collects signals, and produces portfolio decisions.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Covenant Hedge Fund -- AI-governed portfolio analysis",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Ticker symbols to analyze (e.g., AAPL MSFT GOOGL). Crypto supported: BTC ETH SOL etc.",
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
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run analysts sequentially instead of in parallel (default: parallel).",
    )
    parser.add_argument(
        "--llm-lite",
        action="store_true",
        help=(
            "Enable lite LLM backtest: run 4 LLM personas (Buffett, Graham, "
            "Druckenmiller, Taleb) on 5 evenly-spaced rebalance dates. "
            "Requires --backtest. ~60 LLM calls (free via Ollama)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Ollama model to use (e.g., phi4:14b, qwen2.5:32b-instruct). "
            "Overrides OLLAMA_MODEL env var and auto-detection. "
            "Default: auto-selects best model already pulled in Ollama."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the disk-based LLM response cache (forces fresh calls).",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help=(
            "Commission rate per trade as a decimal (default: 0.001 = 10 bps). "
            "Set to 0 for zero-commission backtests."
        ),
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0005,
        help=(
            "Slippage rate per trade as a decimal (default: 0.0005 = 5 bps). "
            "Set to 0 to disable slippage modeling."
        ),
    )
    parser.add_argument(
        "--edge-triggered",
        action="store_true",
        help=(
            "Enable edge-triggered signal filtering in backtest mode. "
            "Prevents overlapping positions: once a position is opened, "
            "subsequent same-direction signals are skipped until the "
            "position is closed or the signal reverses."
        ),
    )
    return parser.parse_args(argv)


def _subtract_months(d: date, months: int) -> date:
    """Subtract months from a date, handling edge cases."""
    month = d.month - months
    year = d.year
    while month <= 0:
        month += 12
        year -= 1
    # Clamp day to valid range for target month
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    day = min(d.day, max_day)
    return date(year, month, day)


def main(argv: list[str] | None = None) -> None:
    """Run the Covenant Hedge Fund."""
    args = parse_args(argv)

    # Apply --model override before any LLM calls
    if args.model:
        from src.llm import set_model
        set_model(args.model)

    # Apply --no-cache flag
    if args.no_cache:
        from src.llm import set_cache_enabled
        set_cache_enabled(False)

    if args.backtest:
        from src.backtest import BacktestEngine

        tickers = [t.upper() for t in args.tickers]
        end_date = args.end_date or date.today()
        start_date = args.start_date or _subtract_months(end_date, 3)

        engine = BacktestEngine(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            initial_cash=args.initial_cash,
            show_reasoning=args.show_reasoning,
            llm_lite=args.llm_lite,
            commission_rate=args.commission,
            slippage_rate=args.slippage,
            edge_triggered=args.edge_triggered,
        )
        engine.run()
        return

    if args.llm_lite:
        print("ERROR: --llm-lite requires --backtest mode.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 1. Initialize
    # -------------------------------------------------------------------------
    tickers = [t.upper() for t in args.tickers]
    end_date = args.end_date or date.today()
    start_date = args.start_date or _subtract_months(end_date, 3)

    print("=" * 70)
    print("COVENANT HEDGE FUND -- Analysis Run")
    print("=" * 70)
    print(f"  Tickers:    {', '.join(tickers)}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Cash:       ${args.initial_cash:,.2f}")

    from src.llm import get_active_model, _check_ollama
    if _check_ollama():
        print(f"  LLM model:  {get_active_model()}")
    else:
        print("  LLM model:  (none -- quant-only mode)")
    print()

    # CF-COMP-030: clear data cache at start of each run
    from src.data.api import (
        get_prices,
        get_financial_metrics,
        get_insider_trades,
        search_line_items,
        clear_cache,
    )
    clear_cache()

    # -------------------------------------------------------------------------
    # 2. Fetch market data
    # -------------------------------------------------------------------------
    from src.data.api import DataFetchError
    from src.data.crypto import cg_get_crypto_metrics, is_crypto, resolve_coin_id
    from src.data.defillama import get_tvl_for_ticker

    print("[1/6] Fetching market data...")

    LINE_ITEMS_TO_FETCH = [
        "revenue", "net_income", "free_cash_flow",
        "operating_cash_flow", "outstanding_shares",
    ]

    market_data: dict[str, dict[str, Any]] = {}
    failed_tickers: list[str] = []

    for ticker in tickers:
        try:
            prices = get_prices(ticker, start_date, end_date)
            if not prices:
                print(f"  WARNING: No price data for {ticker}, skipping.")
                failed_tickers.append(ticker)
                continue

            if is_crypto(ticker):
                # Crypto: no traditional fundamentals, fetch crypto metrics
                md_entry: dict[str, Any] = {
                    "prices": prices,
                    "financial_metrics": [],
                    "insider_trades": [],
                    "line_items": [],
                }
                coin_id = resolve_coin_id(ticker)
                if coin_id:
                    crypto_m = cg_get_crypto_metrics(coin_id)
                    if crypto_m:
                        md_entry["crypto_metrics"] = crypto_m
                # DeFi Llama TVL data
                tvl_data = get_tvl_for_ticker(ticker)
                if tvl_data:
                    md_entry["defi_tvl"] = tvl_data
                # Fear & Greed Index (market-wide, cached)
                from src.data.api import get_fear_greed
                fg_data = get_fear_greed()
                if fg_data.get("current_value") is not None:
                    md_entry["fear_greed"] = fg_data
                market_data[ticker] = md_entry
                print(f"  {ticker}: {len(prices)} price bars (crypto)")
            else:
                financial_metrics = get_financial_metrics(
                    ticker, end_date, period="annual", limit=5,
                )
                insider_trades = get_insider_trades(ticker, end_date)
                line_items = search_line_items(
                    ticker, LINE_ITEMS_TO_FETCH, end_date,
                    period="annual", limit=5,
                )

                market_data[ticker] = {
                    "prices": prices,
                    "financial_metrics": financial_metrics,
                    "insider_trades": insider_trades,
                    "line_items": line_items,
                }
                print(f"  {ticker}: {len(prices)} price bars, "
                      f"{len(financial_metrics)} metric periods, "
                      f"{len(insider_trades)} insider trades")

        except DataFetchError as e:
            print(f"  ERROR [DataFetchError]: {e}")
            failed_tickers.append(ticker)
        except Exception as e:
            print(f"  WARNING: Failed to fetch data for {ticker}: {e}")
            failed_tickers.append(ticker)

    # Remove failed tickers from the active list
    active_tickers = [t for t in tickers if t not in failed_tickers]

    if not active_tickers:
        print("\nERROR: No valid market data for any ticker. Aborting.")
        return

    print()

    # -------------------------------------------------------------------------
    # 3. Run analysts: LLM-first pipeline (quant -> evidence -> LLM -> quorum)
    # -------------------------------------------------------------------------
    from src.agents.quant import QUANT_ANALYSTS
    from src.agents.value import VALUE_ANALYSTS
    from src.agents.macro import MACRO_ANALYSTS
    from src.agents.crypto import CRYPTO_ANALYSTS, CRYPTO_LLM_ANALYSTS
    from src.agents.parallel import run_analysts_parallel, run_analysts_sequential
    from src.evidence import format_evidence_brief

    has_crypto = any(is_crypto(t) for t in active_tickers)

    quant_analysts = [Cls() for Cls in QUANT_ANALYSTS]
    if has_crypto:
        quant_analysts.extend(Cls() for Cls in CRYPTO_ANALYSTS)

    llm_analysts = (
        [Cls() for Cls in VALUE_ANALYSTS]
        + [Cls() for Cls in MACRO_ANALYSTS]
    )
    if has_crypto:
        llm_analysts.extend(Cls() for Cls in CRYPTO_LLM_ANALYSTS)

    all_analysts = quant_analysts + llm_analysts
    analyst_names = [a.name for a in all_analysts]

    run_fn = run_analysts_sequential if args.sequential else run_analysts_parallel
    mode_label = "sequential" if args.sequential else "parallel"

    # Step 1: Run quant analysts first
    print(f"[2/6] Running {len(quant_analysts)} quant analysts ({mode_label})...")
    print(f"  Quant:  {', '.join(a.name for a in quant_analysts)}")

    quant_signals: dict[str, dict[str, Any]]
    quant_signals, quant_elapsed = run_fn(
        quant_analysts, active_tickers, market_data, verbose=True,
    )
    print(f"  Quant analysts complete in {quant_elapsed:.1f}s")

    # Step 2: Build evidence briefs per ticker from quant signals
    evidence_briefs: dict[str, str] = {}
    for ticker in active_tickers:
        ticker_quant = quant_signals.get(ticker, {})
        if ticker_quant:
            evidence_briefs[ticker] = format_evidence_brief(ticker, ticker_quant)

    # Step 3: Run LLM analysts with evidence (or skip if Ollama unavailable)
    ollama_available = _check_ollama()
    all_signals: dict[str, dict[str, Any]] = {}
    llm_elapsed = 0.0

    if ollama_available and llm_analysts:
        print(f"\n  Running {len(llm_analysts)} LLM analysts with quant evidence ({mode_label})...")
        print(f"  Value:  {', '.join(a.name for a in llm_analysts if a.domain == 'value')}")
        print(f"  Macro:  {', '.join(a.name for a in llm_analysts if a.domain == 'macro')}")

        llm_signals, llm_elapsed = run_fn(
            llm_analysts, active_tickers, market_data,
            verbose=True, quant_evidence=evidence_briefs,
        )
        print(f"  LLM analysts complete in {llm_elapsed:.1f}s")

        # LLM-first: quorum uses LLM signals only
        all_signals = llm_signals
        # But keep quant signals for the report
        for ticker in active_tickers:
            for name, sig in quant_signals.get(ticker, {}).items():
                if ticker not in all_signals:
                    all_signals[ticker] = {}
                all_signals[ticker][name] = sig
    else:
        # Fallback: quant-only mode (Ollama not running)
        if not ollama_available:
            print("\n  Ollama unavailable -- falling back to quant-only mode")
        all_signals = quant_signals

    analyst_elapsed = quant_elapsed + llm_elapsed
    print()
    print(f"  All analysts complete in {analyst_elapsed:.1f}s ({mode_label})")
    print(f"  Signals collected for {len(all_signals)} tickers")
    print()

    # -------------------------------------------------------------------------
    # 4. Risk calculations
    # -------------------------------------------------------------------------
    print("[3/6] Computing risk metrics...")

    from src.risk import (
        compute_volatility,
        compute_correlation,
        compute_correlation_cap,
        compute_position_limit,
        compute_allowed_actions,
    )
    from src.portfolio import Portfolio
    from src.models import PortfolioState

    # Build prices_dict: {ticker: [list of close prices]}
    prices_dict: dict[str, list[float]] = {}
    current_prices: dict[str, float] = {}
    for ticker in active_tickers:
        closes = [
            p["close"] for p in market_data[ticker]["prices"]
            if p.get("close") is not None
        ]
        if closes:
            prices_dict[ticker] = closes
            current_prices[ticker] = closes[-1]

    vol_metrics = compute_volatility(prices_dict)
    corr_metrics = compute_correlation(prices_dict)
    corr_caps = compute_correlation_cap(prices_dict)

    # Initialize portfolio
    portfolio = Portfolio(initial_cash=args.initial_cash)
    portfolio_value = args.initial_cash

    # Compute position limits and allowed actions
    position_limits: dict[str, Any] = {}
    allowed_actions: dict[str, list[str]] = {}

    for ticker in active_tickers:
        if ticker not in vol_metrics:
            continue
        limit = compute_position_limit(
            ticker, portfolio_value, vol_metrics[ticker], corr_metrics,
        )
        # Apply correlation-based exposure cap
        cap_mult = corr_caps.get(ticker, 1.0)
        if cap_mult < 1.0:
            limit.final_pct = round(limit.final_pct * cap_mult, 4)
            limit.max_notional = round(portfolio_value * limit.final_pct, 2)
        position_limits[ticker] = limit
        allowed = compute_allowed_actions(
            ticker, portfolio.state, limit, current_prices.get(ticker, 0),
        )
        allowed_actions[ticker] = allowed

    print(f"  Avg correlation: {corr_metrics.avg_correlation:.4f} "
          f"(multiplier: {corr_metrics.multiplier:.2f})")
    for ticker in active_tickers:
        if ticker in vol_metrics:
            vm = vol_metrics[ticker]
            pl = position_limits.get(ticker)
            max_n = f"${pl.max_notional:,.0f}" if pl else "N/A"
            print(f"  {ticker}: vol={vm.annualized_vol:.2%} ({vm.tier}), "
                  f"limit={max_n}")
    print()

    # -------------------------------------------------------------------------
    # 4b. Quorum check and confidence-weighted synthesis
    # -------------------------------------------------------------------------
    print("[4/6] Synthesizing decisions...")

    QUORUM_THRESHOLD = 3  # CF-COMP-021: minimum distinct non-neutral signals
    CRYPTO_QUORUM_THRESHOLD = 2  # Crypto quant-only: 3 analysts, 67% floor
    SCORE_THRESHOLD = 0.3  # Normalized score threshold for action

    # Determine which analyst names are LLM vs quant for quorum filtering
    llm_analyst_names = {a.name for a in llm_analysts} if ollama_available else set()
    quant_analyst_names = {a.name for a in quant_analysts}

    decisions: dict[str, dict[str, Any]] = {}

    for ticker in active_tickers:
        signals = all_signals.get(ticker, {})

        # LLM-first: when LLMs are available, only LLM signals vote in quorum.
        # Quant signals are consumed as evidence in LLM prompts, not counted.
        # Fallback: when no LLMs, all signals (quant-only) vote.
        if llm_analyst_names:
            voting_signals = {
                name: sig for name, sig in signals.items()
                if name in llm_analyst_names
            }
        else:
            voting_signals = dict(signals)

        # Filter out abstained signals -- they don't count toward quorum
        # or synthesis. An abstained signal means "couldn't analyze",
        # not "neutral view".
        active_signals = {
            name: sig for name, sig in voting_signals.items()
            if not sig.abstained
        }
        abstained_count = len(voting_signals) - len(active_signals)

        # Collect non-neutral signals for quorum check
        non_neutral = [
            (name, sig) for name, sig in active_signals.items()
            if sig.signal != "neutral"
        ]

        quorum = CRYPTO_QUORUM_THRESHOLD if is_crypto(ticker) else QUORUM_THRESHOLD
        if len(non_neutral) < quorum:
            abstain_note = f", {abstained_count} abstained" if abstained_count else ""
            decisions[ticker] = {
                "action": "hold",
                "quantity": 0,
                "reasoning": (f"Quorum not met: {len(non_neutral)}/{quorum} "
                              f"non-neutral signals{abstain_note}"),
                "weighted_score": 0.0,
            }
            continue

        # Confidence-weighted majority vote (abstained excluded)
        weighted_sum = 0.0
        confidence_sum = 0.0

        for name, sig in active_signals.items():
            direction = 0.0
            if sig.signal == "bullish":
                direction = 1.0
            elif sig.signal == "bearish":
                direction = -1.0
            # neutral contributes 0

            conf = sig.confidence / 100.0  # normalize to 0-1
            weighted_sum += conf * direction
            confidence_sum += conf

        normalized_score = weighted_sum / confidence_sum if confidence_sum > 0 else 0.0

        # Determine action
        allowed = allowed_actions.get(ticker, ["hold"])
        price = current_prices.get(ticker, 0)

        if normalized_score > SCORE_THRESHOLD and "buy" in allowed:
            # Buy: 50% of position limit on first entry
            pl = position_limits.get(ticker)
            if pl and price > 0:
                target_notional = pl.max_notional * 0.5
                quantity = int(target_notional / price)
                quantity = max(1, quantity)
            else:
                quantity = 0
            decisions[ticker] = {
                "action": "buy",
                "quantity": quantity,
                "reasoning": f"Bullish consensus (score={normalized_score:+.2f})",
                "weighted_score": normalized_score,
            }
        elif normalized_score < -SCORE_THRESHOLD:
            if "sell" in allowed:
                pos = portfolio.state.positions.get(ticker)
                quantity = pos.long_shares if pos else 0
                decisions[ticker] = {
                    "action": "sell",
                    "quantity": quantity,
                    "reasoning": f"Bearish consensus (score={normalized_score:+.2f})",
                    "weighted_score": normalized_score,
                }
            elif "short" in allowed:
                pl = position_limits.get(ticker)
                if pl and price > 0:
                    target_notional = pl.max_notional * 0.5
                    quantity = int(target_notional / price)
                    quantity = max(1, quantity)
                else:
                    quantity = 0
                decisions[ticker] = {
                    "action": "short",
                    "quantity": quantity,
                    "reasoning": f"Bearish consensus (score={normalized_score:+.2f})",
                    "weighted_score": normalized_score,
                }
            else:
                decisions[ticker] = {
                    "action": "hold",
                    "quantity": 0,
                    "reasoning": (f"Bearish (score={normalized_score:+.2f}) "
                                  f"but no sell/short allowed"),
                    "weighted_score": normalized_score,
                }
        else:
            decisions[ticker] = {
                "action": "hold",
                "quantity": 0,
                "reasoning": f"Score within threshold (score={normalized_score:+.2f})",
                "weighted_score": normalized_score,
            }

    # -------------------------------------------------------------------------
    # 5. Execute trades
    # -------------------------------------------------------------------------
    print("[5/6] Executing trades...")

    trades_executed: list[str] = []

    for ticker in active_tickers:
        dec = decisions.get(ticker)
        if not dec or dec["action"] == "hold" or dec["quantity"] == 0:
            continue

        price = current_prices.get(ticker, 0)
        if price <= 0:
            continue

        try:
            if dec["action"] == "buy":
                portfolio.execute_buy(
                    ticker, dec["quantity"], price, reasoning=dec["reasoning"],
                )
                trades_executed.append(
                    f"  BUY  {dec['quantity']:>6} {ticker} @ ${price:.2f} "
                    f"(${dec['quantity'] * price:,.2f})"
                )
            elif dec["action"] == "sell":
                portfolio.execute_sell(
                    ticker, dec["quantity"], price, reasoning=dec["reasoning"],
                )
                trades_executed.append(
                    f"  SELL {dec['quantity']:>6} {ticker} @ ${price:.2f} "
                    f"(${dec['quantity'] * price:,.2f})"
                )
            elif dec["action"] == "short":
                portfolio.execute_short(
                    ticker, dec["quantity"], price, reasoning=dec["reasoning"],
                )
                trades_executed.append(
                    f"  SHORT {dec['quantity']:>5} {ticker} @ ${price:.2f} "
                    f"(${dec['quantity'] * price:,.2f})"
                )
            elif dec["action"] == "cover":
                portfolio.execute_cover(
                    ticker, dec["quantity"], price, reasoning=dec["reasoning"],
                )
                trades_executed.append(
                    f"  COVER {dec['quantity']:>5} {ticker} @ ${price:.2f} "
                    f"(${dec['quantity'] * price:,.2f})"
                )
        except ValueError as e:
            print(f"  WARNING: Trade failed for {ticker}: {e}")
            decisions[ticker]["action"] = "hold"
            decisions[ticker]["reasoning"] += f" [FAILED: {e}]"

    if trades_executed:
        for t in trades_executed:
            print(t)
    else:
        print("  No trades executed.")
    print()

    # -------------------------------------------------------------------------
    # 6. Report
    # -------------------------------------------------------------------------
    print("[6/6] Generating report...")
    print()
    print("=" * 70)
    print("ANALYSIS REPORT")
    print("=" * 70)
    print()

    for ticker in active_tickers:
        print(f"--- {ticker} ---")
        signals = all_signals.get(ticker, {})
        dec = decisions.get(ticker, {})

        # Signals summary
        signal_parts = []
        for name in analyst_names:
            sig = signals.get(name)
            if sig:
                arrow = {"bullish": "+", "bearish": "-", "neutral": "="}[sig.signal]
                signal_parts.append(f"{name}:{arrow}{sig.confidence}")
        print(f"  Signals: {' | '.join(signal_parts)}")

        if args.show_reasoning:
            for name in analyst_names:
                sig = signals.get(name)
                if sig:
                    print(f"    {name}: {sig.reasoning.strip()}")

        # Risk
        if ticker in vol_metrics:
            vm = vol_metrics[ticker]
            pl = position_limits.get(ticker)
            if pl:
                print(f"  Risk: vol={vm.annualized_vol:.2%} ({vm.tier}), "
                      f"limit=${pl.max_notional:,.0f}")

        # Decision
        action = dec.get("action", "hold")
        qty = dec.get("quantity", 0)
        score = dec.get("weighted_score", 0)
        reasoning = dec.get("reasoning", "")
        print(f"  Decision: {action.upper()} {qty} shares "
              f"(score={score:+.2f})")
        print(f"  Reasoning: {reasoning}")
        print()

    # Portfolio state
    print("-" * 70)
    print("PORTFOLIO STATE")
    print("-" * 70)
    final_value = portfolio.compute_portfolio_value(current_prices)
    print(f"  Cash:          ${portfolio.state.cash:,.2f}")
    print(f"  Margin used:   ${portfolio.state.margin_used:,.2f}")
    print(f"  Portfolio val:  ${final_value:,.2f}")
    print(f"  Return:         {((final_value / args.initial_cash) - 1) * 100:+.2f}%")

    if portfolio.state.positions:
        print()
        print("  Positions:")
        for ticker, pos in portfolio.state.positions.items():
            price = current_prices.get(ticker, 0)
            if pos.long_shares > 0:
                mkt_val = pos.long_shares * price
                pnl = (price - pos.avg_long_cost) * pos.long_shares
                print(f"    {ticker} LONG: {pos.long_shares} shares "
                      f"@ ${pos.avg_long_cost:.2f} "
                      f"(mkt ${mkt_val:,.2f}, P&L ${pnl:+,.2f})")
            if pos.short_shares > 0:
                mkt_val = pos.short_shares * price
                pnl = (pos.avg_short_cost - price) * pos.short_shares
                print(f"    {ticker} SHORT: {pos.short_shares} shares "
                      f"@ ${pos.avg_short_cost:.2f} "
                      f"(mkt ${mkt_val:,.2f}, P&L ${pnl:+,.2f})")

    if portfolio.trades:
        print()
        print(f"  Trades executed: {len(portfolio.trades)}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
