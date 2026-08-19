"""Solana trading loop — the governed agent runtime.

Connects the analyst pipeline -> Governor -> Jupiter swaps into a
continuous trading loop. Every trade decision is governance-checked
before execution.

Usage:
    python -m src.solana.trading_loop --wallet <PUBKEY> --tokens SOL JUP BONK
    python -m src.solana.trading_loop --wallet <PUBKEY> --dry-run  # no real trades
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Fix Windows Unicode encoding
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.solana.governor import Governor
from src.solana.jupiter import get_quote, execute_swap, SwapResult
from src.solana.memo import post_governance_memo
from src.solana.tokens import TOKEN_MINTS, USDC
from src.solana.wallet import get_all_balances
from src.solana.signer import load_keypair, make_sign_and_send_fn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Covenant Governed Trading Agent — Solana",
    )
    parser.add_argument(
        "--wallet", required=True,
        help="Solana wallet public key",
    )
    parser.add_argument(
        "--tokens", nargs="+", default=["SOL", "JUP", "BONK"],
        help="Tokens to trade (default: SOL JUP BONK)",
    )
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Seconds between analysis cycles (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Get quotes but don't execute swaps",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run one cycle and exit (don't loop)",
    )
    parser.add_argument(
        "--devnet", action="store_true",
        help="Use Solana devnet",
    )
    parser.add_argument(
        "--model", default=None,
        help="LLM model override",
    )
    parser.add_argument(
        "--max-trade-usdc", type=float, default=50.0,
        help="Maximum trade size in USDC equivalent (default: 50)",
    )
    return parser.parse_args(argv)


def _get_crypto_prices(tokens: list[str]) -> dict[str, float]:
    """Fetch current prices for tokens via CoinGecko."""
    from src.data.crypto import cg_get_current_price, is_crypto
    prices = {}
    for token in tokens:
        try:
            price = cg_get_current_price(token)
            if price:
                prices[token] = price
        except Exception:
            pass
    return prices


def _run_analysts(
    tokens: list[str],
    market_data: dict[str, Any],
    model: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the analyst pipeline on the given tokens."""
    from src.agents.quant import QUANT_ANALYSTS
    from src.agents.crypto import CRYPTO_ANALYSTS, CRYPTO_LLM_ANALYSTS
    from src.agents.value import VALUE_ANALYSTS
    from src.agents.macro import MACRO_ANALYSTS
    from src.agents.parallel import run_analysts_parallel
    from src.evidence import format_evidence_brief

    if model:
        from src.llm import set_model
        set_model(model)

    # Run quant analysts
    quant_analysts = [Cls() for Cls in QUANT_ANALYSTS]
    quant_analysts.extend(Cls() for Cls in CRYPTO_ANALYSTS)
    quant_signals, _ = run_analysts_parallel(quant_analysts, tokens, market_data)

    # Build evidence briefs
    evidence = {}
    for token in tokens:
        ticker_quant = quant_signals.get(token, {})
        if ticker_quant:
            evidence[token] = format_evidence_brief(token, ticker_quant)

    # Run LLM analysts
    from src.llm import get_active_backend
    if get_active_backend() != "none":
        llm_analysts = (
            [Cls() for Cls in VALUE_ANALYSTS]
            + [Cls() for Cls in MACRO_ANALYSTS]
            + [Cls() for Cls in CRYPTO_LLM_ANALYSTS]
        )
        llm_signals, _ = run_analysts_parallel(
            llm_analysts, tokens, market_data, quant_evidence=evidence,
        )
        # Merge
        for token in tokens:
            for name, sig in quant_signals.get(token, {}).items():
                if token not in llm_signals:
                    llm_signals[token] = {}
                llm_signals[token][name] = sig
        return llm_signals
    else:
        return quant_signals


def _synthesize_decision(
    token: str,
    signals: dict[str, Any],
    quorum_threshold: int = 2,
    score_threshold: float = 0.3,
) -> dict[str, Any]:
    """Synthesize analyst signals into a buy/sell/hold decision."""
    active = {n: s for n, s in signals.items() if not s.abstained}
    non_neutral = [(n, s) for n, s in active.items() if s.signal != "neutral"]

    if len(non_neutral) < quorum_threshold:
        return {
            "action": "hold",
            "reasoning": f"Quorum not met: {len(non_neutral)}/{quorum_threshold}",
            "score": 0.0,
            "quorum_count": len(non_neutral),
        }

    weighted_sum = 0.0
    conf_sum = 0.0
    for _, sig in active.items():
        direction = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}[sig.signal]
        conf = sig.confidence / 100.0
        weighted_sum += conf * direction
        conf_sum += conf

    score = weighted_sum / conf_sum if conf_sum > 0 else 0.0

    if score > score_threshold:
        action = "buy"
    elif score < -score_threshold:
        action = "sell"
    else:
        action = "hold"

    return {
        "action": action,
        "reasoning": f"Score {score:+.2f} from {len(active)} analysts",
        "score": score,
        "quorum_count": len(non_neutral),
    }


def run_cycle(
    wallet_pubkey: str,
    tokens: list[str],
    governor: Governor,
    dry_run: bool = True,
    devnet: bool = False,
    model: str | None = None,
    max_trade_usdc: float = 50.0,
    sign_and_send_fn: Any = None,
    memo_keypair: Any = None,
) -> list[dict]:
    """Run one analysis -> governance -> execution cycle.

    Returns a list of decision records for logging/display.
    """
    cycle_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    records: list[dict] = []

    # 1. Get balances
    print(f"\n[{cycle_time}] Starting analysis cycle")
    try:
        balances = get_all_balances(wallet_pubkey, devnet)
        print(f"  Balances: {balances}")
    except Exception as e:
        print(f"  WARNING: Could not fetch balances: {e}")
        balances = {"USDC": max_trade_usdc * 10}  # assume for dry run

    portfolio_value = sum(balances.values())  # rough estimate

    # 2. Fetch market data
    print(f"  Fetching market data for {tokens}...")
    from src.data.api import get_prices, clear_cache
    from src.data.crypto import is_crypto

    clear_cache()
    end_date = date.today()
    start_date = end_date - timedelta(days=90)

    market_data: dict[str, Any] = {}
    for token in tokens:
        try:
            prices = get_prices(token, start_date, end_date)
            if prices:
                market_data[token] = {"prices": prices}
                # Add crypto metrics
                if is_crypto(token):
                    from src.data.crypto import cg_get_crypto_metrics, resolve_coin_id
                    coin_id = resolve_coin_id(token)
                    if coin_id:
                        metrics = cg_get_crypto_metrics(coin_id)
                        if metrics:
                            market_data[token]["crypto_metrics"] = metrics
        except Exception as e:
            print(f"  WARNING: No data for {token}: {e}")

    if not market_data:
        print("  ERROR: No market data available")
        return records

    active_tokens = list(market_data.keys())
    print(f"  Data loaded for: {active_tokens}")

    # 3. Run analysts
    print(f"  Running analysts...")
    try:
        all_signals = _run_analysts(active_tokens, market_data, model)
    except Exception as e:
        print(f"  ERROR: Analyst pipeline failed: {e}")
        return records

    # 4. Synthesize decisions and check governance
    print(f"  Synthesizing decisions...")
    for token in active_tokens:
        signals = all_signals.get(token, {})
        if not signals:
            continue

        decision = _synthesize_decision(token, signals)
        record = {
            "timestamp": cycle_time,
            "token": token,
            "decision": decision,
            "governance": None,
            "execution": None,
        }

        if decision["action"] == "hold":
            print(f"    {token}: HOLD — {decision['reasoning']}")
            records.append(record)
            continue

        # Determine swap direction
        if decision["action"] == "buy":
            input_sym, output_sym = "USDC", token
            amount = min(max_trade_usdc, balances.get("USDC", 0) * 0.1)
        else:
            input_sym, output_sym = token, "USDC"
            amount = balances.get(token, 0) * 0.5

        if amount <= 0:
            print(f"    {token}: {decision['action'].upper()} — insufficient balance")
            records.append(record)
            continue

        # 5. Governor check
        gov_decision = governor.evaluate(
            input_symbol=input_sym,
            output_symbol=output_sym,
            amount=amount,
            portfolio_value=max(portfolio_value, 1),
            balances=balances,
            quorum_count=decision["quorum_count"],
        )
        record["governance"] = {
            "action": gov_decision.action,
            "reasoning": gov_decision.reasoning,
            "rules_checked": gov_decision.rules_checked,
            "rules_violated": gov_decision.rules_violated,
        }

        # Post governance decision to chain as memo
        if memo_keypair and not dry_run:
            memo_data = {
                "action": gov_decision.action,
                "input_symbol": input_sym,
                "output_symbol": output_sym,
                "amount": amount,
                "rules_violated": gov_decision.rules_violated,
                "timestamp": gov_decision.timestamp,
            }
            memo_sig = post_governance_memo(memo_keypair, memo_data, devnet)
            if memo_sig:
                print(f"    Memo TX: {memo_sig}")
                record["memo_tx"] = memo_sig

        if gov_decision.action == "reject":
            print(f"    {token}: BLOCKED by Governor — {gov_decision.reasoning}")
            records.append(record)
            continue

        # 6. Execute swap
        print(f"    {token}: {decision['action'].upper()} "
              f"({amount:.2f} {input_sym} -> {output_sym}) — Governor APPROVED")

        swap_result = execute_swap(
            input_symbol=input_sym,
            output_symbol=output_sym,
            amount=amount,
            user_pubkey=wallet_pubkey,
            sign_and_send_fn=sign_and_send_fn if not dry_run else None,
            dry_run=dry_run,
        )
        record["execution"] = {
            "success": swap_result.success,
            "input": f"{swap_result.input_amount} {swap_result.input_symbol}",
            "output": f"{swap_result.output_amount} {swap_result.output_symbol}",
            "tx_signature": swap_result.tx_signature,
            "error": swap_result.error,
        }

        if swap_result.success:
            mode = "DRY RUN" if dry_run else "EXECUTED"
            print(f"      {mode}: {swap_result.input_amount:.4f} {swap_result.input_symbol} "
                  f"-> {swap_result.output_amount:.4f} {swap_result.output_symbol}")
            if swap_result.tx_signature:
                print(f"      TX: {swap_result.tx_signature}")
        else:
            print(f"      FAILED: {swap_result.error}")

        records.append(record)

    return records


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print("=" * 60)
    print("COVENANT GOVERNED TRADING AGENT — SOLANA")
    print("=" * 60)
    print(f"  Wallet:    {args.wallet}")
    print(f"  Tokens:    {', '.join(args.tokens)}")
    print(f"  Interval:  {args.interval}s")
    print(f"  Max trade: ${args.max_trade_usdc:.2f} USDC")
    print(f"  Mode:      {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  Network:   {'devnet' if args.devnet else 'mainnet'}")
    print()

    # Show governance rules
    governor = Governor()
    print("  Governance rules:")
    for rule in governor.get_rules_summary():
        print(f"    [{rule['id']}] {rule['name']}: {rule['description']}")
    print()

    # Load keypair for live trading
    sign_fn = None
    keypair = None
    if not args.dry_run:
        try:
            keypair = load_keypair()
            sign_fn = make_sign_and_send_fn(keypair, devnet=args.devnet)
            print(f"  Signer:   {keypair.pubkey()} (loaded)")
        except ValueError as e:
            print(f"  WARNING: {e}")
            print(f"  Falling back to dry-run mode.")
            args.dry_run = True

    # Normalize token names
    tokens = [t.upper() for t in args.tokens]

    all_records: list[dict] = []
    cycle_count = 0

    while True:
        cycle_count += 1
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle_count}")
        print(f"{'='*60}")

        try:
            records = run_cycle(
                wallet_pubkey=args.wallet,
                tokens=tokens,
                governor=governor,
                dry_run=args.dry_run,
                devnet=args.devnet,
                sign_and_send_fn=sign_fn,
                memo_keypair=keypair if not args.dry_run else None,
                model=args.model,
                max_trade_usdc=args.max_trade_usdc,
            )
            all_records.extend(records)

            # Save decision log
            log_path = Path("data/trading-log.json")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                json.dumps(all_records[-100:], indent=2, default=str),
                encoding="utf-8",
            )

        except KeyboardInterrupt:
            print("\n\nStopping trading agent...")
            break
        except Exception as e:
            print(f"\nERROR in cycle {cycle_count}: {e}")

        if args.once:
            break

        print(f"\nNext cycle in {args.interval}s...")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nStopping trading agent...")
            break

    # Summary
    print(f"\n{'='*60}")
    print(f"SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Cycles:     {cycle_count}")
    print(f"  Decisions:  {len(all_records)}")
    approved = sum(1 for r in all_records if (r.get("governance") or {}).get("action") == "approve")
    rejected = sum(1 for r in all_records if (r.get("governance") or {}).get("action") == "reject")
    holds = sum(1 for r in all_records if (r.get("decision") or {}).get("action") == "hold")
    print(f"  Approved:   {approved}")
    print(f"  Rejected:   {rejected}")
    print(f"  Holds:      {holds}")
    print()


if __name__ == "__main__":
    main()
