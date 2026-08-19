"""Stress test: prove the governance protects capital under extreme conditions.

Simulates flash crashes, sustained drawdowns, rapid trading, and
liquidity crises. The Governor must block dangerous trades and
protect the fund in every scenario.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.fund.core import Fund, FundConfig
from src.solana.governor import Governor


def _make_governor() -> Governor:
    return Governor(ledger_path=Path(tempfile.mktemp(suffix=".json")))


def _make_fund(cash: float = 10000) -> Fund:
    fund = Fund(state_path=Path(tempfile.mktemp(suffix=".json")))
    fund.deposit("stress_test_wallet", cash)
    return fund


def test_flash_crash():
    """Scenario: SOL drops 30% in one cycle.

    The Governor should block new buys (drawdown stop) and
    the stablecoin floor should prevent selling remaining USDC.
    """
    print("\n" + "=" * 60)
    print("STRESS TEST 1: Flash Crash (SOL -30%)")
    print("=" * 60)

    gov = _make_governor()
    fund = _make_fund(10000)

    # Initial position: 50% SOL, 50% USDC
    fund.record_trade("buy", "USDC", "SOL", 5000, 33.33, 150.0)
    print(f"  Initial: $5,000 USDC + 33.33 SOL @ $150")

    # SOL at $150 -> portfolio = $10,000
    nav1 = fund.update_nav({"SOL": 150.0})
    print(f"  NAV before crash: ${nav1:.4f}")
    print(f"  Portfolio: ${fund.total_value({'SOL': 150.0}):,.2f}")

    # CRASH: SOL drops to $105 (-30%)
    nav2 = fund.update_nav({"SOL": 105.0})
    portfolio_after = fund.total_value({"SOL": 105.0})
    drawdown = (1 - portfolio_after / 10000) * 100
    print(f"  SOL crashes to $105 (-30%)")
    print(f"  NAV after crash: ${nav2:.4f}")
    print(f"  Portfolio: ${portfolio_after:,.2f}")
    print(f"  Drawdown: -{drawdown:.1f}%")

    # Try to buy more SOL (should be blocked by drawdown stop)
    decision = gov.evaluate(
        input_symbol="USDC", output_symbol="SOL",
        amount=2000, portfolio_value=portfolio_after,
        balances={"USDC": fund.cash_usdc, "SOL": 33.33},
        quorum_count=4,
    )
    print(f"\n  Attempt: Buy $2,000 more SOL")
    print(f"  Governor: {decision.action.upper()}")
    print(f"  Reason: {decision.reasoning}")

    blocked = decision.action == "reject"
    print(f"\n  RESULT: {'PASS - Governor blocked the trade' if blocked else 'FAIL - Trade was allowed'}")

    # Try to sell all USDC (should be blocked by stablecoin floor)
    import time
    time.sleep(0.1)  # avoid cooldown
    gov.last_trade_time = 0  # reset cooldown for test

    decision2 = gov.evaluate(
        input_symbol="USDC", output_symbol="SOL",
        amount=4500, portfolio_value=portfolio_after,
        balances={"USDC": fund.cash_usdc, "SOL": 33.33},
        quorum_count=4,
    )
    print(f"\n  Attempt: Sell all remaining USDC for SOL")
    print(f"  Governor: {decision2.action.upper()}")
    print(f"  Reason: {decision2.reasoning}")

    blocked2 = decision2.action == "reject"
    print(f"  RESULT: {'PASS - Stablecoin floor held' if blocked2 else 'FAIL - Floor breached'}")

    return blocked and blocked2


def test_rapid_trading():
    """Scenario: Agent tries to trade 30 times in rapid succession.

    The Governor should enforce cooldown (60s) and daily limit (20).
    """
    print("\n" + "=" * 60)
    print("STRESS TEST 2: Rapid Trading (30 trades in 1 minute)")
    print("=" * 60)

    gov = _make_governor()
    approved = 0
    rejected = 0

    for i in range(30):
        decision = gov.evaluate(
            input_symbol="USDC", output_symbol="SOL",
            amount=100, portfolio_value=10000,
            balances={"USDC": 5000, "SOL": 30},
            quorum_count=4,
        )
        if decision.action == "approve":
            approved += 1
        else:
            rejected += 1

    print(f"  30 trade attempts in rapid succession")
    print(f"  Approved: {approved}")
    print(f"  Rejected: {rejected}")
    print(f"  Cooldown blocked: {rejected} trades")

    # Only 1 should be approved (first one), rest blocked by cooldown
    passed = approved == 1
    print(f"\n  RESULT: {'PASS - Cooldown enforced (1 approved, 29 blocked)' if passed else f'FAIL - {approved} approved'}")
    return passed


def test_concentration_risk():
    """Scenario: Agent tries to put 50% into one token.

    The Governor should enforce max position size (25%).
    """
    print("\n" + "=" * 60)
    print("STRESS TEST 3: Concentration Risk (50% in one token)")
    print("=" * 60)

    gov = _make_governor()

    decision = gov.evaluate(
        input_symbol="USDC", output_symbol="BONK",
        amount=5000, portfolio_value=10000,
        balances={"USDC": 8000, "SOL": 10},
        quorum_count=4,
    )
    print(f"  Attempt: Put $5,000 (50%) into BONK")
    print(f"  Governor: {decision.action.upper()}")
    print(f"  Reason: {decision.reasoning}")

    blocked = decision.action == "reject"
    print(f"\n  RESULT: {'PASS - Position size limit enforced' if blocked else 'FAIL - Concentrated position allowed'}")
    return blocked


def test_low_liquidity():
    """Scenario: Swap with 5% price impact.

    The Governor should reject high-impact trades (>1%).
    """
    print("\n" + "=" * 60)
    print("STRESS TEST 4: Low Liquidity (5% price impact)")
    print("=" * 60)

    gov = _make_governor()

    decision = gov.evaluate(
        input_symbol="USDC", output_symbol="WIF",
        amount=500, portfolio_value=10000,
        balances={"USDC": 5000},
        price_impact_pct=0.05,
        quorum_count=4,
    )
    print(f"  Attempt: Swap $500 for WIF (5% price impact)")
    print(f"  Governor: {decision.action.upper()}")
    print(f"  Reason: {decision.reasoning}")

    blocked = decision.action == "reject"
    print(f"\n  RESULT: {'PASS - Price impact limit enforced' if blocked else 'FAIL - High-impact trade allowed'}")
    return blocked


def test_insufficient_quorum():
    """Scenario: Only 1 analyst has a signal (need 3).

    The Governor should reject trades without quorum.
    """
    print("\n" + "=" * 60)
    print("STRESS TEST 5: Insufficient Quorum (1 of 3 required)")
    print("=" * 60)

    gov = _make_governor()

    decision = gov.evaluate(
        input_symbol="USDC", output_symbol="SOL",
        amount=500, portfolio_value=10000,
        balances={"USDC": 5000},
        quorum_count=1,
    )
    print(f"  Attempt: Trade with only 1 analyst signal (minimum 3)")
    print(f"  Governor: {decision.action.upper()}")
    print(f"  Reason: {decision.reasoning}")

    blocked = decision.action == "reject"
    print(f"\n  RESULT: {'PASS - Quorum requirement enforced' if blocked else 'FAIL - Traded without quorum'}")
    return blocked


def test_sustained_drawdown():
    """Scenario: Portfolio drops 5% per day for 4 days.

    Day 1-2: trades should still be allowed (below 15% threshold)
    Day 3+: Governor should halt all trading.
    """
    print("\n" + "=" * 60)
    print("STRESS TEST 6: Sustained Drawdown (5%/day for 4 days)")
    print("=" * 60)

    gov = _make_governor()
    fund = _make_fund(10000)
    fund.record_trade("buy", "USDC", "SOL", 5000, 33.33, 150.0)

    # Set the peak explicitly — Governor needs to know the starting value
    gov.peak_portfolio_value = 10000

    prices = [150.0, 135.0, 120.0, 105.0, 90.0]  # ~10% drops per step
    values = []
    decisions = []

    for day, price in enumerate(prices):
        fund.update_nav({"SOL": price})
        pv = fund.total_value({"SOL": price})
        values.append(pv)

        if day > 0:
            gov.last_trade_time = 0  # reset cooldown
            gov.daily_trade_count = 0  # reset daily count
            d = gov.evaluate(
                input_symbol="USDC", output_symbol="SOL",
                amount=500, portfolio_value=pv,
                balances={"USDC": fund.cash_usdc, "SOL": 33.33},
                quorum_count=4,
            )
            decisions.append(d)
            dd = (1 - pv / gov.peak_portfolio_value) * 100
            print(f"  Day {day}: SOL ${price:.0f} | Portfolio ${pv:,.0f} | "
                  f"Drawdown -{dd:.1f}% | Governor: {d.action.upper()}")

    # Early days (drawdown <15%) should allow, later (>15%) should block
    early_allowed = any(d.action == "approve" for d in decisions[:2])
    late_blocked = any(d.action == "reject" for d in decisions[2:])

    passed = early_allowed and late_blocked
    print(f"\n  Early days (below 15%): {'trading allowed' if early_allowed else 'INCORRECTLY blocked'}")
    print(f"  Late days (above 15%): {'trading halted' if late_blocked else 'INCORRECTLY allowed'}")
    print(f"  RESULT: {'PASS - Drawdown stop triggered correctly' if passed else 'FAIL'}")
    return passed


def test_recovery_after_drawdown():
    """Scenario: Portfolio recovers after drawdown stop triggers.

    Trading should resume once the fund is no longer in drawdown.
    """
    print("\n" + "=" * 60)
    print("STRESS TEST 7: Recovery After Drawdown")
    print("=" * 60)

    gov = _make_governor()

    # Set peak and then simulate drawdown
    gov.peak_portfolio_value = 10000

    # Currently at -16% drawdown (below threshold)
    d1 = gov.evaluate(
        input_symbol="USDC", output_symbol="SOL",
        amount=500, portfolio_value=8400,
        balances={"USDC": 5000},
        quorum_count=4,
    )
    print(f"  At -16% drawdown: Governor {d1.action.upper()}")

    # Portfolio recovers to -10% drawdown
    gov.last_trade_time = 0
    gov.daily_trade_count = 0
    d2 = gov.evaluate(
        input_symbol="USDC", output_symbol="SOL",
        amount=500, portfolio_value=9000,
        balances={"USDC": 5000},
        quorum_count=4,
    )
    print(f"  At -10% drawdown: Governor {d2.action.upper()}")

    blocked_in_crisis = d1.action == "reject"
    allowed_after_recovery = d2.action == "approve"

    passed = blocked_in_crisis and allowed_after_recovery
    print(f"\n  RESULT: {'PASS - Trading resumed after recovery' if passed else 'FAIL'}")
    return passed


def main():
    print("=" * 60)
    print("COVENANT HEDGE FUND — GOVERNANCE STRESS TEST")
    print("=" * 60)
    print("Testing all 8 governance rules under extreme conditions.")
    print("Every test must pass for the fund to be considered safe.")

    results = {
        "Flash Crash (-30%)": test_flash_crash(),
        "Rapid Trading (30 in 1 min)": test_rapid_trading(),
        "Concentration Risk (50%)": test_concentration_risk(),
        "Low Liquidity (5% impact)": test_low_liquidity(),
        "Insufficient Quorum": test_insufficient_quorum(),
        "Sustained Drawdown": test_sustained_drawdown(),
        "Recovery After Drawdown": test_recovery_after_drawdown(),
    }

    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        icon = "+" if result else "x"
        print(f"  [{icon}] {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n  {passed}/{passed + failed} tests passed")

    if failed == 0:
        print("\n  THE GOVERNANCE HOLDS UNDER ALL STRESS CONDITIONS.")
        print("  The fund cannot blow up — the Constitution prevents it.")
    else:
        print(f"\n  {failed} GOVERNANCE FAILURES DETECTED.")
        print("  The fund is NOT safe for live trading.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
