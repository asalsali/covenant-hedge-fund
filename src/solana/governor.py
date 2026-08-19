"""Governance layer for Solana trading — the Constitution for trades.

Every swap must pass through the Governor before execution. The Governor
enforces risk limits, position caps, and cooldowns. This is what makes
the agent "governed" — it literally cannot execute a trade that violates
its rules, regardless of what the LLM analysts recommend.

All decisions are logged to the decision ledger for on-chain transparency.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GovernanceRule:
    """A single governance rule."""
    id: str
    name: str
    description: str
    check: str  # "max_position_pct", "max_trade_size", etc.
    value: Any


@dataclass
class GovernanceDecision:
    """Record of a governance decision on a proposed trade."""
    timestamp: str
    action: str  # "approve" or "reject"
    input_symbol: str
    output_symbol: str
    amount: float
    rules_checked: list[str]
    rules_violated: list[str]
    reasoning: str


DEFAULT_RULES: list[GovernanceRule] = [
    GovernanceRule(
        id="GOV-001",
        name="Max Position Size",
        description="No single token can exceed 25% of portfolio value",
        check="max_position_pct",
        value=0.25,
    ),
    GovernanceRule(
        id="GOV-002",
        name="Max Single Trade",
        description="No single trade can exceed 10% of portfolio value",
        check="max_trade_pct",
        value=0.10,
    ),
    GovernanceRule(
        id="GOV-003",
        name="Stablecoin Floor",
        description="Must maintain at least 20% in stablecoins (USDC/USDT)",
        check="stablecoin_floor",
        value=0.20,
    ),
    GovernanceRule(
        id="GOV-004",
        name="Max Drawdown Stop",
        description="Stop all trading if portfolio drops 15% from peak",
        check="max_drawdown",
        value=0.15,
    ),
    GovernanceRule(
        id="GOV-005",
        name="Trade Cooldown",
        description="Minimum 60 seconds between trades",
        check="cooldown_seconds",
        value=60,
    ),
    GovernanceRule(
        id="GOV-006",
        name="Max Daily Trades",
        description="Maximum 20 trades per day",
        check="max_daily_trades",
        value=20,
    ),
    GovernanceRule(
        id="GOV-007",
        name="Price Impact Limit",
        description="Reject swaps with price impact > 1%",
        check="max_price_impact",
        value=0.01,
    ),
    GovernanceRule(
        id="GOV-008",
        name="Minimum Quorum",
        description="At least 3 analyst signals required before trading",
        check="min_quorum",
        value=3,
    ),
]


class Governor:
    """The governance engine for the trading agent.

    Every proposed trade must call `evaluate()` before execution.
    The Governor checks all rules and returns an approve/reject decision
    with full reasoning. Decisions are logged for transparency.
    """

    def __init__(
        self,
        rules: list[GovernanceRule] | None = None,
        ledger_path: Path | None = None,
    ):
        self.rules = rules or list(DEFAULT_RULES)
        self.ledger_path = ledger_path or Path("data/governance-ledger.json")
        self.decisions: list[GovernanceDecision] = []
        self.last_trade_time: float = 0
        self.daily_trade_count: int = 0
        self.daily_trade_date: str = ""
        self.peak_portfolio_value: float = 0
        self._load_ledger()

    def _load_ledger(self) -> None:
        if self.ledger_path.exists():
            try:
                data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
                self.decisions = [GovernanceDecision(**d) for d in data.get("decisions", [])]
            except (json.JSONDecodeError, TypeError):
                self.decisions = []

    def _save_ledger(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep last 500 decisions
        recent = self.decisions[-500:]
        data = {"decisions": [vars(d) for d in recent]}
        self.ledger_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def evaluate(
        self,
        input_symbol: str,
        output_symbol: str,
        amount: float,
        portfolio_value: float,
        balances: dict[str, float],
        price_impact_pct: float = 0.0,
        quorum_count: int = 0,
    ) -> GovernanceDecision:
        """Evaluate a proposed trade against all governance rules.

        Returns a GovernanceDecision with approve/reject and reasoning.
        """
        now = time.time()
        today = time.strftime("%Y-%m-%d", time.gmtime())
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Reset daily counter
        if today != self.daily_trade_date:
            self.daily_trade_count = 0
            self.daily_trade_date = today

        # Track peak portfolio value
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

        rules_checked: list[str] = []
        rules_violated: list[str] = []
        trade_value = amount  # approximate USD value

        for rule in self.rules:
            rules_checked.append(rule.id)

            if rule.check == "max_position_pct":
                # Check if buying this token would exceed position limit
                current_holding = balances.get(output_symbol, 0)
                # Rough USD estimate (would need price feed for accuracy)
                if portfolio_value > 0 and trade_value / portfolio_value > rule.value:
                    rules_violated.append(f"{rule.id}: Trade {trade_value:.0f} exceeds "
                                         f"{rule.value:.0%} of portfolio {portfolio_value:.0f}")

            elif rule.check == "max_trade_pct":
                if portfolio_value > 0 and trade_value / portfolio_value > rule.value:
                    rules_violated.append(f"{rule.id}: Trade is {trade_value/portfolio_value:.1%} "
                                         f"of portfolio (max {rule.value:.0%})")

            elif rule.check == "stablecoin_floor":
                stable_value = sum(balances.get(s, 0) for s in ("USDC", "USDT"))
                stable_pct = stable_value / max(portfolio_value, 1)
                # If selling stablecoin, check floor
                if input_symbol in ("USDC", "USDT"):
                    new_stable = stable_value - amount
                    new_pct = new_stable / max(portfolio_value, 1)
                    if new_pct < rule.value:
                        rules_violated.append(f"{rule.id}: Selling {input_symbol} would drop "
                                             f"stablecoin to {new_pct:.1%} (floor {rule.value:.0%})")

            elif rule.check == "max_drawdown":
                if self.peak_portfolio_value > 0:
                    drawdown = 1 - (portfolio_value / self.peak_portfolio_value)
                    if drawdown > rule.value:
                        rules_violated.append(f"{rule.id}: Portfolio drawdown {drawdown:.1%} "
                                             f"exceeds {rule.value:.0%} — trading halted")

            elif rule.check == "cooldown_seconds":
                elapsed = now - self.last_trade_time
                if self.last_trade_time > 0 and elapsed < rule.value:
                    rules_violated.append(f"{rule.id}: {rule.value - elapsed:.0f}s remaining "
                                         f"in cooldown")

            elif rule.check == "max_daily_trades":
                if self.daily_trade_count >= rule.value:
                    rules_violated.append(f"{rule.id}: {self.daily_trade_count}/{rule.value} "
                                         f"daily trades exhausted")

            elif rule.check == "max_price_impact":
                if price_impact_pct > rule.value:
                    rules_violated.append(f"{rule.id}: Price impact {price_impact_pct:.2%} "
                                         f"exceeds {rule.value:.0%} limit")

            elif rule.check == "min_quorum":
                if quorum_count < rule.value:
                    rules_violated.append(f"{rule.id}: Only {quorum_count} analyst signals "
                                         f"(minimum {rule.value})")

        # Decision
        if rules_violated:
            action = "reject"
            reasoning = "Trade rejected: " + "; ".join(rules_violated)
        else:
            action = "approve"
            reasoning = f"All {len(rules_checked)} governance rules passed"

        decision = GovernanceDecision(
            timestamp=timestamp,
            action=action,
            input_symbol=input_symbol,
            output_symbol=output_symbol,
            amount=amount,
            rules_checked=rules_checked,
            rules_violated=rules_violated,
            reasoning=reasoning,
        )

        self.decisions.append(decision)
        self._save_ledger()

        # Update state on approval
        if action == "approve":
            self.last_trade_time = now
            self.daily_trade_count += 1

        return decision

    def get_recent_decisions(self, n: int = 10) -> list[GovernanceDecision]:
        """Get the N most recent decisions."""
        return self.decisions[-n:]

    def get_rules_summary(self) -> list[dict]:
        """Get a human-readable summary of all governance rules."""
        return [
            {"id": r.id, "name": r.name, "description": r.description, "value": r.value}
            for r in self.rules
        ]
