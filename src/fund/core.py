"""Covenant Hedge Fund — Core fund accounting.

NAV tracking, share-based accounting, deposits/withdrawals,
fee structure with high-water mark. The fund that follows rules.

Fund structure:
  - Investors deposit USDC → receive fund shares
  - NAV = total fund value / total shares outstanding
  - Management fee: 2% annual (accrued daily)
  - Performance fee: 20% of profits above high-water mark
  - All positions governed by the 8-rule Constitution
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Investor:
    """A fund investor (wallet address)."""
    address: str
    shares: float = 0.0
    deposited_usdc: float = 0.0
    deposit_nav: float = 1.0  # NAV at time of deposit
    joined_at: str = ""


@dataclass
class FundSnapshot:
    """Point-in-time snapshot of fund state."""
    timestamp: str
    nav: float  # net asset value per share
    total_value: float  # total fund value in USDC
    total_shares: float
    positions: dict[str, float]  # token -> quantity
    cash_usdc: float
    high_water_mark: float
    accrued_mgmt_fee: float
    accrued_perf_fee: float


@dataclass
class FundConfig:
    """Fund configuration — the economic constitution."""
    name: str = "Covenant Governed Fund"
    base_currency: str = "USDC"
    management_fee_annual: float = 0.02  # 2% per year
    performance_fee: float = 0.20  # 20% of profits
    min_deposit: float = 10.0  # minimum deposit in USDC
    max_position_pct: float = 0.25  # max 25% in any single token
    rebalance_interval: int = 3600  # seconds between rebalances
    tokens: list[str] = field(default_factory=lambda: ["SOL", "JUP", "BONK"])


class Fund:
    """The governed hedge fund.

    Manages investor deposits, NAV calculations, fee accrual,
    and position tracking. All trades go through the Governor.
    """

    def __init__(
        self,
        config: FundConfig | None = None,
        state_path: Path | None = None,
    ):
        self.config = config or FundConfig()
        self.state_path = state_path or Path("data/fund-state.json")

        # Fund state
        self.nav: float = 1.0  # per-share NAV, starts at $1
        self.total_shares: float = 0.0
        self.high_water_mark: float = 1.0
        self.positions: dict[str, float] = {}  # token -> quantity
        self.cash_usdc: float = 0.0
        self.investors: dict[str, Investor] = {}  # address -> Investor
        self.snapshots: list[FundSnapshot] = []
        self.accrued_mgmt_fee: float = 0.0
        self.accrued_perf_fee: float = 0.0
        self.inception_time: str = ""
        self.last_fee_accrual: str = ""
        self.trades: list[dict] = []

        self._load_state()

    # ------------------------------------------------------------------
    # Deposits & Withdrawals
    # ------------------------------------------------------------------

    def deposit(self, address: str, usdc_amount: float) -> dict:
        """Investor deposits USDC into the fund. Receives shares at current NAV."""
        if usdc_amount < self.config.min_deposit:
            return {"error": f"Minimum deposit is {self.config.min_deposit} USDC"}

        shares = usdc_amount / self.nav
        self.total_shares += shares
        self.cash_usdc += usdc_amount

        if address not in self.investors:
            self.investors[address] = Investor(
                address=address,
                joined_at=_now(),
            )

        inv = self.investors[address]
        inv.shares += shares
        inv.deposited_usdc += usdc_amount
        inv.deposit_nav = self.nav

        self._save_state()

        return {
            "shares_issued": shares,
            "nav_at_deposit": self.nav,
            "total_shares": inv.shares,
            "fund_total_value": self.total_value(),
        }

    def withdraw(self, address: str, shares: float | None = None) -> dict:
        """Investor withdraws shares from the fund. Receives USDC at current NAV."""
        if address not in self.investors:
            return {"error": "Investor not found"}

        inv = self.investors[address]
        withdraw_shares = shares or inv.shares  # default: withdraw all

        if withdraw_shares > inv.shares:
            return {"error": f"Insufficient shares: have {inv.shares}, requested {withdraw_shares}"}

        usdc_value = withdraw_shares * self.nav
        inv.shares -= withdraw_shares
        self.total_shares -= withdraw_shares
        self.cash_usdc -= min(usdc_value, self.cash_usdc)

        if inv.shares <= 0:
            del self.investors[address]

        self._save_state()

        return {
            "shares_redeemed": withdraw_shares,
            "usdc_value": usdc_value,
            "nav_at_withdrawal": self.nav,
            "remaining_shares": inv.shares if address in self.investors else 0,
        }

    # ------------------------------------------------------------------
    # NAV Calculation
    # ------------------------------------------------------------------

    def total_value(self, prices: dict[str, float] | None = None) -> float:
        """Calculate total fund value in USDC."""
        value = self.cash_usdc
        if prices:
            for token, qty in self.positions.items():
                if token in prices:
                    value += qty * prices[token]
        return value

    def update_nav(self, prices: dict[str, float]) -> float:
        """Recalculate NAV based on current prices."""
        total = self.total_value(prices)

        # Accrue fees
        self._accrue_fees(total)

        # NAV after fees
        if self.total_shares > 0:
            self.nav = (total - self.accrued_mgmt_fee - self.accrued_perf_fee) / self.total_shares
        else:
            self.nav = 1.0

        # Update high-water mark
        if self.nav > self.high_water_mark:
            self.high_water_mark = self.nav

        # Take snapshot
        self.snapshots.append(FundSnapshot(
            timestamp=_now(),
            nav=self.nav,
            total_value=total,
            total_shares=self.total_shares,
            positions=dict(self.positions),
            cash_usdc=self.cash_usdc,
            high_water_mark=self.high_water_mark,
            accrued_mgmt_fee=self.accrued_mgmt_fee,
            accrued_perf_fee=self.accrued_perf_fee,
        ))

        # Keep last 1000 snapshots
        if len(self.snapshots) > 1000:
            self.snapshots = self.snapshots[-1000:]

        self._save_state()
        return self.nav

    # ------------------------------------------------------------------
    # Fee Accrual
    # ------------------------------------------------------------------

    def _accrue_fees(self, total_value: float) -> None:
        """Accrue management and performance fees."""
        now = time.time()

        # Management fee: 2% annual, accrued daily
        # Daily rate = annual_rate / 365
        daily_mgmt_rate = self.config.management_fee_annual / 365
        self.accrued_mgmt_fee += total_value * daily_mgmt_rate

        # Performance fee: 20% of profits above high-water mark
        if self.total_shares > 0:
            current_nav = total_value / self.total_shares
            if current_nav > self.high_water_mark:
                profit_per_share = current_nav - self.high_water_mark
                total_profit = profit_per_share * self.total_shares
                self.accrued_perf_fee += total_profit * self.config.performance_fee

        self.last_fee_accrual = _now()

    # ------------------------------------------------------------------
    # Position Management
    # ------------------------------------------------------------------

    def record_trade(
        self,
        action: str,
        input_token: str,
        output_token: str,
        input_amount: float,
        output_amount: float,
        price: float,
        governance_decision: dict | None = None,
    ) -> None:
        """Record a trade executed by the trading loop."""
        # Update positions
        if input_token == "USDC":
            self.cash_usdc -= input_amount
            self.positions[output_token] = self.positions.get(output_token, 0) + output_amount
        elif output_token == "USDC":
            self.positions[input_token] = self.positions.get(input_token, 0) - input_amount
            self.cash_usdc += output_amount
            # Clean up zero positions
            if self.positions.get(input_token, 0) <= 0:
                self.positions.pop(input_token, None)

        self.trades.append({
            "timestamp": _now(),
            "action": action,
            "input": f"{input_amount} {input_token}",
            "output": f"{output_amount} {output_token}",
            "price": price,
            "governance": governance_decision,
        })

        # Keep last 500 trades
        if len(self.trades) > 500:
            self.trades = self.trades[-500:]

        self._save_state()

    def get_allocations(self, prices: dict[str, float]) -> dict[str, dict]:
        """Get current portfolio allocation percentages."""
        total = self.total_value(prices)
        if total <= 0:
            return {}

        allocs = {"USDC": {"value": self.cash_usdc, "pct": self.cash_usdc / total}}
        for token, qty in self.positions.items():
            value = qty * prices.get(token, 0)
            allocs[token] = {"value": value, "pct": value / total, "quantity": qty}
        return allocs

    # ------------------------------------------------------------------
    # Performance Metrics
    # ------------------------------------------------------------------

    def get_performance(self) -> dict:
        """Calculate fund performance metrics."""
        if len(self.snapshots) < 2:
            return {
                "nav": self.nav,
                "total_return_pct": 0.0,
                "high_water_mark": self.high_water_mark,
                "drawdown_pct": 0.0,
                "sharpe": 0.0,
                "total_value": self.total_value(),
                "investors": len(self.investors),
                "trades": len(self.trades),
            }

        navs = [s.nav for s in self.snapshots]
        first_nav = navs[0]
        last_nav = navs[-1]
        total_return = (last_nav / first_nav - 1) * 100 if first_nav > 0 else 0

        # Max drawdown
        peak = navs[0]
        max_dd = 0.0
        for n in navs:
            if n > peak:
                peak = n
            dd = (peak - n) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Simple Sharpe (daily returns)
        if len(navs) >= 3:
            import math
            returns = [(navs[i] / navs[i-1] - 1) for i in range(1, len(navs)) if navs[i-1] > 0]
            if returns:
                avg_ret = sum(returns) / len(returns)
                std_ret = (sum((r - avg_ret) ** 2 for r in returns) / len(returns)) ** 0.5
                sharpe = (avg_ret / std_ret) * math.sqrt(365) if std_ret > 0 else 0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        return {
            "nav": self.nav,
            "total_return_pct": total_return,
            "high_water_mark": self.high_water_mark,
            "drawdown_pct": max_dd * 100,
            "sharpe": sharpe,
            "total_value": self.total_value(),
            "investors": len(self.investors),
            "trades": len(self.trades),
            "accrued_mgmt_fee": self.accrued_mgmt_fee,
            "accrued_perf_fee": self.accrued_perf_fee,
        }

    # ------------------------------------------------------------------
    # Fund Summary (for dashboard / API)
    # ------------------------------------------------------------------

    def summary(self, prices: dict[str, float] | None = None) -> dict:
        """Full fund summary for display."""
        prices = prices or {}
        perf = self.get_performance()
        allocs = self.get_allocations(prices)

        return {
            "name": self.config.name,
            "nav": self.nav,
            "total_value": self.total_value(prices),
            "total_shares": self.total_shares,
            "high_water_mark": self.high_water_mark,
            "performance": perf,
            "allocations": allocs,
            "investors": len(self.investors),
            "positions": dict(self.positions),
            "cash_usdc": self.cash_usdc,
            "fees": {
                "management": f"{self.config.management_fee_annual:.0%} annual",
                "performance": f"{self.config.performance_fee:.0%} above HWM",
                "accrued_mgmt": self.accrued_mgmt_fee,
                "accrued_perf": self.accrued_perf_fee,
            },
            "governance_rules": 8,
            "recent_trades": self.trades[-10:],
            "snapshot_count": len(self.snapshots),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "nav": self.nav,
            "total_shares": self.total_shares,
            "high_water_mark": self.high_water_mark,
            "positions": self.positions,
            "cash_usdc": self.cash_usdc,
            "accrued_mgmt_fee": self.accrued_mgmt_fee,
            "accrued_perf_fee": self.accrued_perf_fee,
            "inception_time": self.inception_time or _now(),
            "last_fee_accrual": self.last_fee_accrual,
            "investors": {
                addr: {
                    "address": inv.address,
                    "shares": inv.shares,
                    "deposited_usdc": inv.deposited_usdc,
                    "deposit_nav": inv.deposit_nav,
                    "joined_at": inv.joined_at,
                }
                for addr, inv in self.investors.items()
            },
            "trades": self.trades[-500:],
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "nav": s.nav,
                    "total_value": s.total_value,
                    "total_shares": s.total_shares,
                }
                for s in self.snapshots[-100:]
            ],
        }
        self.state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    def _load_state(self) -> None:
        if not self.state_path.exists():
            self.inception_time = _now()
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.nav = state.get("nav", 1.0)
            self.total_shares = state.get("total_shares", 0.0)
            self.high_water_mark = state.get("high_water_mark", 1.0)
            self.positions = state.get("positions", {})
            self.cash_usdc = state.get("cash_usdc", 0.0)
            self.accrued_mgmt_fee = state.get("accrued_mgmt_fee", 0.0)
            self.accrued_perf_fee = state.get("accrued_perf_fee", 0.0)
            self.inception_time = state.get("inception_time", _now())
            self.last_fee_accrual = state.get("last_fee_accrual", "")
            self.trades = state.get("trades", [])

            for addr, inv_data in state.get("investors", {}).items():
                self.investors[addr] = Investor(**inv_data)

            for snap_data in state.get("snapshots", []):
                self.snapshots.append(FundSnapshot(
                    timestamp=snap_data["timestamp"],
                    nav=snap_data["nav"],
                    total_value=snap_data["total_value"],
                    total_shares=snap_data["total_shares"],
                    positions={},
                    cash_usdc=0,
                    high_water_mark=self.high_water_mark,
                    accrued_mgmt_fee=0,
                    accrued_perf_fee=0,
                ))
        except (json.JSONDecodeError, KeyError):
            self.inception_time = _now()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
