"""Portfolio management engine for the Covenant Hedge Fund.

Handles trade execution, position tracking, P&L computation, and
performance metric calculation (Sharpe, Sortino, max drawdown).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

import numpy as np

from src.models import PortfolioState, Position, PerformanceMetrics


@dataclass
class TradeRecord:
    """Immutable record of a single executed trade.

    Attributes:
        ticker: The ticker symbol.
        action: Trade type executed.
        quantity: Number of shares traded.
        price: Execution price per share.
        timestamp: When the trade was executed.
        notional: Total dollar value of the trade.
        reasoning: Why this trade was made (from PortfolioDecision).
    """

    ticker: str
    action: Literal["buy", "sell", "short", "cover"]
    quantity: int
    price: float
    timestamp: str
    notional: float
    reasoning: str


class Portfolio:
    """Portfolio manager with trade execution and performance tracking.

    Maintains a PortfolioState, executes trades against it, records
    trade history, and tracks daily portfolio values for performance
    metric computation.

    Attributes:
        state: Current portfolio state (cash, positions, realized gains).
        trades: Chronological list of all executed trades.
        daily_values: List of (date_str, portfolio_value) tuples for
            performance computation.
        initial_value: Starting portfolio value for return calculations.
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        margin_requirement: float = 0.5,
    ) -> None:
        self.state = PortfolioState(
            cash=initial_cash,
            margin_requirement=margin_requirement,
        )
        self.trades: list[TradeRecord] = []
        self.daily_values: list[tuple[str, float]] = []
        self.initial_value: float = initial_cash

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def execute_buy(
        self,
        ticker: str,
        quantity: int,
        price: float,
        reasoning: str = "",
    ) -> TradeRecord:
        """Buy shares of a ticker (open or add to long position).

        Deducts cash, updates position with weighted average cost.

        Args:
            ticker: Ticker symbol.
            quantity: Number of shares to buy.
            price: Price per share.
            reasoning: Trade rationale.

        Returns:
            TradeRecord of the executed trade.

        Raises:
            ValueError: If insufficient cash.
        """
        notional = quantity * price
        if notional > self.state.cash:
            raise ValueError(
                f"Insufficient cash for buy: need ${notional:.2f}, "
                f"have ${self.state.cash:.2f}"
            )

        self.state.cash -= notional

        pos = self.state.positions.get(ticker, Position(ticker=ticker))
        # Weighted average cost
        total_cost = pos.avg_long_cost * pos.long_shares + notional
        pos.long_shares += quantity
        pos.avg_long_cost = total_cost / pos.long_shares if pos.long_shares > 0 else 0.0
        self.state.positions[ticker] = pos

        trade = TradeRecord(
            ticker=ticker,
            action="buy",
            quantity=quantity,
            price=price,
            timestamp=datetime.utcnow().isoformat(),
            notional=notional,
            reasoning=reasoning,
        )
        self.trades.append(trade)
        return trade

    def execute_sell(
        self,
        ticker: str,
        quantity: int,
        price: float,
        reasoning: str = "",
    ) -> TradeRecord:
        """Sell shares of a ticker (reduce or close long position).

        Returns cash, computes realized P&L.

        Args:
            ticker: Ticker symbol.
            quantity: Number of shares to sell.
            price: Price per share.
            reasoning: Trade rationale.

        Returns:
            TradeRecord of the executed trade.

        Raises:
            ValueError: If insufficient long shares.
        """
        pos = self.state.positions.get(ticker)
        if pos is None or pos.long_shares < quantity:
            held = pos.long_shares if pos else 0
            raise ValueError(
                f"Insufficient long shares for sell: need {quantity}, "
                f"have {held}"
            )

        notional = quantity * price
        cost_basis = quantity * pos.avg_long_cost
        realized_pnl = notional - cost_basis

        self.state.cash += notional
        pos.long_shares -= quantity

        # Track realized gains
        prev_realized = self.state.realized_gains.get(ticker, 0.0)
        self.state.realized_gains[ticker] = prev_realized + realized_pnl

        # Clean up empty position
        if pos.long_shares == 0 and pos.short_shares == 0:
            del self.state.positions[ticker]
        else:
            self.state.positions[ticker] = pos

        trade = TradeRecord(
            ticker=ticker,
            action="sell",
            quantity=quantity,
            price=price,
            timestamp=datetime.utcnow().isoformat(),
            notional=notional,
            reasoning=reasoning,
        )
        self.trades.append(trade)
        return trade

    def execute_short(
        self,
        ticker: str,
        quantity: int,
        price: float,
        reasoning: str = "",
    ) -> TradeRecord:
        """Short sell shares of a ticker (open or add to short position).

        Receives cash from the short sale, increases margin utilization.

        Args:
            ticker: Ticker symbol.
            quantity: Number of shares to short.
            price: Price per share.
            reasoning: Trade rationale.

        Returns:
            TradeRecord of the executed trade.

        Raises:
            ValueError: If insufficient margin available.
        """
        notional = quantity * price
        margin_needed = notional * self.state.margin_requirement

        margin_available = self.state.cash - self.state.margin_used
        if margin_needed > margin_available:
            raise ValueError(
                f"Insufficient margin for short: need ${margin_needed:.2f}, "
                f"available ${margin_available:.2f}"
            )

        self.state.cash += notional
        self.state.margin_used += margin_needed

        pos = self.state.positions.get(ticker, Position(ticker=ticker))
        total_cost = pos.avg_short_cost * pos.short_shares + notional
        pos.short_shares += quantity
        pos.avg_short_cost = total_cost / pos.short_shares if pos.short_shares > 0 else 0.0
        self.state.positions[ticker] = pos

        trade = TradeRecord(
            ticker=ticker,
            action="short",
            quantity=quantity,
            price=price,
            timestamp=datetime.utcnow().isoformat(),
            notional=notional,
            reasoning=reasoning,
        )
        self.trades.append(trade)
        return trade

    def execute_cover(
        self,
        ticker: str,
        quantity: int,
        price: float,
        reasoning: str = "",
    ) -> TradeRecord:
        """Cover (buy to close) short shares of a ticker.

        Deducts cash to buy back shares, releases margin, computes
        realized P&L.

        Args:
            ticker: Ticker symbol.
            quantity: Number of shares to cover.
            price: Price per share.
            reasoning: Trade rationale.

        Returns:
            TradeRecord of the executed trade.

        Raises:
            ValueError: If insufficient short shares.
        """
        pos = self.state.positions.get(ticker)
        if pos is None or pos.short_shares < quantity:
            held = pos.short_shares if pos else 0
            raise ValueError(
                f"Insufficient short shares for cover: need {quantity}, "
                f"have {held}"
            )

        notional = quantity * price
        cost_basis = quantity * pos.avg_short_cost
        # Short P&L is inverted: profit when price drops
        realized_pnl = cost_basis - notional

        self.state.cash -= notional

        # Release margin proportionally
        margin_release = (quantity / pos.short_shares) * (
            pos.short_shares * pos.avg_short_cost * self.state.margin_requirement
        )
        self.state.margin_used = max(0.0, self.state.margin_used - margin_release)

        pos.short_shares -= quantity

        # Track realized gains
        prev_realized = self.state.realized_gains.get(ticker, 0.0)
        self.state.realized_gains[ticker] = prev_realized + realized_pnl

        # Clean up empty position
        if pos.long_shares == 0 and pos.short_shares == 0:
            del self.state.positions[ticker]
        else:
            self.state.positions[ticker] = pos

        trade = TradeRecord(
            ticker=ticker,
            action="cover",
            quantity=quantity,
            price=price,
            timestamp=datetime.utcnow().isoformat(),
            notional=notional,
            reasoning=reasoning,
        )
        self.trades.append(trade)
        return trade

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------

    def compute_portfolio_value(
        self,
        current_prices: dict[str, float],
    ) -> float:
        """Compute total portfolio value at current market prices.

        Value = cash + sum(long_value) - sum(short_value)

        Short positions are liabilities -- if the price goes up, the
        portfolio value decreases.

        Args:
            current_prices: Mapping of ticker to current price.

        Returns:
            Total portfolio value in dollars.
        """
        value = self.state.cash

        for ticker, pos in self.state.positions.items():
            price = current_prices.get(ticker, 0.0)
            # Long positions are assets
            value += pos.long_shares * price
            # Short positions are liabilities: we owe shares at current price
            # but already received cash at avg_short_cost
            # Net effect on value: -(current_price - avg_short_cost) * shares
            value -= pos.short_shares * price

        return value

    def record_daily_value(
        self,
        date_str: str,
        current_prices: dict[str, float],
    ) -> float:
        """Record the portfolio value for a given date.

        Args:
            date_str: Date string (YYYY-MM-DD format).
            current_prices: Current market prices by ticker.

        Returns:
            The computed portfolio value.
        """
        value = self.compute_portfolio_value(current_prices)
        self.daily_values.append((date_str, value))
        return value

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def compute_performance(
        self,
        risk_free_rate: float = 0.04,
        trading_days_per_year: int = 252,
    ) -> PerformanceMetrics:
        """Compute portfolio performance metrics from daily value history.

        Calculates Sharpe ratio, Sortino ratio, maximum drawdown with
        dates, total return, and annualized return.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino.
                Default 0.04 (4%).
            trading_days_per_year: Trading days per year for annualization.

        Returns:
            PerformanceMetrics with all computed values.
        """
        if len(self.daily_values) < 2:
            return PerformanceMetrics()

        values = np.array([v for _, v in self.daily_values], dtype=np.float64)
        dates = [d for d, _ in self.daily_values]

        # Daily returns
        daily_returns = np.diff(values) / values[:-1]

        # Total return
        total_return = (values[-1] - values[0]) / values[0]

        # Annualized return
        n_days = len(daily_returns)
        n_years = n_days / trading_days_per_year
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        # Daily risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1

        # Excess returns
        excess_returns = daily_returns - daily_rf

        # Sharpe ratio (annualized)
        sharpe_ratio: float | None = None
        if len(excess_returns) > 1:
            std = float(np.std(excess_returns, ddof=1))
            if std > 0:
                sharpe_ratio = round(
                    float(np.mean(excess_returns)) / std * np.sqrt(trading_days_per_year),
                    4,
                )

        # Sortino ratio (annualized) -- uses downside deviation only
        sortino_ratio: float | None = None
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 1:
            downside_std = float(np.std(downside_returns, ddof=1))
            if downside_std > 0:
                sortino_ratio = round(
                    float(np.mean(excess_returns)) / downside_std * np.sqrt(trading_days_per_year),
                    4,
                )

        # Maximum drawdown
        cumulative_max = np.maximum.accumulate(values)
        drawdowns = (values - cumulative_max) / cumulative_max
        max_dd_idx = int(np.argmin(drawdowns))
        max_drawdown = round(float(drawdowns[max_dd_idx]), 6)
        max_drawdown_date = dates[max_dd_idx] if max_dd_idx < len(dates) else None

        # Long/short exposure
        long_shares_total = sum(
            p.long_shares for p in self.state.positions.values()
        )
        short_shares_total = sum(
            p.short_shares for p in self.state.positions.values()
        )
        long_short_ratio: float | None = None
        if short_shares_total > 0:
            long_short_ratio = round(long_shares_total / short_shares_total, 4)

        return PerformanceMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_date=max_drawdown_date,
            total_return=round(total_return, 6),
            annualized_return=round(annualized_return, 6),
            long_short_ratio=long_short_ratio,
        )
