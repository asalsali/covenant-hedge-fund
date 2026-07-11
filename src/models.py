"""Pydantic models for the Covenant Hedge Fund signal and portfolio interfaces."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class SignalDirection(str, Enum):
    """Direction of an analyst signal."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class AnalystSignal(BaseModel):
    """Uniform signal produced by every analyst agent.

    This is the standard interface between analysts and the portfolio
    decision layer. All analysts -- whether LLM-augmented or pure
    computation -- produce this exact shape.
    """

    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(ge=0, le=100, description="Confidence level 0-100")
    reasoning: str = Field(max_length=200, description="Concise reasoning for the signal")


class PortfolioDecision(BaseModel):
    """A single trade decision for one ticker."""

    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(ge=0, description="Number of shares")
    confidence: int = Field(ge=0, le=100, description="Decision confidence 0-100")
    reasoning: str = Field(max_length=200, description="Concise reasoning for the decision")


class Position(BaseModel):
    """Current position in a single ticker."""

    ticker: str
    long_shares: int = 0
    short_shares: int = 0
    avg_long_cost: float = 0.0
    avg_short_cost: float = 0.0


class PortfolioState(BaseModel):
    """Full portfolio state at a point in time."""

    cash: float = Field(default=100_000.0, description="Available cash")
    margin_requirement: float = Field(default=0.5, description="Margin requirement ratio")
    margin_used: float = Field(default=0.0, description="Current margin utilization")
    positions: dict[str, Position] = Field(
        default_factory=dict,
        description="Current positions by ticker",
    )
    realized_gains: dict[str, float] = Field(
        default_factory=dict,
        description="Realized P&L by ticker",
    )

    @property
    def total_realized_gains(self) -> float:
        """Sum of all realized gains across tickers."""
        return sum(self.realized_gains.values())


class PerformanceMetrics(BaseModel):
    """Portfolio performance metrics computed over a backtest period."""

    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    max_drawdown: float | None = None
    max_drawdown_date: str | None = None
    total_return: float | None = None
    annualized_return: float | None = None
    long_short_ratio: float | None = None
    gross_exposure: float | None = None
    net_exposure: float | None = None
