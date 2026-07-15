"""Risk calculation engine for the Covenant Hedge Fund.

Computes volatility-adjusted position limits, correlation-based
diversification multipliers, and allowed trading actions based on
current portfolio state. All calculations use numpy for numerical
stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.models import PortfolioState


@dataclass
class VolatilityMetrics:
    """Volatility computation results for a single ticker.

    Attributes:
        ticker: The ticker symbol.
        annualized_vol: Annualized volatility (60-day rolling std * sqrt(252)).
        tier: Classification bucket based on annualized vol.
        multiplier: Position size multiplier derived from volatility tier.
            Lower volatility allows larger positions.
    """

    ticker: str
    annualized_vol: float
    tier: Literal["low", "medium", "high", "extreme"]
    multiplier: float


@dataclass
class CorrelationMetrics:
    """Pairwise correlation results for a set of tickers.

    Attributes:
        avg_correlation: Mean of all pairwise correlations.
        correlation_matrix: Full NxN correlation matrix as nested dict.
        multiplier: Portfolio-level diversification multiplier.
            Lower correlation allows larger aggregate positions.
    """

    avg_correlation: float
    correlation_matrix: dict[str, dict[str, float]]
    multiplier: float


@dataclass
class PositionLimit:
    """Computed position limit for a single ticker.

    Attributes:
        ticker: The ticker symbol.
        base_pct: Base position size as percentage of portfolio (default 20%).
        vol_adjusted_pct: After volatility adjustment.
        final_pct: After both volatility and correlation adjustment.
        max_notional: Maximum dollar value of the position.
    """

    ticker: str
    base_pct: float
    vol_adjusted_pct: float
    final_pct: float
    max_notional: float


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

_VOL_TIERS: list[tuple[float, float, str]] = [
    # (upper_bound, base_multiplier, tier_name)
    (0.15, 1.00, "low"),
    (0.30, 0.75, "medium"),
    (0.50, 0.50, "high"),
    (float("inf"), 0.25, "extreme"),
]


def _interpolate_vol_multiplier(annualized_vol: float) -> tuple[float, str]:
    """Linearly interpolate the volatility multiplier within tier boundaries.

    Within each tier the multiplier transitions linearly from the previous
    tier's multiplier to the current tier's multiplier.  At the exact
    boundary the multiplier equals the tier's base multiplier.

    Returns:
        Tuple of (multiplier, tier_name).
    """
    prev_bound = 0.0
    prev_mult = 1.25  # hypothetical multiplier for vol=0 (extrapolate)

    for upper, base_mult, tier in _VOL_TIERS:
        if annualized_vol <= upper:
            # Linear interpolation within the tier
            if upper == float("inf"):
                return base_mult, tier
            span = upper - prev_bound
            if span == 0:
                return base_mult, tier
            t = (annualized_vol - prev_bound) / span
            interp = prev_mult + t * (base_mult - prev_mult)
            return max(interp, base_mult), tier
        prev_bound = upper
        prev_mult = base_mult

    # Fallback -- should not reach here
    return 0.25, "extreme"


def compute_volatility(
    prices: dict[str, list[float]],
    window: int = 60,
) -> dict[str, VolatilityMetrics]:
    """Compute annualized volatility for each ticker.

    Uses a rolling standard deviation of daily log returns over
    ``window`` trading days, annualized by sqrt(252).

    Args:
        prices: Mapping of ticker -> list of daily closing prices,
            ordered oldest to newest. Must contain at least ``window + 1``
            entries for a valid calculation.
        window: Rolling window size in trading days. Default 60.

    Returns:
        Dict mapping ticker to VolatilityMetrics.
    """
    results: dict[str, VolatilityMetrics] = {}

    for ticker, price_series in prices.items():
        arr = np.array(price_series, dtype=np.float64)

        if len(arr) < window + 1:
            # Not enough data -- return extreme tier as conservative default
            results[ticker] = VolatilityMetrics(
                ticker=ticker,
                annualized_vol=1.0,
                tier="extreme",
                multiplier=0.25,
            )
            continue

        # Daily log returns
        log_returns = np.diff(np.log(arr))

        # Rolling standard deviation over the last `window` returns
        recent_returns = log_returns[-window:]
        daily_std = float(np.std(recent_returns, ddof=1))

        # Annualize
        annualized = daily_std * np.sqrt(252)

        multiplier, tier = _interpolate_vol_multiplier(annualized)

        results[ticker] = VolatilityMetrics(
            ticker=ticker,
            annualized_vol=round(annualized, 6),
            tier=tier,
            multiplier=round(multiplier, 4),
        )

    return results


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

_CORR_BANDS: list[tuple[float, float]] = [
    # (lower_bound_inclusive, multiplier)
    (0.80, 0.70),
    (0.60, 0.85),
    (0.40, 1.00),
    (0.20, 1.05),
    (-1.0, 1.10),
]


def _corr_multiplier(avg_corr: float) -> float:
    """Map average pairwise correlation to a portfolio multiplier."""
    for lower, mult in _CORR_BANDS:
        if avg_corr >= lower:
            return mult
    return 1.10


def compute_correlation(
    prices: dict[str, list[float]],
    window: int = 60,
) -> CorrelationMetrics:
    """Compute average pairwise correlation and diversification multiplier.

    Uses Pearson correlation of daily log returns over the most recent
    ``window`` trading days.

    Args:
        prices: Mapping of ticker -> list of daily closing prices.
        window: Lookback window in trading days.

    Returns:
        CorrelationMetrics with average correlation, full matrix,
        and portfolio-level multiplier.
    """
    tickers = sorted(prices.keys())

    if len(tickers) < 2:
        # Single ticker -- no diversification
        matrix = {t: {t: 1.0} for t in tickers}
        return CorrelationMetrics(
            avg_correlation=1.0,
            correlation_matrix=matrix,
            multiplier=0.70,
        )

    # Build return matrix -- rows are tickers, columns are time
    min_len = min(len(prices[t]) for t in tickers)
    usable = min(min_len - 1, window)

    if usable < 5:
        # Not enough data for meaningful correlation
        matrix = {t1: {t2: 0.0 for t2 in tickers} for t1 in tickers}
        for t in tickers:
            matrix[t][t] = 1.0
        return CorrelationMetrics(
            avg_correlation=0.0,
            correlation_matrix=matrix,
            multiplier=1.10,
        )

    returns_matrix = []
    for t in tickers:
        arr = np.array(prices[t], dtype=np.float64)
        log_ret = np.diff(np.log(arr))
        returns_matrix.append(log_ret[-usable:])

    returns_np = np.array(returns_matrix)  # shape: (n_tickers, usable)
    corr_np = np.corrcoef(returns_np)  # shape: (n_tickers, n_tickers)

    # Build dict matrix and compute average pairwise correlation
    matrix: dict[str, dict[str, float]] = {}
    pairwise_corrs: list[float] = []

    for i, t1 in enumerate(tickers):
        matrix[t1] = {}
        for j, t2 in enumerate(tickers):
            val = float(corr_np[i, j])
            matrix[t1][t2] = round(val, 4)
            if i < j:
                pairwise_corrs.append(val)

    avg_corr = float(np.mean(pairwise_corrs)) if pairwise_corrs else 0.0
    multiplier = _corr_multiplier(avg_corr)

    return CorrelationMetrics(
        avg_correlation=round(avg_corr, 4),
        correlation_matrix=matrix,
        multiplier=multiplier,
    )


# ---------------------------------------------------------------------------
# Correlation-based exposure caps
# ---------------------------------------------------------------------------


def compute_correlation_cap(
    prices: dict[str, list[float]],
    threshold: float = 0.7,
    window: int = 60,
) -> dict[str, float]:
    """Compute per-ticker position limit multipliers based on correlation.

    For each ticker, counts how many other tickers have a pairwise
    correlation above ``threshold``. The more highly-correlated peers
    a ticker has, the lower its multiplier -- capping aggregate
    exposure to correlated clusters.

    Multiplier schedule:
        0 correlated peers  -> 1.00 (no reduction)
        1 correlated peer   -> 0.80
        2 correlated peers  -> 0.60
        3+ correlated peers -> 0.50

    Args:
        prices: Mapping of ticker -> list of daily closing prices.
        threshold: Correlation above which two tickers are considered
            "highly correlated". Default 0.7.
        window: Lookback window for correlation computation.

    Returns:
        Dict mapping ticker -> multiplier (0.0 to 1.0).
    """
    tickers = sorted(prices.keys())

    if len(tickers) < 2:
        return {t: 1.0 for t in tickers}

    # Reuse existing correlation computation
    corr_metrics = compute_correlation(prices, window=window)
    matrix = corr_metrics.correlation_matrix

    result: dict[str, float] = {}
    for ticker in tickers:
        correlated_count = 0
        for other in tickers:
            if other == ticker:
                continue
            corr_val = matrix.get(ticker, {}).get(other, 0.0)
            if abs(corr_val) > threshold:
                correlated_count += 1

        if correlated_count == 0:
            result[ticker] = 1.00
        elif correlated_count == 1:
            result[ticker] = 0.80
        elif correlated_count == 2:
            result[ticker] = 0.60
        else:
            result[ticker] = 0.50

    return result


# ---------------------------------------------------------------------------
# Position limits
# ---------------------------------------------------------------------------

_BASE_POSITION_PCT = 0.20  # 20% of portfolio per position


def compute_position_limit(
    ticker: str,
    portfolio_value: float,
    vol_metrics: VolatilityMetrics,
    corr_metrics: CorrelationMetrics,
    base_pct: float = _BASE_POSITION_PCT,
) -> PositionLimit:
    """Compute the maximum position size for a ticker.

    Combines a base allocation percentage with volatility and
    correlation adjustments:

        final_pct = base_pct * vol_multiplier * corr_multiplier

    Args:
        ticker: The ticker symbol.
        portfolio_value: Total portfolio value in dollars.
        vol_metrics: Pre-computed volatility metrics for this ticker.
        corr_metrics: Pre-computed correlation metrics for the portfolio.
        base_pct: Base position size as fraction of portfolio. Default 0.20.

    Returns:
        PositionLimit with all intermediate calculations.
    """
    vol_adjusted = base_pct * vol_metrics.multiplier
    final = vol_adjusted * corr_metrics.multiplier
    max_notional = portfolio_value * final

    return PositionLimit(
        ticker=ticker,
        base_pct=round(base_pct, 4),
        vol_adjusted_pct=round(vol_adjusted, 4),
        final_pct=round(final, 4),
        max_notional=round(max_notional, 2),
    )


# ---------------------------------------------------------------------------
# Allowed actions
# ---------------------------------------------------------------------------

def compute_allowed_actions(
    ticker: str,
    portfolio: PortfolioState,
    position_limit: PositionLimit,
    current_price: float,
) -> list[Literal["buy", "sell", "short", "cover", "hold"]]:
    """Determine which trading actions are valid for a ticker.

    Rules:
    - **buy**: allowed if adding shares would not exceed position limit
      and sufficient cash exists.
    - **sell**: allowed if the portfolio holds long shares of this ticker.
    - **short**: allowed if adding short shares would not exceed position
      limit and sufficient margin exists.
    - **cover**: allowed if the portfolio holds short shares of this ticker.
    - **hold**: always allowed.

    Args:
        ticker: The ticker symbol.
        portfolio: Current portfolio state.
        position_limit: Pre-computed position limit for this ticker.
        current_price: Current market price per share.

    Returns:
        List of allowed action strings.
    """
    actions: list[Literal["buy", "sell", "short", "cover", "hold"]] = ["hold"]

    if current_price <= 0:
        return actions

    position = portfolio.positions.get(ticker)

    # Current exposure
    long_shares = position.long_shares if position else 0
    short_shares = position.short_shares if position else 0
    long_notional = long_shares * current_price
    short_notional = short_shares * current_price

    max_notional = position_limit.max_notional

    # Buy: can add long if under limit and have cash
    if long_notional < max_notional and portfolio.cash >= current_price:
        actions.append("buy")

    # Sell: can sell if holding long shares
    if long_shares > 0:
        actions.append("sell")

    # Short: can add short if under limit and have margin
    margin_available = portfolio.cash * (1.0 / portfolio.margin_requirement) - portfolio.margin_used
    if short_notional < max_notional and margin_available >= current_price:
        actions.append("short")

    # Cover: can cover if holding short shares
    if short_shares > 0:
        actions.append("cover")

    return actions
