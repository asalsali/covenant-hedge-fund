"""Event study engine for statistical validation of trading signals.

Implements the Cumulative Abnormal Return (CAR) methodology:
1. Estimate a market model (OLS: stock = alpha + beta * SPY) on a
   pre-event estimation window.
2. Compute abnormal returns (actual - expected) around each signal date.
3. Aggregate CARs across event windows and test for significance.

Usage:
    from src.event_study import EventStudy, format_car_results
    es = EventStudy(all_prices, spy_prices)
    results = es.analyze_signals(trades)
    print(format_car_results(results))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MarketModelFit:
    """Result of an OLS market model regression."""
    alpha: float
    beta: float
    residual_std: float
    r_squared: float
    n_obs: int


@dataclass
class WindowResult:
    """CAR result for a single event window."""
    window: tuple[int, int]
    car: float
    t_stat: float
    p_value: float
    n_days: int
    significance: str  # "", "*", "**", "***"


@dataclass
class EventResult:
    """Full result for a single event (signal date + ticker)."""
    ticker: str
    event_date: str
    action: str
    windows: list[WindowResult] = field(default_factory=list)
    model: MarketModelFit | None = None
    bootstrap_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    error: str | None = None


@dataclass
class CARSummary:
    """Aggregate CAR results across all events."""
    events: list[EventResult] = field(default_factory=list)
    aggregate: dict[str, dict[str, float]] = field(default_factory=dict)
    verdict: str = ""
    n_events: int = 0
    n_significant: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normal_sf(x: float) -> float:
    """Survival function for standard normal (1 - CDF)."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def _t_test(car: float, residual_std: float, n_days: int) -> tuple[float, float]:
    """T-statistic and two-tailed p-value for a CAR."""
    if residual_std <= 0 or n_days <= 0:
        return 0.0, 1.0
    car_std = residual_std * math.sqrt(n_days)
    if car_std <= 0:
        return 0.0, 1.0
    t_stat = car / car_std
    try:
        from scipy.stats import t as t_dist
        p_value = float(2 * t_dist.sf(abs(t_stat), df=max(1, n_days - 1)))
    except ImportError:
        p_value = 2.0 * _normal_sf(abs(t_stat))
    return t_stat, p_value


def _significance(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _bootstrap_ci(
    abnormal_returns: np.ndarray,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval for CAR."""
    n = len(abnormal_returns)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed=42)
    boot_cars = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(abnormal_returns, size=n, replace=True)
        boot_cars[i] = np.sum(sample)
    alpha_half = (1 - confidence) / 2
    lower = float(np.percentile(boot_cars, alpha_half * 100))
    upper = float(np.percentile(boot_cars, (1 - alpha_half) * 100))
    return (lower, upper)


# ---------------------------------------------------------------------------
# MarketModel
# ---------------------------------------------------------------------------

class MarketModel:
    """OLS market model: R_stock = alpha + beta * R_market + epsilon."""

    @staticmethod
    def fit(
        stock_returns: np.ndarray,
        spy_returns: np.ndarray,
    ) -> MarketModelFit:
        """Fit OLS on pre-computed return arrays.

        Args:
            stock_returns: Daily returns for the estimation window.
            spy_returns: SPY daily returns, same length.

        Returns:
            MarketModelFit with alpha, beta, residual_std, r_squared.
        """
        n_obs = len(stock_returns)
        if n_obs < 20:
            raise ValueError(f"Need >= 20 observations, got {n_obs}")

        X = np.column_stack([np.ones(n_obs), spy_returns])
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)

        beta_hat = XtX_inv @ (X.T @ stock_returns)
        alpha = float(beta_hat[0])
        beta = float(beta_hat[1])

        residuals = stock_returns - (X @ beta_hat)
        residual_std = float(np.std(residuals, ddof=2))

        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((stock_returns - np.mean(stock_returns)) ** 2))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return MarketModelFit(
            alpha=alpha, beta=beta, residual_std=residual_std,
            r_squared=r_squared, n_obs=n_obs,
        )


# ---------------------------------------------------------------------------
# EventStudy
# ---------------------------------------------------------------------------

class EventStudy:
    """Runs event studies on trading signals to measure abnormal returns."""

    DEFAULT_WINDOWS = [(0, 1), (0, 5), (0, 20)]

    def __init__(
        self,
        all_prices: dict[str, list[dict[str, Any]]],
        spy_prices: list[dict[str, Any]],
    ) -> None:
        self.all_prices = all_prices
        self.spy_prices = spy_prices

        # Build date-indexed Series for each ticker and SPY
        self._stock_series: dict[str, pd.Series] = {}
        for ticker, bars in all_prices.items():
            dates = [b["date"] for b in bars if b.get("close") is not None]
            closes = [b["close"] for b in bars if b.get("close") is not None]
            if dates:
                self._stock_series[ticker] = pd.Series(closes, index=dates)

        spy_dates = [b["date"] for b in spy_prices if b.get("close") is not None]
        spy_closes = [b["close"] for b in spy_prices if b.get("close") is not None]
        self._spy_series = (
            pd.Series(spy_closes, index=spy_dates) if spy_dates
            else pd.Series(dtype=float)
        )

    def analyze_event(
        self,
        ticker: str,
        event_date: str,
        action: str = "buy",
        windows: list[tuple[int, int]] | None = None,
        estimation_window: tuple[int, int] = (-250, -11),
        n_bootstrap: int = 10_000,
    ) -> EventResult:
        """Analyze a single event for one ticker."""
        if windows is None:
            windows = self.DEFAULT_WINDOWS
        result = EventResult(ticker=ticker, event_date=event_date, action=action)

        if ticker not in self._stock_series:
            result.error = f"No price data for {ticker}"
            return result

        stock_s = self._stock_series[ticker]
        spy_s = self._spy_series
        if stock_s.empty or spy_s.empty:
            result.error = "Empty price series"
            return result

        all_dates = sorted(stock_s.index)
        # Snap to nearest available date
        if event_date not in all_dates:
            before = [d for d in all_dates if d <= event_date]
            if not before:
                result.error = f"Event date {event_date} before data"
                return result
            event_date = before[-1]

        event_idx = all_dates.index(event_date)
        max_post = max(w[1] for w in windows)
        needed_before = abs(estimation_window[0])

        if event_idx < needed_before:
            result.error = f"Insufficient pre-event data: {event_idx}/{needed_before}"
            return result
        if event_idx + max_post >= len(all_dates):
            result.error = f"Insufficient post-event data"
            return result

        # Build aligned arrays over the full range needed
        rng_start = max(0, event_idx + estimation_window[0])
        rng_end = min(len(all_dates), event_idx + max_post + 1)
        range_dates = all_dates[rng_start:rng_end]

        a_stock, a_spy, a_dates = [], [], []
        for d in range_dates:
            if d in stock_s.index and d in spy_s.index:
                a_stock.append(float(stock_s[d]))
                a_spy.append(float(spy_s[d]))
                a_dates.append(d)

        if len(a_stock) < 30:
            result.error = f"Too few aligned obs: {len(a_stock)}"
            return result

        stock_arr = np.array(a_stock)
        spy_arr = np.array(a_spy)

        if event_date not in a_dates:
            result.error = f"Event date not in aligned data"
            return result
        event_pos = a_dates.index(event_date)

        # Returns
        stock_ret = np.diff(stock_arr) / stock_arr[:-1]
        spy_ret = np.diff(spy_arr) / spy_arr[:-1]

        # Estimation window
        est_start = max(0, event_pos + estimation_window[0])
        est_end = min(event_pos + estimation_window[1], len(stock_ret))
        est_end = max(est_start + 1, est_end)

        if est_end - est_start < 20:
            result.error = f"Estimation window too short: {est_end - est_start}"
            return result

        # Fit market model
        try:
            model_fit = MarketModel.fit(
                stock_ret[est_start:est_end],
                spy_ret[est_start:est_end],
            )
        except ValueError as e:
            result.error = str(e)
            return result
        result.model = model_fit

        # Compute CARs per window
        for window in windows:
            w_start, w_end = window
            ret_start = event_pos + w_start
            ret_end = event_pos + w_end

            if ret_start < 0 or ret_end >= len(stock_ret):
                result.windows.append(WindowResult(
                    window=window, car=0.0, t_stat=0.0,
                    p_value=1.0, n_days=0, significance="",
                ))
                continue

            actual = stock_ret[ret_start:ret_end + 1]
            market = spy_ret[ret_start:ret_end + 1]
            n_days = len(actual)
            if n_days == 0:
                result.windows.append(WindowResult(
                    window=window, car=0.0, t_stat=0.0,
                    p_value=1.0, n_days=0, significance="",
                ))
                continue

            expected = model_fit.alpha + model_fit.beta * market
            abnormal = actual - expected
            car = float(np.sum(abnormal))

            # Negate for short/sell signals
            if action in ("sell", "short"):
                car = -car

            t_stat, p_value = _t_test(car, model_fit.residual_std, n_days)
            sig = _significance(p_value)

            result.windows.append(WindowResult(
                window=window, car=car, t_stat=t_stat,
                p_value=p_value, n_days=n_days, significance=sig,
            ))

            if n_bootstrap > 0:
                ci = _bootstrap_ci(abnormal, n_bootstrap)
                result.bootstrap_ci[f"({w_start},{w_end})"] = ci

        return result

    def analyze_signals(
        self,
        trades: list[dict[str, Any]],
        windows: list[tuple[int, int]] | None = None,
        n_bootstrap: int = 10_000,
    ) -> CARSummary:
        """Batch analyze all trade entry dates."""
        if windows is None:
            windows = self.DEFAULT_WINDOWS
        summary = CARSummary()

        # Deduplicate events
        seen: set[tuple[str, str, str]] = set()
        events: list[tuple[str, str, str]] = []
        for trade in trades:
            action = trade["action"]
            if action in ("buy",):
                direction = "buy"
            elif action in ("sell", "short"):
                direction = "sell"
            else:
                continue
            key = (trade["ticker"], trade["date"], direction)
            if key not in seen:
                seen.add(key)
                events.append(key)

        for ticker, date, direction in events:
            result = self.analyze_event(
                ticker, date, direction, windows, n_bootstrap=n_bootstrap,
            )
            summary.events.append(result)

        summary.n_events = len([e for e in summary.events if not e.error])

        # Aggregate per window
        for window in windows:
            label = f"({window[0]},{window[1]})"
            cars, p_values = [], []
            for event in summary.events:
                if event.error:
                    continue
                for wr in event.windows:
                    if wr.window == window and wr.n_days > 0:
                        cars.append(wr.car)
                        p_values.append(wr.p_value)

            if not cars:
                continue
            n = len(cars)
            mean_car = float(np.mean(cars))
            median_car = float(np.median(cars))
            std_car = float(np.std(cars, ddof=1)) if n > 1 else 0.0

            if std_car > 0 and n > 1:
                agg_t = mean_car / (std_car / math.sqrt(n))
                try:
                    from scipy.stats import t as t_dist
                    agg_p = float(2 * t_dist.sf(abs(agg_t), df=n - 1))
                except ImportError:
                    agg_p = 2.0 * _normal_sf(abs(agg_t))
            else:
                agg_t, agg_p = 0.0, 1.0

            n_sig = sum(1 for p in p_values if p < 0.05)
            pct_pos = sum(1 for c in cars if c > 0) / n * 100

            summary.aggregate[label] = {
                "mean_car": mean_car, "median_car": median_car,
                "std_car": std_car, "agg_t_stat": agg_t,
                "agg_p_value": agg_p, "n_events": n,
                "n_significant": n_sig, "pct_positive": pct_pos,
            }

        # Count individually significant events
        for event in summary.events:
            if not event.error and any(wr.p_value < 0.05 for wr in event.windows):
                summary.n_significant += 1

        # Verdict
        summary.verdict = _compute_verdict(summary)
        return summary


def _compute_verdict(summary: CARSummary) -> str:
    """Determine the verdict string."""
    if summary.n_events == 0:
        return "No valid events to analyze."
    primary = "(0,5)" if "(0,5)" in summary.aggregate else (
        next(iter(summary.aggregate), None)
    )
    if not primary or primary not in summary.aggregate:
        return "Insufficient data for aggregate significance test."
    agg = summary.aggregate[primary]
    if agg["agg_p_value"] < 0.05 and agg["mean_car"] > 0:
        return (
            "CONFIRMED: Signals produce statistically significant "
            f"abnormal returns (p={agg['agg_p_value']:.4f}, "
            f"mean CAR={agg['mean_car']:+.2%} over {primary} window)."
        )
    if agg["mean_car"] > 0:
        return (
            f"INCONCLUSIVE: Positive mean CAR ({agg['mean_car']:+.2%}) "
            f"but not statistically significant (p={agg['agg_p_value']:.4f})."
        )
    return (
        f"NOT CONFIRMED: Mean CAR is negative ({agg['mean_car']:+.2%}, "
        f"p={agg['agg_p_value']:.4f}). Signals do not produce "
        "abnormal returns beyond market exposure."
    )
