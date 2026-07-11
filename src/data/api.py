"""Market data API client — Yahoo Finance (free) with financialdatasets.ai fallback.

All market data flows through this module. No analyst agent should
call external APIs directly -- they receive data through this interface.

Data source priority:
  1. Yahoo Finance via yfinance (free, no API key needed)
  2. financialdatasets.ai (paid, requires FINANCIAL_DATASETS_API_KEY)

Set DATA_SOURCE=financialdatasets in .env to force the paid API.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date, timedelta
from typing import Any

import httpx
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.financialdatasets.ai"

# Session-level cache: keyed by (function_name, ticker, params_hash).
# Avoids redundant API calls within a single run.
_cache: dict[tuple[str, str, str], Any] = {}


def clear_cache() -> None:
    """Reset the session cache between runs."""
    _cache.clear()


def _cache_key(func_name: str, ticker: str, **params: Any) -> tuple[str, str, str]:
    """Build a deterministic cache key from function name, ticker, and params."""
    serialized = json.dumps(params, sort_keys=True, default=str)
    params_hash = hashlib.md5(serialized.encode()).hexdigest()
    return (func_name, ticker, params_hash)


def _date_str(d: date) -> str:
    """Convert a date object to ISO string for API params."""
    return d.isoformat()


def _use_paid_api() -> bool:
    """Check if the user wants to force the paid API."""
    return os.getenv("DATA_SOURCE", "").lower() == "financialdatasets"


def _get_api_key() -> str:
    """Retrieve the financialdatasets.ai API key from environment."""
    key = os.getenv("FINANCIAL_DATASETS_API_KEY")
    if not key:
        raise EnvironmentError(
            "FINANCIAL_DATASETS_API_KEY not set. "
            "Copy .env.example to .env and add your key."
        )
    return key


def _headers() -> dict[str, str]:
    """Build request headers with authentication."""
    return {
        "X-API-KEY": _get_api_key(),
        "Accept": "application/json",
    }


def _request_get(url: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """Execute a GET request with CF-COMP-033 retry logic."""
    for attempt in range(2):
        try:
            response = httpx.get(url, headers=_headers(), params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, httpx.TimeoutException, Exception):
            if attempt == 0:
                time.sleep(5)
            else:
                return None
    return None


def _request_post(url: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Execute a POST request with CF-COMP-033 retry logic."""
    for attempt in range(2):
        try:
            response = httpx.post(
                url, headers=_headers(), json=payload, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, httpx.TimeoutException, Exception):
            if attempt == 0:
                time.sleep(5)
            else:
                return None
    return None


# ---------------------------------------------------------------------------
# Yahoo Finance implementations
# ---------------------------------------------------------------------------

def _yf_get_prices(
    ticker: str, start_date: date, end_date: date, interval: str = "day",
) -> list[dict[str, Any]]:
    """Fetch OHLCV from Yahoo Finance."""
    interval_map = {"day": "1d", "week": "1wk", "month": "1mo"}
    yf_interval = interval_map.get(interval, "1d")

    t = yf.Ticker(ticker)
    # yfinance end_date is exclusive, add 1 day
    df = t.history(start=start_date.isoformat(),
                   end=(end_date + timedelta(days=1)).isoformat(),
                   interval=yf_interval)
    if df.empty:
        return []

    result = []
    for idx, row in df.iterrows():
        result.append({
            "date": idx.strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": int(row["Volume"]),
        })
    return result


def _yf_get_financial_metrics(
    ticker: str, end_date: date, period: str = "annual", limit: int = 10,
) -> list[dict[str, Any]]:
    """Derive financial metrics from Yahoo Finance data."""
    t = yf.Ticker(ticker)
    info = t.info or {}

    # Build a single metrics entry from current info
    metrics = {
        "report_period": end_date.isoformat(),
        "period": period,
        "return_on_equity": info.get("returnOnEquity"),
        "return_on_assets": info.get("returnOnAssets"),
        "debt_to_equity": (info.get("debtToEquity") or 0) / 100.0 if info.get("debtToEquity") else None,
        "current_ratio": info.get("currentRatio"),
        "net_margin": info.get("profitMargins"),
        "gross_margin": info.get("grossMargins"),
        "operating_margin": info.get("operatingMargins"),
        "price_to_earnings": info.get("trailingPE") or info.get("forwardPE"),
        "ev_to_ebitda": info.get("enterpriseToEbitda"),
        "market_cap": info.get("marketCap"),
        "earnings_growth": info.get("earningsGrowth"),
        "revenue_growth": info.get("revenueGrowth"),
        "asset_turnover": None,  # not directly available
    }

    # Try to get historical financials for multi-period data
    results = [metrics]

    if period == "annual":
        financials = t.financials  # annual income statement
        bs = t.balance_sheet
    else:
        financials = t.quarterly_financials
        bs = t.quarterly_balance_sheet

    if financials is not None and not financials.empty:
        for col in financials.columns[1:limit]:
            period_metrics = {
                "report_period": col.strftime("%Y-%m-%d") if hasattr(col, 'strftime') else str(col),
                "period": period,
                "return_on_equity": None,
                "return_on_assets": None,
                "debt_to_equity": None,
                "current_ratio": None,
                "net_margin": None,
                "gross_margin": None,
                "market_cap": info.get("marketCap"),
            }
            # Derive margins from income statement
            rev = financials.loc["Total Revenue", col] if "Total Revenue" in financials.index else None
            ni = financials.loc["Net Income", col] if "Net Income" in financials.index else None
            gp = financials.loc["Gross Profit", col] if "Gross Profit" in financials.index else None
            if rev and ni:
                period_metrics["net_margin"] = float(ni / rev)
            if rev and gp:
                period_metrics["gross_margin"] = float(gp / rev)
            results.append(period_metrics)

    return results[:limit]


def _yf_search_line_items(
    ticker: str, line_items: list[str], end_date: date,
    period: str = "annual", limit: int = 10,
) -> list[dict[str, Any]]:
    """Get financial statement line items from Yahoo Finance."""
    t = yf.Ticker(ticker)

    if period == "annual":
        financials = t.financials
        cf = t.cashflow
        bs = t.balance_sheet
    else:
        financials = t.quarterly_financials
        cf = t.quarterly_cashflow
        bs = t.quarterly_balance_sheet

    # Map our line item names to yfinance row labels
    yf_map = {
        "revenue": ("Total Revenue", financials),
        "net_income": ("Net Income", financials),
        "free_cash_flow": ("Free Cash Flow", cf),
        "operating_cash_flow": ("Operating Cash Flow", cf),
        "outstanding_shares": ("Ordinary Shares Number", bs),
    }

    results = []
    frames_to_check = {}
    for item in line_items:
        mapping = yf_map.get(item)
        if mapping:
            frames_to_check[item] = mapping

    # Collect data across periods
    # Determine available periods from the first non-empty dataframe
    period_cols = []
    for item, (row_name, df) in frames_to_check.items():
        if df is not None and not df.empty:
            period_cols = list(df.columns[:limit])
            break

    for col in period_cols:
        period_data: dict[str, Any] = {
            "period_end_date": col.strftime("%Y-%m-%d") if hasattr(col, 'strftime') else str(col),
        }
        for item, (row_name, df) in frames_to_check.items():
            if df is not None and not df.empty and row_name in df.index:
                try:
                    val = df.loc[row_name, col]
                    period_data[item] = float(val) if val is not None else None
                except (KeyError, TypeError):
                    period_data[item] = None
            else:
                period_data[item] = None
        results.append(period_data)

    return results[:limit]


def _yf_get_insider_trades(
    ticker: str, end_date: date, limit: int = 50,
) -> list[dict[str, Any]]:
    """Get insider transactions from Yahoo Finance."""
    t = yf.Ticker(ticker)
    try:
        insiders = t.insider_transactions
    except Exception:
        return []

    if insiders is None or insiders.empty:
        return []

    results = []
    for _, row in insiders.head(limit).iterrows():
        tx_type = str(row.get("Text", row.get("Transaction", "")))
        results.append({
            "transaction_type": tx_type,
            "shares": abs(int(row.get("Shares", 0))) if row.get("Shares") else 0,
            "price_per_share": float(row.get("Value", 0)) / max(abs(int(row.get("Shares", 1))), 1) if row.get("Value") else 0,
            "date": str(row.get("Start Date", row.get("Date", "")))[:10],
            "owner": str(row.get("Insider", "")),
        })
    return results[:limit]


# ---------------------------------------------------------------------------
# Public API (same interface regardless of data source)
# ---------------------------------------------------------------------------

def get_prices(
    ticker: str,
    start_date: date,
    end_date: date,
    interval: str = "day",
) -> list[dict[str, Any]]:
    """Fetch OHLCV price bars for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        start_date: Start of date range.
        end_date: End of date range.
        interval: Bar interval -- "day", "week", or "month".

    Returns:
        List of price bar dicts with keys: date, open, high, low,
        close, volume.
    """
    key = _cache_key(
        "get_prices", ticker,
        start_date=start_date, end_date=end_date, interval=interval,
    )
    if key in _cache:
        return _cache[key]

    if _use_paid_api():
        params = {
            "ticker": ticker,
            "interval": interval,
            "interval_multiplier": 1,
            "start_date": _date_str(start_date),
            "end_date": _date_str(end_date),
        }
        data = _request_get(f"{BASE_URL}/prices/", params)
        result = data.get("prices", []) if data else []
    else:
        result = _yf_get_prices(ticker, start_date, end_date, interval)

    _cache[key] = result
    return result


def get_financial_metrics(
    ticker: str,
    end_date: date,
    period: str = "annual",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Fetch financial ratios and metrics for a ticker.

    Returns metrics including P/E, ROE, margins, debt ratios,
    growth rates, and efficiency metrics.

    Args:
        ticker: Stock ticker symbol.
        end_date: Fetch metrics reported on or before this date.
        period: Reporting period -- "annual", "quarterly", or "ttm".
        limit: Maximum number of periods to return.

    Returns:
        List of metric dicts, most recent first.
    """
    key = _cache_key(
        "get_financial_metrics", ticker,
        end_date=end_date, period=period, limit=limit,
    )
    if key in _cache:
        return _cache[key]

    if _use_paid_api():
        params = {
            "ticker": ticker,
            "report_period": period,
            "limit": limit,
            "end_date": _date_str(end_date),
        }
        data = _request_get(f"{BASE_URL}/financial-metrics/", params)
        result = data.get("financial_metrics", []) if data else []
    else:
        result = _yf_get_financial_metrics(ticker, end_date, period, limit)

    _cache[key] = result
    return result


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: date,
    period: str = "annual",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search for specific financial statement line items.

    Args:
        ticker: Stock ticker symbol.
        line_items: List of line item names to search for.
        end_date: Fetch items reported on or before this date.
        period: Reporting period -- "annual", "quarterly", or "ttm".
        limit: Maximum number of periods to return.

    Returns:
        List of line item dicts with period, name, and value.
    """
    key = _cache_key(
        "search_line_items", ticker,
        line_items=line_items, end_date=end_date, period=period, limit=limit,
    )
    if key in _cache:
        return _cache[key]

    if _use_paid_api():
        payload = {
            "tickers": [ticker],
            "line_items": line_items,
            "end_date": _date_str(end_date),
            "period": period,
            "limit": limit,
        }
        data = _request_post(f"{BASE_URL}/financials/search/line-items", payload)
        result = data.get("search_results", []) if data else []
    else:
        result = _yf_search_line_items(ticker, line_items, end_date, period, limit)

    _cache[key] = result
    return result


def get_insider_trades(
    ticker: str,
    end_date: date,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fetch insider trading transactions for a ticker.

    Args:
        ticker: Stock ticker symbol.
        end_date: Fetch trades on or before this date.
        limit: Maximum number of trades to return.

    Returns:
        List of insider trade dicts with transaction_type,
        shares, price, owner details.
    """
    key = _cache_key(
        "get_insider_trades", ticker,
        end_date=end_date, limit=limit,
    )
    if key in _cache:
        return _cache[key]

    if _use_paid_api():
        params = {
            "ticker": ticker,
            "end_date": _date_str(end_date),
            "limit": limit,
        }
        data = _request_get(f"{BASE_URL}/insider-trades/", params)
        result = data.get("insider_trades", []) if data else []
    else:
        result = _yf_get_insider_trades(ticker, end_date, limit)

    _cache[key] = result
    return result


def get_company_news(
    ticker: str,
    end_date: date,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fetch company news articles with sentiment tags.

    Args:
        ticker: Stock ticker symbol.
        end_date: Fetch news on or before this date.
        limit: Maximum number of articles to return.

    Returns:
        List of news dicts with title, summary, sentiment,
        source, published_at.
    """
    key = _cache_key(
        "get_company_news", ticker,
        end_date=end_date, limit=limit,
    )
    if key in _cache:
        return _cache[key]

    if _use_paid_api():
        params = {
            "ticker": ticker,
            "end_date": _date_str(end_date),
            "limit": limit,
        }
        data = _request_get(f"{BASE_URL}/news/", params)
        result = data.get("news", []) if data else []
    else:
        # yfinance news doesn't have structured sentiment -- return empty
        result = []

    _cache[key] = result
    return result


def get_market_cap(
    ticker: str,
    end_date: date,
) -> float | None:
    """Fetch the market capitalization for a ticker.

    Args:
        ticker: Stock ticker symbol.
        end_date: Fetch market cap as of this date.

    Returns:
        Market cap in dollars, or None if unavailable.
    """
    metrics = get_financial_metrics(ticker, end_date, period="quarterly", limit=1)
    if not metrics:
        return None
    latest = metrics[0]
    cap = latest.get("market_cap")
    if cap is not None:
        return float(cap)
    return None
