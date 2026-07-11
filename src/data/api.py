"""Market data API client for financialdatasets.ai.

All market data flows through this module. No analyst agent should
call external APIs directly -- they receive data through this interface.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.financialdatasets.ai"


def _get_api_key() -> str:
    """Retrieve the API key from environment."""
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
    # TODO: Implement API call to /prices/ endpoint
    # TODO: Add session-level caching to avoid redundant calls
    raise NotImplementedError("get_prices not yet implemented")


def get_financial_metrics(
    ticker: str,
    end_date: date,
    period: str = "annual",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Fetch financial ratios and metrics for a ticker.

    Returns 50+ metrics including P/E, ROE, margins, debt ratios,
    growth rates, and efficiency metrics.

    Args:
        ticker: Stock ticker symbol.
        end_date: Fetch metrics reported on or before this date.
        period: Reporting period -- "annual", "quarterly", or "ttm".
        limit: Maximum number of periods to return.

    Returns:
        List of metric dicts, most recent first.
    """
    # TODO: Implement API call to /financial-metrics/ endpoint
    raise NotImplementedError("get_financial_metrics not yet implemented")


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
    # TODO: Implement API call to /financials/search/line-items endpoint
    raise NotImplementedError("search_line_items not yet implemented")


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
    # TODO: Implement API call to /insider-trades/ endpoint
    raise NotImplementedError("get_insider_trades not yet implemented")


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
    # TODO: Implement API call to /news/ endpoint
    raise NotImplementedError("get_company_news not yet implemented")


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
    # TODO: Implement via /company/facts/ or derive from financial metrics
    raise NotImplementedError("get_market_cap not yet implemented")
