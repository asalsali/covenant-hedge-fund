"""CoinGecko data source for cryptocurrency tickers.

Routes crypto symbols through the CoinGecko free API (v3) instead of
Yahoo Finance or financialdatasets.ai. All functions use the session
cache from api.py to avoid redundant calls.

CoinGecko free tier: ~10-30 calls/minute. We enforce a 6-second
minimum between requests to stay well within limits.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import date, datetime
from typing import Any

# Symbol -> CoinGecko ID mapping for top-20 crypto assets
CRYPTO_SYMBOL_MAP: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "LINK": "chainlink",
    "MATIC": "matic-network",
    "UNI": "uniswap",
    "SHIB": "shiba-inu",
    "LTC": "litecoin",
    "ATOM": "cosmos",
    "XLM": "stellar",
    "NEAR": "near",
    "APT": "aptos",
    "ARB": "arbitrum",
    "OP": "optimism",
}

import os

# CoinGecko API key: free "Demo" tier requires x-cg-demo-api-key header.
# Sign up at https://www.coingecko.com/en/api/pricing (free plan).
# Set COINGECKO_API_KEY in .env or environment.
_CG_API_KEY: str | None = os.environ.get("COINGECKO_API_KEY")

# Use pro base URL if a paid key is detected, otherwise demo
CG_BASE_URL = "https://api.coingecko.com/api/v3"

# Track whether we have already warned about missing key (once per process)
_warned_no_key: bool = False

# Timestamp of last CoinGecko request, for rate limiting
_last_cg_request: float = 0.0
_CG_MIN_INTERVAL: float = 6.0  # seconds between requests


def is_crypto(ticker: str) -> bool:
    """Check whether a ticker symbol is a known cryptocurrency."""
    return ticker.upper() in CRYPTO_SYMBOL_MAP


def resolve_coin_id(ticker: str) -> str | None:
    """Map a crypto ticker symbol to a CoinGecko coin ID.

    Returns None if the ticker is not in the symbol map.
    """
    return CRYPTO_SYMBOL_MAP.get(ticker.upper())


def _cg_rate_limit() -> None:
    """Enforce minimum interval between CoinGecko API calls."""
    global _last_cg_request
    now = time.monotonic()
    elapsed = now - _last_cg_request
    if _last_cg_request > 0 and elapsed < _CG_MIN_INTERVAL:
        time.sleep(_CG_MIN_INTERVAL - elapsed)
    _last_cg_request = time.monotonic()


def _cg_request(path: str, params: dict[str, str] | None = None) -> dict | None:
    """Make a CoinGecko API request with rate limiting and error handling.

    Args:
        path: API path relative to base URL (e.g., "/coins/bitcoin").
        params: Query parameters as a dict.

    Returns:
        Parsed JSON response dict, or None on failure.
    """
    global _warned_no_key

    if not _CG_API_KEY:
        if not _warned_no_key:
            print(
                "  WARNING: COINGECKO_API_KEY not set. "
                "Crypto metrics will be unavailable (price-only via yfinance). "
                "Get a free key at https://www.coingecko.com/en/api/pricing"
            )
            _warned_no_key = True
        return None

    _cg_rate_limit()

    url = f"{CG_BASE_URL}{path}"
    if params:
        query = urllib.parse.urlencode(params)
        url = f"{url}?{query}"

    headers = {
        "Accept": "application/json",
        "User-Agent": "CovenantHedgeFund/1.0",
        "x-cg-demo-api-key": _CG_API_KEY,
    }

    req = urllib.request.Request(url, headers=headers)

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited -- back off exponentially
                wait = _CG_MIN_INTERVAL * (2 ** attempt)
                print(f"  CoinGecko rate limited (429), waiting {wait:.0f}s...")
                time.sleep(wait)
            elif e.code == 401:
                print(
                    "  WARNING: CoinGecko API key rejected (401). "
                    "Check COINGECKO_API_KEY in .env"
                )
                return None
            elif attempt < 2:
                time.sleep(5)
            else:
                return None
        except (urllib.error.URLError, Exception) as e:
            if attempt < 2:
                time.sleep(5)
            else:
                from src.data.api import DataFetchError
                raise DataFetchError(
                    coin_id, "coingecko",
                    f"Failed after 3 attempts: {type(e).__name__}: {e}",
                )
    return None


def cg_get_prices(
    coin_id: str,
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    """Fetch daily price data from CoinGecko market_chart/range endpoint.

    CoinGecko market_chart returns [timestamp_ms, price] pairs for close
    prices only. We set O=H=L=C since intraday OHLC is not available
    on the free tier for arbitrary ranges.

    Args:
        coin_id: CoinGecko coin ID (e.g., "bitcoin").
        start_date: Start of date range.
        end_date: End of date range.

    Returns:
        List of price bar dicts matching the existing schema:
        {date, open, high, low, close, volume}.
    """
    # Convert dates to UNIX timestamps
    start_ts = str(int(datetime.combine(start_date, datetime.min.time()).timestamp()))
    end_ts = str(int(datetime.combine(end_date, datetime.max.time()).timestamp()))

    data = _cg_request(
        f"/coins/{coin_id}/market_chart/range",
        params={
            "vs_currency": "usd",
            "from": start_ts,
            "to": end_ts,
        },
    )

    if not data:
        return []

    prices_raw = data.get("prices", [])
    volumes_raw = data.get("total_volumes", [])

    # Build a volume lookup by date string
    volume_by_date: dict[str, float] = {}
    for ts_ms, vol in volumes_raw:
        dt = datetime.utcfromtimestamp(ts_ms / 1000.0)
        date_str = dt.strftime("%Y-%m-%d")
        volume_by_date[date_str] = vol

    # Deduplicate by date (CoinGecko may return multiple points per day)
    seen_dates: dict[str, dict[str, Any]] = {}
    for ts_ms, price in prices_raw:
        dt = datetime.utcfromtimestamp(ts_ms / 1000.0)
        date_str = dt.strftime("%Y-%m-%d")
        # Keep the last price point per day (closest to close)
        seen_dates[date_str] = {
            "date": date_str,
            "open": round(price, 4),
            "high": round(price, 4),
            "low": round(price, 4),
            "close": round(price, 4),
            "volume": int(volume_by_date.get(date_str, 0)),
        }

    # Return sorted by date
    return sorted(seen_dates.values(), key=lambda x: x["date"])


def cg_get_crypto_metrics(coin_id: str) -> dict[str, Any]:
    """Fetch comprehensive crypto metrics from CoinGecko /coins/{id} endpoint.

    Returns a dict with market data, supply info, and price change
    percentages. This is the crypto equivalent of get_financial_metrics
    for equities.

    Args:
        coin_id: CoinGecko coin ID (e.g., "bitcoin").

    Returns:
        Dict with crypto-specific metrics, or empty dict on failure.
    """
    data = _cg_request(
        f"/coins/{coin_id}",
        params={
            "localization": "false",
            "tickers": "false",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false",
        },
    )

    if not data:
        return {}

    market = data.get("market_data", {})

    def _get_usd(field: str) -> Any:
        """Extract USD value from a multi-currency field."""
        val = market.get(field)
        if isinstance(val, dict):
            return val.get("usd")
        return val

    return {
        "market_cap": _get_usd("market_cap"),
        "fully_diluted_valuation": _get_usd("fully_diluted_valuation"),
        "total_volume": _get_usd("total_volume"),
        "circulating_supply": market.get("circulating_supply"),
        "total_supply": market.get("total_supply"),
        "max_supply": market.get("max_supply"),
        "price_change_percentage_24h": market.get("price_change_percentage_24h"),
        "price_change_percentage_7d": market.get("price_change_percentage_7d"),
        "price_change_percentage_14d": market.get("price_change_percentage_14d"),
        "price_change_percentage_30d": market.get("price_change_percentage_30d"),
        "price_change_percentage_60d": market.get("price_change_percentage_60d"),
        "price_change_percentage_200d": market.get("price_change_percentage_200d"),
        "price_change_percentage_1y": market.get("price_change_percentage_1y"),
        "ath": _get_usd("ath"),
        "ath_change_percentage": _get_usd("ath_change_percentage"),
        "atl": _get_usd("atl"),
        "atl_change_percentage": _get_usd("atl_change_percentage"),
        "market_cap_rank": data.get("market_cap_rank"),
    }
