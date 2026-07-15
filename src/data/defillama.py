"""DeFi Llama data source for total value locked (TVL) metrics.

Free API, no key required, no rate limit.
Base URL: https://api.llama.fi

Provides chain-level TVL data that complements CoinGecko price/supply
metrics. TVL trends signal capital flows into/out of ecosystems.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Any

# ---------------------------------------------------------------------------
# Ticker -> DeFi Llama chain name mapping
# ---------------------------------------------------------------------------

TICKER_TO_CHAIN: dict[str, str] = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "BNB": "BSC",
    "SOL": "Solana",
    "AVAX": "Avalanche",
    "MATIC": "Polygon",
    "ARB": "Arbitrum",
    "OP": "Optimism",
    "ADA": "Cardano",
    "DOT": "Polkadot",
    "ATOM": "Cosmos",
    "NEAR": "Near",
    "APT": "Aptos",
    "XLM": "Stellar",
    "LINK": "Ethereum",    # Chainlink lives on Ethereum
    "UNI": "Ethereum",     # Uniswap lives on Ethereum
}

DEFILLAMA_BASE = "https://api.llama.fi"

# Session-level cache: avoid redundant calls within a single run.
_dl_cache: dict[str, Any] = {}


def clear_defillama_cache() -> None:
    """Reset the DeFi Llama session cache between runs."""
    _dl_cache.clear()


def _dl_request(path: str) -> Any | None:
    """Make a DeFi Llama API GET request.

    Args:
        path: API path (e.g., "/v2/historicalChainTvl/Ethereum").

    Returns:
        Parsed JSON response, or None on failure.
    """
    if path in _dl_cache:
        return _dl_cache[path]

    url = f"{DEFILLAMA_BASE}{path}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "CovenantHedgeFund/1.0",
        },
    )

    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                _dl_cache[path] = data
                return data
        except urllib.error.HTTPError:
            if attempt == 0:
                time.sleep(2)
            else:
                return None
        except (urllib.error.URLError, Exception):
            if attempt == 0:
                time.sleep(2)
            else:
                return None
    return None


def get_chain_tvl(chain: str) -> dict[str, Any]:
    """Fetch current and historical TVL for a blockchain.

    Args:
        chain: DeFi Llama chain name (e.g., "Ethereum", "Solana").

    Returns:
        Dict with keys: current_tvl, tvl_30d_ago, tvl_change_pct_30d,
        tvl_7d_ago, tvl_change_pct_7d. Empty dict on failure.
    """
    data = _dl_request(f"/v2/historicalChainTvl/{chain}")

    if not data or not isinstance(data, list) or len(data) < 2:
        return {}

    # Data is sorted chronologically: [{date: unix_ts, tvl: float}, ...]
    current_tvl = data[-1].get("tvl", 0)
    if current_tvl <= 0:
        return {}

    result: dict[str, Any] = {"current_tvl": current_tvl}

    # Find TVL ~7 days ago
    if len(data) >= 7:
        tvl_7d = data[-7].get("tvl", 0)
        if tvl_7d > 0:
            result["tvl_7d_ago"] = tvl_7d
            result["tvl_change_pct_7d"] = (
                (current_tvl - tvl_7d) / tvl_7d * 100
            )

    # Find TVL ~30 days ago
    if len(data) >= 30:
        tvl_30d = data[-30].get("tvl", 0)
        if tvl_30d > 0:
            result["tvl_30d_ago"] = tvl_30d
            result["tvl_change_pct_30d"] = (
                (current_tvl - tvl_30d) / tvl_30d * 100
            )

    return result


def get_tvl_for_ticker(ticker: str) -> dict[str, Any]:
    """Fetch TVL data for a crypto ticker symbol.

    Maps the ticker to a DeFi Llama chain name and fetches TVL.
    Returns empty dict if the ticker has no chain mapping or on failure.

    Args:
        ticker: Crypto ticker symbol (e.g., "ETH", "SOL").

    Returns:
        TVL data dict (see get_chain_tvl), or empty dict.
    """
    chain = TICKER_TO_CHAIN.get(ticker.upper())
    if not chain:
        return {}

    # Cache key includes ticker to handle LINK/UNI sharing Ethereum
    cache_key = f"_ticker_tvl_{ticker.upper()}"
    if cache_key in _dl_cache:
        return _dl_cache[cache_key]

    result = get_chain_tvl(chain)
    _dl_cache[cache_key] = result
    return result
