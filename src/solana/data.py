"""Solana-native data sources: Jupiter Price API + Helius.

Replaces CoinGecko for Solana token data. Free, no-key-required for
Jupiter prices. Helius needs a free API key for enriched data
(free tier: 1M credits/month, hackathon teams get extra credits).

Data flow:
  Jupiter Price API  -> real-time prices (no key)
  Helius DAS API     -> token metadata, holders, supply (free key)
  DexScreener API    -> volume, liquidity, pair data (no key)
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from src.solana.tokens import TOKEN_MINTS, MINT_TO_SYMBOL, get_mint


# ---------------------------------------------------------------------------
# Jupiter Price API (free, no key)
# ---------------------------------------------------------------------------

JUPITER_PRICE_URL = "https://price.jup.ag/v6/price"

# Rate limiting
_last_jup_request: float = 0
_JUP_MIN_INTERVAL: float = 1.0


def _jup_rate_limit() -> None:
    global _last_jup_request
    elapsed = time.time() - _last_jup_request
    if elapsed < _JUP_MIN_INTERVAL:
        time.sleep(_JUP_MIN_INTERVAL - elapsed)
    _last_jup_request = time.time()


def get_token_prices(symbols: list[str]) -> dict[str, float]:
    """Get current USD prices for tokens via Jupiter Price API.

    Args:
        symbols: List of token symbols (e.g. ["SOL", "JUP", "BONK"])

    Returns:
        Dict of symbol -> USD price
    """
    _jup_rate_limit()

    # Jupiter accepts both symbols and mint addresses
    ids = ",".join(symbols)
    try:
        resp = httpx.get(
            JUPITER_PRICE_URL,
            params={"ids": ids},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        prices: dict[str, float] = {}
        for symbol, info in data.get("data", {}).items():
            if info and "price" in info:
                prices[symbol] = float(info["price"])
        return prices
    except Exception:
        return {}


def get_token_price(symbol: str) -> float | None:
    """Get current USD price for a single token."""
    prices = get_token_prices([symbol])
    return prices.get(symbol)


# ---------------------------------------------------------------------------
# Helius API (free key, 1M credits/month)
# ---------------------------------------------------------------------------

HELIUS_API_KEY = os.environ.get("HELIUS_API_KEY", "")
HELIUS_RPC_URL = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}" if HELIUS_API_KEY else ""

_last_helius_request: float = 0
_HELIUS_MIN_INTERVAL: float = 0.2


def _helius_rate_limit() -> None:
    global _last_helius_request
    elapsed = time.time() - _last_helius_request
    if elapsed < _HELIUS_MIN_INTERVAL:
        time.sleep(_HELIUS_MIN_INTERVAL - elapsed)
    _last_helius_request = time.time()


def _helius_available() -> bool:
    return bool(HELIUS_API_KEY)


def get_token_metadata(mint: str) -> dict[str, Any] | None:
    """Get token metadata via Helius DAS getAsset.

    Returns: name, symbol, supply, decimals, holder count, etc.
    """
    if not _helius_available():
        return None

    _helius_rate_limit()
    try:
        resp = httpx.post(
            HELIUS_RPC_URL,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAsset",
                "params": {"id": mint},
            },
            timeout=10,
        )
        result = resp.json()
        if "result" in result:
            asset = result["result"]
            return {
                "name": asset.get("content", {}).get("metadata", {}).get("name", ""),
                "symbol": asset.get("content", {}).get("metadata", {}).get("symbol", ""),
                "supply": asset.get("token_info", {}).get("supply"),
                "decimals": asset.get("token_info", {}).get("decimals"),
                "price_info": asset.get("token_info", {}).get("price_info", {}),
            }
    except Exception:
        pass
    return None


def get_token_holders_count(mint: str) -> int | None:
    """Get approximate holder count for a token via Helius."""
    if not _helius_available():
        return None

    _helius_rate_limit()
    try:
        resp = httpx.post(
            HELIUS_RPC_URL,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccounts",
                "params": {
                    "mint": mint,
                    "limit": 1,
                    "showZeroBalance": False,
                },
            },
            timeout=10,
        )
        result = resp.json()
        if "result" in result:
            # The total field gives us holder count
            return result["result"].get("total", 0)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# DexScreener API (free, no key)
# ---------------------------------------------------------------------------

DEXSCREENER_URL = "https://api.dexscreener.com/latest/dex"

_last_dex_request: float = 0
_DEX_MIN_INTERVAL: float = 1.0


def _dex_rate_limit() -> None:
    global _last_dex_request
    elapsed = time.time() - _last_dex_request
    if elapsed < _DEX_MIN_INTERVAL:
        time.sleep(_DEX_MIN_INTERVAL - elapsed)
    _last_dex_request = time.time()


def get_dex_data(symbol: str) -> dict[str, Any] | None:
    """Get DEX trading data from DexScreener.

    Returns: volume_24h, liquidity, price_change_24h, pair info.
    """
    _dex_rate_limit()
    try:
        mint = get_mint(symbol)
        resp = httpx.get(
            f"{DEXSCREENER_URL}/tokens/{mint}",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        pairs = data.get("pairs", [])
        if not pairs:
            return None

        # Get the highest-liquidity pair
        best = max(pairs, key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0))
        return {
            "price_usd": float(best.get("priceUsd", 0) or 0),
            "volume_24h": float(best.get("volume", {}).get("h24", 0) or 0),
            "liquidity_usd": float(best.get("liquidity", {}).get("usd", 0) or 0),
            "price_change_24h": float(best.get("priceChange", {}).get("h24", 0) or 0),
            "price_change_6h": float(best.get("priceChange", {}).get("h6", 0) or 0),
            "price_change_1h": float(best.get("priceChange", {}).get("h1", 0) or 0),
            "txns_24h_buys": best.get("txns", {}).get("h24", {}).get("buys", 0),
            "txns_24h_sells": best.get("txns", {}).get("h24", {}).get("sells", 0),
            "pair_address": best.get("pairAddress", ""),
            "dex": best.get("dexId", ""),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Combined data feed for analysts
# ---------------------------------------------------------------------------

def get_solana_market_data(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch comprehensive market data for Solana tokens.

    Combines Jupiter (prices), DexScreener (volume/liquidity),
    and Helius (metadata/holders) into a single data bundle
    per token for analyst consumption.
    """
    result: dict[str, dict[str, Any]] = {}

    # 1. Jupiter prices (batch)
    prices = get_token_prices(symbols)

    for symbol in symbols:
        token_data: dict[str, Any] = {
            "symbol": symbol,
            "price_usd": prices.get(symbol),
        }

        # 2. DexScreener data
        dex = get_dex_data(symbol)
        if dex:
            token_data["volume_24h"] = dex["volume_24h"]
            token_data["liquidity_usd"] = dex["liquidity_usd"]
            token_data["price_change_24h"] = dex["price_change_24h"]
            token_data["price_change_6h"] = dex["price_change_6h"]
            token_data["price_change_1h"] = dex["price_change_1h"]
            token_data["txns_24h_buys"] = dex["txns_24h_buys"]
            token_data["txns_24h_sells"] = dex["txns_24h_sells"]
            token_data["buy_sell_ratio"] = (
                dex["txns_24h_buys"] / max(dex["txns_24h_sells"], 1)
            )
            token_data["dex"] = dex["dex"]

        # 3. Helius metadata (if key available)
        if _helius_available():
            try:
                mint = get_mint(symbol)
                meta = get_token_metadata(mint)
                if meta:
                    token_data["supply"] = meta.get("supply")
                    token_data["decimals"] = meta.get("decimals")

                holders = get_token_holders_count(mint)
                if holders:
                    token_data["holders"] = holders
            except (KeyError, Exception):
                pass

        result[symbol] = token_data

    return result
