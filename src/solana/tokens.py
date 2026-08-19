"""Solana token registry — mint addresses for tradeable tokens."""

from __future__ import annotations

# Native SOL wrapped address
WSOL = "So11111111111111111111111111111111111111112"

# Stablecoins
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"

# Major Solana ecosystem tokens
TOKEN_MINTS: dict[str, str] = {
    "SOL": WSOL,
    "USDC": USDC,
    "USDT": USDT,
    "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "WIF": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    "JTO": "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL",
    "PYTH": "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3",
    "RAY": "4k3Dyjzvzp8eMZFUEDRexMhn8UrYNA77DUerFKaYLwEa",
    "ORCA": "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
    "MSOL": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
    "ANSEM": "HeLp6NuQkmYB4pYWo2zYs22mESHXPQYzXbB8n4V98jwC",
}

# Reverse lookup: mint -> symbol
MINT_TO_SYMBOL: dict[str, str] = {v: k for k, v in TOKEN_MINTS.items()}


def get_mint(symbol: str) -> str:
    """Get mint address for a token symbol. Raises KeyError if unknown."""
    symbol = symbol.upper()
    if symbol in TOKEN_MINTS:
        return TOKEN_MINTS[symbol]
    raise KeyError(f"Unknown token: {symbol}. Available: {list(TOKEN_MINTS.keys())}")


def get_symbol(mint: str) -> str:
    """Get symbol for a mint address. Returns truncated mint if unknown."""
    return MINT_TO_SYMBOL.get(mint, mint[:8] + "...")
