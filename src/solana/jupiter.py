"""Jupiter DEX aggregator integration for Solana swaps.

Handles quoting and swap execution through Jupiter's public API.
All swaps go through the governance layer before execution.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from src.solana.tokens import get_mint, get_symbol, USDC, WSOL


JUPITER_API = os.environ.get("JUPITER_API_URL", "https://quote-api.jup.ag/v6")


@dataclass
class SwapQuote:
    """A quote from Jupiter for a token swap."""
    input_mint: str
    output_mint: str
    input_symbol: str
    output_symbol: str
    input_amount: int  # in smallest unit (lamports/token decimals)
    output_amount: int  # expected output in smallest unit
    output_amount_ui: float  # human-readable output amount
    price_impact_pct: float
    slippage_bps: int
    route_plan: list[dict]
    raw_quote: dict  # full Jupiter response for swap execution


@dataclass
class SwapResult:
    """Result of an executed swap."""
    success: bool
    tx_signature: str | None = None
    input_symbol: str = ""
    output_symbol: str = ""
    input_amount: float = 0.0
    output_amount: float = 0.0
    price_impact_pct: float = 0.0
    error: str | None = None
    timestamp: str = ""


# Token decimals for UI amount conversion
TOKEN_DECIMALS: dict[str, int] = {
    "SOL": 9,
    "USDC": 6,
    "USDT": 6,
    "BONK": 5,
    "JUP": 6,
    "WIF": 6,
    "JTO": 9,
    "PYTH": 6,
    "RAY": 6,
    "ORCA": 6,
    "MSOL": 9,
    "ANSEM": 6,
}


def _get_decimals(symbol: str) -> int:
    return TOKEN_DECIMALS.get(symbol.upper(), 6)


def to_smallest_unit(amount: float, symbol: str) -> int:
    """Convert a human-readable amount to smallest unit."""
    return int(amount * (10 ** _get_decimals(symbol)))


def from_smallest_unit(amount: int, symbol: str) -> float:
    """Convert smallest unit to human-readable amount."""
    return amount / (10 ** _get_decimals(symbol))


def get_quote(
    input_symbol: str,
    output_symbol: str,
    amount: float,
    slippage_bps: int = 50,
) -> SwapQuote:
    """Get a swap quote from Jupiter.

    Args:
        input_symbol: Token to sell (e.g. "SOL", "USDC")
        output_symbol: Token to buy (e.g. "JUP", "BONK")
        amount: Amount to sell in human-readable units
        slippage_bps: Slippage tolerance in basis points (50 = 0.5%)
    """
    input_mint = get_mint(input_symbol)
    output_mint = get_mint(output_symbol)
    input_amount = to_smallest_unit(amount, input_symbol)

    resp = httpx.get(
        f"{JUPITER_API}/quote",
        params={
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(input_amount),
            "slippageBps": str(slippage_bps),
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    output_amount = int(data["outAmount"])

    return SwapQuote(
        input_mint=input_mint,
        output_mint=output_mint,
        input_symbol=input_symbol.upper(),
        output_symbol=output_symbol.upper(),
        input_amount=input_amount,
        output_amount=output_amount,
        output_amount_ui=from_smallest_unit(output_amount, output_symbol),
        price_impact_pct=float(data.get("priceImpactPct", 0)),
        slippage_bps=slippage_bps,
        route_plan=data.get("routePlan", []),
        raw_quote=data,
    )


def get_swap_transaction(
    quote: SwapQuote,
    user_pubkey: str,
) -> str:
    """Get a serialized swap transaction from Jupiter.

    Returns base64-encoded transaction ready to be signed and sent.
    """
    resp = httpx.post(
        f"{JUPITER_API}/swap",
        json={
            "quoteResponse": quote.raw_quote,
            "userPublicKey": user_pubkey,
            "wrapAndUnwrapSol": True,
            "dynamicComputeUnitLimit": True,
            "prioritizationFeeLamports": "auto",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["swapTransaction"]


def execute_swap(
    input_symbol: str,
    output_symbol: str,
    amount: float,
    user_pubkey: str,
    sign_and_send_fn: Any = None,
    slippage_bps: int = 50,
    dry_run: bool = False,
) -> SwapResult:
    """Execute a full swap: quote → transaction → sign → send.

    Args:
        input_symbol: Token to sell
        output_symbol: Token to buy
        amount: Amount to sell (human-readable)
        user_pubkey: Wallet public key
        sign_and_send_fn: Callable(base64_tx) -> tx_signature
            If None, returns the quote without executing (dry run).
        slippage_bps: Slippage tolerance in basis points
        dry_run: If True, get quote but don't execute
    """
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        quote = get_quote(input_symbol, output_symbol, amount, slippage_bps)
    except Exception as e:
        return SwapResult(
            success=False,
            error=f"Quote failed: {e}",
            input_symbol=input_symbol,
            output_symbol=output_symbol,
            timestamp=timestamp,
        )

    if dry_run or sign_and_send_fn is None:
        return SwapResult(
            success=True,
            input_symbol=quote.input_symbol,
            output_symbol=quote.output_symbol,
            input_amount=amount,
            output_amount=quote.output_amount_ui,
            price_impact_pct=quote.price_impact_pct,
            timestamp=timestamp,
        )

    try:
        swap_tx = get_swap_transaction(quote, user_pubkey)
        tx_sig = sign_and_send_fn(swap_tx)
        return SwapResult(
            success=True,
            tx_signature=tx_sig,
            input_symbol=quote.input_symbol,
            output_symbol=quote.output_symbol,
            input_amount=amount,
            output_amount=quote.output_amount_ui,
            price_impact_pct=quote.price_impact_pct,
            timestamp=timestamp,
        )
    except Exception as e:
        return SwapResult(
            success=False,
            error=f"Swap execution failed: {e}",
            input_symbol=quote.input_symbol,
            output_symbol=quote.output_symbol,
            input_amount=amount,
            timestamp=timestamp,
        )
