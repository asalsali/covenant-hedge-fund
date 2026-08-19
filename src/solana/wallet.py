"""Solana wallet management for the governed trading agent.

Handles keypair loading, balance checking, and transaction signing.
Private keys are NEVER logged or stored in the repo — loaded from
environment or encrypted keyfile only.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import httpx

from src.solana.tokens import WSOL, USDC, TOKEN_MINTS, get_symbol


# Solana RPC endpoint
RPC_URL = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

# Devnet for testing
DEVNET_RPC_URL = "https://api.devnet.solana.com"


def get_rpc_url(devnet: bool = False) -> str:
    """Get the appropriate RPC URL."""
    if devnet:
        return DEVNET_RPC_URL
    return RPC_URL


def get_sol_balance(pubkey: str, devnet: bool = False) -> float:
    """Get SOL balance for a wallet address."""
    rpc = get_rpc_url(devnet)
    resp = httpx.post(rpc, json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getBalance",
        "params": [pubkey],
    }, timeout=10)
    result = resp.json()
    if "result" in result:
        return result["result"]["value"] / 1e9  # lamports to SOL
    return 0.0


def get_token_balances(pubkey: str, devnet: bool = False) -> dict[str, float]:
    """Get all SPL token balances for a wallet."""
    rpc = get_rpc_url(devnet)
    resp = httpx.post(rpc, json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenAccountsByOwner",
        "params": [
            pubkey,
            {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
            {"encoding": "jsonParsed"},
        ],
    }, timeout=10)
    result = resp.json()
    balances: dict[str, float] = {}
    if "result" in result:
        for account in result["result"]["value"]:
            info = account["account"]["data"]["parsed"]["info"]
            mint = info["mint"]
            amount = float(info["tokenAmount"]["uiAmount"] or 0)
            if amount > 0:
                symbol = get_symbol(mint)
                balances[symbol] = amount
    return balances


def get_all_balances(pubkey: str, devnet: bool = False) -> dict[str, float]:
    """Get SOL + all SPL token balances."""
    balances = {"SOL": get_sol_balance(pubkey, devnet)}
    balances.update(get_token_balances(pubkey, devnet))
    return {k: v for k, v in balances.items() if v > 0}
