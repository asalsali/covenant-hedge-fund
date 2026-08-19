"""Solana transaction signing and submission.

Handles keypair loading, transaction signing, and RPC submission.
Private keys are loaded from environment variable or encrypted file only.
NEVER logged, printed, or stored in the repo.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import httpx
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction

from src.solana.wallet import get_rpc_url


def load_keypair(
    private_key_env: str = "SOLANA_PRIVATE_KEY",
    keyfile_path: str | None = None,
) -> Keypair:
    """Load a Solana keypair from environment variable or keyfile.

    Priority:
    1. Environment variable (base58-encoded private key)
    2. Keyfile path (JSON array of bytes, same as Solana CLI format)

    Raises ValueError if no key source is available.
    """
    # Try environment variable first (base58 secret key)
    env_key = os.environ.get(private_key_env)
    if env_key:
        return Keypair.from_base58_string(env_key)

    # Try keyfile (Solana CLI format: JSON array of 64 bytes)
    if keyfile_path:
        path = Path(keyfile_path)
    else:
        # Default: ~/.config/solana/id.json
        path = Path.home() / ".config" / "solana" / "id.json"

    if path.exists():
        key_bytes = json.loads(path.read_text(encoding="utf-8"))
        return Keypair.from_bytes(bytes(key_bytes))

    raise ValueError(
        f"No Solana keypair found. Set {private_key_env} environment variable "
        f"or create a keyfile at {path}"
    )


def sign_and_send(
    serialized_tx_b64: str,
    keypair: Keypair | None = None,
    devnet: bool = False,
) -> str:
    """Sign a base64-encoded transaction and submit to the network.

    Args:
        serialized_tx_b64: Base64-encoded transaction from Jupiter
        keypair: Solana keypair for signing. If None, loads from env/keyfile.
        devnet: Use devnet RPC if True.

    Returns:
        Transaction signature as a string.
    """
    if keypair is None:
        keypair = load_keypair()

    # Decode the transaction
    tx_bytes = base64.b64decode(serialized_tx_b64)
    tx = VersionedTransaction.from_bytes(tx_bytes)

    # Sign the transaction
    signed_tx = VersionedTransaction(tx.message, [keypair])

    # Submit via RPC (using httpx, no async dependency)
    rpc_url = get_rpc_url(devnet)
    signed_bytes = bytes(signed_tx)
    encoded = base64.b64encode(signed_bytes).decode("utf-8")

    resp = httpx.post(
        rpc_url,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendTransaction",
            "params": [
                encoded,
                {"encoding": "base64", "skipPreflight": False},
            ],
        },
        timeout=30,
    )
    result = resp.json()

    if "error" in result:
        raise RuntimeError(f"RPC error: {result['error']}")

    return str(result["result"])


def make_sign_and_send_fn(
    keypair: Keypair | None = None,
    devnet: bool = False,
):
    """Create a sign_and_send callable for use with jupiter.execute_swap().

    Usage:
        from src.solana.signer import load_keypair, make_sign_and_send_fn
        kp = load_keypair()
        fn = make_sign_and_send_fn(kp)
        result = execute_swap("USDC", "SOL", 10.0, str(kp.pubkey()), sign_and_send_fn=fn)
    """
    if keypair is None:
        keypair = load_keypair()

    def _fn(serialized_tx_b64: str) -> str:
        return sign_and_send(serialized_tx_b64, keypair, devnet)

    return _fn
