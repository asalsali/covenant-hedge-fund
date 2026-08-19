"""On-chain memo logging for governance decisions.

Posts governance decisions to the Solana blockchain as memo transactions.
Every approve/reject is permanently recorded on-chain — full transparency.

Memo Program ID: MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solders.hash import Hash

from src.solana.wallet import get_rpc_url


# SPL Memo Program v2
MEMO_PROGRAM_ID = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")


def _get_recent_blockhash(devnet: bool = False) -> Hash:
    """Fetch the latest blockhash from RPC."""
    rpc_url = get_rpc_url(devnet)
    resp = httpx.post(
        rpc_url,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getLatestBlockhash",
            "params": [{"commitment": "finalized"}],
        },
        timeout=10,
    )
    result = resp.json()
    blockhash_str = result["result"]["value"]["blockhash"]
    return Hash.from_string(blockhash_str)


def build_memo_instruction(signer: Pubkey, message: str) -> Instruction:
    """Build a Memo program instruction.

    The memo is UTF-8 text, max ~566 bytes to fit in a transaction.
    """
    # Truncate to fit in transaction
    msg_bytes = message.encode("utf-8")[:500]

    return Instruction(
        program_id=MEMO_PROGRAM_ID,
        accounts=[AccountMeta(pubkey=signer, is_signer=True, is_writable=False)],
        data=msg_bytes,
    )


def post_governance_memo(
    keypair: Keypair,
    decision: dict[str, Any],
    devnet: bool = False,
) -> str | None:
    """Post a governance decision to the Solana blockchain as a memo.

    Args:
        keypair: Signer keypair
        decision: Governance decision dict with action, reasoning, etc.
        devnet: Use devnet if True

    Returns:
        Transaction signature, or None if failed.
    """
    # Build compact memo text
    memo_text = _format_governance_memo(decision)

    try:
        # Build instruction
        ix = build_memo_instruction(keypair.pubkey(), memo_text)

        # Get recent blockhash
        blockhash = _get_recent_blockhash(devnet)

        # Build versioned message
        msg = MessageV0.try_compile(
            payer=keypair.pubkey(),
            instructions=[ix],
            address_lookup_table_accounts=[],
            recent_blockhash=blockhash,
        )

        # Sign
        tx = VersionedTransaction(msg, [keypair])

        # Submit
        rpc_url = get_rpc_url(devnet)
        encoded = base64.b64encode(bytes(tx)).decode("utf-8")

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
            return None

        return str(result["result"])

    except Exception:
        return None


def _format_governance_memo(decision: dict[str, Any]) -> str:
    """Format a governance decision as a compact memo string.

    Format: CVT|<action>|<input>→<output>|<amount>|<rules_violated or OK>|<timestamp>
    """
    action = decision.get("action", "?")
    input_sym = decision.get("input_symbol", "?")
    output_sym = decision.get("output_symbol", "?")
    amount = decision.get("amount", 0)
    timestamp = decision.get("timestamp", "")
    violated = decision.get("rules_violated", [])

    if violated:
        rules_str = ",".join(v.split(":")[0] for v in violated)  # Just rule IDs
    else:
        rules_str = "OK"

    return f"CVT|{action}|{input_sym}>{output_sym}|{amount}|{rules_str}|{timestamp[:19]}"


def post_batch_memos(
    keypair: Keypair,
    decisions: list[dict[str, Any]],
    devnet: bool = False,
) -> list[str | None]:
    """Post multiple governance decisions as memos.

    Returns list of transaction signatures (None for failures).
    """
    results = []
    for decision in decisions:
        sig = post_governance_memo(keypair, decision, devnet)
        results.append(sig)
        if sig:
            time.sleep(0.5)  # Rate limit
    return results
