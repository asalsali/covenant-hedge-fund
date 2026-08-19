"""Covenant Hedge Fund API — investor-facing endpoints.

Endpoints:
    GET  /fund                — Fund summary (NAV, performance, allocations)
    GET  /fund/performance    — Detailed performance metrics
    GET  /fund/positions      — Current positions and allocations
    GET  /fund/governance     — Governance rules and recent decisions
    GET  /fund/trades         — Recent trade history
    GET  /fund/nav-history    — NAV snapshots for charting
    POST /fund/deposit        — Deposit USDC (requires wallet signature)
    POST /fund/withdraw       — Withdraw shares (requires wallet signature)
    GET  /fund/investor/:addr — Investor position details
    GET  /health              — Health check

Run:
    python -m src.fund.api
"""

from __future__ import annotations

import os
import sys

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.fund.core import Fund, FundConfig
from src.solana.governor import Governor


app = FastAPI(
    title="Covenant Governed Hedge Fund",
    description="AI-governed hedge fund on Solana — every trade passes through 8 governance rules",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize fund and governor
fund = Fund()
governor = Governor()


# ------------------------------------------------------------------
# Request/Response models
# ------------------------------------------------------------------

class DepositRequest(BaseModel):
    address: str = Field(..., description="Investor wallet address")
    amount: float = Field(..., gt=0, description="USDC amount to deposit")


class WithdrawRequest(BaseModel):
    address: str = Field(..., description="Investor wallet address")
    shares: float | None = Field(None, description="Shares to withdraw (default: all)")


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "fund": fund.config.name, "nav": fund.nav}


@app.get("/fund")
def fund_summary():
    """Full fund summary — NAV, performance, allocations, fees."""
    # Get current prices for accurate valuation
    try:
        from src.solana.data import get_token_prices
        prices = get_token_prices(fund.config.tokens)
    except Exception:
        prices = {}

    return fund.summary(prices)


@app.get("/fund/performance")
def fund_performance():
    """Detailed performance metrics."""
    return fund.get_performance()


@app.get("/fund/positions")
def fund_positions():
    """Current positions and allocations."""
    try:
        from src.solana.data import get_token_prices
        prices = get_token_prices(fund.config.tokens)
    except Exception:
        prices = {}

    return {
        "positions": fund.positions,
        "cash_usdc": fund.cash_usdc,
        "allocations": fund.get_allocations(prices),
        "total_value": fund.total_value(prices),
    }


@app.get("/fund/governance")
def fund_governance():
    """Governance rules and recent decisions."""
    return {
        "rules": governor.get_rules_summary(),
        "recent_decisions": [
            {
                "timestamp": d.timestamp,
                "action": d.action,
                "trade": f"{d.input_symbol} -> {d.output_symbol}",
                "amount": d.amount,
                "reasoning": d.reasoning,
            }
            for d in governor.get_recent_decisions(20)
        ],
        "total_decisions": len(governor.decisions),
    }


@app.get("/fund/trades")
def fund_trades():
    """Recent trade history."""
    return {"trades": fund.trades[-50:]}


@app.get("/fund/nav-history")
def fund_nav_history():
    """NAV snapshots for charting."""
    return {
        "snapshots": [
            {"timestamp": s.timestamp, "nav": s.nav, "total_value": s.total_value}
            for s in fund.snapshots[-200:]
        ]
    }


@app.post("/fund/deposit")
def deposit(req: DepositRequest):
    """Deposit USDC into the fund."""
    result = fund.deposit(req.address, req.amount)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/fund/withdraw")
def withdraw(req: WithdrawRequest):
    """Withdraw shares from the fund."""
    result = fund.withdraw(req.address, req.shares)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/fund/investor/{address}")
def investor_details(address: str):
    """Get investor position details."""
    if address not in fund.investors:
        raise HTTPException(status_code=404, detail="Investor not found")

    inv = fund.investors[address]
    current_value = inv.shares * fund.nav
    pnl = current_value - inv.deposited_usdc

    return {
        "address": inv.address,
        "shares": inv.shares,
        "deposited_usdc": inv.deposited_usdc,
        "current_value": current_value,
        "pnl": pnl,
        "pnl_pct": (pnl / inv.deposited_usdc * 100) if inv.deposited_usdc > 0 else 0,
        "joined_at": inv.joined_at,
    }


# ------------------------------------------------------------------
# Serve dashboard static files
# ------------------------------------------------------------------

DASHBOARD_DIR = Path(__file__).resolve().parent.parent.parent / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    import uvicorn
    port = int(os.environ.get("FUND_PORT", 8000))
    print(f"Covenant Governed Hedge Fund API")
    print(f"  URL:       http://localhost:{port}")
    print(f"  Dashboard: http://localhost:{port}/dashboard/")
    print(f"  Docs:      http://localhost:{port}/docs")
    print(f"  Fund:      {fund.config.name}")
    print(f"  NAV:       ${fund.nav:.4f}")
    print(f"  Tokens:    {', '.join(fund.config.tokens)}")
    print()
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
