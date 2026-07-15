"""Pre-computed fundamentals snapshot for LLM analysts.

LLMs are bad at arithmetic. This module pre-computes financial
aggregates in Python so the LLM receives ready-made ratios, trends,
and growth rates instead of raw data it would need to calculate itself.

The snapshot is built from data ALREADY fetched by the data layer --
no new API calls. It replaces the raw JSON dump that value analysts
previously received.

Inspired by virattt/ai-hedge-fund v2's FundamentalsSnapshot pattern.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


# ---------------------------------------------------------------------------
# Sector WACC reference tables
# ---------------------------------------------------------------------------
# Static data mapping sectors to weighted-average cost of capital ranges.
# Used by DCF/valuation analysts to apply sector-appropriate discount rates
# instead of a blanket 10% for everything.
#
# Sources: Damodaran's annual industry WACC tables, cross-referenced with
# dexter project sector mappings. Ranges reflect typical spread for
# established companies in each sector.

SECTOR_WACC: dict[str, dict[str, float]] = {
    "Technology":             {"low": 0.09, "mid": 0.105, "high": 0.12},
    "Healthcare":             {"low": 0.08, "mid": 0.095, "high": 0.11},
    "Financial Services":     {"low": 0.07, "mid": 0.085, "high": 0.10},
    "Financials":             {"low": 0.07, "mid": 0.085, "high": 0.10},
    "Consumer Discretionary": {"low": 0.08, "mid": 0.095, "high": 0.11},
    "Consumer Staples":       {"low": 0.06, "mid": 0.075, "high": 0.09},
    "Energy":                 {"low": 0.09, "mid": 0.11,  "high": 0.13},
    "Utilities":              {"low": 0.05, "mid": 0.06,  "high": 0.07},
    "Industrials":            {"low": 0.08, "mid": 0.095, "high": 0.11},
    "Materials":              {"low": 0.08, "mid": 0.095, "high": 0.11},
    "Real Estate":            {"low": 0.06, "mid": 0.075, "high": 0.09},
    "Communication Services": {"low": 0.08, "mid": 0.095, "high": 0.11},
    "Information Technology":  {"low": 0.09, "mid": 0.105, "high": 0.12},
}

# Default WACC when sector is unknown or unmapped
_DEFAULT_WACC: dict[str, float] = {"low": 0.08, "mid": 0.10, "high": 0.12}


def get_sector_wacc(sector: str | None) -> dict[str, float]:
    """Return the WACC range for a sector.

    Args:
        sector: Sector name (e.g., "Technology", "Healthcare").
                Case-insensitive partial matching is attempted.

    Returns:
        Dict with "low", "mid", "high" WACC values as decimals.
        Returns default range (8-12%) if sector is unknown.
    """
    if not sector:
        return _DEFAULT_WACC.copy()

    # Exact match
    if sector in SECTOR_WACC:
        return SECTOR_WACC[sector].copy()

    # Case-insensitive match
    sector_lower = sector.lower()
    for key, wacc in SECTOR_WACC.items():
        if key.lower() == sector_lower:
            return wacc.copy()

    # Partial match (sector name contained in key or vice versa)
    for key, wacc in SECTOR_WACC.items():
        if sector_lower in key.lower() or key.lower() in sector_lower:
            return wacc.copy()

    return _DEFAULT_WACC.copy()


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_pct(val: Any) -> str | None:
    """Format a decimal as a percentage string for LLM readability."""
    v = _safe_float(val)
    if v is None:
        return None
    return f"{v * 100:.1f}%"


def _compute_cagr(start: float, end: float, periods: int) -> float | None:
    """Compute compound annual growth rate.

    Args:
        start: Starting value (oldest period).
        end: Ending value (most recent period).
        periods: Number of periods between start and end.

    Returns:
        CAGR as a decimal, or None if computation is impossible.
    """
    if periods <= 0 or start <= 0 or end <= 0:
        return None
    try:
        return (end / start) ** (1.0 / periods) - 1.0
    except (ZeroDivisionError, OverflowError, ValueError):
        return None


def _compute_average(values: list[float | None]) -> float | None:
    """Compute average of non-None values."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _compute_trend(values: list[float | None]) -> float | None:
    """Compute trend as latest minus oldest (directional change).

    Values are ordered most-recent-first (as returned by the data layer).
    """
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return None
    return valid[0] - valid[-1]


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

class FundamentalsSnapshot:
    """Pre-computed financial aggregates for a single ticker.

    Built from data already fetched by the data layer. All arithmetic
    is done here in Python so the LLM receives computed values, not
    raw numbers it would need to calculate.

    Attributes:
        ticker: The stock ticker symbol.
        metrics: Dict of pre-computed financial metrics.
        content_hash: SHA-256 hash of the input data, usable as cache key.
    """

    def __init__(self, ticker: str, metrics: dict[str, Any], raw_hash: str) -> None:
        self.ticker = ticker
        self.metrics = metrics
        self._content_hash = raw_hash

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the input data. Doubles as a cache key."""
        return self._content_hash

    def render(self) -> str:
        """Format the snapshot as a compact text block for LLM prompts.

        Returns a human-readable table that replaces the raw JSON dump
        previously sent to value analysts.
        """
        m = self.metrics
        lines = [
            f"=== {self.ticker} Fundamentals Snapshot ===",
            "",
        ]

        # Price
        if m.get("current_price") is not None:
            line = f"  Price:          ${m['current_price']:.2f}"
            if m.get("price_change_pct") is not None:
                line += f"  ({m['price_change_pct']:+.1f}% period)"
            lines.append(line)

        # Valuation
        val_parts = []
        if m.get("price_to_earnings") is not None:
            val_parts.append(f"P/E {m['price_to_earnings']:.1f}")
        if m.get("price_to_book") is not None:
            val_parts.append(f"P/B {m['price_to_book']:.1f}")
        if m.get("ev_to_ebitda") is not None:
            val_parts.append(f"EV/EBITDA {m['ev_to_ebitda']:.1f}")
        if val_parts:
            lines.append(f"  Valuation:      {' | '.join(val_parts)}")

        # Market cap
        if m.get("market_cap") is not None:
            cap = m["market_cap"]
            if cap >= 1e12:
                lines.append(f"  Market Cap:     ${cap / 1e12:.1f}T")
            elif cap >= 1e9:
                lines.append(f"  Market Cap:     ${cap / 1e9:.1f}B")
            elif cap >= 1e6:
                lines.append(f"  Market Cap:     ${cap / 1e6:.0f}M")
            else:
                lines.append(f"  Market Cap:     ${cap:,.0f}")

        # Profitability
        prof_parts = []
        if m.get("roe_avg") is not None:
            prof_parts.append(f"ROE avg {m['roe_avg']}")
        if m.get("net_margin_avg") is not None:
            prof_parts.append(f"Net margin avg {m['net_margin_avg']}")
        if m.get("gross_margin_latest") is not None:
            prof_parts.append(f"Gross margin {m['gross_margin_latest']}")
        if prof_parts:
            lines.append(f"  Profitability:  {' | '.join(prof_parts)}")

        # Trends
        if m.get("gross_margin_trend") is not None:
            direction = "improving" if m["gross_margin_trend"] > 0 else "declining"
            lines.append(
                f"  Margin Trend:   {m['gross_margin_trend']:+.1f}pp ({direction})"
            )

        # Growth
        growth_parts = []
        if m.get("revenue_cagr") is not None:
            growth_parts.append(f"Rev CAGR {m['revenue_cagr']}")
        if m.get("earnings_growth") is not None:
            growth_parts.append(f"Earnings growth {m['earnings_growth']}")
        if growth_parts:
            lines.append(f"  Growth:         {' | '.join(growth_parts)}")

        # Balance sheet
        bs_parts = []
        if m.get("debt_to_equity") is not None:
            bs_parts.append(f"D/E {m['debt_to_equity']:.2f}")
        if m.get("current_ratio") is not None:
            bs_parts.append(f"Current ratio {m['current_ratio']:.2f}")
        if bs_parts:
            lines.append(f"  Balance Sheet:  {' | '.join(bs_parts)}")

        # WACC (for DCF context)
        if m.get("sector_wacc") is not None:
            w = m["sector_wacc"]
            lines.append(
                f"  Sector WACC:    {w['low']:.0%}-{w['high']:.0%} "
                f"(mid {w['mid']:.0%})"
            )

        # Cash flows (last 3 periods, pre-formatted)
        if m.get("cash_flow_summary"):
            lines.append("  Cash Flows:")
            for cf in m["cash_flow_summary"]:
                period = cf.get("period", "?")
                fcf = cf.get("free_cash_flow")
                rev = cf.get("revenue")
                parts = [f"    {period}:"]
                if rev is not None:
                    parts.append(f"Rev ${rev / 1e9:.1f}B" if abs(rev) >= 1e9
                                 else f"Rev ${rev / 1e6:.0f}M")
                if fcf is not None:
                    parts.append(f"FCF ${fcf / 1e9:.1f}B" if abs(fcf) >= 1e9
                                 else f"FCF ${fcf / 1e6:.0f}M")
                lines.append(" ".join(parts))

        # Data quality
        period_count = m.get("periods_available", 0)
        lines.append(f"  Data:           {period_count} periods available")

        lines.append("")
        lines.append("All ratios pre-computed. Do not recalculate.")

        return "\n".join(lines)


def build_snapshot(
    ticker: str,
    data: dict[str, Any],
    sector: str | None = None,
) -> FundamentalsSnapshot:
    """Build a pre-computed fundamentals snapshot from fetched data.

    Args:
        ticker: Stock ticker symbol.
        data: Market data dict as structured by main.py, with keys:
              "prices", "financial_metrics", "line_items",
              optionally "insider_trades".
        sector: Optional sector name for WACC lookup. If None,
                default WACC range is used.

    Returns:
        FundamentalsSnapshot with all aggregates pre-computed.
    """
    # Compute content hash for caching
    raw_hash = hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    prices = data.get("prices", [])
    metrics_list = data.get("financial_metrics", [])
    line_items = data.get("line_items", [])

    cur = metrics_list[0] if metrics_list else {}

    # --- Price ---
    closes = [p["close"] for p in prices if p.get("close") is not None]
    current_price = closes[-1] if closes else None
    price_start = closes[0] if closes else None
    price_change_pct = None
    if current_price and price_start and price_start > 0:
        price_change_pct = round((current_price / price_start - 1) * 100, 2)

    # --- ROE average across periods ---
    roe_values = [_safe_float(m.get("return_on_equity")) for m in metrics_list]
    roe_avg = _compute_average(roe_values)

    # --- Net margin average ---
    net_margin_values = [_safe_float(m.get("net_margin")) for m in metrics_list]
    net_margin_avg = _compute_average(net_margin_values)

    # --- Gross margin: latest and trend ---
    gross_margin_values = [_safe_float(m.get("gross_margin")) for m in metrics_list]
    gross_margin_latest = _safe_float(cur.get("gross_margin"))
    gross_margin_trend = _compute_trend(gross_margin_values)
    # Convert to percentage points for readability
    if gross_margin_trend is not None:
        gross_margin_trend = round(gross_margin_trend * 100, 1)

    # --- Revenue CAGR ---
    revenues = [_safe_float(li.get("revenue")) for li in line_items]
    revenues_valid = [r for r in revenues if r is not None and r > 0]
    revenue_cagr = None
    if len(revenues_valid) >= 2:
        # line_items are most-recent-first
        revenue_cagr = _compute_cagr(
            start=revenues_valid[-1],
            end=revenues_valid[0],
            periods=len(revenues_valid) - 1,
        )

    # --- Valuation multiples ---
    pe = _safe_float(cur.get("price_to_earnings"))
    ev_ebitda = _safe_float(cur.get("ev_to_ebitda"))

    # P/B: compute from price and book value if available
    # yfinance doesn't always provide P/B directly, but we can derive it
    # from price_to_earnings and return_on_equity: P/B = P/E * ROE
    pb = None
    if pe is not None and roe_avg is not None and roe_avg > 0:
        pb = round(pe * roe_avg, 2)

    # --- Balance sheet ---
    de = _safe_float(cur.get("debt_to_equity"))
    cr = _safe_float(cur.get("current_ratio"))

    # --- Market cap ---
    market_cap = _safe_float(cur.get("market_cap"))

    # --- Earnings growth ---
    earnings_growth = _safe_float(cur.get("earnings_growth"))

    # --- WACC ---
    sector_wacc = get_sector_wacc(sector)

    # --- Cash flow summary (last 3 periods, pre-formatted) ---
    cash_flow_summary = []
    for li in line_items[:3]:
        entry: dict[str, Any] = {
            "period": li.get("period_end_date") or li.get("report_period"),
        }
        rev = _safe_float(li.get("revenue"))
        if rev is not None:
            entry["revenue"] = rev
        fcf = _safe_float(li.get("free_cash_flow"))
        if fcf is not None:
            entry["free_cash_flow"] = fcf
        ocf = _safe_float(li.get("operating_cash_flow"))
        if ocf is not None:
            entry["operating_cash_flow"] = ocf
        ni = _safe_float(li.get("net_income"))
        if ni is not None:
            entry["net_income"] = ni
        cash_flow_summary.append(entry)

    # --- Assemble metrics dict ---
    computed: dict[str, Any] = {
        "current_price": current_price,
        "price_change_pct": price_change_pct,
        "roe_avg": _safe_pct(roe_avg),
        "net_margin_avg": _safe_pct(net_margin_avg),
        "gross_margin_latest": _safe_pct(gross_margin_latest),
        "gross_margin_trend": gross_margin_trend,
        "revenue_cagr": _safe_pct(revenue_cagr),
        "debt_to_equity": de,
        "current_ratio": cr,
        "price_to_earnings": pe,
        "price_to_book": pb,
        "ev_to_ebitda": ev_ebitda,
        "market_cap": market_cap,
        "earnings_growth": _safe_pct(earnings_growth),
        "sector_wacc": sector_wacc,
        "cash_flow_summary": cash_flow_summary,
        "periods_available": len(metrics_list),
    }

    return FundamentalsSnapshot(ticker, computed, raw_hash)
